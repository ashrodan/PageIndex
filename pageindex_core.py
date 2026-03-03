# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "openai>=1.0", "pymupdf>=1.24", "PyPDF2>=3.0",
#   "tiktoken>=0.5", "python-dotenv>=1.0", "pyyaml>=6.0", "rich>=13.0",
# ]
# ///
"""pageindex_core.py — self-contained single-file distillation of the PageIndex package.

Consolidates pageindex/utils.py, pageindex/page_index.py, and pageindex/page_index_md.py
into one runnable script.  Drop-in replacement for the `pageindex` CLI.

Usage:
    uv run pageindex_core.py --pdf_path doc.pdf
    uv run pageindex_core.py --md_path doc.md
    uv run pageindex_core.py --help
"""

import os
import sys
import json
import copy
import math
import random
import re
import asyncio
import time
import logging
import argparse
from datetime import datetime
from io import BytesIO
from types import SimpleNamespace as config
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
import tiktoken
import PyPDF2
import pymupdf
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("CHATGPT_API_KEY")


# ---------------------------------------------------------------------------
# 1. LLM helpers — single implementation, three public aliases
# ---------------------------------------------------------------------------

def _call_llm(model, prompt, api_key=None, *, chat_history=None, return_finish_reason=False):
    """Synchronous LLM call with exponential-backoff retry (up to 10 attempts)."""
    _key = api_key or API_KEY
    messages = [*(chat_history or []), {"role": "user", "content": prompt}]
    client = openai.OpenAI(api_key=_key)
    for i in range(10):
        try:
            resp = client.chat.completions.create(model=model, messages=messages, temperature=0)
            content = resp.choices[0].message.content
            if return_finish_reason:
                finish = "max_output_reached" if resp.choices[0].finish_reason == "length" else "finished"
                return content, finish
            return content
        except Exception as e:
            print("************* Retrying *************")
            logging.error(f"LLM error (attempt {i+1}): {e}")
            if i < 9:
                time.sleep(min(2 ** i + random.uniform(0, 1), 60))
            else:
                logging.error("Max retries reached for prompt: " + prompt)
    return ("Error", "error") if return_finish_reason else "Error"


async def _call_llm_async(model, prompt, api_key=None):
    """Async LLM call with exponential-backoff retry (up to 10 attempts)."""
    _key = api_key or API_KEY
    messages = [{"role": "user", "content": prompt}]
    client = openai.AsyncOpenAI(api_key=_key)
    for i in range(10):
        try:
            resp = await client.chat.completions.create(model=model, messages=messages, temperature=0)
            return resp.choices[0].message.content
        except Exception as e:
            print("************* Retrying *************")
            logging.error(f"LLM error (attempt {i+1}): {e}")
            if i < 9:
                await asyncio.sleep(min(2 ** i + random.uniform(0, 1), 60))
            else:
                logging.error("Max retries reached for prompt: " + prompt)
    return "Error"


# Public aliases matching original names
ChatGPT_API = _call_llm
ChatGPT_API_async = _call_llm_async


def ChatGPT_API_with_finish_reason(model, prompt, api_key=None, chat_history=None):
    return _call_llm(model, prompt, api_key, chat_history=chat_history, return_finish_reason=True)


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def count_tokens(text, model=None):
    if not text:
        return 0
    enc = tiktoken.encoding_for_model(model or "gpt-4o")
    return len(enc.encode(text))


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def get_json_content(response):
    start_idx = response.find("```json")
    if start_idx != -1:
        start_idx += 7
        response = response[start_idx:]
    end_idx = response.rfind("```")
    if end_idx != -1:
        response = response[:end_idx]
    return response.strip()


def extract_json(content):
    try:
        start_idx = content.find("```json")
        if start_idx != -1:
            start_idx += 7
            end_idx = content.rfind("```")
            json_content = content[start_idx:end_idx].strip()
        else:
            json_content = content.strip()
        json_content = json_content.replace('None', 'null')
        json_content = json_content.replace('\n', ' ').replace('\r', ' ')
        json_content = ' '.join(json_content.split())
        return json.loads(json_content)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to extract JSON: {e}")
        try:
            json_content = json_content.replace(',]', ']').replace(',}', '}')
            return json.loads(json_content)
        except Exception:
            logging.error("Failed to parse JSON even after cleanup")
            return {}
    except Exception as e:
        logging.error(f"Unexpected error while extracting JSON: {e}")
        return {}


# ---------------------------------------------------------------------------
# 2. Tree walkers — single _walk generator, three public facades
# ---------------------------------------------------------------------------

def _walk(structure):
    """Yield every node dict in the tree (references, not copies)."""
    if isinstance(structure, dict):
        yield structure
        yield from _walk(structure.get('nodes') or [])
    else:
        for item in (structure or []):
            yield from _walk(item)


def structure_to_list(structure):
    return list(_walk(structure))


def write_node_id(data, node_id=0):
    if isinstance(data, dict):
        data['node_id'] = str(node_id).zfill(4)
        node_id += 1
        for key in list(data.keys()):
            if 'nodes' in key:
                node_id = write_node_id(data[key], node_id)
    elif isinstance(data, list):
        for index in range(len(data)):
            node_id = write_node_id(data[index], node_id)
    return node_id


# ---------------------------------------------------------------------------
# 4. Single find_all_children (used in markdown pipeline)
# ---------------------------------------------------------------------------

def _find_children(node_list, parent_index, parent_level):
    """Return indices of all descendants of node at parent_index."""
    return [
        i for i in range(parent_index + 1, len(node_list))
        if node_list[i]['level'] > parent_level
        and all(node_list[j]['level'] > parent_level
                for j in range(parent_index + 1, i + 1))
    ]


# ---------------------------------------------------------------------------
# PDF utilities
# ---------------------------------------------------------------------------

def sanitize_filename(filename, replacement='-'):
    return filename.replace('/', replacement)


def get_pdf_name(pdf_path):
    if isinstance(pdf_path, str):
        return os.path.basename(pdf_path)
    elif isinstance(pdf_path, BytesIO):
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        meta = pdf_reader.metadata
        pdf_name = meta.title if meta and meta.title else 'Untitled'
        return sanitize_filename(pdf_name)


def get_page_tokens(pdf_path, model="gpt-4o-2024-11-20", pdf_parser="PyPDF2"):
    enc = tiktoken.encoding_for_model(model)
    if pdf_parser == "PyPDF2":
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        page_list = []
        for page_num in range(len(pdf_reader.pages)):
            page_text = pdf_reader.pages[page_num].extract_text()
            page_list.append((page_text, len(enc.encode(page_text))))
        return page_list
    elif pdf_parser == "PyMuPDF":
        if isinstance(pdf_path, BytesIO):
            doc = pymupdf.open(stream=pdf_path, filetype="pdf")
        else:
            doc = pymupdf.open(pdf_path)
        page_list = []
        for page in doc:
            page_text = page.get_text()
            page_list.append((page_text, len(enc.encode(page_text))))
        return page_list
    else:
        raise ValueError(f"Unsupported PDF parser: {pdf_parser}")


# 3. Merged get_text_of_pdf_pages pair
def get_text_of_pdf_pages(pdf_pages, start_page, end_page, labels=False):
    text = ""
    for page_num in range(start_page - 1, end_page):
        if labels:
            text += f"<physical_index_{page_num+1}>\n{pdf_pages[page_num][0]}\n<physical_index_{page_num+1}>\n"
        else:
            text += pdf_pages[page_num][0]
    return text


def get_text_of_pdf_pages_with_labels(pdf_pages, start_page, end_page):
    return get_text_of_pdf_pages(pdf_pages, start_page, end_page, labels=True)


# 3. Merged add_node_text pair
def add_node_text(node, pdf_pages, labels=False):
    if isinstance(node, dict):
        start_page = node.get('start_index')
        end_page = node.get('end_index')
        node['text'] = get_text_of_pdf_pages(pdf_pages, start_page, end_page, labels=labels)
        if 'nodes' in node:
            add_node_text(node['nodes'], pdf_pages, labels=labels)
    elif isinstance(node, list):
        for index in range(len(node)):
            add_node_text(node[index], pdf_pages, labels=labels)


# ---------------------------------------------------------------------------
# Data-transformation utilities
# ---------------------------------------------------------------------------

def list_to_tree(data):
    def get_parent_structure(structure):
        if not structure:
            return None
        parts = str(structure).split('.')
        return '.'.join(parts[:-1]) if len(parts) > 1 else None

    nodes = {}
    root_nodes = []
    for item in data:
        structure = item.get('structure')
        node = {
            'title': item.get('title'),
            'start_index': item.get('start_index'),
            'end_index': item.get('end_index'),
            'nodes': [],
        }
        nodes[structure] = node
        parent_structure = get_parent_structure(structure)
        if parent_structure:
            if parent_structure in nodes:
                nodes[parent_structure]['nodes'].append(node)
            else:
                root_nodes.append(node)
        else:
            root_nodes.append(node)

    def clean_node(node):
        if not node['nodes']:
            del node['nodes']
        else:
            for child in node['nodes']:
                clean_node(child)
        return node

    return [clean_node(node) for node in root_nodes]


def add_preface_if_needed(data):
    if not isinstance(data, list) or not data:
        return data
    if data[0]['physical_index'] is not None and data[0]['physical_index'] > 1:
        data.insert(0, {"structure": "0", "title": "Preface", "physical_index": 1})
    return data


def post_processing(structure, end_physical_index):
    for i, item in enumerate(structure):
        item['start_index'] = item.get('physical_index')
        if i < len(structure) - 1:
            if structure[i + 1].get('appear_start') == 'yes':
                item['end_index'] = structure[i + 1]['physical_index'] - 1
            else:
                item['end_index'] = structure[i + 1]['physical_index']
        else:
            item['end_index'] = end_physical_index
    tree = list_to_tree(structure)
    if tree:
        return tree
    for node in structure:
        node.pop('appear_start', None)
        node.pop('physical_index', None)
    return structure


def remove_structure_text(data):
    if isinstance(data, dict):
        data.pop('text', None)
        if 'nodes' in data:
            remove_structure_text(data['nodes'])
    elif isinstance(data, list):
        for item in data:
            remove_structure_text(item)
    return data


def remove_page_number(data):
    if isinstance(data, dict):
        data.pop('page_number', None)
        for key in list(data.keys()):
            if 'nodes' in key:
                remove_page_number(data[key])
    elif isinstance(data, list):
        for item in data:
            remove_page_number(item)
    return data


def convert_physical_index_to_int(data):
    if isinstance(data, list):
        for i in range(len(data)):
            if isinstance(data[i], dict) and 'physical_index' in data[i]:
                if isinstance(data[i]['physical_index'], str):
                    val = data[i]['physical_index']
                    if val.startswith('<physical_index_'):
                        data[i]['physical_index'] = int(val.split('_')[-1].rstrip('>').strip())
                    elif val.startswith('physical_index_'):
                        data[i]['physical_index'] = int(val.split('_')[-1].strip())
    elif isinstance(data, str):
        if data.startswith('<physical_index_'):
            data = int(data.split('_')[-1].rstrip('>').strip())
        elif data.startswith('physical_index_'):
            data = int(data.split('_')[-1].strip())
        if isinstance(data, int):
            return data
        return None
    return data


def convert_page_to_int(data):
    for item in data:
        if 'page' in item and isinstance(item['page'], str):
            try:
                item['page'] = int(item['page'])
            except ValueError:
                pass
    return data


def reorder_dict(data, key_order):
    if not key_order:
        return data
    return {key: data[key] for key in key_order if key in data}


def format_structure(structure, order=None):
    if not order:
        return structure
    if isinstance(structure, dict):
        if 'nodes' in structure:
            structure['nodes'] = format_structure(structure['nodes'], order)
        if not structure.get('nodes'):
            structure.pop('nodes', None)
        structure = reorder_dict(structure, order)
    elif isinstance(structure, list):
        structure = [format_structure(item, order) for item in structure]
    return structure


def create_clean_structure_for_description(structure):
    if isinstance(structure, dict):
        clean_node = {}
        for key in ['title', 'node_id', 'summary', 'prefix_summary']:
            if key in structure:
                clean_node[key] = structure[key]
        if 'nodes' in structure and structure['nodes']:
            clean_node['nodes'] = create_clean_structure_for_description(structure['nodes'])
        return clean_node
    elif isinstance(structure, list):
        return [create_clean_structure_for_description(item) for item in structure]
    return structure


def generate_doc_description(structure, model=None):
    prompt = f"""Your are an expert in generating descriptions for a document.
    You are given a structure of a document. Your task is to generate a one-sentence description for the document, which makes it easy to distinguish the document from other documents.

    Document Structure: {structure}

    Directly return the description, do not include any other text.
    """
    return ChatGPT_API(model, prompt)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class JsonLogger:
    WRITE_THRESHOLD = 20

    def __init__(self, file_path):
        pdf_name = get_pdf_name(file_path)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"{pdf_name}_{current_time}.json"
        os.makedirs("./logs", exist_ok=True)
        self.log_data = []

    def log(self, level, message, **kwargs):
        self.log_data.append(message if isinstance(message, dict) else {'message': message})
        if len(self.log_data) % self.WRITE_THRESHOLD == 0:
            self._flush()

    def _flush(self):
        with open(self._filepath(), "w") as f:
            json.dump(self.log_data, f, indent=2)

    def close(self):
        self._flush()

    def __del__(self):
        try:
            self._flush()
        except Exception:
            pass

    def info(self, message, **kwargs):
        self.log("INFO", message, **kwargs)

    def error(self, message, **kwargs):
        self.log("ERROR", message, **kwargs)

    def debug(self, message, **kwargs):
        self.log("DEBUG", message, **kwargs)

    def exception(self, message, **kwargs):
        kwargs["exception"] = True
        self.log("ERROR", message, **kwargs)

    def _filepath(self):
        return os.path.join("logs", self.filename)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

async def generate_node_summary(node, model=None):
    prompt = f"""You are given a part of a document, your task is to generate a description of the partial document about what are main points covered in the partial document.

    Partial Document Text: {node['text']}

    Directly return the description, do not include any other text.
    """
    return await ChatGPT_API_async(model, prompt)


async def generate_summaries_for_structure(structure, model=None):
    nodes = structure_to_list(structure)
    tasks = [generate_node_summary(node, model=model) for node in nodes]
    summaries = await asyncio.gather(*tasks)
    for node, summary in zip(nodes, summaries):
        node['summary'] = summary
    return structure


# ---------------------------------------------------------------------------
# Config (inline defaults — no config.yaml dependency in single-file mode)
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = {
    "model": "gpt-4o-2024-11-20",
    "toc_check_page_num": 20,
    "max_page_num_each_node": 10,
    "max_token_num_each_node": 20000,
    "if_add_node_id": "yes",
    "if_add_node_summary": "yes",
    "if_add_doc_description": "no",
    "if_add_node_text": "no",
}


def _load_config(user_opt=None):
    if user_opt is None:
        user_dict = {}
    elif isinstance(user_opt, config):
        user_dict = vars(user_opt)
    elif isinstance(user_opt, dict):
        user_dict = user_opt
    else:
        raise TypeError("user_opt must be dict, config(SimpleNamespace) or None")
    unknown = set(user_dict) - set(_DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    return config(**{**_DEFAULT_CONFIG, **user_dict})


# ---------------------------------------------------------------------------
# 5. TOC extraction — merged generate_toc_init / generate_toc_continue
# ---------------------------------------------------------------------------

_TOC_EXTRACTION_PROMPT_BODY = """
    The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

    For the title, you need to extract the original title from the text, only fix the space inconsistency.

    The provided text contains tags like <physical_index_X> and <physical_index_X> to indicate the start and end of page X.
    For the physical_index, you need to extract the physical index of the start of the section from the text. Keep the <physical_index_X> format.

    The response should be in the following format:
        [
            {
                "structure": <structure index, "x.x.x"> (string),
                "title": <title of the section, keep the original title>,
                "physical_index": "<physical_index_X> (keep the format)"
            },
            ...
        ]"""


def _generate_toc_part(part, model, previous_structure=None):
    """Generate (init) or extend (continue) a TOC from a document chunk."""
    if previous_structure is None:
        header = "You are an expert in extracting hierarchical tree structure, your task is to generate the tree structure of the document."
        footer = "Directly return the final JSON structure. Do not output anything else."
    else:
        header = (
            "You are an expert in extracting hierarchical tree structure.\n"
            "    You are given a tree structure of the previous part and the text of the current part.\n"
            "    Your task is to continue the tree structure from the previous part to include the current part."
        )
        footer = "Directly return the additional part of the final JSON structure. Do not output anything else."

    prompt = header + "\n" + _TOC_EXTRACTION_PROMPT_BODY + "\n    " + footer
    prompt += '\nGiven text\n:' + part
    if previous_structure is not None:
        prompt += '\nPrevious tree structure\n:' + json.dumps(previous_structure, indent=2)

    response, finish_reason = _call_llm(model, prompt, return_finish_reason=True)
    if finish_reason != 'finished':
        raise Exception(f'finish reason: {finish_reason}')
    return extract_json(response)


def generate_toc_init(part, model=None):
    print('start generate_toc_init')
    return _generate_toc_part(part, model, previous_structure=None)


def generate_toc_continue(toc_content, part, model="gpt-4o-2024-11-20"):
    print('start generate_toc_continue')
    return _generate_toc_part(part, model, previous_structure=toc_content)


# ---------------------------------------------------------------------------
# TOC detection helpers
# ---------------------------------------------------------------------------

def toc_detector_single_page(content, model=None):
    prompt = f"""
    Your job is to detect if there is a table of content provided in the given text.

    Given text: {content}

    return the following JSON format:
    {{
        "thinking": <why do you think there is a table of content in the given text>
        "toc_detected": "<yes or no>",
    }}

    Directly return the final JSON structure. Do not output anything else.
    Please note: abstract,summary, notation list, figure list, table list, etc. are not table of contents."""

    response = ChatGPT_API(model=model, prompt=prompt)
    json_content = extract_json(response)
    return json_content['toc_detected']


def check_if_toc_extraction_is_complete(content, toc, model=None):
    prompt = f"""
    You are given a partial document  and a  table of contents.
    Your job is to check if the  table of contents is complete, which it contains all the main sections in the partial document.

    Reply format:
    {{
        "thinking": <why do you think the table of contents is complete or not>
        "completed": "yes" or "no"
    }}
    Directly return the final JSON structure. Do not output anything else."""

    prompt = prompt + '\n Document:\n' + content + '\n Table of contents:\n' + toc
    response = ChatGPT_API(model=model, prompt=prompt)
    json_content = extract_json(response)
    return json_content['completed']


def check_if_toc_transformation_is_complete(content, toc, model=None):
    prompt = f"""
    You are given a raw table of contents and a  table of contents.
    Your job is to check if the  table of contents is complete.

    Reply format:
    {{
        "thinking": <why do you think the cleaned table of contents is complete or not>
        "completed": "yes" or "no"
    }}
    Directly return the final JSON structure. Do not output anything else."""

    prompt = prompt + '\n Raw Table of contents:\n' + content + '\n Cleaned Table of contents:\n' + toc
    response = ChatGPT_API(model=model, prompt=prompt)
    json_content = extract_json(response)
    return json_content['completed']


def detect_page_index(toc_content, model=None):
    print('start detect_page_index')
    prompt = f"""
    You will be given a table of contents.

    Your job is to detect if there are page numbers/indices given within the table of contents.

    Given text: {toc_content}

    Reply format:
    {{
        "thinking": <why do you think there are page numbers/indices given within the table of contents>
        "page_index_given_in_toc": "<yes or no>"
    }}
    Directly return the final JSON structure. Do not output anything else."""

    response = ChatGPT_API(model=model, prompt=prompt)
    json_content = extract_json(response)
    return json_content['page_index_given_in_toc']


def toc_extractor(page_list, toc_page_list, model):
    def transform_dots_to_colon(text):
        text = re.sub(r'\.{5,}', ': ', text)
        text = re.sub(r'(?:\. ){5,}\.?', ': ', text)
        return text

    toc_content = ""
    for page_index in toc_page_list:
        toc_content += page_list[page_index][0]
    toc_content = transform_dots_to_colon(toc_content)
    has_page_index = detect_page_index(toc_content, model=model)
    return {
        "toc_content": toc_content,
        "page_index_given_in_toc": has_page_index,
    }


def toc_index_extractor(toc, content, model=None):
    print('start toc_index_extractor')
    toc_extractor_prompt = """
    You are given a table of contents in a json format and several pages of a document, your job is to add the physical_index to the table of contents in the json format.

    The provided pages contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X.

    The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

    The response should be in the following JSON format:
    [
        {
            "structure": <structure index, "x.x.x" or None> (string),
            "title": <title of the section>,
            "physical_index": "<physical_index_X>" (keep the format)
        },
        ...
    ]

    Only add the physical_index to the sections that are in the provided pages.
    If the section is not in the provided pages, do not add the physical_index to it.
    Directly return the final JSON structure. Do not output anything else."""

    prompt = toc_extractor_prompt + '\nTable of contents:\n' + str(toc) + '\nDocument pages:\n' + content
    response = ChatGPT_API(model=model, prompt=prompt)
    return extract_json(response)


def toc_transformer(toc_content, model=None):
    print('start toc_transformer')
    init_prompt = """
    You are given a table of contents, You job is to transform the whole table of content into a JSON format included table_of_contents.

    structure is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

    The response should be in the following JSON format:
    {
    table_of_contents: [
        {
            "structure": <structure index, "x.x.x" or None> (string),
            "title": <title of the section>,
            "page": <page number or None>,
        },
        ...
        ],
    }
    You should transform the full table of contents in one go.
    Directly return the final JSON structure, do not output anything else. """

    prompt = init_prompt + '\n Given table of contents\n:' + toc_content
    last_complete, finish_reason = ChatGPT_API_with_finish_reason(model=model, prompt=prompt)
    if_complete = check_if_toc_transformation_is_complete(toc_content, last_complete, model)
    if if_complete == "yes" and finish_reason == "finished":
        last_complete = extract_json(last_complete)
        return convert_page_to_int(last_complete['table_of_contents'])

    last_complete = get_json_content(last_complete)
    while not (if_complete == "yes" and finish_reason == "finished"):
        position = last_complete.rfind('}')
        if position != -1:
            last_complete = last_complete[:position + 2]
        prompt = f"""
        Your task is to continue the table of contents json structure, directly output the remaining part of the json structure.
        The response should be in the following JSON format:

        The raw table of contents json structure is:
        {toc_content}

        The incomplete transformed table of contents json structure is:
        {last_complete}

        Please continue the json structure, directly output the remaining part of the json structure."""

        new_complete, finish_reason = ChatGPT_API_with_finish_reason(model=model, prompt=prompt)
        if new_complete.startswith('```json'):
            new_complete = get_json_content(new_complete)
            last_complete = last_complete + new_complete
        if_complete = check_if_toc_transformation_is_complete(toc_content, last_complete, model)

    last_complete = json.loads(last_complete)
    return convert_page_to_int(last_complete['table_of_contents'])


def find_toc_pages(start_page_index, page_list, opt, logger=None):
    print('start find_toc_pages')
    last_page_is_yes = False
    toc_page_list = []
    i = start_page_index

    while i < len(page_list):
        if i >= opt.toc_check_page_num and not last_page_is_yes:
            break
        detected_result = toc_detector_single_page(page_list[i][0], model=opt.model)
        if detected_result == 'yes':
            if logger:
                logger.info(f'Page {i} has toc')
            toc_page_list.append(i)
            last_page_is_yes = True
        elif detected_result == 'no' and last_page_is_yes:
            if logger:
                logger.info(f'Found the last page with toc: {i-1}')
            break
        i += 1

    if not toc_page_list and logger:
        logger.info('No toc found')
    return toc_page_list


def extract_matching_page_pairs(toc_page, toc_physical_index, start_page_index):
    pairs = []
    for phy_item in toc_physical_index:
        for page_item in toc_page:
            if phy_item.get('title') == page_item.get('title'):
                physical_index = phy_item.get('physical_index')
                if physical_index is not None and int(physical_index) >= start_page_index:
                    pairs.append({
                        'title': phy_item.get('title'),
                        'page': page_item.get('page'),
                        'physical_index': physical_index,
                    })
    return pairs


def calculate_page_offset(pairs):
    differences = []
    for pair in pairs:
        try:
            differences.append(pair['physical_index'] - pair['page'])
        except (KeyError, TypeError):
            continue
    if not differences:
        return None
    counts = {}
    for d in differences:
        counts[d] = counts.get(d, 0) + 1
    return max(counts.items(), key=lambda x: x[1])[0]


def add_page_offset_to_toc_json(data, offset):
    for i in range(len(data)):
        if data[i].get('page') is not None and isinstance(data[i]['page'], int):
            data[i]['physical_index'] = data[i]['page'] + offset
            del data[i]['page']
    return data


def page_list_to_group_text(page_contents, token_lengths, max_tokens=20000, overlap_page=1):
    num_tokens = sum(token_lengths)
    if num_tokens <= max_tokens:
        return ["".join(page_contents)]

    subsets = []
    current_subset = []
    current_token_count = 0
    expected_parts_num = math.ceil(num_tokens / max_tokens)
    average_tokens_per_part = math.ceil(((num_tokens / expected_parts_num) + max_tokens) / 2)

    for i, (page_content, page_tokens) in enumerate(zip(page_contents, token_lengths)):
        if current_token_count + page_tokens > average_tokens_per_part:
            subsets.append(''.join(current_subset))
            overlap_start = max(i - overlap_page, 0)
            current_subset = page_contents[overlap_start:i]
            current_token_count = sum(token_lengths[overlap_start:i])
        current_subset.append(page_content)
        current_token_count += page_tokens

    if current_subset:
        subsets.append(''.join(current_subset))

    print('divide page_list to groups', len(subsets))
    return subsets


def add_page_number_to_toc(part, structure, model=None):
    fill_prompt_seq = """
    You are given an JSON structure of a document and a partial part of the document. Your task is to check if the title that is described in the structure is started in the partial given document.

    The provided text contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X.

    If the full target section starts in the partial given document, insert the given JSON structure with the "start": "yes", and "start_index": "<physical_index_X>".

    If the full target section does not start in the partial given document, insert "start": "no",  "start_index": None.

    The response should be in the following format.
        [
            {
                "structure": <structure index, "x.x.x" or None> (string),
                "title": <title of the section>,
                "start": "<yes or no>",
                "physical_index": "<physical_index_X> (keep the format)" or None
            },
            ...
        ]
    The given structure contains the result of the previous part, you need to fill the result of the current part, do not change the previous result.
    Directly return the final JSON structure. Do not output anything else."""

    prompt = fill_prompt_seq + f"\n\nCurrent Partial Document:\n{part}\n\nGiven Structure\n{json.dumps(structure, indent=2)}\n"
    current_json_raw = ChatGPT_API(model=model, prompt=prompt)
    json_result = extract_json(current_json_raw)
    for item in json_result:
        if 'start' in item:
            del item['start']
    return json_result


def _build_page_groups(page_list, start_index, model):
    """Label each page with physical_index tags and chunk by token budget."""
    page_contents = []
    token_lengths = []
    for page_index in range(start_index, start_index + len(page_list)):
        page_text = f"<physical_index_{page_index}>\n{page_list[page_index-start_index][0]}\n<physical_index_{page_index}>\n\n"
        page_contents.append(page_text)
        token_lengths.append(count_tokens(page_text, model))
    return page_list_to_group_text(page_contents, token_lengths)


def process_no_toc(page_list, start_index=1, model=None, logger=None):
    group_texts = _build_page_groups(page_list, start_index, model)
    logger.info(f'len(group_texts): {len(group_texts)}')

    toc_with_page_number = generate_toc_init(group_texts[0], model)
    for group_text in group_texts[1:]:
        toc_with_page_number_additional = generate_toc_continue(toc_with_page_number, group_text, model)
        toc_with_page_number.extend(toc_with_page_number_additional)
    logger.info(f'generate_toc: {toc_with_page_number}')

    toc_with_page_number = convert_physical_index_to_int(toc_with_page_number)
    logger.info(f'convert_physical_index_to_int: {toc_with_page_number}')
    return toc_with_page_number


def process_toc_no_page_numbers(toc_content, toc_page_list, page_list, start_index=1, model=None, logger=None):
    toc_content = toc_transformer(toc_content, model)
    logger.info(f'toc_transformer: {toc_content}')
    group_texts = _build_page_groups(page_list, start_index, model)
    logger.info(f'len(group_texts): {len(group_texts)}')

    toc_with_page_number = copy.deepcopy(toc_content)
    for group_text in group_texts:
        toc_with_page_number = add_page_number_to_toc(group_text, toc_with_page_number, model)
    logger.info(f'add_page_number_to_toc: {toc_with_page_number}')

    toc_with_page_number = convert_physical_index_to_int(toc_with_page_number)
    logger.info(f'convert_physical_index_to_int: {toc_with_page_number}')
    return toc_with_page_number


def process_toc_with_page_numbers(toc_content, toc_page_list, page_list, toc_check_page_num=None, model=None, logger=None):
    toc_with_page_number = toc_transformer(toc_content, model)
    logger.info(f'toc_with_page_number: {toc_with_page_number}')

    toc_no_page_number = remove_page_number(copy.deepcopy(toc_with_page_number))
    start_page_index = toc_page_list[-1] + 1
    main_content = ""
    for page_index in range(start_page_index, min(start_page_index + toc_check_page_num, len(page_list))):
        main_content += f"<physical_index_{page_index+1}>\n{page_list[page_index][0]}\n<physical_index_{page_index+1}>\n\n"

    toc_with_physical_index = toc_index_extractor(toc_no_page_number, main_content, model)
    logger.info(f'toc_with_physical_index: {toc_with_physical_index}')

    toc_with_physical_index = convert_physical_index_to_int(toc_with_physical_index)
    logger.info(f'toc_with_physical_index: {toc_with_physical_index}')

    matching_pairs = extract_matching_page_pairs(toc_with_page_number, toc_with_physical_index, start_page_index)
    logger.info(f'matching_pairs: {matching_pairs}')

    offset = calculate_page_offset(matching_pairs)
    logger.info(f'offset: {offset}')

    toc_with_page_number = add_page_offset_to_toc_json(toc_with_page_number, offset)
    logger.info(f'toc_with_page_number: {toc_with_page_number}')

    toc_with_page_number = process_none_page_numbers(toc_with_page_number, page_list, model=model)
    logger.info(f'toc_with_page_number: {toc_with_page_number}')
    return toc_with_page_number


def process_none_page_numbers(toc_items, page_list, start_index=1, model=None):
    for i, item in enumerate(toc_items):
        if "physical_index" not in item:
            prev_physical_index = 0
            for j in range(i - 1, -1, -1):
                if toc_items[j].get('physical_index') is not None:
                    prev_physical_index = toc_items[j]['physical_index']
                    break

            next_physical_index = -1
            for j in range(i + 1, len(toc_items)):
                if toc_items[j].get('physical_index') is not None:
                    next_physical_index = toc_items[j]['physical_index']
                    break

            page_contents = []
            for page_index in range(prev_physical_index, next_physical_index + 1):
                list_index = page_index - start_index
                if 0 <= list_index < len(page_list):
                    page_text = f"<physical_index_{page_index}>\n{page_list[list_index][0]}\n<physical_index_{page_index}>\n\n"
                    page_contents.append(page_text)
            item_copy = copy.deepcopy(item)
            del item_copy['page']
            result = add_page_number_to_toc(page_contents, item_copy, model)
            if isinstance(result[0]['physical_index'], str) and result[0]['physical_index'].startswith('<physical_index'):
                item['physical_index'] = int(result[0]['physical_index'].split('_')[-1].rstrip('>').strip())
                del item['page']
    return toc_items


def check_toc(page_list, opt=None):
    toc_page_list = find_toc_pages(start_page_index=0, page_list=page_list, opt=opt)
    if not toc_page_list:
        print('no toc found')
        return {'toc_content': None, 'toc_page_list': [], 'page_index_given_in_toc': 'no'}
    print('toc found')
    toc_json = toc_extractor(page_list, toc_page_list, opt.model)

    if toc_json['page_index_given_in_toc'] == 'yes':
        print('index found')
        return {
            'toc_content': toc_json['toc_content'],
            'toc_page_list': toc_page_list,
            'page_index_given_in_toc': 'yes',
        }

    current_start_index = toc_page_list[-1] + 1
    while (toc_json['page_index_given_in_toc'] == 'no'
           and current_start_index < len(page_list)
           and current_start_index < opt.toc_check_page_num):
        additional_toc_pages = find_toc_pages(
            start_page_index=current_start_index, page_list=page_list, opt=opt
        )
        if not additional_toc_pages:
            break
        additional_toc_json = toc_extractor(page_list, additional_toc_pages, opt.model)
        if additional_toc_json['page_index_given_in_toc'] == 'yes':
            print('index found')
            return {
                'toc_content': additional_toc_json['toc_content'],
                'toc_page_list': additional_toc_pages,
                'page_index_given_in_toc': 'yes',
            }
        current_start_index = additional_toc_pages[-1] + 1

    print('index not found')
    return {
        'toc_content': toc_json['toc_content'],
        'toc_page_list': toc_page_list,
        'page_index_given_in_toc': 'no',
    }


# ---------------------------------------------------------------------------
# TOC verification and fixing
# ---------------------------------------------------------------------------

async def check_title_appearance(item, page_list, start_index=1, model=None):
    title = item['title']
    if 'physical_index' not in item or item['physical_index'] is None:
        return {'list_index': item.get('list_index'), 'answer': 'no', 'title': title, 'page_number': None}

    page_number = item['physical_index']
    page_text = page_list[page_number - start_index][0]
    prompt = f"""
    Your job is to check if the given section appears or starts in the given page_text.

    Note: do fuzzy matching, ignore any space inconsistency in the page_text.

    The given section title is {title}.
    The given page_text is {page_text}.

    Reply format:
    {{

        "thinking": <why do you think the section appears or starts in the page_text>
        "answer": "yes or no" (yes if the section appears or starts in the page_text, no otherwise)
    }}
    Directly return the final JSON structure. Do not output anything else."""

    response = await ChatGPT_API_async(model=model, prompt=prompt)
    response = extract_json(response)
    answer = response.get('answer', 'no')
    return {'list_index': item['list_index'], 'answer': answer, 'title': title, 'page_number': page_number}


async def check_title_appearance_in_start(title, page_text, model=None, logger=None):
    prompt = f"""
    You will be given the current section title and the current page_text.
    Your job is to check if the current section starts in the beginning of the given page_text.
    If there are other contents before the current section title, then the current section does not start in the beginning of the given page_text.
    If the current section title is the first content in the given page_text, then the current section starts in the beginning of the given page_text.

    Note: do fuzzy matching, ignore any space inconsistency in the page_text.

    The given section title is {title}.
    The given page_text is {page_text}.

    reply format:
    {{
        "thinking": <why do you think the section appears or starts in the page_text>
        "start_begin": "yes or no" (yes if the section starts in the beginning of the page_text, no otherwise)
    }}
    Directly return the final JSON structure. Do not output anything else."""

    response = await ChatGPT_API_async(model=model, prompt=prompt)
    response = extract_json(response)
    if logger:
        logger.info(f"Response: {response}")
    return response.get("start_begin", "no")


async def check_title_appearance_in_start_concurrent(structure, page_list, model=None, logger=None):
    if logger:
        logger.info("Checking title appearance in start concurrently")

    for item in structure:
        if item.get('physical_index') is None:
            item['appear_start'] = 'no'

    tasks = []
    valid_items = []
    for item in structure:
        if item.get('physical_index') is not None:
            page_text = page_list[item['physical_index'] - 1][0]
            tasks.append(check_title_appearance_in_start(item['title'], page_text, model=model, logger=logger))
            valid_items.append(item)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for item, result in zip(valid_items, results):
        if isinstance(result, Exception):
            if logger:
                logger.error(f"Error checking start for {item['title']}: {result}")
            item['appear_start'] = 'no'
        else:
            item['appear_start'] = result

    return structure


def single_toc_item_index_fixer(section_title, content, model="gpt-4o-2024-11-20"):
    toc_extractor_prompt = """
    You are given a section title and several pages of a document, your job is to find the physical index of the start page of the section in the partial document.

    The provided pages contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X.

    Reply in a JSON format:
    {
        "thinking": <explain which page, started and closed by <physical_index_X>, contains the start of this section>,
        "physical_index": "<physical_index_X>" (keep the format)
    }
    Directly return the final JSON structure. Do not output anything else."""

    prompt = toc_extractor_prompt + '\nSection Title:\n' + str(section_title) + '\nDocument pages:\n' + content
    response = ChatGPT_API(model=model, prompt=prompt)
    json_content = extract_json(response)
    return convert_physical_index_to_int(json_content['physical_index'])


async def fix_incorrect_toc(toc_with_page_number, page_list, incorrect_results, start_index=1, model=None, logger=None):
    print(f'start fix_incorrect_toc with {len(incorrect_results)} incorrect results')
    incorrect_indices = {result['list_index'] for result in incorrect_results}
    end_index = len(page_list) + start_index - 1
    incorrect_results_and_range_logs = []

    async def process_and_check_item(incorrect_item):
        list_index = incorrect_item['list_index']
        if list_index < 0 or list_index >= len(toc_with_page_number):
            return {
                'list_index': list_index,
                'title': incorrect_item['title'],
                'physical_index': incorrect_item.get('physical_index'),
                'is_valid': False,
            }

        prev_correct = None
        for i in range(list_index - 1, -1, -1):
            if i not in incorrect_indices and 0 <= i < len(toc_with_page_number):
                physical_index = toc_with_page_number[i].get('physical_index')
                if physical_index is not None:
                    prev_correct = physical_index
                    break
        if prev_correct is None:
            prev_correct = start_index - 1

        next_correct = None
        for i in range(list_index + 1, len(toc_with_page_number)):
            if i not in incorrect_indices and 0 <= i < len(toc_with_page_number):
                physical_index = toc_with_page_number[i].get('physical_index')
                if physical_index is not None:
                    next_correct = physical_index
                    break
        if next_correct is None:
            next_correct = end_index

        incorrect_results_and_range_logs.append({
            'list_index': list_index,
            'title': incorrect_item['title'],
            'prev_correct': prev_correct,
            'next_correct': next_correct,
        })

        page_contents = []
        for page_index in range(prev_correct, next_correct + 1):
            page_list_idx = page_index - start_index
            if 0 <= page_list_idx < len(page_list):
                page_text = f"<physical_index_{page_index}>\n{page_list[page_list_idx][0]}\n<physical_index_{page_index}>\n\n"
                page_contents.append(page_text)
        content_range = ''.join(page_contents)

        physical_index_int = single_toc_item_index_fixer(incorrect_item['title'], content_range, model)
        check_item = incorrect_item.copy()
        check_item['physical_index'] = physical_index_int
        check_result = await check_title_appearance(check_item, page_list, start_index, model)

        return {
            'list_index': list_index,
            'title': incorrect_item['title'],
            'physical_index': physical_index_int,
            'is_valid': check_result['answer'] == 'yes',
        }

    tasks = [process_and_check_item(item) for item in incorrect_results]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    invalid_results = []
    for result in results:
        if isinstance(result, Exception):
            print(f"Processing generated an exception: {result}")
            continue
        if result['is_valid']:
            list_idx = result['list_index']
            if 0 <= list_idx < len(toc_with_page_number):
                toc_with_page_number[list_idx]['physical_index'] = result['physical_index']
            else:
                invalid_results.append(result)
        else:
            invalid_results.append(result)

    logger.info(f'incorrect_results_and_range_logs: {incorrect_results_and_range_logs}')
    logger.info(f'invalid_results: {invalid_results}')
    return toc_with_page_number, invalid_results


async def fix_incorrect_toc_with_retries(toc_with_page_number, page_list, incorrect_results, start_index=1, max_attempts=3, model=None, logger=None):
    print('start fix_incorrect_toc')
    fix_attempt = 0
    current_toc = toc_with_page_number
    current_incorrect = incorrect_results

    while current_incorrect:
        print(f"Fixing {len(current_incorrect)} incorrect results")
        current_toc, current_incorrect = await fix_incorrect_toc(
            current_toc, page_list, current_incorrect, start_index, model, logger
        )
        fix_attempt += 1
        if fix_attempt >= max_attempts:
            logger.info("Maximum fix attempts reached")
            break
    return current_toc, current_incorrect


async def verify_toc(page_list, list_result, start_index=1, N=None, model=None):
    print('start verify_toc')
    last_physical_index = None
    for item in reversed(list_result):
        if item.get('physical_index') is not None:
            last_physical_index = item['physical_index']
            break

    if last_physical_index is None or last_physical_index < len(page_list) / 2:
        return 0, []

    if N is None:
        print('check all items')
        sample_indices = range(0, len(list_result))
    else:
        N = min(N, len(list_result))
        print(f'check {N} items')
        sample_indices = random.sample(range(0, len(list_result)), N)

    indexed_sample_list = []
    for idx in sample_indices:
        item = list_result[idx]
        if item.get('physical_index') is not None:
            item_with_index = item.copy()
            item_with_index['list_index'] = idx
            indexed_sample_list.append(item_with_index)

    tasks = [check_title_appearance(item, page_list, start_index, model) for item in indexed_sample_list]
    results = await asyncio.gather(*tasks)

    correct_count = 0
    incorrect_results = []
    for result in results:
        if result['answer'] == 'yes':
            correct_count += 1
        else:
            incorrect_results.append(result)

    checked_count = len(results)
    accuracy = correct_count / checked_count if checked_count > 0 else 0
    print(f"accuracy: {accuracy*100:.2f}%")
    return accuracy, incorrect_results


def validate_and_truncate_physical_indices(toc_with_page_number, page_list_length, start_index=1, logger=None):
    if not toc_with_page_number:
        return toc_with_page_number

    max_allowed_page = page_list_length + start_index - 1
    truncated_items = []

    for item in toc_with_page_number:
        if item.get('physical_index') is not None:
            original_index = item['physical_index']
            if original_index > max_allowed_page:
                item['physical_index'] = None
                truncated_items.append({'title': item.get('title', 'Unknown'), 'original_index': original_index})
                if logger:
                    logger.info(f"Removed physical_index for '{item.get('title', 'Unknown')}' (was {original_index}, too far beyond document)")

    if truncated_items and logger:
        logger.info(f"Total removed items: {len(truncated_items)}")

    print(f"Document validation: {page_list_length} pages, max allowed index: {max_allowed_page}")
    if truncated_items:
        print(f"Truncated {len(truncated_items)} TOC items that exceeded document length")

    return toc_with_page_number


# ---------------------------------------------------------------------------
# Main PDF processing pipeline
# ---------------------------------------------------------------------------

async def meta_processor(page_list, mode=None, toc_content=None, toc_page_list=None, start_index=1, opt=None, logger=None):
    print(mode)
    print(f'start_index: {start_index}')

    if mode == 'process_toc_with_page_numbers':
        toc_with_page_number = process_toc_with_page_numbers(
            toc_content, toc_page_list, page_list,
            toc_check_page_num=opt.toc_check_page_num, model=opt.model, logger=logger
        )
    elif mode == 'process_toc_no_page_numbers':
        toc_with_page_number = process_toc_no_page_numbers(
            toc_content, toc_page_list, page_list, model=opt.model, logger=logger
        )
    else:
        toc_with_page_number = process_no_toc(page_list, start_index=start_index, model=opt.model, logger=logger)

    toc_with_page_number = [item for item in toc_with_page_number if item.get('physical_index') is not None]
    toc_with_page_number = validate_and_truncate_physical_indices(
        toc_with_page_number, len(page_list), start_index=start_index, logger=logger
    )

    accuracy, incorrect_results = await verify_toc(
        page_list, toc_with_page_number, start_index=start_index, model=opt.model
    )
    logger.info({
        'mode': 'process_toc_with_page_numbers',
        'accuracy': accuracy,
        'incorrect_results': incorrect_results,
    })

    if accuracy == 1.0 and not incorrect_results:
        return toc_with_page_number
    if accuracy > 0.6 and incorrect_results:
        toc_with_page_number, incorrect_results = await fix_incorrect_toc_with_retries(
            toc_with_page_number, page_list, incorrect_results,
            start_index=start_index, max_attempts=3, model=opt.model, logger=logger
        )
        return toc_with_page_number

    if mode == 'process_toc_with_page_numbers':
        return await meta_processor(
            page_list, mode='process_toc_no_page_numbers',
            toc_content=toc_content, toc_page_list=toc_page_list,
            start_index=start_index, opt=opt, logger=logger
        )
    elif mode == 'process_toc_no_page_numbers':
        return await meta_processor(
            page_list, mode='process_no_toc', start_index=start_index, opt=opt, logger=logger
        )
    else:
        logger.info("Warning: process_no_toc accuracy below threshold; returning best available result")
        return toc_with_page_number


async def process_large_node_recursively(node, page_list, opt=None, logger=None):
    node_page_list = page_list[node['start_index'] - 1:node['end_index']]
    token_num = sum(page[1] for page in node_page_list)

    if node['end_index'] - node['start_index'] > opt.max_page_num_each_node and token_num >= opt.max_token_num_each_node:
        print('large node:', node['title'], 'start_index:', node['start_index'], 'end_index:', node['end_index'], 'token_num:', token_num)
        node_toc_tree = await meta_processor(
            node_page_list, mode='process_no_toc', start_index=node['start_index'], opt=opt, logger=logger
        )
        node_toc_tree = await check_title_appearance_in_start_concurrent(
            node_toc_tree, page_list, model=opt.model, logger=logger
        )
        valid_node_toc_items = [item for item in node_toc_tree if item.get('physical_index') is not None]

        if valid_node_toc_items and node['title'].strip() == valid_node_toc_items[0]['title'].strip():
            node['nodes'] = post_processing(valid_node_toc_items[1:], node['end_index'])
            node['end_index'] = valid_node_toc_items[1]['start_index'] if len(valid_node_toc_items) > 1 else node['end_index']
        else:
            node['nodes'] = post_processing(valid_node_toc_items, node['end_index'])
            node['end_index'] = valid_node_toc_items[0]['start_index'] if valid_node_toc_items else node['end_index']

    if node.get('nodes'):
        tasks = [
            process_large_node_recursively(child_node, page_list, opt, logger=logger)
            for child_node in node['nodes']
        ]
        await asyncio.gather(*tasks)
    return node


async def tree_parser(page_list, opt, doc=None, logger=None):
    check_toc_result = check_toc(page_list, opt)
    logger.info(check_toc_result)

    if (check_toc_result.get("toc_content")
            and check_toc_result["toc_content"].strip()
            and check_toc_result["page_index_given_in_toc"] == "yes"):
        toc_with_page_number = await meta_processor(
            page_list, mode='process_toc_with_page_numbers', start_index=1,
            toc_content=check_toc_result['toc_content'],
            toc_page_list=check_toc_result['toc_page_list'],
            opt=opt, logger=logger
        )
    else:
        toc_with_page_number = await meta_processor(
            page_list, mode='process_no_toc', start_index=1, opt=opt, logger=logger
        )

    toc_with_page_number = add_preface_if_needed(toc_with_page_number)
    toc_with_page_number = await check_title_appearance_in_start_concurrent(
        toc_with_page_number, page_list, model=opt.model, logger=logger
    )

    valid_toc_items = [item for item in toc_with_page_number if item.get('physical_index') is not None]
    toc_tree = post_processing(valid_toc_items, len(page_list))

    tasks = [
        process_large_node_recursively(node, page_list, opt, logger=logger)
        for node in toc_tree
    ]
    await asyncio.gather(*tasks)
    return toc_tree


def page_index_main(doc, opt=None):
    logger = JsonLogger(doc)

    is_valid_pdf = (
        (isinstance(doc, str) and os.path.isfile(doc) and doc.lower().endswith(".pdf"))
        or isinstance(doc, BytesIO)
    )
    if not is_valid_pdf:
        raise ValueError("Unsupported input type. Expected a PDF file path or BytesIO object.")

    print('Parsing PDF...')
    page_list = get_page_tokens(doc)
    logger.info({'total_page_number': len(page_list)})
    logger.info({'total_token': sum(page[1] for page in page_list)})

    async def page_index_builder():
        structure = await tree_parser(page_list, opt, doc=doc, logger=logger)
        if opt.if_add_node_id == 'yes':
            write_node_id(structure)
        if opt.if_add_node_text == 'yes':
            add_node_text(structure, page_list)
        if opt.if_add_node_summary == 'yes':
            if opt.if_add_node_text == 'no':
                add_node_text(structure, page_list)
            await generate_summaries_for_structure(structure, model=opt.model)
            if opt.if_add_node_text == 'no':
                remove_structure_text(structure)
            if opt.if_add_doc_description == 'yes':
                clean_structure = create_clean_structure_for_description(structure)
                doc_description = generate_doc_description(clean_structure, model=opt.model)
                return {
                    'doc_name': get_pdf_name(doc),
                    'doc_description': doc_description,
                    'structure': structure,
                }
        return {
            'doc_name': get_pdf_name(doc),
            'structure': structure,
        }

    return asyncio.run(page_index_builder())


def page_index(doc, model=None, toc_check_page_num=None, max_page_num_each_node=None,
               max_token_num_each_node=None, if_add_node_id=None, if_add_node_summary=None,
               if_add_doc_description=None, if_add_node_text=None):
    user_opt = {arg: value for arg, value in locals().items() if arg != "doc" and value is not None}
    opt = _load_config(user_opt)
    return page_index_main(doc, opt)


# ---------------------------------------------------------------------------
# Markdown pipeline (from page_index_md.py)
# ---------------------------------------------------------------------------

async def _get_node_summary_md(node, summary_token_threshold=200, model=None):
    node_text = node.get('text')
    if count_tokens(node_text, model=model) < summary_token_threshold:
        return node_text
    return await generate_node_summary(node, model=model)


async def generate_summaries_for_structure_md(structure, summary_token_threshold, model=None):
    nodes = structure_to_list(structure)
    tasks = [_get_node_summary_md(node, summary_token_threshold=summary_token_threshold, model=model) for node in nodes]
    summaries = await asyncio.gather(*tasks)
    for node, summary in zip(nodes, summaries):
        if not node.get('nodes'):
            node['summary'] = summary
        else:
            node['prefix_summary'] = summary
    return structure


def extract_nodes_from_markdown(markdown_content):
    header_pattern = r'^(#{1,6})\s+(.+)$'
    code_block_pattern = r'^```'
    node_list = []
    lines = markdown_content.split('\n')
    in_code_block = False

    for line_num, line in enumerate(lines, 1):
        stripped_line = line.strip()
        if re.match(code_block_pattern, stripped_line):
            in_code_block = not in_code_block
            continue
        if not stripped_line:
            continue
        if not in_code_block:
            match = re.match(header_pattern, stripped_line)
            if match:
                node_list.append({'node_title': match.group(2).strip(), 'line_num': line_num})

    return node_list, lines


def extract_node_text_content(node_list, markdown_lines):
    all_nodes = []
    for node in node_list:
        line_content = markdown_lines[node['line_num'] - 1]
        header_match = re.match(r'^(#{1,6})', line_content)
        if header_match is None:
            print(f"Warning: Line {node['line_num']} does not contain a valid header: '{line_content}'")
            continue
        all_nodes.append({
            'title': node['node_title'],
            'line_num': node['line_num'],
            'level': len(header_match.group(1)),
        })

    for i, node in enumerate(all_nodes):
        start_line = node['line_num'] - 1
        end_line = all_nodes[i + 1]['line_num'] - 1 if i + 1 < len(all_nodes) else len(markdown_lines)
        node['text'] = '\n'.join(markdown_lines[start_line:end_line]).strip()
    return all_nodes


def update_node_list_with_text_token_count(node_list, model=None):
    result_list = node_list.copy()
    for i in range(len(result_list) - 1, -1, -1):
        current_node = result_list[i]
        children_indices = _find_children(result_list, i, current_node['level'])
        total_text = current_node.get('text', '')
        for child_index in children_indices:
            child_text = result_list[child_index].get('text', '')
            if child_text:
                total_text += '\n' + child_text
        result_list[i]['text_token_count'] = count_tokens(total_text, model=model)
    return result_list


def tree_thinning_for_index(node_list, min_node_token=None, model=None):
    result_list = node_list.copy()
    nodes_to_remove = set()

    for i in range(len(result_list) - 1, -1, -1):
        if i in nodes_to_remove:
            continue
        current_node = result_list[i]
        current_level = current_node['level']
        total_tokens = current_node.get('text_token_count', 0)

        if total_tokens < min_node_token:
            children_indices = _find_children(result_list, i, current_level)
            children_texts = []
            for child_index in sorted(children_indices):
                if child_index not in nodes_to_remove:
                    child_text = result_list[child_index].get('text', '')
                    if child_text.strip():
                        children_texts.append(child_text)
                    nodes_to_remove.add(child_index)
            if children_texts:
                parent_text = current_node.get('text', '')
                merged_text = parent_text
                for child_text in children_texts:
                    if merged_text and not merged_text.endswith('\n'):
                        merged_text += '\n\n'
                    merged_text += child_text
                result_list[i]['text'] = merged_text
                result_list[i]['text_token_count'] = count_tokens(merged_text, model=model)

    for index in sorted(nodes_to_remove, reverse=True):
        result_list.pop(index)
    return result_list


def build_tree_from_nodes(node_list):
    if not node_list:
        return []
    stack = []
    root_nodes = []
    node_counter = 1

    for node in node_list:
        current_level = node['level']
        tree_node = {
            'title': node['title'],
            'node_id': str(node_counter).zfill(4),
            'text': node['text'],
            'line_num': node['line_num'],
            'nodes': [],
        }
        node_counter += 1
        while stack and stack[-1][1] >= current_level:
            stack.pop()
        if not stack:
            root_nodes.append(tree_node)
        else:
            stack[-1][0]['nodes'].append(tree_node)
        stack.append((tree_node, current_level))
    return root_nodes


async def md_to_tree(md_path, if_thinning=False, min_token_threshold=None, if_add_node_summary='no',
                     summary_token_threshold=None, model=None, if_add_doc_description='no',
                     if_add_node_text='no', if_add_node_id='yes'):
    with open(md_path, 'r', encoding='utf-8') as f:
        markdown_content = f.read()

    print("Extracting nodes from markdown...")
    node_list, markdown_lines = extract_nodes_from_markdown(markdown_content)

    print("Extracting text content from nodes...")
    nodes_with_content = extract_node_text_content(node_list, markdown_lines)

    if if_thinning:
        nodes_with_content = update_node_list_with_text_token_count(nodes_with_content, model=model)
        print("Thinning nodes...")
        nodes_with_content = tree_thinning_for_index(nodes_with_content, min_token_threshold, model=model)

    print("Building tree from nodes...")
    tree_structure = build_tree_from_nodes(nodes_with_content)

    if if_add_node_id == 'yes':
        write_node_id(tree_structure)

    print("Formatting tree structure...")

    if if_add_node_summary == 'yes':
        tree_structure = format_structure(
            tree_structure,
            order=['title', 'node_id', 'summary', 'prefix_summary', 'text', 'line_num', 'nodes']
        )
        print("Generating summaries for each node...")
        tree_structure = await generate_summaries_for_structure_md(
            tree_structure, summary_token_threshold=summary_token_threshold, model=model
        )
        if if_add_node_text == 'no':
            tree_structure = format_structure(
                tree_structure,
                order=['title', 'node_id', 'summary', 'prefix_summary', 'line_num', 'nodes']
            )
        if if_add_doc_description == 'yes':
            print("Generating document description...")
            clean_structure = create_clean_structure_for_description(tree_structure)
            doc_description = generate_doc_description(clean_structure, model=model)
            return {
                'doc_name': os.path.splitext(os.path.basename(md_path))[0],
                'doc_description': doc_description,
                'structure': tree_structure,
            }
    else:
        if if_add_node_text == 'yes':
            tree_structure = format_structure(
                tree_structure,
                order=['title', 'node_id', 'summary', 'prefix_summary', 'text', 'line_num', 'nodes']
            )
        else:
            tree_structure = format_structure(
                tree_structure,
                order=['title', 'node_id', 'summary', 'prefix_summary', 'line_num', 'nodes']
            )

    return {
        'doc_name': os.path.splitext(os.path.basename(md_path))[0],
        'structure': tree_structure,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Process PDF or Markdown document and generate structure')
    parser.add_argument('--pdf_path', type=str, help='Path to the PDF file')
    parser.add_argument('--md_path', type=str, help='Path to the Markdown file')
    parser.add_argument('--model', type=str, default='gpt-4o-2024-11-20', help='Model to use')
    parser.add_argument('--toc-check-pages', type=int, default=20,
                        help='Number of pages to check for table of contents (PDF only)')
    parser.add_argument('--max-pages-per-node', type=int, default=10,
                        help='Maximum number of pages per node (PDF only)')
    parser.add_argument('--max-tokens-per-node', type=int, default=20000,
                        help='Maximum number of tokens per node (PDF only)')
    parser.add_argument('--if-add-node-id', type=str, default='yes',
                        help='Whether to add node id to the node')
    parser.add_argument('--if-add-node-summary', type=str, default='yes',
                        help='Whether to add summary to the node')
    parser.add_argument('--if-add-doc-description', type=str, default='no',
                        help='Whether to add doc description to the doc')
    parser.add_argument('--if-add-node-text', type=str, default='no',
                        help='Whether to add text to the node')
    parser.add_argument('--if-thinning', type=str, default='no',
                        help='Whether to apply tree thinning for markdown (markdown only)')
    parser.add_argument('--thinning-threshold', type=int, default=5000,
                        help='Minimum token threshold for thinning (markdown only)')
    parser.add_argument('--summary-token-threshold', type=int, default=200,
                        help='Token threshold for generating summaries (markdown only)')
    parser.add_argument('--index-only', action='store_true',
                        help='Build index only (skip summaries), overrides --if-add-node-summary')
    args = parser.parse_args()

    if not args.pdf_path and not args.md_path:
        raise ValueError("Either --pdf_path or --md_path must be specified")
    if args.pdf_path and args.md_path:
        raise ValueError("Only one of --pdf_path or --md_path can be specified")

    if args.index_only:
        args.if_add_node_summary = 'no'

    if args.pdf_path:
        if not args.pdf_path.lower().endswith('.pdf'):
            raise ValueError("PDF file must have .pdf extension")
        if not os.path.isfile(args.pdf_path):
            raise ValueError(f"PDF file not found: {args.pdf_path}")

        opt = config(
            model=args.model,
            toc_check_page_num=args.toc_check_pages,
            max_page_num_each_node=args.max_pages_per_node,
            max_token_num_each_node=args.max_tokens_per_node,
            if_add_node_id=args.if_add_node_id,
            if_add_node_summary=args.if_add_node_summary,
            if_add_doc_description=args.if_add_doc_description,
            if_add_node_text=args.if_add_node_text,
        )
        result = page_index_main(args.pdf_path, opt)
        print('Parsing done, saving to file...')

        pdf_name = os.path.splitext(os.path.basename(args.pdf_path))[0]
        output_dir = './results'
        output_file = f'{output_dir}/{pdf_name}_structure.json'
        os.makedirs(output_dir, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        print(f'Tree structure saved to: {output_file}')

    elif args.md_path:
        if not args.md_path.lower().endswith(('.md', '.markdown')):
            raise ValueError("Markdown file must have .md or .markdown extension")
        if not os.path.isfile(args.md_path):
            raise ValueError(f"Markdown file not found: {args.md_path}")

        print('Processing markdown file...')
        result = asyncio.run(md_to_tree(
            md_path=args.md_path,
            if_thinning=args.if_thinning.lower() == 'yes',
            min_token_threshold=args.thinning_threshold,
            if_add_node_summary=args.if_add_node_summary,
            summary_token_threshold=args.summary_token_threshold,
            model=args.model,
            if_add_doc_description=args.if_add_doc_description,
            if_add_node_text=args.if_add_node_text,
            if_add_node_id=args.if_add_node_id,
        ))
        print('Parsing done, saving to file...')

        md_name = os.path.splitext(os.path.basename(args.md_path))[0]
        output_dir = './results'
        output_file = f'{output_dir}/{md_name}_structure.json'
        os.makedirs(output_dir, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f'Tree structure saved to: {output_file}')


if __name__ == "__main__":
    main()
