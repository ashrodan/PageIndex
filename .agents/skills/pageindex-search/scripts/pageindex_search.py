#!/usr/bin/env python3
"""
PageIndex Search: tree-first vectorless retrieval with optional LLM stages.

Examples:
  uv run .claude/skills/pageindex-search/scripts/pageindex_search.py tests/results/q1-fy25-earnings_structure.json "What was total revenue?"
  uv run .claude/skills/pageindex-search/scripts/pageindex_search.py tests/results/q1-fy25-earnings_structure.json "What was total revenue?" --answer-mode llm --llm-rerank
  uv run .claude/skills/pageindex-search/scripts/pageindex_search.py tests/results/q1-fy25-earnings_structure.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


REPO_ROOT = Path(__file__).resolve().parents[4]
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

TREE_SEARCH_PROMPT = """\
You are given a query and the tree structure of a document (PageIndex).
Find all leaf/intermediate nodes most likely to contain the answer.

Query: {query}

Document tree structure:
{tree}

Reply ONLY with valid JSON:
{{
  "thinking": "<brief reasoning>",
  "node_list": ["<node_id1>", "<node_id2>"]
}}
"""

RERANK_PROMPT = """\
You are given a question and candidate nodes from a document tree.
Select the most relevant node IDs for answering the question.

Question: {query}

Candidates:
{candidates}

Reply ONLY with valid JSON:
{{
  "thinking": "<brief reasoning>",
  "node_list": ["<node_id1>", "<node_id2>"]
}}
"""

ANSWER_PROMPT = """\
Answer the user's question using only the retrieved document context.
If context is insufficient, say that explicitly.
Cite used nodes in square brackets, for example [0007].

Question:
{query}

Retrieved context:
{context}
"""

INLINE_EVAL_PROMPT = """\
Evaluate retrieval/answer quality for this PageIndex query run.

Question:
{query}

Retrieved nodes:
{retrieved_nodes}

Answer:
{answer}

Return ONLY valid JSON:
{{
  "retrieval_quality": "high|medium|low",
  "answer_grounded": "yes|partial|no",
  "coverage_gaps": ["<missing point 1>", "<missing point 2>"],
  "recommended_next_step": "<one practical action>"
}}
"""


@dataclass
class PipelineConfig:
    retrieval_mode: str
    answer_mode: str
    llm_rerank: bool
    eval_inline: bool
    top_k: int
    candidate_k: int
    retrieve_model: str
    answer_model: str
    eval_model: str


@dataclass
class QueryOutcome:
    retrieved_ids: list[str]
    answer: str | None
    retrieval_result: dict[str, Any]
    rerank_result: dict[str, Any] | None
    eval_result: dict[str, Any] | None


def flatten_nodes(
    structure: list[dict[str, Any]],
    acc: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    if acc is None:
        acc = {}
    for node in structure:
        node_id = node.get("node_id")
        if node_id:
            acc[node_id] = node
        children = node.get("nodes") or []
        if children:
            flatten_nodes(children, acc)
    return acc


def compact_tree(structure: list[dict[str, Any]], depth: int = 0) -> str:
    lines: list[str] = []
    for node in structure:
        node_id = node.get("node_id", "-")
        title = node.get("title", "Untitled")
        start = node.get("start_index", node.get("line_num", "?"))
        end = node.get("end_index", node.get("line_num", "?"))
        summary = node.get("summary") or node.get("prefix_summary") or ""
        snippet = (summary[:120] + "...") if len(summary) > 120 else summary
        indent = "  " * depth
        lines.append(f"{indent}[{node_id}] {title} ({start}-{end})")
        if snippet:
            lines.append(f"{indent}  - {snippet}")
        children = node.get("nodes") or []
        if children:
            lines.extend(compact_tree(children, depth + 1).splitlines())
    return "\n".join(lines)


def parse_json_from_model(raw: str) -> dict[str, Any]:
    payload = raw.strip()
    if payload.startswith("```"):
        payload = payload.split("```", maxsplit=2)[1]
        if payload.startswith("json"):
            payload = payload[4:]
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return {"thinking": payload, "node_list": []}


def run_llm_tree_search(
    query: str,
    index: dict[str, Any],
    model: str,
    api_key: str,
) -> dict[str, Any]:
    import openai

    prompt = TREE_SEARCH_PROMPT.format(query=query, tree=compact_tree(index.get("structure", [])))
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    result = parse_json_from_model(response.choices[0].message.content or "")
    result["_tokens_in"] = response.usage.prompt_tokens
    result["_tokens_out"] = response.usage.completion_tokens
    result["_method"] = "llm_tree"
    return result


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text)]


def build_node_corpus(node: dict[str, Any]) -> dict[str, Any]:
    title = str(node.get("title") or "")
    summary = " ".join(
        part for part in [node.get("summary"), node.get("prefix_summary")] if part
    )
    text = str(node.get("text") or "")
    return {
        "title": title,
        "summary": summary,
        "text": text,
        "title_tokens": set(tokenize(title)),
        "summary_tokens": set(tokenize(summary)),
        "text_tokens": set(tokenize(text)),
        "full_text": f"{title}\n{summary}\n{text}".lower(),
    }


def score_node(query_tokens: set[str], query_norm: str, corpus: dict[str, Any]) -> tuple[float, dict[str, int]]:
    title_hits = sum(1 for tok in query_tokens if tok in corpus["title_tokens"])
    summary_hits = sum(1 for tok in query_tokens if tok in corpus["summary_tokens"])
    text_hits = sum(1 for tok in query_tokens if tok in corpus["text_tokens"])

    phrase_bonus = 6 if query_norm and query_norm in corpus["full_text"] else 0
    score = (title_hits * 4) + (summary_hits * 2) + text_hits + phrase_bonus
    return float(score), {
        "title_hits": title_hits,
        "summary_hits": summary_hits,
        "text_hits": text_hits,
        "phrase_bonus": phrase_bonus,
    }


def run_tree_retrieval(
    query: str,
    all_nodes: dict[str, dict[str, Any]],
    candidate_k: int,
) -> dict[str, Any]:
    q_tokens = set(tokenize(query))
    q_norm = " ".join(tokenize(query))

    scored: list[tuple[float, str, dict[str, int]]] = []
    for node_id, node in all_nodes.items():
        corpus = build_node_corpus(node)
        score, components = score_node(q_tokens, q_norm, corpus)
        if score > 0:
            scored.append((score, node_id, components))

    scored.sort(key=lambda item: (-item[0], item[1]))
    if not scored:
        fallback_ids = sorted(all_nodes.keys())[:candidate_k]
        return {
            "thinking": "No lexical overlap found; using deterministic node-id fallback.",
            "node_list": fallback_ids,
            "scores": {nid: 0.0 for nid in fallback_ids},
            "_method": "tree_lexical_fallback",
        }

    node_list = [node_id for _, node_id, _ in scored[:candidate_k]]
    scores = {node_id: score for score, node_id, _ in scored[:candidate_k]}
    return {
        "thinking": "Tree lexical retrieval based on title/summary/text token overlap.",
        "node_list": node_list,
        "scores": scores,
        "_method": "tree_lexical",
    }


def run_llm_rerank(
    query: str,
    candidate_ids: list[str],
    all_nodes: dict[str, dict[str, Any]],
    model: str,
    api_key: str,
    top_k: int,
) -> dict[str, Any]:
    import openai

    candidate_lines: list[str] = []
    for node_id in candidate_ids:
        node = all_nodes.get(node_id)
        if not node:
            continue
        title = node.get("title", "Untitled")
        start = node.get("start_index", node.get("line_num", "?"))
        end = node.get("end_index", node.get("line_num", "?"))
        body = node.get("summary") or node.get("text") or ""
        body = textwrap.shorten(str(body).replace("\n", " "), width=300, placeholder="...")
        candidate_lines.append(f"[{node_id}] {title} ({start}-{end})\n{body}")

    prompt = RERANK_PROMPT.format(query=query, candidates="\n\n".join(candidate_lines))
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    result = parse_json_from_model(response.choices[0].message.content or "")
    raw_ids = result.get("node_list") or []
    allowed = set(candidate_ids)
    node_list = [node_id for node_id in raw_ids if node_id in allowed]
    if not node_list:
        node_list = candidate_ids[:top_k]

    result["node_list"] = node_list[:top_k]
    result["_tokens_in"] = response.usage.prompt_tokens
    result["_tokens_out"] = response.usage.completion_tokens
    result["_method"] = "llm_rerank"
    return result


def run_answer_llm(query: str, context: str, model: str, api_key: str) -> str:
    import openai

    prompt = ANSWER_PROMPT.format(query=query, context=context)
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return (response.choices[0].message.content or "").strip()


def run_inline_eval(
    query: str,
    node_ids: list[str],
    all_nodes: dict[str, dict[str, Any]],
    answer: str | None,
    model: str,
    api_key: str,
) -> dict[str, Any]:
    import openai

    retrieved_chunks: list[str] = []
    for node_id in node_ids:
        node = all_nodes.get(node_id)
        if not node:
            continue
        title = node.get("title", "Untitled")
        start = node.get("start_index", node.get("line_num", "?"))
        end = node.get("end_index", node.get("line_num", "?"))
        text = node.get("text") or node.get("summary") or node.get("prefix_summary") or ""
        text = textwrap.shorten(str(text).replace("\n", " "), width=220, placeholder="...")
        retrieved_chunks.append(f"[{node_id}] {title} ({start}-{end}): {text}")

    prompt = INLINE_EVAL_PROMPT.format(
        query=query,
        retrieved_nodes="\n".join(retrieved_chunks) if retrieved_chunks else "(none)",
        answer=answer or "(no answer stage was run)",
    )
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    result = parse_json_from_model(response.choices[0].message.content or "")
    result["_tokens_in"] = response.usage.prompt_tokens
    result["_tokens_out"] = response.usage.completion_tokens
    result["_method"] = "llm_inline_eval"
    return result


def run_answer_extractive(query: str, node_ids: list[str], all_nodes: dict[str, dict[str, Any]], top_k: int) -> str:
    del query  # Query currently used only during retrieval.

    if not node_ids:
        return "Insufficient retrieved context to answer this question."

    lines = ["Best matching tree nodes (extractive mode):"]
    for node_id in node_ids[:top_k]:
        node = all_nodes.get(node_id)
        if not node:
            continue
        title = node.get("title", "Untitled")
        start = node.get("start_index", node.get("line_num", "?"))
        end = node.get("end_index", node.get("line_num", "?"))
        text = node.get("text") or node.get("summary") or node.get("prefix_summary") or ""
        text = textwrap.shorten(str(text).replace("\n", " "), width=260, placeholder="...")
        lines.append(f"- [{node_id}] {title} ({start}-{end}): {text}")
    return "\n".join(lines)


def run_cmd(cmd: list[str], env: dict[str, str]) -> None:
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"$ {printable}")
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def resolve_api_key(explicit_key: str | None) -> str | None:
    return explicit_key or os.getenv("CHATGPT_API_KEY") or os.getenv("OPENAI_API_KEY")


def build_or_use_index(document_path: Path, model: str, env: dict[str, str]) -> Path:
    suffix = document_path.suffix.lower()

    if suffix == ".json":
        return document_path

    if suffix in {".pdf", ".md", ".markdown"}:
        cmd = [
            "uv",
            "run",
            "pageindex",
            "--model",
            model,
            "--if-add-node-id",
            "yes",
            "--if-add-node-summary",
            "yes",
        ]
        if suffix == ".pdf":
            cmd.extend(["--pdf_path", str(document_path)])
        else:
            cmd.extend(["--md_path", str(document_path)])

        run_cmd(cmd, env)
        out_path = REPO_ROOT / "results" / f"{document_path.stem}_structure.json"
        if not out_path.exists():
            raise FileNotFoundError(f"Expected index file not found: {out_path}")
        return out_path

    raise ValueError("document must be .pdf, .md/.markdown, or .json")


def build_context(node_ids: list[str], all_nodes: dict[str, dict[str, Any]], top_k: int) -> str:
    chunks: list[str] = []
    for node_id in node_ids[:top_k]:
        node = all_nodes.get(node_id)
        if not node:
            continue
        title = node.get("title", "Untitled")
        start = node.get("start_index", node.get("line_num", "?"))
        end = node.get("end_index", node.get("line_num", "?"))
        text = node.get("text") or node.get("summary") or node.get("prefix_summary") or ""
        text = textwrap.shorten(str(text).replace("\n", " "), width=1200, placeholder="...")
        chunks.append(f"[{node_id}] {title} ({start}-{end})\n{text}")
    return "\n\n".join(chunks)


def print_retrieved_nodes(
    node_ids: list[str],
    all_nodes: dict[str, dict[str, Any]],
    scores: dict[str, float] | None = None,
) -> None:
    print("\nRetrieved nodes:")
    if not node_ids:
        print("- none")
        return

    for node_id in node_ids:
        node = all_nodes.get(node_id)
        if not node:
            print(f"- [{node_id}] NOT FOUND IN INDEX")
            continue
        start = node.get("start_index", node.get("line_num", "?"))
        end = node.get("end_index", node.get("line_num", "?"))
        score_part = ""
        if scores and node_id in scores:
            score_part = f" score={scores[node_id]:.2f}"
        print(f"- [{node_id}] {node.get('title', 'Untitled')} ({start}-{end}){score_part}")


def ensure_llm_available(api_key: str | None, reason: str) -> str:
    if not api_key:
        raise ValueError(f"API key required for {reason}. Set CHATGPT_API_KEY/OPENAI_API_KEY or pass --api-key.")
    return api_key


def execute_query(
    query: str,
    index: dict[str, Any],
    all_nodes: dict[str, dict[str, Any]],
    cfg: PipelineConfig,
    api_key: str | None,
) -> QueryOutcome:
    if cfg.retrieval_mode == "llm":
        key = ensure_llm_available(api_key, "LLM retrieval")
        retrieval_result = run_llm_tree_search(query, index, cfg.retrieve_model, key)
    else:
        retrieval_result = run_tree_retrieval(query, all_nodes, candidate_k=max(1, cfg.candidate_k))

    retrieved_ids = [nid for nid in retrieval_result.get("node_list", []) if nid in all_nodes]
    retrieved_ids = retrieved_ids[: max(1, cfg.candidate_k)]

    rerank_result: dict[str, Any] | None = None
    if cfg.llm_rerank:
        key = ensure_llm_available(api_key, "LLM reranking")
        rerank_result = run_llm_rerank(
            query=query,
            candidate_ids=retrieved_ids,
            all_nodes=all_nodes,
            model=cfg.retrieve_model,
            api_key=key,
            top_k=max(1, cfg.top_k),
        )
        retrieved_ids = [nid for nid in rerank_result.get("node_list", []) if nid in all_nodes]

    retrieved_ids = retrieved_ids[: max(1, cfg.top_k)]

    answer: str | None
    if cfg.answer_mode == "none":
        answer = None
    elif cfg.answer_mode == "extractive":
        answer = run_answer_extractive(query, retrieved_ids, all_nodes, max(1, cfg.top_k))
    else:
        key = ensure_llm_available(api_key, "LLM answer synthesis")
        context = build_context(retrieved_ids, all_nodes, top_k=max(1, cfg.top_k))
        if not context:
            answer = "Insufficient retrieved context to answer this question."
        else:
            answer = run_answer_llm(query, context, cfg.answer_model, key)

    eval_result: dict[str, Any] | None = None
    if cfg.eval_inline:
        key = ensure_llm_available(api_key, "inline LLM evaluation")
        eval_result = run_inline_eval(
            query=query,
            node_ids=retrieved_ids,
            all_nodes=all_nodes,
            answer=answer,
            model=cfg.eval_model,
            api_key=key,
        )

    return QueryOutcome(
        retrieved_ids=retrieved_ids,
        answer=answer,
        retrieval_result=retrieval_result,
        rerank_result=rerank_result,
        eval_result=eval_result,
    )


def print_query_outcome(outcome: QueryOutcome, all_nodes: dict[str, dict[str, Any]]) -> None:
    scores = outcome.retrieval_result.get("scores")
    print_retrieved_nodes(outcome.retrieved_ids, all_nodes, scores=scores if isinstance(scores, dict) else None)

    method = outcome.retrieval_result.get("_method", "unknown")
    print(f"\nRetrieval method: {method}")

    tokens_in = outcome.retrieval_result.get("_tokens_in")
    tokens_out = outcome.retrieval_result.get("_tokens_out")
    if tokens_in is not None and tokens_out is not None:
        print(f"Retriever tokens in/out: {tokens_in}/{tokens_out}")

    if outcome.rerank_result:
        rerank_in = outcome.rerank_result.get("_tokens_in")
        rerank_out = outcome.rerank_result.get("_tokens_out")
        if rerank_in is not None and rerank_out is not None:
            print(f"Reranker tokens in/out: {rerank_in}/{rerank_out}")

    if outcome.answer is not None:
        print("\nAnswer:")
        print(outcome.answer)

    if outcome.eval_result:
        eval_tokens_in = outcome.eval_result.get("_tokens_in")
        eval_tokens_out = outcome.eval_result.get("_tokens_out")
        if eval_tokens_in is not None and eval_tokens_out is not None:
            print(f"Inline eval tokens in/out: {eval_tokens_in}/{eval_tokens_out}")
        visible_eval = {k: v for k, v in outcome.eval_result.items() if not str(k).startswith("_")}
        print("\nInline evaluation:")
        print(json.dumps(visible_eval, indent=2))


def print_config(cfg: PipelineConfig) -> None:
    print("Current pipeline:")
    print(f"- retrieval_mode: {cfg.retrieval_mode}")
    print(f"- llm_rerank: {'on' if cfg.llm_rerank else 'off'}")
    print(f"- answer_mode: {cfg.answer_mode}")
    print(f"- eval_inline: {'on' if cfg.eval_inline else 'off'}")
    print(f"- top_k: {cfg.top_k}")
    print(f"- candidate_k: {cfg.candidate_k}")
    print(f"- retrieve_model: {cfg.retrieve_model}")
    print(f"- answer_model: {cfg.answer_model}")
    print(f"- eval_model: {cfg.eval_model}")


def run_conversation_repl(
    index: dict[str, Any],
    all_nodes: dict[str, dict[str, Any]],
    cfg: PipelineConfig,
    api_key: str | None,
) -> None:
    print("\nPageIndex Search REPL")
    print("Commands: :help :tree :nodes :config :retrieval <tree|llm> :answer <none|extractive|llm> :rerank <on|off> :eval <on|off> :topk <n> :quit")
    print_config(cfg)

    while True:
        try:
            raw = input("\nquery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        if not raw:
            continue
        if raw in {":quit", ":q", "quit", "exit"}:
            print("Bye.")
            return
        if raw == ":help":
            print("- Ask any question to run retrieval/answer with current pipeline.")
            print("- :tree prints compact tree.")
            print("- :nodes lists node IDs and titles.")
            print("- :config prints current pipeline settings.")
            print("- :retrieval <tree|llm> switches retrieval strategy.")
            print("- :answer <none|extractive|llm> switches answer stage.")
            print("- :rerank <on|off> toggles LLM rerank stage.")
            print("- :eval <on|off> toggles inline LLM evaluation stage.")
            print("- :topk <n> controls answer context size.")
            continue
        if raw == ":tree":
            print(compact_tree(index.get("structure", [])))
            continue
        if raw == ":nodes":
            for node_id in sorted(all_nodes):
                print(f"- [{node_id}] {all_nodes[node_id].get('title', 'Untitled')}")
            continue
        if raw == ":config":
            print_config(cfg)
            continue

        if raw.startswith(":retrieval "):
            value = raw.split(maxsplit=1)[1].strip().lower()
            if value in {"tree", "llm"}:
                cfg.retrieval_mode = value
                print(f"retrieval_mode set to {cfg.retrieval_mode}")
            else:
                print("Error: retrieval mode must be 'tree' or 'llm'.")
            continue

        if raw.startswith(":answer "):
            value = raw.split(maxsplit=1)[1].strip().lower()
            if value in {"none", "extractive", "llm"}:
                cfg.answer_mode = value
                print(f"answer_mode set to {cfg.answer_mode}")
            else:
                print("Error: answer mode must be 'none', 'extractive', or 'llm'.")
            continue

        if raw.startswith(":rerank "):
            value = raw.split(maxsplit=1)[1].strip().lower()
            if value in {"on", "off"}:
                cfg.llm_rerank = value == "on"
                print(f"llm_rerank set to {'on' if cfg.llm_rerank else 'off'}")
            else:
                print("Error: rerank must be 'on' or 'off'.")
            continue

        if raw.startswith(":eval "):
            value = raw.split(maxsplit=1)[1].strip().lower()
            if value in {"on", "off"}:
                cfg.eval_inline = value == "on"
                print(f"eval_inline set to {'on' if cfg.eval_inline else 'off'}")
            else:
                print("Error: eval must be 'on' or 'off'.")
            continue

        if raw.startswith(":topk "):
            value = raw.split(maxsplit=1)[1].strip()
            try:
                cfg.top_k = max(1, int(value))
                print(f"top_k set to {cfg.top_k}")
            except ValueError:
                print("Error: topk must be an integer.")
            continue

        try:
            outcome = execute_query(raw, index, all_nodes, cfg, api_key)
            print_query_outcome(outcome, all_nodes)
        except Exception as exc:
            print(f"Error: {exc}")


def needs_api_key(
    document_path: Path,
    index_only: bool,
    query: str | None,
    cfg: PipelineConfig,
) -> bool:
    suffix = document_path.suffix.lower()
    if suffix in {".pdf", ".md", ".markdown"}:
        return True
    if index_only:
        return False

    if query is None:
        # REPL may still need key if default pipeline uses LLM.
        return cfg.retrieval_mode == "llm" or cfg.llm_rerank or cfg.answer_mode == "llm" or cfg.eval_inline

    return cfg.retrieval_mode == "llm" or cfg.llm_rerank or cfg.answer_mode == "llm" or cfg.eval_inline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tree-first vectorless search over PageIndex structures with optional LLM plug-ins."
    )
    parser.add_argument("document", help="Path to .pdf/.md/.markdown document, or an existing *_structure.json")
    parser.add_argument("query", nargs="?", help="Optional one-shot question")
    parser.add_argument("--query-text", help="Alias for positional query")

    parser.add_argument("--retrieve-model", default="gpt-4o-mini", help="Model for index build, LLM retrieval, and rerank")
    parser.add_argument("--answer-model", default="gpt-4o-mini", help="Model for LLM answer synthesis")

    parser.add_argument("--retrieval-mode", choices=["tree", "llm"], default="tree", help="Retrieval strategy")
    parser.add_argument("--llm-rerank", action="store_true", help="Apply LLM reranking over retrieved candidates")
    parser.add_argument("--answer-mode", choices=["none", "extractive", "llm"], default="extractive", help="Answer stage")
    parser.add_argument("--eval-inline", action="store_true", help="Run inline LLM evaluation of retrieval/answer quality")

    parser.add_argument("--top-k", type=int, default=5, help="Final node count passed to answer stage")
    parser.add_argument("--candidate-k", type=int, default=12, help="Candidate node count before optional rerank")

    parser.add_argument("--no-answer", action="store_true", help="Shortcut for --answer-mode none")
    parser.add_argument("--index-only", action="store_true", help="Only build/resolve index")
    parser.add_argument("--eval-model", default="gpt-4o-mini", help="Model for inline LLM evaluation")
    parser.add_argument("--api-key", help="API key, otherwise use CHATGPT_API_KEY or OPENAI_API_KEY")
    args = parser.parse_args()

    query = args.query_text or args.query
    document_path = Path(args.document).expanduser().resolve()
    if not document_path.exists():
        print(f"Error: path not found: {document_path}", file=sys.stderr)
        sys.exit(1)

    answer_mode = "none" if args.no_answer else args.answer_mode
    cfg = PipelineConfig(
        retrieval_mode=args.retrieval_mode,
        answer_mode=answer_mode,
        llm_rerank=args.llm_rerank,
        eval_inline=args.eval_inline,
        top_k=max(1, args.top_k),
        candidate_k=max(1, args.candidate_k),
        retrieve_model=args.retrieve_model,
        answer_model=args.answer_model,
        eval_model=args.eval_model,
    )

    api_key = resolve_api_key(args.api_key)
    if needs_api_key(document_path, args.index_only, query, cfg) and not api_key:
        print("Error: selected pipeline requires API key. Set CHATGPT_API_KEY/OPENAI_API_KEY or pass --api-key.", file=sys.stderr)
        sys.exit(1)

    env = os.environ.copy()
    if api_key:
        env.setdefault("OPENAI_API_KEY", api_key)
        env.setdefault("CHATGPT_API_KEY", api_key)

    try:
        index_path = build_or_use_index(document_path, args.retrieve_model, env)
    except Exception as exc:
        print(f"Error while building/loading index: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Index: {index_path}")
    if args.index_only:
        return

    index = json.loads(index_path.read_text())
    all_nodes = flatten_nodes(index.get("structure", []))

    if query:
        try:
            outcome = execute_query(query, index, all_nodes, cfg, api_key)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        print_query_outcome(outcome, all_nodes)
        return

    run_conversation_repl(index, all_nodes, cfg, api_key)


if __name__ == "__main__":
    main()
