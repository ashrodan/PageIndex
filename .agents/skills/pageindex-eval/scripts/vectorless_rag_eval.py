#!/usr/bin/env python3
"""
Run PageIndex vectorless RAG on a provided document or existing index.

Examples:
  uv run .claude/skills/pageindex-eval/scripts/vectorless_rag_eval.py tests/pdfs/q1-fy25-earnings.pdf "What was total revenue for Q1 FY2025?"
  uv run .claude/skills/pageindex-eval/scripts/vectorless_rag_eval.py tests/results/q1-fy25-earnings_structure.json --query-text "How did Sports perform?"
  uv run .claude/skills/pageindex-eval/scripts/vectorless_rag_eval.py tests/results/q1-fy25-earnings_structure.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


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


ANSWER_PROMPT = """\
Answer the user question using only the retrieved document context.
If the context is insufficient, say so explicitly.
Keep the answer concise and include node citations in square brackets like [0002].

Question:
{query}

Retrieved context:
{context}
"""


REPO_ROOT = Path(__file__).resolve().parents[4]


def flatten_nodes(structure: list[dict[str, Any]], acc: dict[str, dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
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
        start = node.get("start_index", "?")
        end = node.get("end_index", "?")
        summary = node.get("summary", "")
        summary_snippet = (summary[:120] + "...") if len(summary) > 120 else summary
        indent = "  " * depth
        lines.append(f"{indent}[{node_id}] {title} (p{start}-{end})")
        if summary_snippet:
            lines.append(f"{indent}  - {summary_snippet}")
        children = node.get("nodes") or []
        if children:
            lines.extend(compact_tree(children, depth + 1).splitlines())
    return "\n".join(lines)


def parse_json_from_model(raw: str) -> dict[str, Any]:
    payload = raw.strip()
    if payload.startswith("```"):
        payload = payload.split("```")[1]
        if payload.startswith("json"):
            payload = payload[4:]
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return {"thinking": payload, "node_list": []}


def run_tree_search(query: str, index: dict[str, Any], model: str, api_key: str) -> dict[str, Any]:
    import openai

    prompt = TREE_SEARCH_PROMPT.format(query=query, tree=compact_tree(index.get("structure", [])))
    client = openai.OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    out = parse_json_from_model(resp.choices[0].message.content or "")
    out["_tokens_in"] = resp.usage.prompt_tokens
    out["_tokens_out"] = resp.usage.completion_tokens
    return out


def run_answer(query: str, context: str, model: str, api_key: str) -> str:
    import openai

    prompt = ANSWER_PROMPT.format(query=query, context=context)
    client = openai.OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return (resp.choices[0].message.content or "").strip()


def resolve_api_key(explicit_key: str | None) -> str | None:
    return explicit_key or os.getenv("CHATGPT_API_KEY") or os.getenv("OPENAI_API_KEY")


def run_cmd(cmd: list[str], env: dict[str, str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def build_or_use_index(document_path: Path, retrieve_model: str, env: dict[str, str]) -> Path:
    suffix = document_path.suffix.lower()
    if suffix == ".json":
        return document_path

    if suffix in {".pdf", ".md", ".markdown"}:
        cmd = ["uv", "run", "pageindex", "--model", retrieve_model, "--if-add-node-id", "yes", "--if-add-node-summary", "yes"]
        if suffix == ".pdf":
            cmd.extend(["--pdf_path", str(document_path)])
        else:
            cmd.extend(["--md_path", str(document_path)])
        run_cmd(cmd, env)

        generated = REPO_ROOT / "results" / f"{document_path.stem}_structure.json"
        if not generated.exists():
            raise FileNotFoundError(f"Expected generated index not found: {generated}")
        return generated

    raise ValueError("document must be .pdf, .md/.markdown, or .json")


def build_context(node_ids: list[str], all_nodes: dict[str, dict[str, Any]], top_k: int) -> str:
    parts: list[str] = []
    for node_id in node_ids[:top_k]:
        node = all_nodes.get(node_id)
        if not node:
            continue
        title = node.get("title", "Untitled")
        start = node.get("start_index", "?")
        end = node.get("end_index", "?")
        text = node.get("text") or node.get("summary") or ""
        text = textwrap.shorten(str(text).replace("\n", " "), width=1200, placeholder="...")
        parts.append(f"[{node_id}] {title} (p{start}-{end})\n{text}")
    return "\n\n".join(parts)


def launch_repl(index_path: Path, retrieve_model: str, api_key: str, env: dict[str, str]) -> None:
    cmd = ["uv", "run", "eval_repl.py", str(index_path), "--model", retrieve_model, "--api-key", api_key]
    run_cmd(cmd, env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Vectorless RAG runner for a provided document using PageIndex.")
    parser.add_argument("document", help="Path to a .pdf/.md/.markdown document or existing *_structure.json")
    parser.add_argument("query", nargs="?", help="One-shot question to answer from the document")
    parser.add_argument("--query-text", help="Alias for positional query")
    parser.add_argument("--retrieve-model", default="gpt-4o-mini", help="Model used for tree retrieval and optional indexing")
    parser.add_argument("--answer-model", default="gpt-4o-mini", help="Model used for final answer synthesis")
    parser.add_argument("--top-k", type=int, default=5, help="Max retrieved nodes to include in answer context")
    parser.add_argument("--no-answer", action="store_true", help="Skip final answer synthesis and return retrieval only")
    parser.add_argument("--index-only", action="store_true", help="Build/resolve index and exit")
    parser.add_argument("--api-key", help="OpenAI API key (falls back to CHATGPT_API_KEY or OPENAI_API_KEY)")
    args = parser.parse_args()

    query = args.query_text or args.query
    document_path = Path(args.document).expanduser().resolve()
    if not document_path.exists():
        print(f"Error: path not found: {document_path}", file=sys.stderr)
        sys.exit(1)

    api_key = resolve_api_key(args.api_key)
    need_model_calls = not args.index_only or document_path.suffix.lower() in {".pdf", ".md", ".markdown"}
    if need_model_calls and not api_key:
        print("Error: set CHATGPT_API_KEY or OPENAI_API_KEY, or pass --api-key", file=sys.stderr)
        sys.exit(1)

    env = os.environ.copy()
    if api_key:
        env.setdefault("OPENAI_API_KEY", api_key)
        env.setdefault("CHATGPT_API_KEY", api_key)

    try:
        index_path = build_or_use_index(document_path, args.retrieve_model, env)
    except Exception as exc:
        print(f"Error while resolving index: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Index: {index_path}")

    if args.index_only:
        return

    if not query:
        launch_repl(index_path, args.retrieve_model, api_key or "", env)
        return

    index = json.loads(index_path.read_text())
    all_nodes = flatten_nodes(index.get("structure", []))
    result = run_tree_search(query, index, args.retrieve_model, api_key or "")
    node_list = result.get("node_list", [])

    print("\nRetrieved nodes:")
    if not node_list:
        print("- none")
    for node_id in node_list:
        node = all_nodes.get(node_id)
        if not node:
            print(f"- [{node_id}] NOT FOUND IN INDEX")
            continue
        print(
            f"- [{node_id}] {node.get('title', 'Untitled')} "
            f"(p{node.get('start_index', '?')}-{node.get('end_index', '?')})"
        )
    print(f"\nRetriever tokens in/out: {result.get('_tokens_in', 0)}/{result.get('_tokens_out', 0)}")

    if args.no_answer:
        return

    context = build_context(node_list, all_nodes, top_k=max(1, args.top_k))
    if not context:
        print("\nAnswer:\nInsufficient retrieved context to answer this question.")
        return

    answer = run_answer(query, context, args.answer_model, api_key or "")
    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()
