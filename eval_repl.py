#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "openai>=1.0",
#   "python-dotenv>=1.0",
#   "rich>=13.0",
# ]
# ///
"""
PageIndex Eval REPL — standalone eval script, separate from the core package.

Works with pre-built index JSON files only (no indexing).
For indexing + eval see run.py.

Usage:
    uv run eval_repl.py report_structure.json
    uv run eval_repl.py ./tests/results/          # folder picker
    uv run eval_repl.py report.json --bench bench.json
    uv run eval_repl.py report.json --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.table import Table
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None  # type: ignore[assignment]

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ─── UI helpers ──────────────────────────────────────────────────────────────

def _print(msg: str, style: str = "") -> None:
    if HAS_RICH:
        console.print(f"[{style}]{msg}[/]" if style else msg)
    else:
        print(msg)


# ─── Index resolution ────────────────────────────────────────────────────────

def resolve_index(raw: str) -> Path:
    """Return path to a JSON index. Folder → interactive picker."""
    p = Path(raw)
    if p.is_file():
        return p
    if p.is_dir():
        files = sorted(f for f in p.rglob("*.json") if not f.name.endswith(".results.json"))
        if not files:
            _print(f"No JSON files found in {p}", "red"); sys.exit(1)
        _print(f"\n[bold]Indices in [cyan]{p}[/]:[/]" if HAS_RICH else f"\nIndices in {p}:")
        for i, f in enumerate(files, 1):
            _print(f"  [cyan]{i}[/]. {f.name}" if HAS_RICH else f"  {i}. {f.name}")
        raw2 = (Prompt.ask("\nSelect", default="1") if HAS_RICH else input("\nSelect [1]: ").strip()) or "1"
        try:
            return files[int(raw2) - 1]
        except (ValueError, IndexError):
            _print(f"Invalid selection '{raw2}'", "red"); sys.exit(1)
    _print(f"Path not found: {p}", "red"); sys.exit(1)


# ─── Tree search ─────────────────────────────────────────────────────────────

_PROMPT = """\
You are given a query and a PageIndex document tree. Find all nodes most likely \
to contain the answer.

Query: {query}

Tree:
{tree}

Reply ONLY with valid JSON:
{{"thinking": "<brief reasoning>", "node_list": ["<node_id>", ...]}}
"""


def flatten_nodes(structure: list, _acc: dict | None = None) -> dict[str, dict]:
    if _acc is None:
        _acc = {}
    for node in structure:
        if nid := node.get("node_id"):
            _acc[nid] = node
        if node.get("nodes"):
            flatten_nodes(node["nodes"], _acc)
    return _acc


def compact_tree(structure: list, depth: int = 0) -> str:
    lines = []
    for node in structure:
        nid = node.get("node_id", "—")
        title = node.get("title", "Untitled")
        s, e = node.get("start_index", "?"), node.get("end_index", "?")
        summary = node.get("summary", "")
        snippet = (summary[:120] + "…") if len(summary) > 120 else summary
        indent = "  " * depth
        lines.append(f"{indent}[{nid}] {title} (p{s}–{e})")
        if snippet:
            lines.append(f"{indent}     └ {snippet}")
        if node.get("nodes"):
            lines.extend(compact_tree(node["nodes"], depth + 1).splitlines())
    return "\n".join(lines)


def tree_search(query: str, index: dict, model: str, api_key: str) -> dict:
    import openai
    prompt = _PROMPT.format(query=query, tree=compact_tree(index.get("structure", [])))
    client = openai.OpenAI(api_key=api_key)
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], temperature=0,
    )
    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"thinking": raw, "node_list": []}
    result.update(_elapsed=time.perf_counter() - t0,
                  _tokens_in=resp.usage.prompt_tokens,
                  _tokens_out=resp.usage.completion_tokens)
    return result


# ─── Display ─────────────────────────────────────────────────────────────────

def print_result(result: dict, all_nodes: dict) -> None:
    node_ids = result.get("node_list", [])
    elapsed = result.get("_elapsed", 0)
    tok_in, tok_out = result.get("_tokens_in", 0), result.get("_tokens_out", 0)
    if HAS_RICH:
        console.print(f"\n[bold cyan]Reasoning:[/] {result.get('thinking', '')}\n")
        console.print(f"[dim]⏱  {elapsed:.2f}s  |  tokens in/out: {tok_in}/{tok_out}[/]\n")
        if not node_ids:
            console.print("[yellow]No nodes retrieved.[/]"); return
        t = Table(title=f"Retrieved {len(node_ids)} node(s)", show_lines=True)
        t.add_column("ID", style="bold magenta", width=8)
        t.add_column("Title", style="bold")
        t.add_column("Pages", width=8)
        t.add_column("Summary", max_width=60)
        for nid in node_ids:
            n = all_nodes.get(nid)
            if n:
                s = n.get("summary", "—")
                t.add_row(nid, n.get("title", "—"),
                          f"{n.get('start_index','?')}–{n.get('end_index','?')}",
                          s[:200] + "…" if len(s) > 200 else s)
            else:
                t.add_row(nid, "[red]NOT FOUND[/]", "—", "—")
        console.print(t)
    else:
        print(f"\nReasoning: {result.get('thinking', '')}")
        print(f"Time: {elapsed:.2f}s | tokens: {tok_in}/{tok_out}")
        for nid in node_ids:
            n = all_nodes.get(nid, {})
            print(f"  [{nid}] {n.get('title','?')} p{n.get('start_index','?')}–{n.get('end_index','?')}")


# ─── Benchmark ───────────────────────────────────────────────────────────────

def run_benchmark(bench_path: str, index: dict, all_nodes: dict, model: str, api_key: str) -> None:
    cases = json.loads(Path(bench_path).read_text())
    hits, results = 0, []
    for i, case in enumerate(cases, 1):
        query = case["query"]
        expected = set(case.get("expected_nodes", []))
        if HAS_RICH:
            console.rule(f"[{i}/{len(cases)}] {query[:70]}")
        else:
            print(f"\n[{i}/{len(cases)}] {query}")
        result = tree_search(query, index, model, api_key)
        retrieved = set(result.get("node_list", []))
        hit = bool(expected & retrieved) if expected else True
        if hit: hits += 1
        precision = len(expected & retrieved) / len(retrieved) if retrieved else 0
        recall = len(expected & retrieved) / len(expected) if expected else 1.0
        results.append(dict(query=query, expected=list(expected), retrieved=list(retrieved),
                            hit=hit, precision=precision, recall=recall, elapsed=result["_elapsed"]))
        print_result(result, all_nodes)
        color = "green" if hit else "red"
        _print(f"[{color}]{'✓' if hit else '✗'} P={precision:.2f} R={recall:.2f}[/]" if HAS_RICH
               else f"{'✓' if hit else '✗'} P={precision:.2f} R={recall:.2f}")

    acc = hits / len(cases) if cases else 0
    avg = sum(r["elapsed"] for r in results) / len(results) if results else 0
    if HAS_RICH:
        console.rule("Summary")
        t = Table(); t.add_column("Metric"); t.add_column("Value", style="bold")
        t.add_row("Queries", str(len(cases))); t.add_row("Hit rate", f"{acc:.1%}")
        t.add_row("Avg latency", f"{avg:.2f}s"); console.print(t)
    else:
        print(f"\nQueries: {len(cases)}  Hit rate: {acc:.1%}  Avg: {avg:.2f}s")
    out = Path(bench_path).with_suffix(".results.json")
    out.write_text(json.dumps(results, indent=2))
    _print(f"Results → [cyan]{out}[/]" if HAS_RICH else f"Results → {out}")


# ─── REPL ────────────────────────────────────────────────────────────────────

def run_repl(index: dict, all_nodes: dict, model: str, api_key: str) -> None:
    doc_name = index.get("doc_name", "document")
    if HAS_RICH:
        console.print(Panel(
            f"[bold]PageIndex Eval REPL[/]\n"
            f"Document: [cyan]{doc_name}[/]  |  Nodes: [cyan]{len(all_nodes)}[/]  |  Model: [cyan]{model}[/]\n\n"
            "Commands: [bold]:tree[/]  [bold]:nodes[/]  [bold]:quit[/]",
            title="[bold green]PageIndex[/]",
        ))
    else:
        print(f"\n=== PageIndex  |  {doc_name}  |  {len(all_nodes)} nodes  |  {model} ===")
        print("Commands: :tree  :nodes  :quit\n")

    while True:
        try:
            query = (Prompt.ask("\n[bold green]query[/]") if HAS_RICH else input("\nquery> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye."); break
        if not query: continue
        if query in (":quit", ":q", "exit", "quit"):
            print("Bye."); break
        if query == ":tree":
            txt = compact_tree(index.get("structure", []))
            (console.print(txt) if HAS_RICH else print(txt)); continue
        if query == ":nodes":
            for nid, n in all_nodes.items():
                print(f"  [{nid}] {n.get('title','?')}"); continue
        try:
            print_result(tree_search(query, index, model, api_key), all_nodes)
        except Exception as e:
            _print(f"Error: {e}", "red")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PageIndex Eval REPL — query pre-built index JSONs interactively",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run eval_repl.py tests/results/q1-fy25-earnings_structure.json
  uv run eval_repl.py tests/results/          # folder picker
  uv run eval_repl.py report.json --bench bench.json
  uv run eval_repl.py report.json --model gpt-4o-mini
        """,
    )
    parser.add_argument("path", nargs="?", help="JSON index file or folder of indices")
    parser.add_argument("--model", default="gpt-4o-2024-11-20")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--bench", help="Benchmark JSON for batch eval")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("CHATGPT_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        _print("Error: set CHATGPT_API_KEY or OPENAI_API_KEY, or pass --api-key", "red")
        sys.exit(1)

    raw = args.path or str(Path(__file__).parent / "tests" / "results")
    index_path = resolve_index(raw)
    index = json.loads(index_path.read_text())
    all_nodes = flatten_nodes(index.get("structure", []))

    if args.bench:
        run_benchmark(args.bench, index, all_nodes, args.model, api_key)
    else:
        run_repl(index, all_nodes, args.model, api_key)


if __name__ == "__main__":
    main()
