#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pageindex",
#   "openai>=1.0",
#   "pymupdf>=1.24",
#   "PyPDF2>=3.0",
#   "tiktoken>=0.5",
#   "python-dotenv>=1.0",
#   "pyyaml>=6.0",
#   "rich>=13.0",
# ]
#
# [tool.uv.sources]
# pageindex = { path = "." }   # local dev; remove once published to PyPI
# ///
"""
PageIndex pipeline — index a PDF (or folder) then open the eval REPL.

Usage:
    uv run run.py report.pdf              # index then eval
    uv run run.py report.pdf --index-only # index only
    uv run run.py report.json             # eval an existing index
    uv run run.py ./docs/                 # pick from folder
    uv run run.py report.pdf --model gpt-4o-mini
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
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.prompt import Prompt
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

# eval REPL lives outside the package — import from sibling script
sys.path.insert(0, str(Path(__file__).parent))
from eval_repl import flatten_nodes, resolve_index, run_repl, run_benchmark  # noqa: E402


# ─── UI helpers ──────────────────────────────────────────────────────────────

def _print(msg: str, style: str = "") -> None:
    if HAS_RICH:
        console.print(f"[{style}]{msg}[/]" if style else msg)
    else:
        print(msg)


# ─── Input resolution (file/folder, PDF or JSON) ─────────────────────────────

def resolve_input(raw: str) -> tuple[Path, str]:
    """Return (path, kind) where kind is 'pdf' or 'json'."""
    p = Path(raw)
    if p.is_file():
        ext = p.suffix.lower()
        if ext == ".pdf":  return p, "pdf"
        if ext == ".json": return p, "json"
        _print(f"Unsupported file type: {ext}", "red"); sys.exit(1)

    if p.is_dir():
        json_files = sorted(p.rglob("*_structure.json"))
        pdf_files  = sorted(p.rglob("*.pdf"))
        if json_files and pdf_files:
            _print("Folder has both PDFs and indices.\n  1. JSON indices\n  2. PDF files")
            raw2 = (Prompt.ask("Select", default="1") if HAS_RICH else input("Select [1]: ").strip()) or "1"
            use_json = raw2.strip() not in ("2", "pdf")
        else:
            use_json = bool(json_files)
        files = json_files if use_json else pdf_files
        if not files:
            _print(f"No files found in {p}", "red"); sys.exit(1)
        _print(f"\n[bold]{'Indices' if use_json else 'PDFs'} in [cyan]{p}[/]:[/]" if HAS_RICH
               else f"\n{'Indices' if use_json else 'PDFs'} in {p}:")
        for i, f in enumerate(files, 1):
            _print(f"  [cyan]{i}[/]. {f.name}" if HAS_RICH else f"  {i}. {f.name}")
        raw3 = (Prompt.ask("\nSelect", default="1") if HAS_RICH else input("\nSelect [1]: ").strip()) or "1"
        try:
            return files[int(raw3) - 1], "json" if use_json else "pdf"
        except (ValueError, IndexError):
            _print(f"Invalid selection", "red"); sys.exit(1)

    _print(f"Path not found: {p}", "red"); sys.exit(1)


# ─── Indexing (calls the pageindex package) ───────────────────────────────────

def build_index(pdf_path: Path, model: str, out_dir: Path) -> Path:
    from pageindex import page_index_main
    from pageindex.utils import ConfigLoader

    opt = ConfigLoader().load({"model": model})
    _print(f"\n[bold]Indexing[/] [cyan]{pdf_path.name}[/] with [cyan]{model}[/]…" if HAS_RICH
           else f"\nIndexing {pdf_path.name} with {model}…")
    t0 = time.perf_counter()
    if HAS_RICH:
        with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                      TimeElapsedColumn(), console=console) as prog:
            prog.add_task("Building tree index…", total=None)
            result = page_index_main(str(pdf_path), opt)
    else:
        result = page_index_main(str(pdf_path), opt)

    _print(f"Done in {time.perf_counter() - t0:.1f}s", "green")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pdf_path.stem}_structure.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    _print(f"Saved → [cyan]{out_path}[/]" if HAS_RICH else f"Saved → {out_path}")
    return out_path


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PageIndex — index a PDF then query it interactively",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run run.py report.pdf                # index then open eval REPL
  uv run run.py report.pdf --index-only   # index only
  uv run run.py report_structure.json     # eval an existing index
  uv run run.py ./docs/                   # pick from folder
  uv run run.py report.pdf --model gpt-4o-mini
  uv run run.py report.json --bench bench.json
        """,
    )
    parser.add_argument("path", help="PDF file, JSON index, or folder")
    parser.add_argument("--model", default="gpt-4o-2024-11-20")
    parser.add_argument("--out", default="./results", help="Output dir for built indices")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--bench", help="Benchmark JSON for batch eval")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("CHATGPT_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        _print("Error: set CHATGPT_API_KEY or OPENAI_API_KEY, or pass --api-key", "red")
        sys.exit(1)

    file_path, kind = resolve_input(args.path)

    index_path = build_index(file_path, args.model, Path(args.out)) if kind == "pdf" else file_path

    if args.index_only:
        return

    index = json.loads(index_path.read_text())
    all_nodes = flatten_nodes(index.get("structure", []))

    if args.bench:
        run_benchmark(args.bench, index, all_nodes, args.model, api_key)
    else:
        run_repl(index, all_nodes, args.model, api_key)


if __name__ == "__main__":
    main()
