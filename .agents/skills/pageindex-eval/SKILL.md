---
name: pageindex-eval
description: Run PageIndex vectorless RAG and retrieval evaluation on a user-provided document path via `uv` scripts. Use when the user wants to process a provided PDF/Markdown/structure JSON, ask questions against it, benchmark retrieval quality, or open an interactive PageIndex REPL for document-tree search.
---

# PageIndex Eval

Run vectorless RAG on a provided document and evaluate retrieval behavior.

## Primary Command

```bash
# One-shot vectorless RAG on a provided document/index
uv run .Codex/skills/pageindex-eval/scripts/vectorless_rag_eval.py <document-path> "<question>"

# Retrieval-only mode (no answer synthesis)
uv run .Codex/skills/pageindex-eval/scripts/vectorless_rag_eval.py <document-path> "<question>" --no-answer

# Build or resolve index only
uv run .Codex/skills/pageindex-eval/scripts/vectorless_rag_eval.py <document-path> --index-only

# Interactive REPL after indexing/resolving
uv run .Codex/skills/pageindex-eval/scripts/vectorless_rag_eval.py <document-path>
```

`<document-path>` can be:
- `.pdf` (build index then query)
- `.md` / `.markdown` (build index then query)
- `*_structure.json` (query existing index)

## Run Preflight Checks

1. Confirm `CHATGPT_API_KEY` or `OPENAI_API_KEY` is set, or pass `--api-key`.
2. Use `--retrieve-model gpt-4o-mini` for lower-cost retrieval iteration.
3. Use `--answer-model gpt-4o` when answer quality matters more than cost.
4. Keep prompts specific so retrieval returns fewer, high-signal nodes.

## REPL Commands

- `:help`: Show available commands.
- `:tree`: Print the full tree.
- `:nodes`: List all node IDs and titles.
- `:quit` or `:q`: Exit.

## Direct Eval Commands

```bash
# Benchmark against expected node IDs
uv run eval_repl.py <path/to/index.json> --bench tests/bench_template.json

# Query index or folder of indices in REPL
uv run eval_repl.py <path/to/index.json-or-folder>
```

Benchmark case format:

```json
[
  {
    "query": "What was the total revenue for Q1 FY2025?",
    "expected_nodes": ["0001", "0002"],
    "notes": "Revenue appears in the Financial Results section."
  }
]
```

Interpret metrics as:
- `hit`: at least one expected node retrieved.
- `precision`: overlap divided by retrieved count.
- `recall`: overlap divided by expected count.

## Common Paths

- `tests/results/`: sample `*_structure.json` files for quick eval.
- `tests/bench_template.json`: benchmark starter template.
- `.Codex/skills/pageindex-eval/scripts/vectorless_rag_eval.py`: primary uv script for provided documents.

## Troubleshoot Quickly

- Fix `set CHATGPT_API_KEY or OPENAI_API_KEY` errors by exporting a key or passing `--api-key`.
- Treat empty `node_list` as retrieval miss; tighten query wording or try a stronger retrieval model.
- If indexing a PDF/Markdown file fails, confirm path and run `uv sync` from repo root.
