---
name: pageindex-search
description: Use PageIndex for tree-first vectorless search on user-provided documents via `uv` scripts, and plug in LLM retrieval/rerank/answer stages only when needed.
---

# PageIndex Search

Run tree-centric retrieval over a provided path, with optional LLM stages.

## Primary Workflow

Use this command first:

```bash
uv run .claude/skills/pageindex-search/scripts/pageindex_search.py <document-path> "<question>"
```

Supported inputs:
- `.pdf`: build PageIndex structure then query.
- `.md` / `.markdown`: build structure then query.
- `*_structure.json`: query existing structure directly.

Default pipeline is tree-first and API-key-free for existing JSON:
- `retrieval_mode=tree`
- `answer_mode=extractive`
- `llm_rerank=off`

## Common Commands

```bash
# Tree-only retrieval (no final answer stage)
uv run .claude/skills/pageindex-search/scripts/pageindex_search.py <document-path> "<question>" --no-answer

# Tree retrieval + LLM answer synthesis
uv run .claude/skills/pageindex-search/scripts/pageindex_search.py <document-path> "<question>" --answer-mode llm

# Tree retrieval + LLM rerank + LLM answer
uv run .claude/skills/pageindex-search/scripts/pageindex_search.py <document-path> "<question>" --llm-rerank --answer-mode llm

# Same call, with inline LLM quality evaluation in the same run
uv run .claude/skills/pageindex-search/scripts/pageindex_search.py <document-path> "<question>" --llm-rerank --answer-mode llm --eval-inline

# Pure LLM retrieval + LLM answer
uv run .claude/skills/pageindex-search/scripts/pageindex_search.py <document-path> "<question>" --retrieval-mode llm --answer-mode llm

# Build/resolve index only
uv run .claude/skills/pageindex-search/scripts/pageindex_search.py <document-path> --index-only

# Open interactive conversation REPL over the index
uv run .claude/skills/pageindex-search/scripts/pageindex_search.py <document-path>
```

## REPL Controls

When running without a query, use one conversation and switch modes live:
- `:help`
- `:tree`
- `:nodes`
- `:config`
- `:retrieval <tree|llm>`
- `:answer <none|extractive|llm>`
- `:rerank <on|off>`
- `:eval <on|off>`
- `:topk <n>`
- `:quit`

## Options

- `--retrieval-mode`: `tree` or `llm` (default `tree`)
- `--llm-rerank`: apply LLM reranking over candidates
- `--answer-mode`: `none`, `extractive`, or `llm` (default `extractive`)
- `--eval-inline`: run inline LLM evaluation over retrieval/answer quality
- `--retrieve-model`: model for index build/LLM retrieval/rerank (default `gpt-4o-mini`)
- `--answer-model`: model for LLM answer synthesis (default `gpt-4o-mini`)
- `--eval-model`: model for inline evaluation (default `gpt-4o-mini`)
- `--top-k`: final node count passed to answer stage (default `5`)
- `--candidate-k`: candidate nodes before optional rerank (default `12`)
- `--api-key`: explicit API key; otherwise use `CHATGPT_API_KEY` or `OPENAI_API_KEY`

## Preflight Checks

1. Confirm `uv` is available and run from repo root.
2. If indexing `.pdf`/`.md`, confirm API key env is set.
3. If using `--retrieval-mode llm`, `--llm-rerank`, `--answer-mode llm`, or `--eval-inline`, confirm API key env is set.
4. Confirm the input path exists.

## Fallback Tools

For retrieval eval benchmarks:

```bash
uv run eval_repl.py <path/to/index.json> --bench tests/bench_template.json
```

Use benchmark mode for precision/recall checks; use `pageindex_search.py` for tree-first search and one-session mode switching.
