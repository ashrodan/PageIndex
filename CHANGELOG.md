# Change Log
All notable changes to this project will be documented in this file.

## Beta - 2026-03-04

### Added

- **`pageindex_core.py`** — Self-contained, single-file distillation of the entire PageIndex package (`utils.py` + `page_index.py` + `page_index_md.py`) into one portable `uv run`-able script with inline `/// script` dependency declarations. Drop-in replacement for the `pageindex` CLI that requires no install step.

- **`run.py`** — Unified pipeline entry point. Accepts a PDF, Markdown file, existing `*_structure.json`, or a folder (interactive picker). Indexes the document if needed and then opens the eval REPL. Supports `--index-only` to skip the REPL stage. Uses inline `/// script` deps.

- **`eval_repl.py`** (standalone, uplift) — Reworked as a minimal standalone script with only `openai`, `python-dotenv`, and `rich` as dependencies. Works with pre-built index JSON only; for full index-then-eval use `run.py`.

- **`.claude/skills/pageindex-eval/`** — New Claude Code skill wrapping the eval REPL via `vectorless_rag_eval.py`. Triggers when a user wants to interactively query a `*_structure.json` index or run batch benchmarks.

- **`.claude/skills/pageindex-search/`** — New Claude Code skill for tree-first vectorless search (`pageindex_search.py`). Default pipeline is API-key-free for existing JSON (`retrieval_mode=tree`, `answer_mode=extractive`). Optional stages: `--llm-rerank`, `--answer-mode llm`.

- **`Makefile`** — New makefile with targets for packaging and uploading skills to OpenAI Hosted Skills: `skill-zip`, `skill-upload-new`, `skill-upload-version`, `skill-list`.

- **Expanded test suite** — `tests/test_pageindex_core.py` (unit tests for `pageindex_core.py`) and `tests/test_pageindex_core_integration.py` (integration tests). Existing `test_utils.py` and `test_page_index.py` also expanded.

### Changed

- **`pyproject.toml`** — Added `pageindex-eval = "pageindex.eval_repl:main"` script entry point. `uv.lock` regenerated.

## Beta - 2026-03-03

### Fixed

- **Silent data corruption in `fix_incorrect_toc`** (`page_index.py`) — The inner loop variable `list_index = page_index - start_index` was shadowing the outer `list_index = incorrect_item['list_index']`. The return dict then held the wrong value, causing `toc_with_page_number` to update the wrong TOC entry. Renamed inner variable to `page_list_idx`.

- **Infinite-loop risk in `extract_toc_content`** (`page_index.py`) — The termination guard `if len(chat_history) > 5` never fired because `chat_history` is rebuilt as a fresh 2-element list on every iteration. Replaced with an explicit `attempt` counter that raises after 5 while-loop iterations.

- **OpenAI client created on every API call** (`utils.py`) — Every call to `ChatGPT_API`, `ChatGPT_API_with_finish_reason`, and `ChatGPT_API_async` opened a new `openai.OpenAI` / `openai.AsyncOpenAI` instance and a new HTTP connection pool. A 200-page document can trigger 300+ calls. Replaced with module-level lazy singletons via `_get_sync_client()` / `_get_async_client()`.

- **`async with` misuse in `ChatGPT_API_async`** (`utils.py`) — The async client was opened and closed inside the retry loop using `async with openai.AsyncOpenAI(...) as client:`, destroying it on every attempt. Removed the context manager; the singleton now persists across retries.

- **Static 1-second retry sleep in all API functions** (`utils.py`) — All three API wrappers slept a flat `1s` between retries, hammering rate limits at constant cadence. Replaced with exponential backoff with jitter: `min(2**i + random(), 60)s`.

- **`count_tokens` crashes when `model=None`** (`utils.py`) — `tiktoken.encoding_for_model(None)` raised `TypeError`. Many callers pass `model=None`. Added fallback: `model or "gpt-4o"`.

- **`meta_processor` raises instead of returning best-effort result** (`page_index.py`) — When all three processing modes were exhausted and `process_no_toc` accuracy was still below 0.6, the code raised `Exception('Processing failed')`, discarding the partial result entirely. Now logs a warning and returns the best available result.

- **Duplicate `import os`** (`page_index.py`) — Removed the redundant second `import os` at line 8.

### Changed

- **`JsonLogger` write strategy** (`utils.py`) — Every log call previously serialised and wrote the entire log array to disk, giving O(n²) I/O for log-heavy runs. Now buffers in memory and flushes only every 20 entries (configurable via `WRITE_THRESHOLD`), plus on explicit `close()` or `__del__`.

- **`cli.py` entry point** (`cli.py`, `run_pageindex.py`) — `runpy.run_path("run_pageindex.py")` only worked when invoked from the repo root; a pip-installed `pageindex` command would fail with "file not found". All argparse and dispatch logic moved into `pageindex.cli:main()`. `run_pageindex.py` is now a 4-line shim that calls `main()`.

### Added

- **Pytest suite** (`tests/test_utils.py`, `tests/test_page_index.py`) — 78 unit tests covering all fixes above, plus utility helpers (`extract_json`, `get_json_content`, `write_node_id`, `get_nodes`, `get_leaf_nodes`, `sanitize_filename`, `list_to_tree`). All tests run without hitting the OpenAI API. Run with `uv run pytest tests/`.

## Beta - 2025-04-23

### Fixed
- [x] Fixed a bug introduced on April 18 where `start_index` was incorrectly passed.

## Beta - 2025-04-03

### Added
- [x] Add node_id, node summary
- [x] Add document discription

### Changed
- [x] Change "child_nodes" -> "nodes" to simplify the structure
