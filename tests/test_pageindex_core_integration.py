"""
Integration tests for pageindex_core.py using real PDFs and synthetic markdown.

All tests that touch the LLM pipeline mock only the OpenAI call surface
(core.ChatGPT_API, core._call_llm, core.ChatGPT_API_async) — everything
else (PDF reading, tokenisation, chunking, tree building, post-processing,
markdown parsing) runs against real code.

PDF used:      tests/pdfs/2023-annual-report-truncated.pdf (50 pages, ~3.4 MB)
Markdown used: synthetic fixture written to tmp_path per test
"""

import asyncio
import json
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
import pageindex_core as core

PDF = _ROOT / "tests" / "pdfs" / "2023-annual-report-truncated.pdf"
PDF_PATH = str(PDF)

# ---------------------------------------------------------------------------
# Fake TOC: two sections with physical indices in the second half of the
# document so verify_toc doesn't short-circuit (needs last idx >= pages/2).
# ---------------------------------------------------------------------------
_FAKE_TOC_JSON = json.dumps([
    {"structure": "1", "title": "Introduction",  "physical_index": "<physical_index_5>"},
    {"structure": "2", "title": "Main Content",  "physical_index": "<physical_index_30>"},
])


def _sync_llm_mock(model, prompt, api_key=None, *, chat_history=None, return_finish_reason=False):
    """
    Smart sync LLM mock.  Distinguishes call sites by prompt keywords:
      - _generate_toc_part (continue): returns empty list (no extra sections)
      - _generate_toc_part (init):     returns FAKE_TOC
      - any other:                     returns '{}'
    """
    if "continue the tree structure" in prompt:
        response = "[]"
    elif "generate the tree structure" in prompt:
        response = _FAKE_TOC_JSON
    else:
        response = "{}"
    return (response, "finished") if return_finish_reason else response


async def _async_llm_mock(model=None, prompt=None, api_key=None):
    """
    Smart async LLM mock. Handles:
      - check_title_appearance_in_start  →  start_begin: yes
      - check_title_appearance           →  answer: yes
      - generate_node_summary            →  plain summary string
    """
    if prompt and "start_begin" in prompt:
        return '{"thinking": "yes", "start_begin": "yes"}'
    if prompt and "answer" in prompt:
        return '{"thinking": "yes it appears", "answer": "yes"}'
    return "A concise summary of this document section."


def _no_toc_chatgpt_api(model=None, prompt=None, api_key=None, chat_history=None):
    """Always signals 'no TOC' for toc_detector_single_page."""
    return '{"thinking": "no toc detected", "toc_detected": "no"}'


def _patch_llm(monkeypatch):
    """Apply all three LLM patches in one call."""
    monkeypatch.setattr(core, "_call_llm", _sync_llm_mock)
    monkeypatch.setattr(core, "ChatGPT_API", _no_toc_chatgpt_api)
    monkeypatch.setattr(core, "ChatGPT_API_async", _async_llm_mock)


# ===========================================================================
# 1. PDF reading — purely local, no mocks needed
# ===========================================================================

class TestGetPageTokens:
    def test_returns_correct_page_count(self):
        page_list = core.get_page_tokens(PDF_PATH)
        assert len(page_list) == 50

    def test_each_item_is_text_token_tuple(self):
        page_list = core.get_page_tokens(PDF_PATH)
        for text, tokens in page_list:
            assert isinstance(text, str)
            assert isinstance(tokens, int)

    def test_token_counts_positive(self):
        page_list = core.get_page_tokens(PDF_PATH)
        # At least half the pages should have some text
        non_empty = [t for _, t in page_list if t > 0]
        assert len(non_empty) >= 25

    def test_total_tokens_in_expected_range(self):
        page_list = core.get_page_tokens(PDF_PATH)
        total = sum(t for _, t in page_list)
        # A 50-page annual report should be between 5k and 200k tokens
        assert 5_000 < total < 200_000

    def test_first_page_contains_text(self):
        page_list = core.get_page_tokens(PDF_PATH)
        first_text = page_list[0][0]
        assert len(first_text.strip()) > 0

    def test_pymupdf_parser_returns_same_count(self):
        pages_pypdf2 = core.get_page_tokens(PDF_PATH, pdf_parser="PyPDF2")
        pages_pymupdf = core.get_page_tokens(PDF_PATH, pdf_parser="PyMuPDF")
        assert len(pages_pypdf2) == len(pages_pymupdf) == 50

    def test_pymupdf_pages_have_text_and_tokens(self):
        page_list = core.get_page_tokens(PDF_PATH, pdf_parser="PyMuPDF")
        for text, tokens in page_list:
            assert isinstance(text, str)
            assert isinstance(tokens, int)

    def test_model_parameter_affects_token_count(self):
        # Different models may use different encodings — just check it doesn't crash
        pages = core.get_page_tokens(PDF_PATH, model="gpt-4o")
        assert len(pages) == 50


# ===========================================================================
# 2. check_toc with real pages, LLM mocked
# ===========================================================================

class TestCheckTocWithRealPages:
    """
    Verifies that the TOC detection pipeline correctly processes real page
    text and returns the right shape — LLM answers are mocked to "no TOC".
    """

    def test_returns_dict_with_required_keys(self, monkeypatch):
        monkeypatch.setattr(core, "ChatGPT_API", _no_toc_chatgpt_api)
        page_list = core.get_page_tokens(PDF_PATH)
        opt = core._load_config()
        result = core.check_toc(page_list, opt)
        assert "toc_content" in result
        assert "toc_page_list" in result
        assert "page_index_given_in_toc" in result

    def test_no_toc_path_returns_empty_page_list(self, monkeypatch):
        monkeypatch.setattr(core, "ChatGPT_API", _no_toc_chatgpt_api)
        page_list = core.get_page_tokens(PDF_PATH)
        opt = core._load_config()
        result = core.check_toc(page_list, opt)
        assert result["toc_page_list"] == []

    def test_no_toc_content_is_none(self, monkeypatch):
        monkeypatch.setattr(core, "ChatGPT_API", _no_toc_chatgpt_api)
        page_list = core.get_page_tokens(PDF_PATH)
        opt = core._load_config()
        result = core.check_toc(page_list, opt)
        assert result["toc_content"] is None


# ===========================================================================
# 3. _build_page_groups with real PDF pages
# ===========================================================================

class TestBuildPageGroupsWithRealPages:
    """Real page text, real token counting, no mocks."""

    def test_returns_at_least_one_group(self):
        page_list = core.get_page_tokens(PDF_PATH)
        groups = core._build_page_groups(page_list, start_index=1, model=None)
        assert len(groups) >= 1

    def test_all_groups_are_strings(self):
        page_list = core.get_page_tokens(PDF_PATH)
        groups = core._build_page_groups(page_list, start_index=1, model=None)
        assert all(isinstance(g, str) for g in groups)

    def test_first_page_label_in_first_group(self):
        page_list = core.get_page_tokens(PDF_PATH)
        groups = core._build_page_groups(page_list, start_index=1, model=None)
        assert "<physical_index_1>" in groups[0]

    def test_all_page_labels_present_across_groups(self):
        page_list = core.get_page_tokens(PDF_PATH)
        groups = core._build_page_groups(page_list, start_index=1, model=None)
        combined = "".join(groups)
        # Every page should be labeled
        for i in range(1, 51):
            assert f"<physical_index_{i}>" in combined

    def test_start_index_offset_applied(self):
        page_list = core.get_page_tokens(PDF_PATH)[:5]
        groups = core._build_page_groups(page_list, start_index=10, model=None)
        combined = "".join(groups)
        assert "<physical_index_10>" in combined
        assert "<physical_index_14>" in combined
        assert "<physical_index_1>" not in combined


# ===========================================================================
# 4. Full pipeline smoke test — real PDF, mocked LLM
# ===========================================================================

class TestPageIndexMainSmoke:
    """
    End-to-end smoke test: real PDF bytes → real page extraction → real
    chunking/post-processing → mocked LLM responses → valid output structure.

    Assertions are structural only (shape, types, invariants), not content.
    """

    @pytest.fixture(autouse=True)
    def _patch(self, monkeypatch):
        _patch_llm(monkeypatch)

    def _run(self, **opt_overrides):
        defaults = dict(
            model="gpt-4o",
            if_add_node_summary="no",    # skip async summary generation
            if_add_node_text="no",
            if_add_doc_description="no",
            if_add_node_id="yes",
            max_page_num_each_node=100,  # prevent recursive splitting
            max_token_num_each_node=10_000_000,
        )
        defaults.update(opt_overrides)
        opt = core._load_config(defaults)
        return core.page_index_main(PDF_PATH, opt)

    def test_output_has_doc_name_and_structure(self):
        result = self._run()
        assert "doc_name" in result
        assert "structure" in result

    def test_doc_name_matches_filename(self):
        result = self._run()
        assert "2023-annual-report-truncated" in result["doc_name"]

    def test_structure_is_nonempty_list(self):
        result = self._run()
        assert isinstance(result["structure"], list)
        assert len(result["structure"]) > 0

    def test_every_node_has_title_and_indices(self):
        result = self._run()
        for node in core.structure_to_list(result["structure"]):
            assert "title" in node
            assert "start_index" in node
            assert "end_index" in node

    def test_every_node_has_node_id(self):
        result = self._run()
        for node in core.structure_to_list(result["structure"]):
            assert "node_id" in node
            assert len(node["node_id"]) == 4  # zero-padded

    def test_node_ids_are_unique(self):
        result = self._run()
        ids = [n["node_id"] for n in core.structure_to_list(result["structure"])]
        assert len(ids) == len(set(ids))

    def test_start_index_lte_end_index(self):
        result = self._run()
        for node in core.structure_to_list(result["structure"]):
            assert node["start_index"] <= node["end_index"], (
                f"Node '{node['title']}' has start={node['start_index']} > end={node['end_index']}"
            )

    def test_indices_within_document_bounds(self):
        result = self._run()
        for node in core.structure_to_list(result["structure"]):
            assert node["start_index"] >= 1
            assert node["end_index"] <= 50

    def test_no_text_field_when_disabled(self):
        result = self._run(if_add_node_text="no")
        for node in core.structure_to_list(result["structure"]):
            assert "text" not in node

    def test_text_field_present_when_enabled(self):
        result = self._run(if_add_node_text="yes")
        for node in core.structure_to_list(result["structure"]):
            assert "text" in node
            assert isinstance(node["text"], str)

    def test_result_is_json_serialisable(self):
        result = self._run()
        # Must not raise
        serialised = json.dumps(result)
        assert len(serialised) > 0


# ===========================================================================
# 5. md_to_tree — markdown indexing pipeline
# ===========================================================================

# Synthetic markdown used across all md_to_tree tests.
# Structure:
#   # Document Title              ← level 1 root
#   ## Chapter 1                  ← level 2
#   ### Section 1.1               ← level 3
#   ### Section 1.2               ← level 3
#   ## Chapter 2                  ← level 2  (has code block with fake header)
#   ## Chapter 3                  ← level 2  (short, good for thinning tests)
_SAMPLE_MD = """\
# Document Title

Introduction paragraph at the top level.

## Chapter 1

Chapter 1 has substantial body text to ensure it exceeds small token thresholds.

### Section 1.1

Section 1.1 covers the first sub-topic in detail.

### Section 1.2

Section 1.2 covers the second sub-topic in detail.

## Chapter 2

```python
# This header inside a code block must NOT become a node
def example():
    pass
```

Chapter 2 continues after the code block.

## Chapter 3

Short chapter.
"""


class TestMdToTree:
    """
    Tests for the full md_to_tree async pipeline.

    Real: file I/O, markdown parsing, token counting, tree building, formatting.
    Mocked: ChatGPT_API_async (summaries) and ChatGPT_API (doc description).
    """

    @pytest.fixture()
    def md_file(self, tmp_path):
        """Write the sample markdown to a temp file and return its path."""
        p = tmp_path / "sample_doc.md"
        p.write_text(_SAMPLE_MD, encoding="utf-8")
        return str(p)

    def _run(self, md_file, **kwargs):
        return asyncio.run(core.md_to_tree(md_file, **kwargs))

    # --- output shape ---

    def test_returns_doc_name_and_structure(self, md_file):
        result = self._run(md_file)
        assert "doc_name" in result
        assert "structure" in result

    def test_doc_name_is_filename_stem(self, md_file):
        result = self._run(md_file)
        assert result["doc_name"] == "sample_doc"

    def test_structure_is_nonempty_list(self, md_file):
        result = self._run(md_file)
        assert isinstance(result["structure"], list)
        assert len(result["structure"]) > 0

    # --- node content ---

    def test_code_block_header_not_extracted(self, md_file):
        result = self._run(md_file)
        all_titles = [n["title"] for n in core.structure_to_list(result["structure"])]
        assert "This header inside a code block must NOT become a node" not in all_titles

    def test_real_headers_all_present(self, md_file):
        result = self._run(md_file)
        all_titles = {n["title"] for n in core.structure_to_list(result["structure"])}
        for expected in ["Document Title", "Chapter 1", "Section 1.1", "Section 1.2",
                         "Chapter 2", "Chapter 3"]:
            assert expected in all_titles, f"Missing: {expected}"

    def test_sections_nested_under_chapters(self, md_file):
        result = self._run(md_file)
        # Chapter 1 (level 2) is a child of Document Title (level 1),
        # so search the full flattened tree, not just root level.
        all_nodes = core.structure_to_list(result["structure"])
        ch1 = next(n for n in all_nodes if n["title"] == "Chapter 1")
        child_titles = {n["title"] for n in ch1.get("nodes", [])}
        assert "Section 1.1" in child_titles
        assert "Section 1.2" in child_titles

    # --- node_id ---

    def test_node_ids_present_by_default(self, md_file):
        result = self._run(md_file)
        for node in core.structure_to_list(result["structure"]):
            assert "node_id" in node

    def test_node_ids_unique_and_sequential(self, md_file):
        result = self._run(md_file)
        ids = [n["node_id"] for n in core.structure_to_list(result["structure"])]
        assert ids == sorted(ids)
        assert len(ids) == len(set(ids))

    def test_node_id_zero_based_when_write_node_id_runs(self, md_file):
        # if_add_node_id='yes' calls write_node_id which resets to 0-based counting
        result = self._run(md_file, if_add_node_id="yes")
        assert result["structure"][0]["node_id"] == "0000"

    def test_node_id_one_based_when_write_node_id_skipped(self, md_file):
        # build_tree_from_nodes always adds node_id with a 1-based counter;
        # if_add_node_id='no' skips the write_node_id re-run so the 1-based IDs remain
        result = self._run(md_file, if_add_node_id="no")
        assert result["structure"][0]["node_id"] == "0001"

    # --- text field ---

    def test_no_text_field_by_default(self, md_file):
        result = self._run(md_file)
        for node in core.structure_to_list(result["structure"]):
            assert "text" not in node

    def test_text_field_present_when_enabled(self, md_file):
        result = self._run(md_file, if_add_node_text="yes")
        for node in core.structure_to_list(result["structure"]):
            assert "text" in node
            assert isinstance(node["text"], str)

    def test_text_contains_header(self, md_file):
        result = self._run(md_file, if_add_node_text="yes")
        ch1 = next(n for n in core.structure_to_list(result["structure"])
                   if n["title"] == "Chapter 1")
        assert "Chapter 1" in ch1["text"]

    # --- thinning ---

    def test_thinning_with_high_threshold_merges_small_nodes(self, md_file):
        # With a very high threshold all small leaf nodes should merge up
        result_default = self._run(md_file)
        result_thinned = self._run(md_file, if_thinning=True, min_token_threshold=50_000)
        default_count = len(core.structure_to_list(result_default["structure"]))
        thinned_count = len(core.structure_to_list(result_thinned["structure"]))
        assert thinned_count <= default_count

    def test_thinning_with_zero_threshold_leaves_tree_unchanged(self, md_file):
        result_default = self._run(md_file)
        result_thinned = self._run(md_file, if_thinning=True, min_token_threshold=0)
        assert (len(core.structure_to_list(result_default["structure"])) ==
                len(core.structure_to_list(result_thinned["structure"])))

    # --- summaries (LLM mocked) ---

    def test_summary_fields_added_when_enabled(self, md_file, monkeypatch):
        async def fake_async(model=None, prompt=None, api_key=None):
            return "A short summary."
        monkeypatch.setattr(core, "ChatGPT_API_async", fake_async)

        result = self._run(md_file, if_add_node_summary="yes", summary_token_threshold=1)
        nodes = core.structure_to_list(result["structure"])
        # Every node should have either summary (leaf) or prefix_summary (parent)
        for node in nodes:
            has_summary = "summary" in node or "prefix_summary" in node
            assert has_summary, f"Node '{node.get('title')}' missing summary field"

    def test_no_text_in_summary_mode_when_disabled(self, md_file, monkeypatch):
        async def fake_async(model=None, prompt=None, api_key=None):
            return "Summary."
        monkeypatch.setattr(core, "ChatGPT_API_async", fake_async)

        result = self._run(md_file, if_add_node_summary="yes",
                           summary_token_threshold=1, if_add_node_text="no")
        for node in core.structure_to_list(result["structure"]):
            assert "text" not in node

    # --- doc description (LLM mocked) ---

    def test_doc_description_in_result_when_enabled(self, md_file, monkeypatch):
        async def fake_async(model=None, prompt=None, api_key=None):
            return "A short summary."
        monkeypatch.setattr(core, "ChatGPT_API_async", fake_async)
        monkeypatch.setattr(core, "ChatGPT_API",
                            lambda model, prompt, **kw: "A document about chapters.")

        result = self._run(md_file, if_add_node_summary="yes",
                           summary_token_threshold=1, if_add_doc_description="yes")
        assert "doc_description" in result
        assert isinstance(result["doc_description"], str)

    # --- JSON serialisable ---

    def test_result_is_json_serialisable(self, md_file):
        result = self._run(md_file)
        serialised = json.dumps(result, ensure_ascii=False)
        assert len(serialised) > 0
