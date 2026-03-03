"""
Integration tests for pageindex_core.py using real PDFs from tests/pdfs/.

All tests that touch the LLM pipeline mock only the OpenAI call surface
(core.ChatGPT_API, core._call_llm, core.ChatGPT_API_async) — everything
else (PDF reading, tokenisation, chunking, tree building, post-processing)
runs against real code and real PDF bytes.

PDF used: tests/pdfs/2023-annual-report-truncated.pdf (50 pages, ~3.4 MB)
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
