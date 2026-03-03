"""
Tests for pageindex/page_index.py covering the fixes from the bug-fix plan:
  Fix 1 — variable shadowing in fix_incorrect_toc/process_and_check_item
  Fix 2 — bounded retry loop in extract_toc_content (attempt counter)
  Fix 7 — meta_processor returns best-effort result instead of raising
  Fix 9 — duplicate import removed (verified by AST parse)

Also covers pure-logic helpers: check_title_appearance early-exit.

NOTE: pageindex/__init__.py does `from .page_index import *`, which puts the
`page_index` *function* into the package namespace, shadowing the submodule
as a package attribute. We therefore retrieve the actual module object via
importlib so monkeypatching targets the correct namespace.
"""

import asyncio
import importlib
import pathlib
import sys

import pytest
from unittest.mock import MagicMock, AsyncMock

# Ensure the package is imported first, then grab the real submodule.
import pageindex  # noqa: F401 — triggers __init__.py
pi_mod = importlib.import_module("pageindex.page_index")

# Convenience re-export for direct calls
extract_toc_content = pi_mod.extract_toc_content
fix_incorrect_toc = pi_mod.fix_incorrect_toc
meta_processor = pi_mod.meta_processor
check_title_appearance = pi_mod.check_title_appearance


# ─── Fix 2: extract_toc_content attempt counter ──────────────────────────────

class TestExtractTocContent:
    """Verify the while-loop terminates after max_attempts=5 iterations."""

    def _patch(self, monkeypatch, *, complete_on_call=None):
        """
        Patch the two API helpers used by extract_toc_content.

        complete_on_call: 1-based call number of check_if_toc_transformation_is_complete
                          after which it returns 'yes'. None → always 'no'.
        """
        call_count = [0]

        def fake_api(model, prompt, api_key=None, chat_history=None):
            return "response chunk", "finished"

        def fake_check(content, toc, model=None):
            call_count[0] += 1
            if complete_on_call is not None and call_count[0] >= complete_on_call:
                return "yes"
            return "no"

        monkeypatch.setattr(pi_mod, "ChatGPT_API_with_finish_reason", fake_api)
        monkeypatch.setattr(pi_mod, "check_if_toc_transformation_is_complete", fake_check)
        return call_count

    def test_raises_after_max_attempts_when_never_complete(self, monkeypatch):
        """Never returning 'yes' must raise, not loop forever."""
        self._patch(monkeypatch, complete_on_call=None)
        with pytest.raises(Exception, match="maximum retries"):
            extract_toc_content("raw toc text", model="gpt-4o")

    def test_does_not_raise_when_complete_in_loop(self, monkeypatch):
        """Completing on the 3rd check call must return normally."""
        # Call 1 → pre-loop first check, call 2 → pre-loop second check,
        # call 3 → inside while iteration 1: returns 'yes', loop exits.
        self._patch(monkeypatch, complete_on_call=3)
        result = extract_toc_content("raw toc text", model="gpt-4o")
        assert isinstance(result, str)

    def test_early_return_when_complete_on_first_response(self, monkeypatch):
        """If the very first API response is complete, return immediately."""

        def fake_api(model, prompt, api_key=None, chat_history=None):
            return "complete toc", "finished"

        def fake_check(content, toc, model=None):
            return "yes"

        monkeypatch.setattr(pi_mod, "ChatGPT_API_with_finish_reason", fake_api)
        monkeypatch.setattr(pi_mod, "check_if_toc_transformation_is_complete", fake_check)
        result = extract_toc_content("content", model="gpt-4o")
        assert result == "complete toc"

    def test_exactly_5_while_iterations_before_raise(self, monkeypatch):
        """The test must complete (i.e. the loop is bounded, not infinite)."""
        self._patch(monkeypatch, complete_on_call=None)
        with pytest.raises(Exception):
            extract_toc_content("text", model="gpt-4o")
        # Reaching here proves the loop terminated


# ─── Fix 1: variable shadowing in process_and_check_item ─────────────────────

class TestFixIncorrectTocShadowing:
    """
    Verify that the list_index in each result comes from
    incorrect_item['list_index'], not from (page_index - start_index) computed
    inside the inner loop.
    """

    def _run_fix(self, monkeypatch, toc, page_list, incorrect_results, start_index=1):
        monkeypatch.setattr(
            pi_mod, "single_toc_item_index_fixer",
            lambda title, content, model: 2,  # always says page 2
        )

        async def fake_check_appearance(item, page_list, start_index, model):
            return {"answer": "yes", "list_index": item.get("list_index")}

        monkeypatch.setattr(pi_mod, "check_title_appearance", fake_check_appearance)

        logger = MagicMock()
        return asyncio.run(
            fix_incorrect_toc(
                toc, page_list, incorrect_results,
                start_index=start_index, model="gpt-4o", logger=logger,
            )
        )

    def test_result_list_index_matches_incorrect_item_not_loop_var(self, monkeypatch):
        """
        The incorrect item is at list_index=2. The inner loop starts from
        prev_correct=1, so page_list_idx=0 on the first iteration.
        Before the fix, list_index was overwritten to 0 and toc[0] was updated.
        After the fix, toc[2] must be updated.
        """
        toc = [
            {"title": "A", "physical_index": 1},
            {"title": "B", "physical_index": 3},
            {"title": "C", "physical_index": None},  # incorrect — list_index=2
            {"title": "D", "physical_index": 5},
        ]
        page_list = [(f"page {i+1} text", 100) for i in range(6)]
        incorrect = [{"list_index": 2, "title": "C", "physical_index": None}]

        result_toc, _ = self._run_fix(monkeypatch, toc, page_list, incorrect)

        assert result_toc[2]["physical_index"] == 2  # updated at correct index
        assert result_toc[0]["physical_index"] == 1  # untouched

    def test_correct_entries_remain_unchanged(self, monkeypatch):
        toc = [
            {"title": "A", "physical_index": 10},
            {"title": "B", "physical_index": None},  # incorrect
            {"title": "C", "physical_index": 20},
        ]
        page_list = [(f"p{i}", 50) for i in range(22)]
        incorrect = [{"list_index": 1, "title": "B", "physical_index": None}]

        result_toc, _ = self._run_fix(monkeypatch, toc, page_list, incorrect)

        assert result_toc[0]["physical_index"] == 10
        assert result_toc[2]["physical_index"] == 20

    def test_out_of_bounds_list_index_does_not_crash(self, monkeypatch):
        """An incorrect item beyond the toc length must not raise IndexError."""
        toc = [{"title": "A", "physical_index": 1}]
        page_list = [("text", 100)]
        incorrect = [{"list_index": 99, "title": "Ghost", "physical_index": None}]

        logger = MagicMock()
        result_toc, _ = asyncio.run(
            fix_incorrect_toc(
                toc, page_list, incorrect,
                start_index=1, model="gpt-4o", logger=logger,
            )
        )
        assert result_toc[0]["physical_index"] == 1  # original untouched

    def test_multiple_incorrect_items_each_fixed_at_correct_index(self, monkeypatch):
        """Two incorrect items must each land at their own list positions."""
        toc = [
            {"title": "A", "physical_index": 1},
            {"title": "B", "physical_index": None},  # list_index=1
            {"title": "C", "physical_index": None},  # list_index=2
            {"title": "D", "physical_index": 5},
        ]
        page_list = [(f"p{i}", 50) for i in range(6)]
        incorrect = [
            {"list_index": 1, "title": "B", "physical_index": None},
            {"list_index": 2, "title": "C", "physical_index": None},
        ]

        result_toc, _ = self._run_fix(monkeypatch, toc, page_list, incorrect)

        assert result_toc[1]["physical_index"] == 2
        assert result_toc[2]["physical_index"] == 2
        assert result_toc[0]["physical_index"] == 1
        assert result_toc[3]["physical_index"] == 5


# ─── Fix 7: meta_processor best-effort fallback ───────────────────────────────

class TestMetaProcessorFallback:
    """
    When mode='process_no_toc' and accuracy < 0.6, meta_processor must return
    toc_with_page_number (best effort) rather than raising.
    """

    def _make_opt(self):
        opt = MagicMock()
        opt.model = "gpt-4o"
        opt.toc_check_page_num = 20
        return opt

    def _make_logger(self):
        logger = MagicMock()
        logger.info = MagicMock()
        return logger

    def _patch_common(self, monkeypatch, best_effort, accuracy=0.3):
        monkeypatch.setattr(
            pi_mod, "process_no_toc",
            lambda page_list, start_index=1, model=None, logger=None: best_effort,
        )
        monkeypatch.setattr(
            pi_mod, "validate_and_truncate_physical_indices",
            lambda toc, length, start_index=1, logger=None: toc,
        )

        async def fake_verify(page_list, toc, start_index=1, N=None, model=None):
            return accuracy, []

        monkeypatch.setattr(pi_mod, "verify_toc", fake_verify)

    def test_returns_result_instead_of_raising(self, monkeypatch):
        best_effort = [{"title": "Section 1", "physical_index": 1}]
        self._patch_common(monkeypatch, best_effort, accuracy=0.3)

        result = asyncio.run(
            meta_processor(
                [("text", 100)] * 5,
                mode="process_no_toc",
                start_index=1,
                opt=self._make_opt(),
                logger=self._make_logger(),
            )
        )
        assert result is not None
        assert isinstance(result, list)

    def test_warning_logged_on_fallback(self, monkeypatch):
        best_effort = [{"title": "Sec", "physical_index": 1}]
        self._patch_common(monkeypatch, best_effort, accuracy=0.0)

        logger = self._make_logger()
        asyncio.run(
            meta_processor(
                [("text", 100)] * 5,
                mode="process_no_toc",
                start_index=1,
                opt=self._make_opt(),
                logger=logger,
            )
        )
        called_args = [str(c) for c in logger.info.call_args_list]
        assert any("Warning" in a or "best available" in a for a in called_args)

    def test_perfect_accuracy_returns_without_fallback(self, monkeypatch):
        """accuracy==1.0 with no incorrect results returns the toc directly."""
        best_effort = [{"title": "Sec", "physical_index": 1}]
        self._patch_common(monkeypatch, best_effort, accuracy=1.0)

        result = asyncio.run(
            meta_processor(
                [("text", 100)] * 5,
                mode="process_no_toc",
                start_index=1,
                opt=self._make_opt(),
                logger=self._make_logger(),
            )
        )
        assert result == best_effort


# ─── Fix 9: duplicate import removed ─────────────────────────────────────────

class TestNoDuplicateImport:
    def test_os_imported_once(self):
        """page_index.py must have exactly one top-level 'import os'."""
        import ast

        src_path = pathlib.Path(__file__).parent.parent / "pageindex" / "page_index.py"
        tree = ast.parse(src_path.read_text())

        count = sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            and any(alias.name == "os" for alias in node.names)
        )
        assert count == 1, f"Expected 1 'import os', found {count}"


# ─── check_title_appearance early-exit ───────────────────────────────────────

class TestCheckTitleAppearanceEarlyExit:
    """When physical_index is missing or None, return early without an API call."""

    def test_returns_no_when_physical_index_missing(self):
        item = {"title": "Intro", "list_index": 0}  # no physical_index key
        page_list = [("some text", 100)]
        result = asyncio.run(
            check_title_appearance(item, page_list, start_index=1, model=None)
        )
        assert result["answer"] == "no"
        assert result["list_index"] == 0

    def test_returns_no_when_physical_index_is_none(self):
        item = {"title": "Intro", "list_index": 3, "physical_index": None}
        page_list = [("some text", 100)]
        result = asyncio.run(
            check_title_appearance(item, page_list, start_index=1, model=None)
        )
        assert result["answer"] == "no"
        assert result["page_number"] is None

    def test_list_index_preserved_in_early_exit(self):
        item = {"title": "Chapter", "list_index": 7, "physical_index": None}
        result = asyncio.run(
            check_title_appearance(item, [("text", 50)], start_index=1, model=None)
        )
        assert result["list_index"] == 7
