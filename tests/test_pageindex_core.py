"""
Tests for pageindex_core.py — standalone single-file distillation.

Covers (no LLM calls for pure-logic; mocks for LLM-backed paths):
  1.  Module isolation — importable and AST-parseable in isolation
  2.  _load_config — defaults, overrides, unknown-key rejection
  3.  count_tokens — None/empty guards, model fallback
  4.  JSON helpers — get_json_content, extract_json
  5.  Tree walkers — _walk, structure_to_list, write_node_id
  6.  Data transforms — list_to_tree, add_preface_if_needed, post_processing,
                        convert_physical_index_to_int, convert_page_to_int,
                        calculate_page_offset, add_page_offset_to_toc_json,
                        remove_structure_text, remove_page_number,
                        reorder_dict, format_structure
  7.  page_list_to_group_text — single group, multi-group chunking
  8.  _build_page_groups — label format, start_index offset, delegation
  9.  validate_and_truncate_physical_indices — keep vs. truncate
  10. Markdown pipeline — extract_nodes_from_markdown, extract_node_text_content,
                          build_tree_from_nodes
  11. LLM helpers (mocked) — _call_llm retry/backoff, _call_llm_async
  12. check_title_appearance — early-exit when physical_index absent/None
"""

import ast
import asyncio
import importlib
import json
import pathlib
import sys

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

# ---------------------------------------------------------------------------
# Import the standalone module
# ---------------------------------------------------------------------------
_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
import pageindex_core as core


# ===========================================================================
# 1. Module isolation
# ===========================================================================

class TestModuleIsolation:
    def test_ast_parses_cleanly(self):
        src = (_ROOT / "pageindex_core.py").read_text()
        tree = ast.parse(src)
        assert tree is not None

    def test_module_importable(self):
        """Re-importing must not raise."""
        importlib.reload(core)

    def test_no_dead_functions_present(self):
        """Functions deleted in the plan must not exist on the module."""
        dead = [
            "get_nodes", "get_leaf_nodes", "is_leaf_node", "get_last_node",
            "get_pdf_title", "extract_text_from_pdf", "get_number_of_pages",
            "get_text_of_pages", "add_node_text_with_labels",
            "_get_start_page", "get_first_start_page_from_text",
            "get_last_start_page_from_text", "clean_structure_post",
            "remove_fields", "print_toc", "print_json", "check_token_limit",
            "extract_toc_content", "remove_first_physical_index_section",
            "clean_tree_for_output",
        ]
        for fn in dead:
            assert not hasattr(core, fn), f"{fn!r} should have been deleted"

    def test_new_helper_exists(self):
        assert callable(core._build_page_groups)


# ===========================================================================
# 2. _load_config
# ===========================================================================

class TestLoadConfig:
    def test_defaults_returned_when_no_opt(self):
        cfg = core._load_config()
        assert cfg.model == "gpt-4o-2024-11-20"
        assert cfg.max_page_num_each_node == 10
        assert cfg.if_add_node_summary == "yes"

    def test_dict_override_applied(self):
        cfg = core._load_config({"model": "gpt-4o-mini"})
        assert cfg.model == "gpt-4o-mini"
        # Unset keys remain at default
        assert cfg.max_page_num_each_node == 10

    def test_namespace_override_applied(self):
        ns = core.config(model="gpt-3.5-turbo")
        cfg = core._load_config(ns)
        assert cfg.model == "gpt-3.5-turbo"

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown config keys"):
            core._load_config({"nonexistent_key": 99})

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            core._load_config("bad_type")

    def test_all_default_keys_present(self):
        cfg = core._load_config()
        for key in core._DEFAULT_CONFIG:
            assert hasattr(cfg, key)


# ===========================================================================
# 3. count_tokens
# ===========================================================================

class TestCountTokens:
    def test_none_returns_zero(self):
        assert core.count_tokens(None) == 0

    def test_empty_string_returns_zero(self):
        assert core.count_tokens("") == 0

    def test_model_none_does_not_raise(self):
        result = core.count_tokens("hello world", model=None)
        assert isinstance(result, int) and result > 0

    def test_model_none_matches_gpt4o(self):
        assert core.count_tokens("test", model=None) == core.count_tokens("test", model="gpt-4o")

    def test_longer_text_more_tokens(self):
        assert core.count_tokens("x " * 100) > core.count_tokens("x")


# ===========================================================================
# 4. JSON helpers
# ===========================================================================

class TestGetJsonContent:
    def test_strips_json_code_block(self):
        assert core.get_json_content('```json\n{"x":1}\n```') == '{"x":1}'

    def test_plain_string_returned_stripped(self):
        assert core.get_json_content('  {"x":1}  ') == '{"x":1}'

    def test_no_code_block_end_returns_rest(self):
        # rfind of ``` not found → takes whole string
        result = core.get_json_content('no backtick here')
        assert 'no backtick here' in result


class TestExtractJson:
    def test_plain_json(self):
        assert core.extract_json('{"a": 1}') == {"a": 1}

    def test_json_in_code_block(self):
        assert core.extract_json('```json\n{"a":1}\n```') == {"a": 1}

    def test_none_replaced_with_null(self):
        assert core.extract_json('{"k": None}') == {"k": None}

    def test_trailing_comma_in_object(self):
        assert core.extract_json('{"a": 1,}') == {"a": 1}

    def test_trailing_comma_in_array(self):
        assert core.extract_json('{"a": [1,2,]}') == {"a": [1, 2]}

    def test_invalid_returns_empty_dict(self):
        assert core.extract_json("not json!!!") == {}

    def test_list_json(self):
        assert core.extract_json('[{"x": 1}, {"x": 2}]') == [{"x": 1}, {"x": 2}]


# ===========================================================================
# 5. Tree walkers
# ===========================================================================

class TestWalk:
    def test_single_dict_yields_self(self):
        node = {"title": "A"}
        assert list(core._walk(node)) == [node]

    def test_dict_with_children(self):
        tree = {"title": "Root", "nodes": [{"title": "Child"}]}
        nodes = list(core._walk(tree))
        titles = [n["title"] for n in nodes]
        assert "Root" in titles and "Child" in titles

    def test_list_of_dicts(self):
        tree = [{"title": "A"}, {"title": "B"}]
        titles = {n["title"] for n in core._walk(tree)}
        assert titles == {"A", "B"}

    def test_empty_list(self):
        assert list(core._walk([])) == []

    def test_deeply_nested(self):
        tree = {"title": "R", "nodes": [{"title": "A", "nodes": [{"title": "A1"}]}]}
        titles = {n["title"] for n in core._walk(tree)}
        assert titles == {"R", "A", "A1"}


class TestStructureToList:
    def test_flat(self):
        data = [{"title": "A"}, {"title": "B"}]
        result = core.structure_to_list(data)
        assert len(result) == 2

    def test_nested_all_nodes_returned(self):
        data = {"title": "R", "nodes": [{"title": "C"}]}
        result = core.structure_to_list(data)
        assert len(result) == 2


class TestWriteNodeId:
    def test_flat_list_sequential(self):
        data = [{"title": "A"}, {"title": "B"}]
        core.write_node_id(data)
        assert data[0]["node_id"] == "0000"
        assert data[1]["node_id"] == "0001"

    def test_nested_depth_first(self):
        data = {"title": "Root", "nodes": [{"title": "Child"}]}
        core.write_node_id(data)
        assert data["node_id"] == "0000"
        assert data["nodes"][0]["node_id"] == "0001"

    def test_zero_padded_to_four_digits(self):
        data = [{"title": str(i)} for i in range(12)]
        core.write_node_id(data)
        assert data[9]["node_id"] == "0009"
        assert data[10]["node_id"] == "0010"


# ===========================================================================
# 6. Data transforms
# ===========================================================================

class TestListToTree:
    def test_flat_roots(self):
        data = [
            {"structure": "1", "title": "Ch1", "start_index": 1, "end_index": 5},
            {"structure": "2", "title": "Ch2", "start_index": 6, "end_index": 10},
        ]
        tree = core.list_to_tree(data)
        assert len(tree) == 2
        assert tree[0]["title"] == "Ch1"

    def test_parent_child(self):
        data = [
            {"structure": "1", "title": "Ch", "start_index": 1, "end_index": 10},
            {"structure": "1.1", "title": "Sec", "start_index": 1, "end_index": 5},
        ]
        tree = core.list_to_tree(data)
        assert len(tree) == 1
        assert tree[0]["nodes"][0]["title"] == "Sec"

    def test_leaf_has_no_nodes_key(self):
        data = [{"structure": "1", "title": "Solo", "start_index": 1, "end_index": 5}]
        tree = core.list_to_tree(data)
        assert "nodes" not in tree[0]


class TestAddPrefaceIfNeeded:
    def test_adds_preface_when_first_page_gt_1(self):
        data = [{"physical_index": 3, "title": "Ch1"}]
        result = core.add_preface_if_needed(data)
        assert result[0]["title"] == "Preface"
        assert result[0]["physical_index"] == 1

    def test_no_preface_when_first_page_is_1(self):
        data = [{"physical_index": 1, "title": "Ch1"}]
        result = core.add_preface_if_needed(data)
        assert result[0]["title"] == "Ch1"

    def test_no_preface_when_physical_index_none(self):
        data = [{"physical_index": None, "title": "Ch1"}]
        result = core.add_preface_if_needed(data)
        assert result[0]["title"] == "Ch1"

    def test_empty_list_returned_unchanged(self):
        assert core.add_preface_if_needed([]) == []


class TestConvertPhysicalIndexToInt:
    def test_list_of_dicts_with_tag_format(self):
        data = [{"physical_index": "<physical_index_5>"}]
        result = core.convert_physical_index_to_int(data)
        assert result[0]["physical_index"] == 5

    def test_list_no_tag_prefix(self):
        data = [{"physical_index": "physical_index_7"}]
        result = core.convert_physical_index_to_int(data)
        assert result[0]["physical_index"] == 7

    def test_string_tag_format(self):
        result = core.convert_physical_index_to_int("<physical_index_10>")
        assert result == 10

    def test_string_no_prefix_returns_none(self):
        result = core.convert_physical_index_to_int("something_else")
        assert result is None


class TestConvertPageToInt:
    def test_string_page_becomes_int(self):
        data = [{"page": "5"}]
        result = core.convert_page_to_int(data)
        assert result[0]["page"] == 5

    def test_non_numeric_string_left_as_is(self):
        data = [{"page": "N/A"}]
        result = core.convert_page_to_int(data)
        assert result[0]["page"] == "N/A"

    def test_already_int_unchanged(self):
        data = [{"page": 3}]
        result = core.convert_page_to_int(data)
        assert result[0]["page"] == 3


class TestCalculatePageOffset:
    def test_single_pair(self):
        pairs = [{"physical_index": 5, "page": 3}]
        assert core.calculate_page_offset(pairs) == 2

    def test_mode_wins(self):
        pairs = [
            {"physical_index": 5, "page": 3},  # diff=2
            {"physical_index": 6, "page": 4},  # diff=2
            {"physical_index": 8, "page": 5},  # diff=3
        ]
        assert core.calculate_page_offset(pairs) == 2

    def test_empty_pairs_returns_none(self):
        assert core.calculate_page_offset([]) is None

    def test_missing_keys_skipped(self):
        pairs = [{"physical_index": 5}, {"physical_index": 8, "page": 4}]
        # Only second pair is valid
        assert core.calculate_page_offset(pairs) == 4


class TestAddPageOffsetToTocJson:
    def test_adds_offset_and_removes_page(self):
        data = [{"page": 3, "title": "A"}, {"page": 7, "title": "B"}]
        result = core.add_page_offset_to_toc_json(data, offset=2)
        assert result[0]["physical_index"] == 5
        assert "page" not in result[0]

    def test_none_page_skipped(self):
        data = [{"page": None, "title": "A"}]
        result = core.add_page_offset_to_toc_json(data, offset=2)
        assert "physical_index" not in result[0]

    def test_string_page_skipped(self):
        data = [{"page": "N/A", "title": "A"}]
        result = core.add_page_offset_to_toc_json(data, offset=2)
        assert "physical_index" not in result[0]


class TestRemoveStructureText:
    def test_text_removed_from_dict(self):
        data = {"title": "A", "text": "content", "nodes": []}
        core.remove_structure_text(data)
        assert "text" not in data

    def test_text_removed_from_nested_nodes(self):
        data = {"title": "R", "nodes": [{"title": "C", "text": "stuff"}]}
        core.remove_structure_text(data)
        assert "text" not in data["nodes"][0]

    def test_text_removed_from_list(self):
        data = [{"text": "a"}, {"text": "b"}]
        core.remove_structure_text(data)
        assert all("text" not in n for n in data)


class TestRemovePageNumber:
    def test_page_number_removed(self):
        data = {"title": "A", "page_number": 5}
        core.remove_page_number(data)
        assert "page_number" not in data

    def test_nested_removal(self):
        data = {"page_number": 1, "nodes": [{"page_number": 2}]}
        core.remove_page_number(data)
        assert "page_number" not in data
        assert "page_number" not in data["nodes"][0]


class TestReorderDict:
    def test_reorders_keys(self):
        data = {"b": 2, "a": 1, "c": 3}
        result = core.reorder_dict(data, ["a", "b", "c"])
        assert list(result.keys()) == ["a", "b", "c"]

    def test_missing_keys_skipped(self):
        data = {"a": 1, "c": 3}
        result = core.reorder_dict(data, ["a", "b", "c"])
        assert list(result.keys()) == ["a", "c"]

    def test_empty_order_returns_as_is(self):
        data = {"b": 2, "a": 1}
        assert core.reorder_dict(data, []) is data


class TestFormatStructure:
    def test_reorders_dict_keys(self):
        s = {"b": 2, "a": 1}
        result = core.format_structure(s, order=["a", "b"])
        assert list(result.keys()) == ["a", "b"]

    def test_no_order_returns_unchanged(self):
        s = {"b": 2, "a": 1}
        assert core.format_structure(s) is s

    def test_empty_nodes_key_removed(self):
        s = {"title": "A", "nodes": []}
        result = core.format_structure(s, order=["title", "nodes"])
        assert "nodes" not in result

    def test_nested_structure_reordered(self):
        s = [{"b": 2, "a": 1}]
        result = core.format_structure(s, order=["a", "b"])
        assert list(result[0].keys()) == ["a", "b"]


# ===========================================================================
# 7. page_list_to_group_text
# ===========================================================================

class TestPageListToGroupText:
    def test_single_page_under_limit_one_group(self):
        pages = ["hello world"]
        tokens = [2]
        result = core.page_list_to_group_text(pages, tokens, max_tokens=20000)
        assert len(result) == 1
        assert result[0] == "hello world"

    def test_multiple_pages_under_limit_joined(self):
        pages = ["a", "b", "c"]
        tokens = [1, 1, 1]
        result = core.page_list_to_group_text(pages, tokens, max_tokens=20000)
        assert len(result) == 1
        assert result[0] == "abc"

    def test_over_limit_splits_into_multiple_groups(self, monkeypatch):
        # 4 pages × 6000 tokens each = 24000 > 20000 → should split
        pages = [f"page{i}" for i in range(4)]
        tokens = [6000] * 4
        result = core.page_list_to_group_text(pages, tokens, max_tokens=20000)
        assert len(result) > 1

    def test_result_is_list_of_strings(self):
        result = core.page_list_to_group_text(["x"], [1])
        assert isinstance(result, list)
        assert all(isinstance(g, str) for g in result)


# ===========================================================================
# 8. _build_page_groups
# ===========================================================================

class TestBuildPageGroups:
    """The new helper extracted from process_no_toc / process_toc_no_page_numbers."""

    def _make_page_list(self, n=3, text="some page text"):
        return [(text, 10)] * n

    def test_returns_list(self):
        page_list = self._make_page_list(2)
        result = core._build_page_groups(page_list, start_index=1, model=None)
        assert isinstance(result, list)

    def test_label_format_present_in_output(self):
        page_list = [("content here", 10)]
        result = core._build_page_groups(page_list, start_index=1, model=None)
        combined = "".join(result)
        assert "<physical_index_1>" in combined
        assert "content here" in combined

    def test_start_index_offsets_labels(self):
        page_list = [("pg text", 5)]
        result = core._build_page_groups(page_list, start_index=5, model=None)
        combined = "".join(result)
        assert "<physical_index_5>" in combined
        assert "<physical_index_1>" not in combined

    def test_multiple_pages_labeled_sequentially(self):
        page_list = [("a", 5), ("b", 5), ("c", 5)]
        result = core._build_page_groups(page_list, start_index=10, model=None)
        combined = "".join(result)
        assert "<physical_index_10>" in combined
        assert "<physical_index_11>" in combined
        assert "<physical_index_12>" in combined

    def test_delegates_to_page_list_to_group_text(self, monkeypatch):
        """_build_page_groups must call page_list_to_group_text."""
        calls = []

        def spy(page_contents, token_lengths):
            calls.append((page_contents, token_lengths))
            return ["grouped"]

        monkeypatch.setattr(core, "page_list_to_group_text", spy)
        result = core._build_page_groups([("text", 5)], start_index=1, model=None)
        assert len(calls) == 1
        assert result == ["grouped"]

    def test_token_count_per_page(self, monkeypatch):
        """count_tokens must be called once per page."""
        call_count = [0]
        original = core.count_tokens

        def counting_count_tokens(text, model=None):
            call_count[0] += 1
            return original(text, model)

        monkeypatch.setattr(core, "count_tokens", counting_count_tokens)
        page_list = [("a", 5), ("b", 5), ("c", 5)]
        core._build_page_groups(page_list, start_index=1, model=None)
        assert call_count[0] == 3


# ===========================================================================
# 9. validate_and_truncate_physical_indices
# ===========================================================================

class TestValidateAndTruncate:
    def test_within_range_unchanged(self):
        toc = [{"title": "A", "physical_index": 3}]
        result = core.validate_and_truncate_physical_indices(toc, page_list_length=5)
        assert result[0]["physical_index"] == 3

    def test_exceeds_range_set_to_none(self):
        toc = [{"title": "A", "physical_index": 10}]
        result = core.validate_and_truncate_physical_indices(toc, page_list_length=5)
        assert result[0]["physical_index"] is None

    def test_exact_max_allowed_kept(self):
        # max_allowed = page_list_length + start_index - 1 = 5 + 1 - 1 = 5
        toc = [{"title": "A", "physical_index": 5}]
        result = core.validate_and_truncate_physical_indices(toc, page_list_length=5)
        assert result[0]["physical_index"] == 5

    def test_empty_toc_returned_unchanged(self):
        assert core.validate_and_truncate_physical_indices([], page_list_length=10) == []

    def test_none_physical_index_left_as_none(self):
        toc = [{"title": "A", "physical_index": None}]
        result = core.validate_and_truncate_physical_indices(toc, page_list_length=5)
        assert result[0]["physical_index"] is None


# ===========================================================================
# 10. Markdown pipeline
# ===========================================================================

class TestExtractNodesFromMarkdown:
    def test_finds_headers(self):
        md = "# Title\n## Section\n### Sub"
        nodes, _ = core.extract_nodes_from_markdown(md)
        titles = [n["node_title"] for n in nodes]
        assert "Title" in titles
        assert "Section" in titles
        assert "Sub" in titles

    def test_skips_headers_inside_code_blocks(self):
        md = "# Real\n```\n# Fake\n```\n## Also Real"
        nodes, _ = core.extract_nodes_from_markdown(md)
        titles = [n["node_title"] for n in nodes]
        assert "Real" in titles
        assert "Also Real" in titles
        assert "Fake" not in titles

    def test_skips_empty_lines(self):
        md = "\n\n# Header\n\n"
        nodes, _ = core.extract_nodes_from_markdown(md)
        assert len(nodes) == 1

    def test_returns_line_numbers(self):
        md = "# A\n## B"
        nodes, _ = core.extract_nodes_from_markdown(md)
        assert nodes[0]["line_num"] == 1
        assert nodes[1]["line_num"] == 2


class TestBuildTreeFromNodes:
    def test_flat_nodes_all_roots(self):
        nodes = [
            {"title": "A", "level": 1, "text": "a text", "line_num": 1},
            {"title": "B", "level": 1, "text": "b text", "line_num": 3},
        ]
        tree = core.build_tree_from_nodes(nodes)
        assert len(tree) == 2

    def test_nested_parent_child(self):
        nodes = [
            {"title": "Parent", "level": 1, "text": "p", "line_num": 1},
            {"title": "Child", "level": 2, "text": "c", "line_num": 2},
        ]
        tree = core.build_tree_from_nodes(nodes)
        assert len(tree) == 1
        assert tree[0]["nodes"][0]["title"] == "Child"

    def test_empty_input_returns_empty(self):
        assert core.build_tree_from_nodes([]) == []

    def test_node_ids_assigned(self):
        nodes = [{"title": "A", "level": 1, "text": "t", "line_num": 1}]
        tree = core.build_tree_from_nodes(nodes)
        assert "node_id" in tree[0]

    def test_sibling_then_child(self):
        nodes = [
            {"title": "A", "level": 1, "text": "a", "line_num": 1},
            {"title": "A1", "level": 2, "text": "a1", "line_num": 2},
            {"title": "B", "level": 1, "text": "b", "line_num": 3},
        ]
        tree = core.build_tree_from_nodes(nodes)
        assert len(tree) == 2
        assert tree[0]["nodes"][0]["title"] == "A1"


# ===========================================================================
# 11. LLM helpers (mocked)
# ===========================================================================

def _make_openai_response(content, finish_reason="stop"):
    resp = MagicMock()
    resp.choices[0].message.content = content
    resp.choices[0].finish_reason = finish_reason
    return resp


class TestCallLlm:
    def _patch_client(self, monkeypatch, *, side_effect=None, return_value=None):
        mock = MagicMock()
        if side_effect is not None:
            mock.chat.completions.create.side_effect = side_effect
        else:
            mock.chat.completions.create.return_value = return_value
        monkeypatch.setattr(core.openai, "OpenAI", lambda api_key: mock)
        return mock

    def test_returns_content_on_success(self, monkeypatch):
        self._patch_client(monkeypatch, return_value=_make_openai_response("hello"))
        with patch("time.sleep"):
            result = core._call_llm("gpt-4o", "prompt", api_key="k")
        assert result == "hello"

    def test_retries_on_exception_then_succeeds(self, monkeypatch):
        self._patch_client(
            monkeypatch,
            side_effect=[RuntimeError("boom"), _make_openai_response("ok")],
        )
        with patch("time.sleep"):
            result = core._call_llm("gpt-4o", "prompt", api_key="k")
        assert result == "ok"

    def test_max_retries_returns_error(self, monkeypatch):
        self._patch_client(monkeypatch, side_effect=RuntimeError("always"))
        with patch("time.sleep"):
            result = core._call_llm("gpt-4o", "prompt", api_key="k")
        assert result == "Error"

    def test_finish_reason_length_returns_max_output_reached(self, monkeypatch):
        self._patch_client(monkeypatch, return_value=_make_openai_response("partial", "length"))
        with patch("time.sleep"):
            content, reason = core._call_llm("gpt-4o", "p", api_key="k", return_finish_reason=True)
        assert reason == "max_output_reached"

    def test_finish_reason_stop_returns_finished(self, monkeypatch):
        self._patch_client(monkeypatch, return_value=_make_openai_response("ok", "stop"))
        with patch("time.sleep"):
            _, reason = core._call_llm("gpt-4o", "p", api_key="k", return_finish_reason=True)
        assert reason == "finished"

    def test_exponential_backoff(self, monkeypatch):
        self._patch_client(
            monkeypatch,
            side_effect=[RuntimeError("e")] * 3 + [_make_openai_response("done")],
        )
        sleeps = []
        with patch("time.sleep", side_effect=lambda t: sleeps.append(t)):
            core._call_llm("gpt-4o", "p", api_key="k")
        assert len(sleeps) == 3
        assert sleeps[1] > sleeps[0]

    def test_backoff_capped_at_60(self, monkeypatch):
        self._patch_client(
            monkeypatch,
            side_effect=[RuntimeError("e")] * 7 + [_make_openai_response("ok")],
        )
        sleeps = []
        with patch("time.sleep", side_effect=lambda t: sleeps.append(t)):
            core._call_llm("gpt-4o", "p", api_key="k")
        assert all(s <= 61 for s in sleeps)


class TestCallLlmAsync:
    def _patch_async_client(self, monkeypatch, *, side_effect=None, return_value=None):
        mock = AsyncMock()
        if side_effect is not None:
            mock.chat.completions.create.side_effect = side_effect
        else:
            mock.chat.completions.create.return_value = return_value
        monkeypatch.setattr(core.openai, "AsyncOpenAI", lambda api_key: mock)
        return mock

    def test_returns_content_on_success(self, monkeypatch):
        self._patch_async_client(monkeypatch, return_value=_make_openai_response("async ok"))
        result = asyncio.run(core._call_llm_async("gpt-4o", "prompt", api_key="k"))
        assert result == "async ok"

    def test_retries_on_exception(self, monkeypatch):
        self._patch_async_client(
            monkeypatch,
            side_effect=[RuntimeError("boom"), _make_openai_response("ok")],
        )
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = asyncio.run(core._call_llm_async("gpt-4o", "p", api_key="k"))
        assert result == "ok"

    def test_max_retries_returns_error(self, monkeypatch):
        self._patch_async_client(monkeypatch, side_effect=RuntimeError("always"))
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = asyncio.run(core._call_llm_async("gpt-4o", "p", api_key="k"))
        assert result == "Error"


# ===========================================================================
# 12. check_title_appearance — early-exit
# ===========================================================================

class TestCheckTitleAppearance:
    def test_early_exit_when_no_physical_index_key(self):
        item = {"title": "Intro", "list_index": 0}
        page_list = [("text", 100)]
        result = asyncio.run(
            core.check_title_appearance(item, page_list, start_index=1, model=None)
        )
        assert result["answer"] == "no"
        assert result["list_index"] == 0

    def test_early_exit_when_physical_index_is_none(self):
        item = {"title": "Ch1", "list_index": 2, "physical_index": None}
        page_list = [("text", 100)]
        result = asyncio.run(
            core.check_title_appearance(item, page_list, start_index=1, model=None)
        )
        assert result["answer"] == "no"
        assert result["page_number"] is None

    def test_list_index_preserved(self):
        item = {"title": "Appendix", "list_index": 7, "physical_index": None}
        result = asyncio.run(
            core.check_title_appearance(item, [("t", 50)], start_index=1, model=None)
        )
        assert result["list_index"] == 7
