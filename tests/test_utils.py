"""
Tests for pageindex/utils.py covering the fixes from the bug-fix plan:
  Fix 3/4 — module-level client singletons
  Fix 5   — exponential backoff with jitter
  Fix 6   — count_tokens model=None fallback
  Fix 8   — JsonLogger buffered writes

Also covers utility helpers: extract_json, get_json_content, write_node_id,
get_nodes, get_leaf_nodes, sanitize_filename, list_to_tree.
"""

import asyncio
import json
import os

import openai
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

import pageindex.utils as utils_mod
from pageindex.utils import (
    ChatGPT_API,
    ChatGPT_API_async,
    ChatGPT_API_with_finish_reason,
    JsonLogger,
    _get_async_client,
    _get_sync_client,
    count_tokens,
    extract_json,
    get_json_content,
    get_leaf_nodes,
    get_nodes,
    list_to_tree,
    sanitize_filename,
    write_node_id,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_response(content, finish_reason="stop"):
    """Build a minimal mock OpenAI chat-completion response."""
    resp = MagicMock()
    resp.choices[0].message.content = content
    resp.choices[0].finish_reason = finish_reason
    return resp


# ─── count_tokens (Fix 6) ─────────────────────────────────────────────────────

class TestCountTokens:
    def test_empty_string_returns_zero(self):
        assert count_tokens("") == 0

    def test_none_returns_zero(self):
        assert count_tokens(None) == 0

    def test_model_none_does_not_raise(self):
        result = count_tokens("hello world", model=None)
        assert isinstance(result, int)
        assert result > 0

    def test_explicit_model_works(self):
        result = count_tokens("hello world", model="gpt-4o")
        assert isinstance(result, int)
        assert result > 0

    def test_longer_text_more_tokens(self):
        short = count_tokens("hi")
        long = count_tokens("hi " * 50)
        assert long > short

    def test_result_consistent_model_none_vs_gpt4o(self):
        # model=None should fall back to gpt-4o encoding — same result
        assert count_tokens("test", model=None) == count_tokens("test", model="gpt-4o")


# ─── Client singletons (Fixes 3 & 4) ────────────────────────────────────────

class TestClientSingletons:
    def setup_method(self):
        # Isolate each test: reset module-level globals
        utils_mod._sync_client = None
        utils_mod._async_client = None

    def teardown_method(self):
        utils_mod._sync_client = None
        utils_mod._async_client = None

    def test_get_sync_client_returns_same_instance(self, monkeypatch):
        mock = MagicMock()
        monkeypatch.setattr(openai, "OpenAI", lambda api_key: mock)
        assert _get_sync_client("k") is _get_sync_client("k")

    def test_get_sync_client_constructor_called_once(self, monkeypatch):
        calls = []
        monkeypatch.setattr(openai, "OpenAI", lambda api_key: calls.append(1) or MagicMock())
        _get_sync_client("k")
        _get_sync_client("k")
        _get_sync_client("k")
        assert len(calls) == 1

    def test_get_async_client_returns_same_instance(self, monkeypatch):
        mock = MagicMock()
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda api_key: mock)
        assert _get_async_client("k") is _get_async_client("k")

    def test_get_async_client_constructor_called_once(self, monkeypatch):
        calls = []
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda api_key: calls.append(1) or MagicMock())
        _get_async_client("k")
        _get_async_client("k")
        assert len(calls) == 1

    def test_reset_global_allows_new_client(self, monkeypatch):
        clients = []
        monkeypatch.setattr(openai, "OpenAI", lambda api_key: clients.append(MagicMock()) or clients[-1])
        _get_sync_client("k")
        utils_mod._sync_client = None  # manual reset
        _get_sync_client("k")
        assert len(clients) == 2


# ─── ChatGPT_API (Fixes 3 & 5) ───────────────────────────────────────────────

class TestChatGPTAPI:
    def setup_method(self):
        utils_mod._sync_client = None

    def _patch_client(self, monkeypatch, *, side_effect=None, return_value=None):
        mock = MagicMock()
        if side_effect is not None:
            mock.chat.completions.create.side_effect = side_effect
        else:
            mock.chat.completions.create.return_value = return_value
        monkeypatch.setattr(utils_mod, "_sync_client", mock)
        return mock

    def test_returns_content_on_success(self, monkeypatch):
        self._patch_client(monkeypatch, return_value=_make_response("hello"))
        assert ChatGPT_API("gpt-4o", "prompt", api_key="k") == "hello"

    def test_retries_on_exception_then_succeeds(self, monkeypatch):
        self._patch_client(monkeypatch, side_effect=[RuntimeError("boom"), _make_response("ok")])
        with patch("time.sleep"):
            result = ChatGPT_API("gpt-4o", "prompt", api_key="k")
        assert result == "ok"

    def test_exponential_backoff_increases(self, monkeypatch):
        # Fail 3 times, succeed on 4th
        errors = [RuntimeError("e")] * 3
        self._patch_client(monkeypatch, side_effect=errors + [_make_response("done")])
        sleeps = []
        with patch("time.sleep", side_effect=lambda t: sleeps.append(t)):
            ChatGPT_API("gpt-4o", "prompt", api_key="k")
        assert len(sleeps) == 3
        # Each wait should be larger than the previous (base 2^i grows)
        assert sleeps[1] > sleeps[0]
        assert sleeps[2] > sleeps[1]

    def test_backoff_capped_at_60_seconds(self, monkeypatch):
        # i=6 → 2^6=64 → capped to 60+jitter≤61
        errors = [RuntimeError("e")] * 6
        self._patch_client(monkeypatch, side_effect=errors + [_make_response("ok")])
        sleeps = []
        with patch("time.sleep", side_effect=lambda t: sleeps.append(t)):
            ChatGPT_API("gpt-4o", "prompt", api_key="k")
        assert all(s <= 61 for s in sleeps)

    def test_max_retries_exhausted_returns_error(self, monkeypatch):
        self._patch_client(monkeypatch, side_effect=RuntimeError("always"))
        with patch("time.sleep"):
            result = ChatGPT_API("gpt-4o", "prompt", api_key="k")
        assert result == "Error"

    def test_chat_history_is_appended(self, monkeypatch):
        mock = self._patch_client(monkeypatch, return_value=_make_response("r"))
        history = [{"role": "user", "content": "prior"}]
        ChatGPT_API("gpt-4o", "follow up", api_key="k", chat_history=history)
        msgs = mock.chat.completions.create.call_args.kwargs["messages"]
        assert msgs[-1] == {"role": "user", "content": "follow up"}
        assert msgs[0] == {"role": "user", "content": "prior"}


# ─── ChatGPT_API_with_finish_reason (Fixes 3 & 5) ───────────────────────────

class TestChatGPTAPIWithFinishReason:
    def setup_method(self):
        utils_mod._sync_client = None

    def _patch_client(self, monkeypatch, *, side_effect=None, return_value=None):
        mock = MagicMock()
        if side_effect is not None:
            mock.chat.completions.create.side_effect = side_effect
        else:
            mock.chat.completions.create.return_value = return_value
        monkeypatch.setattr(utils_mod, "_sync_client", mock)
        return mock

    def test_stop_reason_returns_finished(self, monkeypatch):
        self._patch_client(monkeypatch, return_value=_make_response("result", "stop"))
        content, reason = ChatGPT_API_with_finish_reason("gpt-4o", "p", api_key="k")
        assert content == "result"
        assert reason == "finished"

    def test_length_reason_returns_max_output_reached(self, monkeypatch):
        self._patch_client(monkeypatch, return_value=_make_response("partial", "length"))
        _, reason = ChatGPT_API_with_finish_reason("gpt-4o", "p", api_key="k")
        assert reason == "max_output_reached"

    def test_retries_on_exception(self, monkeypatch):
        self._patch_client(monkeypatch, side_effect=[RuntimeError("err"), _make_response("ok")])
        with patch("time.sleep"):
            content, _ = ChatGPT_API_with_finish_reason("gpt-4o", "p", api_key="k")
        assert content == "ok"

    def test_max_retries_returns_error_string(self, monkeypatch):
        self._patch_client(monkeypatch, side_effect=RuntimeError("fail"))
        with patch("time.sleep"):
            result = ChatGPT_API_with_finish_reason("gpt-4o", "p", api_key="k")
        assert result == "Error"

    def test_chat_history_included_in_messages(self, monkeypatch):
        mock = self._patch_client(monkeypatch, return_value=_make_response("r"))
        history = [{"role": "assistant", "content": "prior reply"}]
        ChatGPT_API_with_finish_reason("gpt-4o", "next", api_key="k", chat_history=history)
        msgs = mock.chat.completions.create.call_args.kwargs["messages"]
        assert any(m["content"] == "next" for m in msgs)


# ─── ChatGPT_API_async (Fixes 4 & 5) ─────────────────────────────────────────

class TestChatGPTAPIAsync:
    def setup_method(self):
        utils_mod._async_client = None

    def teardown_method(self):
        utils_mod._async_client = None

    def _patch_async_client(self, monkeypatch, *, side_effect=None, return_value=None):
        mock = AsyncMock()
        if side_effect is not None:
            mock.chat.completions.create.side_effect = side_effect
        else:
            mock.chat.completions.create.return_value = return_value
        monkeypatch.setattr(utils_mod, "_async_client", mock)
        return mock

    def test_returns_content_on_success(self, monkeypatch):
        self._patch_async_client(monkeypatch, return_value=_make_response("async ok"))
        result = asyncio.run(ChatGPT_API_async("gpt-4o", "prompt", api_key="k"))
        assert result == "async ok"

    def test_retries_on_exception_then_succeeds(self, monkeypatch):
        self._patch_async_client(
            monkeypatch,
            side_effect=[RuntimeError("boom"), _make_response("ok")],
        )
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = asyncio.run(ChatGPT_API_async("gpt-4o", "prompt", api_key="k"))
        assert result == "ok"

    def test_max_retries_returns_error(self, monkeypatch):
        self._patch_async_client(monkeypatch, side_effect=RuntimeError("always"))
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = asyncio.run(ChatGPT_API_async("gpt-4o", "prompt", api_key="k"))
        assert result == "Error"

    def test_client_fetched_once_not_per_retry(self, monkeypatch):
        """_get_async_client must be called exactly once, not once per retry loop."""
        get_calls = []
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = _make_response("ok")

        def counting_getter(api_key):
            get_calls.append(api_key)
            return mock_client

        monkeypatch.setattr(utils_mod, "_get_async_client", counting_getter)
        asyncio.run(ChatGPT_API_async("gpt-4o", "prompt", api_key="k"))
        assert len(get_calls) == 1, "Client factory must only be called once per invocation"

    def test_exponential_backoff_async(self, monkeypatch):
        errors = [RuntimeError("e")] * 2
        self._patch_async_client(monkeypatch, side_effect=errors + [_make_response("done")])
        sleeps = []

        async def fake_sleep(t):
            sleeps.append(t)

        with patch("asyncio.sleep", side_effect=fake_sleep):
            asyncio.run(ChatGPT_API_async("gpt-4o", "prompt", api_key="k"))

        assert len(sleeps) == 2
        assert sleeps[1] > sleeps[0]


# ─── JsonLogger (Fix 8) ──────────────────────────────────────────────────────

class TestJsonLogger:
    def _make_logger(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        return JsonLogger("test.pdf")

    def _read_log(self, tmp_path, logger):
        return json.loads((tmp_path / "logs" / logger.filename).read_text())

    def test_no_disk_write_below_threshold(self, tmp_path, monkeypatch):
        logger = self._make_logger(tmp_path, monkeypatch)
        for i in range(19):
            logger.info(f"msg {i}")
        assert not (tmp_path / "logs" / logger.filename).exists()

    def test_flush_at_threshold_20(self, tmp_path, monkeypatch):
        logger = self._make_logger(tmp_path, monkeypatch)
        for i in range(20):
            logger.info(f"msg {i}")
        data = self._read_log(tmp_path, logger)
        assert len(data) == 20

    def test_flush_at_40_entries(self, tmp_path, monkeypatch):
        logger = self._make_logger(tmp_path, monkeypatch)
        for i in range(40):
            logger.info(f"msg {i}")
        data = self._read_log(tmp_path, logger)
        assert len(data) == 40

    def test_close_flushes_partial_buffer(self, tmp_path, monkeypatch):
        logger = self._make_logger(tmp_path, monkeypatch)
        for i in range(7):  # below threshold
            logger.info(f"msg {i}")
        logger.close()
        data = self._read_log(tmp_path, logger)
        assert len(data) == 7

    def test_string_message_wrapped_in_dict(self, tmp_path, monkeypatch):
        logger = self._make_logger(tmp_path, monkeypatch)
        logger.log("INFO", "plain string")
        logger.close()
        data = self._read_log(tmp_path, logger)
        assert data[0] == {"message": "plain string"}

    def test_dict_message_stored_directly(self, tmp_path, monkeypatch):
        logger = self._make_logger(tmp_path, monkeypatch)
        msg = {"key": "value", "count": 42}
        logger.log("INFO", msg)
        logger.close()
        data = self._read_log(tmp_path, logger)
        assert data[0] == msg

    def test_info_error_debug_all_recorded(self, tmp_path, monkeypatch):
        logger = self._make_logger(tmp_path, monkeypatch)
        logger.info("info msg")
        logger.error("error msg")
        logger.debug("debug msg")
        logger.close()
        data = self._read_log(tmp_path, logger)
        messages = [d["message"] for d in data]
        assert "info msg" in messages
        assert "error msg" in messages
        assert "debug msg" in messages

    def test_log_data_accumulates_in_memory(self, tmp_path, monkeypatch):
        logger = self._make_logger(tmp_path, monkeypatch)
        for i in range(5):
            logger.info(f"msg{i}")
        assert len(logger.log_data) == 5

    def test_threshold_not_triggered_at_19(self, tmp_path, monkeypatch):
        """Threshold is 20, not 19."""
        logger = self._make_logger(tmp_path, monkeypatch)
        for i in range(19):
            logger.info(f"msg{i}")
        assert not (tmp_path / "logs" / logger.filename).exists()
        logger.info("msg19")  # 20th — must flush now
        assert (tmp_path / "logs" / logger.filename).exists()

    def test_flush_writes_valid_json(self, tmp_path, monkeypatch):
        logger = self._make_logger(tmp_path, monkeypatch)
        logger.info({"structured": True, "value": 99})
        logger.close()
        data = self._read_log(tmp_path, logger)
        assert data[0]["structured"] is True
        assert data[0]["value"] == 99


# ─── extract_json ─────────────────────────────────────────────────────────────

class TestExtractJson:
    def test_plain_json_object(self):
        assert extract_json('{"key": "value"}') == {"key": "value"}

    def test_json_in_code_block(self):
        assert extract_json('```json\n{"key": "value"}\n```') == {"key": "value"}

    def test_invalid_json_returns_empty_dict(self):
        assert extract_json("not json at all!!!") == {}

    def test_python_none_replaced_with_null(self):
        result = extract_json('{"key": None}')
        assert result == {"key": None}

    def test_trailing_comma_in_object_cleaned(self):
        assert extract_json('{"a": 1,}') == {"a": 1}

    def test_trailing_comma_in_array_cleaned(self):
        assert extract_json('{"a": [1,2,]}') == {"a": [1, 2]}


# ─── get_json_content ─────────────────────────────────────────────────────────

class TestGetJsonContent:
    def test_strips_json_code_block(self):
        raw = '```json\n{"x": 1}\n```'
        assert get_json_content(raw) == '{"x": 1}'

    def test_returns_plain_string_unchanged(self):
        raw = '{"x": 1}'
        assert get_json_content(raw) == '{"x": 1}'

    def test_strips_surrounding_whitespace(self):
        raw = '   {"x": 1}   '
        assert get_json_content(raw) == '{"x": 1}'


# ─── write_node_id ────────────────────────────────────────────────────────────

class TestWriteNodeId:
    def test_flat_list_sequential_ids(self):
        data = [{"title": "A", "nodes": []}, {"title": "B", "nodes": []}]
        write_node_id(data)
        assert data[0]["node_id"] == "0000"
        assert data[1]["node_id"] == "0001"

    def test_nested_nodes_depth_first(self):
        data = {"title": "Root", "nodes": [{"title": "Child", "nodes": []}]}
        write_node_id(data)
        assert data["node_id"] == "0000"
        assert data["nodes"][0]["node_id"] == "0001"

    def test_ids_zero_padded_to_four_digits(self):
        data = [{"title": str(i), "nodes": []} for i in range(12)]
        write_node_id(data)
        assert data[9]["node_id"] == "0009"
        assert data[10]["node_id"] == "0010"

    def test_deeply_nested_ids_sequential(self):
        data = {
            "title": "Root",
            "nodes": [
                {"title": "A", "nodes": [{"title": "A1", "nodes": []}]},
                {"title": "B", "nodes": []},
            ],
        }
        write_node_id(data)
        assert data["node_id"] == "0000"
        assert data["nodes"][0]["node_id"] == "0001"
        assert data["nodes"][0]["nodes"][0]["node_id"] == "0002"
        assert data["nodes"][1]["node_id"] == "0003"


# ─── get_nodes / get_leaf_nodes ───────────────────────────────────────────────

class TestGetNodes:
    def test_flat_list_all_titles_present(self):
        data = [{"title": "A", "nodes": []}, {"title": "B", "nodes": []}]
        nodes = get_nodes(data)
        titles = {n["title"] for n in nodes}
        assert titles == {"A", "B"}

    def test_parent_node_has_no_nodes_key(self):
        data = {"title": "Root", "nodes": [{"title": "Child", "nodes": []}]}
        nodes = get_nodes(data)
        root = next(n for n in nodes if n["title"] == "Root")
        assert "nodes" not in root

    def test_both_parent_and_child_returned(self):
        data = {"title": "Root", "nodes": [{"title": "Child", "nodes": []}]}
        nodes = get_nodes(data)
        titles = {n["title"] for n in nodes}
        assert "Root" in titles and "Child" in titles


class TestGetLeafNodes:
    def test_returns_only_leaves(self):
        data = [
            {"title": "A", "nodes": [{"title": "A1", "nodes": []}]},
            {"title": "B", "nodes": []},
        ]
        leaves = get_leaf_nodes(data)
        titles = {n["title"] for n in leaves}
        assert "A1" in titles
        assert "B" in titles
        assert "A" not in titles

    def test_single_root_no_children_is_leaf(self):
        data = [{"title": "Solo", "nodes": []}]
        leaves = get_leaf_nodes(data)
        assert len(leaves) == 1
        assert leaves[0]["title"] == "Solo"


# ─── sanitize_filename ────────────────────────────────────────────────────────

class TestSanitizeFilename:
    def test_slash_replaced_with_hyphen(self):
        assert sanitize_filename("foo/bar") == "foo-bar"

    def test_multiple_slashes(self):
        assert sanitize_filename("a/b/c") == "a-b-c"

    def test_clean_name_unchanged(self):
        assert sanitize_filename("report_2024.pdf") == "report_2024.pdf"

    def test_custom_replacement_char(self):
        assert sanitize_filename("a/b", replacement="_") == "a_b"


# ─── list_to_tree ─────────────────────────────────────────────────────────────

class TestListToTree:
    def test_flat_list_produces_root_nodes(self):
        data = [
            {"structure": "1", "title": "Ch 1", "start_index": 1, "end_index": 5},
            {"structure": "2", "title": "Ch 2", "start_index": 6, "end_index": 10},
        ]
        tree = list_to_tree(data)
        assert len(tree) == 2
        assert tree[0]["title"] == "Ch 1"
        assert tree[1]["title"] == "Ch 2"

    def test_nested_structure_builds_parent_child(self):
        data = [
            {"structure": "1", "title": "Ch 1", "start_index": 1, "end_index": 10},
            {"structure": "1.1", "title": "Sec 1.1", "start_index": 1, "end_index": 5},
            {"structure": "1.2", "title": "Sec 1.2", "start_index": 6, "end_index": 10},
        ]
        tree = list_to_tree(data)
        assert len(tree) == 1
        assert len(tree[0]["nodes"]) == 2

    def test_leaf_nodes_have_no_nodes_key(self):
        data = [{"structure": "1", "title": "Only", "start_index": 1, "end_index": 5}]
        tree = list_to_tree(data)
        assert "nodes" not in tree[0]

    def test_three_level_nesting(self):
        data = [
            {"structure": "1", "title": "Part", "start_index": 1, "end_index": 20},
            {"structure": "1.1", "title": "Chapter", "start_index": 1, "end_index": 10},
            {"structure": "1.1.1", "title": "Section", "start_index": 1, "end_index": 5},
        ]
        tree = list_to_tree(data)
        assert tree[0]["nodes"][0]["nodes"][0]["title"] == "Section"
