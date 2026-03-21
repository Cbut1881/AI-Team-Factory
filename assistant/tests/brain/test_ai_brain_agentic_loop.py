"""
Tests for the AIBrain agentic tool-use loop.

Covers single/multi tool calls, multi-turn chains, tool result formatting,
exception handling, and the MAX_AGENTIC_ITERATIONS safety limit.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from brain.ai_brain import (
    AIBrain,
    ConversationMessage,
    MAX_AGENTIC_ITERATIONS,
    MessageRole,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text_block(text: str) -> MagicMock:
    b = MagicMock()
    b.type = "text"
    b.text = text
    return b


def _tool_block(tool_id: str, name: str, inp: dict) -> MagicMock:
    b = MagicMock()
    b.type = "tool_use"
    b.id = tool_id
    b.name = name
    b.input = inp
    return b


def _make_response(
    blocks: list,
    stop_reason: str = "end_turn",
) -> MagicMock:
    msg = MagicMock()
    msg.content = blocks
    msg.stop_reason = stop_reason
    return msg


# =========================================================================
# Single-turn — no tools
# =========================================================================

class TestSingleTurnNoTools:
    """Agentic loop exits immediately on pure text responses."""

    @pytest.mark.asyncio
    async def test_single_text_response(self, brain_with_client):
        resp = _make_response([_text_block("Just text")])
        brain_with_client.client.messages.create = MagicMock(return_value=resp)

        brain_with_client.conversation.append(
            ConversationMessage(role=MessageRole.USER, content=[{"type": "text", "text": "hi"}])
        )
        result = await brain_with_client._agentic_loop()
        assert result == "Just text"

    @pytest.mark.asyncio
    async def test_empty_text_returns_default(self, brain_with_client):
        resp = _make_response([_text_block("")])
        brain_with_client.client.messages.create = MagicMock(return_value=resp)

        brain_with_client.conversation.append(
            ConversationMessage(role=MessageRole.USER, content=[{"type": "text", "text": "x"}])
        )
        result = await brain_with_client._agentic_loop()
        assert result == "(Nova completed the requested actions.)"

    @pytest.mark.asyncio
    async def test_whitespace_only_text(self, brain_with_client):
        resp = _make_response([_text_block("   \n  ")])
        brain_with_client.client.messages.create = MagicMock(return_value=resp)

        brain_with_client.conversation.append(
            ConversationMessage(role=MessageRole.USER, content=[{"type": "text", "text": "x"}])
        )
        result = await brain_with_client._agentic_loop()
        assert result == "(Nova completed the requested actions.)"

    @pytest.mark.asyncio
    async def test_multi_text_blocks_joined(self, brain_with_client):
        resp = _make_response([_text_block("Part 1"), _text_block("Part 2")])
        brain_with_client.client.messages.create = MagicMock(return_value=resp)

        brain_with_client.conversation.append(
            ConversationMessage(role=MessageRole.USER, content=[{"type": "text", "text": "x"}])
        )
        result = await brain_with_client._agentic_loop()
        assert result == "Part 1\nPart 2"


# =========================================================================
# Single tool call
# =========================================================================

class TestSingleToolCall:
    """Agentic loop with a single tool call then final text."""

    @pytest.mark.asyncio
    async def test_one_tool_then_text(self, brain_with_client):
        tool_resp = _make_response(
            [_tool_block("t1", "run_shell_command", {"command": "ls"})],
            stop_reason="tool_use",
        )
        final_resp = _make_response([_text_block("Here are the files")])

        brain_with_client.client.messages.create = MagicMock(
            side_effect=[tool_resp, final_resp]
        )
        brain_with_client.execute_tool = MagicMock(return_value="file1\nfile2")

        brain_with_client.conversation.append(
            ConversationMessage(role=MessageRole.USER, content=[{"type": "text", "text": "list files"}])
        )
        result = await brain_with_client._agentic_loop()

        assert result == "Here are the files"
        brain_with_client.execute_tool.assert_called_once_with(
            "run_shell_command", {"command": "ls"}
        )

    @pytest.mark.asyncio
    async def test_tool_result_appended_as_user_message(self, brain_with_client):
        tool_resp = _make_response(
            [_tool_block("t1", "get_clipboard", {})],
            stop_reason="tool_use",
        )
        final_resp = _make_response([_text_block("Done")])

        brain_with_client.client.messages.create = MagicMock(
            side_effect=[tool_resp, final_resp]
        )
        brain_with_client.execute_tool = MagicMock(return_value="clipboard text")

        brain_with_client.conversation.append(
            ConversationMessage(role=MessageRole.USER, content=[{"type": "text", "text": "paste"}])
        )
        await brain_with_client._agentic_loop()

        # Should have: user, assistant(tool_use), user(tool_result), assistant(text)
        roles = [m.role for m in brain_with_client.conversation]
        assert roles == ["user", "assistant", "user", "assistant"]


# =========================================================================
# Multi tool calls in single response
# =========================================================================

class TestMultiToolCalls:
    """Multiple tool_use blocks in a single Claude response."""

    @pytest.mark.asyncio
    async def test_two_tools_in_one_response(self, brain_with_client):
        tool_resp = _make_response(
            [
                _tool_block("t1", "run_shell_command", {"command": "pwd"}),
                _tool_block("t2", "get_clipboard", {}),
            ],
            stop_reason="tool_use",
        )
        final_resp = _make_response([_text_block("All done")])

        brain_with_client.client.messages.create = MagicMock(
            side_effect=[tool_resp, final_resp]
        )
        call_count = 0

        def mock_exec(name, inp):
            nonlocal call_count
            call_count += 1
            return f"result_{call_count}"

        brain_with_client.execute_tool = mock_exec

        brain_with_client.conversation.append(
            ConversationMessage(role=MessageRole.USER, content=[{"type": "text", "text": "do both"}])
        )
        result = await brain_with_client._agentic_loop()

        assert result == "All done"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_tool_results_have_correct_ids(self, brain_with_client):
        tool_resp = _make_response(
            [
                _tool_block("id_a", "tool_a", {}),
                _tool_block("id_b", "tool_b", {}),
            ],
            stop_reason="tool_use",
        )
        final_resp = _make_response([_text_block("Done")])

        brain_with_client.client.messages.create = MagicMock(
            side_effect=[tool_resp, final_resp]
        )
        brain_with_client.execute_tool = MagicMock(return_value="ok")

        brain_with_client.conversation.append(
            ConversationMessage(role=MessageRole.USER, content=[{"type": "text", "text": "go"}])
        )
        await brain_with_client._agentic_loop()

        # The tool result user message should have two tool_result blocks
        tool_result_msg = brain_with_client.conversation[2]
        assert tool_result_msg.role == "user"
        ids = [b["tool_use_id"] for b in tool_result_msg.content]
        assert ids == ["id_a", "id_b"]


# =========================================================================
# Multi-turn tool chains
# =========================================================================

class TestMultiTurnToolChain:
    """Multiple iterations of tool_use -> tool_result -> tool_use."""

    @pytest.mark.asyncio
    async def test_two_turn_tool_chain(self, brain_with_client):
        resp1 = _make_response(
            [_tool_block("t1", "run_shell_command", {"command": "ls"})],
            stop_reason="tool_use",
        )
        resp2 = _make_response(
            [_text_block("Checking..."), _tool_block("t2", "read_file", {"path": "a.txt"})],
            stop_reason="tool_use",
        )
        resp3 = _make_response([_text_block("File contents shown")])

        brain_with_client.client.messages.create = MagicMock(
            side_effect=[resp1, resp2, resp3]
        )
        brain_with_client.execute_tool = MagicMock(return_value="data")

        brain_with_client.conversation.append(
            ConversationMessage(role=MessageRole.USER, content=[{"type": "text", "text": "chain"}])
        )
        result = await brain_with_client._agentic_loop()

        assert result == "File contents shown"
        assert brain_with_client.execute_tool.call_count == 2

    @pytest.mark.asyncio
    async def test_three_turn_chain(self, brain_with_client):
        responses = []
        for i in range(3):
            responses.append(_make_response(
                [_tool_block(f"t{i}", f"tool_{i}", {})],
                stop_reason="tool_use",
            ))
        responses.append(_make_response([_text_block("Final")]))

        brain_with_client.client.messages.create = MagicMock(side_effect=responses)
        brain_with_client.execute_tool = MagicMock(return_value="ok")

        brain_with_client.conversation.append(
            ConversationMessage(role=MessageRole.USER, content=[{"type": "text", "text": "go"}])
        )
        result = await brain_with_client._agentic_loop()

        assert result == "Final"
        assert brain_with_client.execute_tool.call_count == 3


# =========================================================================
# Tool result formatting
# =========================================================================

class TestToolResultFormatting:
    """Tests for _format_tool_result with various result types."""

    def test_none_result(self, brain_with_client):
        result = brain_with_client._format_tool_result(None)
        assert result == "Done (no output)."

    def test_string_result(self, brain_with_client):
        result = brain_with_client._format_tool_result("hello world")
        assert result == "hello world"

    def test_dict_result_json(self, brain_with_client):
        result = brain_with_client._format_tool_result({"key": "value"})
        parsed = json.loads(result)
        assert parsed == {"key": "value"}

    def test_dict_with_image_base64(self, brain_with_client):
        result = brain_with_client._format_tool_result({
            "image_base64": "AAAA",
            "media_type": "image/png",
        })
        assert isinstance(result, list)
        assert result[0]["type"] == "image"
        assert result[0]["source"]["data"] == "AAAA"

    def test_dict_with_image_and_text(self, brain_with_client):
        result = brain_with_client._format_tool_result({
            "image_base64": "BBBB",
            "text": "Screenshot taken",
        })
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[1]["type"] == "text"
        assert result[1]["text"] == "Screenshot taken"

    def test_dict_image_default_media_type(self, brain_with_client):
        result = brain_with_client._format_tool_result({"image_base64": "CC"})
        assert result[0]["source"]["media_type"] == "image/png"

    def test_list_result_json(self, brain_with_client):
        result = brain_with_client._format_tool_result([1, 2, 3])
        parsed = json.loads(result)
        assert parsed == [1, 2, 3]

    def test_tuple_result_json(self, brain_with_client):
        result = brain_with_client._format_tool_result((1, "a"))
        parsed = json.loads(result)
        assert parsed == [1, "a"]

    def test_integer_result_stringified(self, brain_with_client):
        result = brain_with_client._format_tool_result(42)
        assert result == "42"

    def test_bool_result_stringified(self, brain_with_client):
        result = brain_with_client._format_tool_result(True)
        assert result == "True"


# =========================================================================
# Tool execution exceptions
# =========================================================================

class TestToolExecutionExceptions:
    """Tests for error handling when tools raise exceptions."""

    @pytest.mark.asyncio
    async def test_tool_exception_returns_error_block(self, brain_with_client):
        tool_resp = _make_response(
            [_tool_block("t1", "bad_tool", {})],
            stop_reason="tool_use",
        )
        final_resp = _make_response([_text_block("Handled error")])

        brain_with_client.client.messages.create = MagicMock(
            side_effect=[tool_resp, final_resp]
        )
        brain_with_client.execute_tool = MagicMock(
            side_effect=RuntimeError("tool broke")
        )

        brain_with_client.conversation.append(
            ConversationMessage(role=MessageRole.USER, content=[{"type": "text", "text": "try"}])
        )
        result = await brain_with_client._agentic_loop()

        assert result == "Handled error"
        # Check the tool result was marked as error
        tool_result_msg = brain_with_client.conversation[2]
        error_block = tool_result_msg.content[0]
        assert error_block["is_error"] is True
        assert "RuntimeError" in error_block["content"]

    @pytest.mark.asyncio
    async def test_tool_exception_preserves_tool_use_id(self, brain_with_client):
        tool_resp = _make_response(
            [_tool_block("myid123", "failing", {})],
            stop_reason="tool_use",
        )
        final_resp = _make_response([_text_block("ok")])

        brain_with_client.client.messages.create = MagicMock(
            side_effect=[tool_resp, final_resp]
        )
        brain_with_client.execute_tool = MagicMock(side_effect=ValueError("bad"))

        brain_with_client.conversation.append(
            ConversationMessage(role=MessageRole.USER, content=[{"type": "text", "text": "x"}])
        )
        await brain_with_client._agentic_loop()

        tool_result_msg = brain_with_client.conversation[2]
        assert tool_result_msg.content[0]["tool_use_id"] == "myid123"

    @pytest.mark.asyncio
    async def test_tool_type_error_captured(self, brain_with_client):
        tool_resp = _make_response(
            [_tool_block("t1", "tool", {})],
            stop_reason="tool_use",
        )
        final_resp = _make_response([_text_block("ok")])

        brain_with_client.client.messages.create = MagicMock(
            side_effect=[tool_resp, final_resp]
        )
        brain_with_client.execute_tool = MagicMock(
            side_effect=TypeError("wrong type")
        )

        brain_with_client.conversation.append(
            ConversationMessage(role=MessageRole.USER, content=[{"type": "text", "text": "x"}])
        )
        await brain_with_client._agentic_loop()

        tool_result_msg = brain_with_client.conversation[2]
        assert "TypeError" in tool_result_msg.content[0]["content"]


# =========================================================================
# MAX_AGENTIC_ITERATIONS safety limit
# =========================================================================

class TestMaxAgenticIterations:
    """Safety limit prevents infinite tool loops."""

    @pytest.mark.asyncio
    async def test_loop_exhaustion_returns_pause_message(self, brain_with_client):
        # Every response requests another tool call
        infinite_tool_resp = _make_response(
            [_tool_block("t1", "tool", {})],
            stop_reason="tool_use",
        )
        brain_with_client.client.messages.create = MagicMock(
            return_value=infinite_tool_resp
        )
        brain_with_client.execute_tool = MagicMock(return_value="ok")

        brain_with_client.conversation.append(
            ConversationMessage(role=MessageRole.USER, content=[{"type": "text", "text": "loop"}])
        )
        result = await brain_with_client._agentic_loop()

        assert "pause" in result.lower() or "while" in result.lower()
        assert brain_with_client.execute_tool.call_count == MAX_AGENTIC_ITERATIONS

    @pytest.mark.asyncio
    async def test_loop_exits_before_max_when_text_returned(self, brain_with_client):
        tool_resp = _make_response(
            [_tool_block("t1", "tool", {})],
            stop_reason="tool_use",
        )
        final_resp = _make_response([_text_block("Done early")])

        # Tool on first call, text on second
        brain_with_client.client.messages.create = MagicMock(
            side_effect=[tool_resp, final_resp]
        )
        brain_with_client.execute_tool = MagicMock(return_value="ok")

        brain_with_client.conversation.append(
            ConversationMessage(role=MessageRole.USER, content=[{"type": "text", "text": "go"}])
        )
        result = await brain_with_client._agentic_loop()

        assert result == "Done early"
        assert brain_with_client.execute_tool.call_count == 1


# =========================================================================
# Mixed text + tool_use blocks
# =========================================================================

class TestMixedTextAndToolUse:
    """Claude responses containing both text and tool_use blocks."""

    @pytest.mark.asyncio
    async def test_text_plus_tool_use(self, brain_with_client):
        mixed_resp = _make_response(
            [
                _text_block("Let me check..."),
                _tool_block("t1", "run_shell_command", {"command": "ls"}),
            ],
            stop_reason="tool_use",
        )
        final_resp = _make_response([_text_block("Found the files")])

        brain_with_client.client.messages.create = MagicMock(
            side_effect=[mixed_resp, final_resp]
        )
        brain_with_client.execute_tool = MagicMock(return_value="file1")

        brain_with_client.conversation.append(
            ConversationMessage(role=MessageRole.USER, content=[{"type": "text", "text": "find"}])
        )
        result = await brain_with_client._agentic_loop()

        assert result == "Found the files"

    @pytest.mark.asyncio
    async def test_assistant_content_has_both_types(self, brain_with_client):
        mixed_resp = _make_response(
            [
                _text_block("Working on it"),
                _tool_block("t1", "tool", {}),
            ],
            stop_reason="tool_use",
        )
        final_resp = _make_response([_text_block("Done")])

        brain_with_client.client.messages.create = MagicMock(
            side_effect=[mixed_resp, final_resp]
        )
        brain_with_client.execute_tool = MagicMock(return_value="ok")

        brain_with_client.conversation.append(
            ConversationMessage(role=MessageRole.USER, content=[{"type": "text", "text": "x"}])
        )
        await brain_with_client._agentic_loop()

        # First assistant message should have text + tool_use
        asst_msg = brain_with_client.conversation[1]
        types = [b["type"] for b in asst_msg.content]
        assert "text" in types
        assert "tool_use" in types

    @pytest.mark.asyncio
    async def test_end_turn_with_tool_blocks_still_exits(self, brain_with_client):
        """If stop_reason is end_turn even with tool blocks, loop exits."""
        resp = _make_response(
            [
                _text_block("No need to run tools"),
                _tool_block("t1", "tool", {}),
            ],
            stop_reason="end_turn",
        )
        brain_with_client.client.messages.create = MagicMock(return_value=resp)

        brain_with_client.conversation.append(
            ConversationMessage(role=MessageRole.USER, content=[{"type": "text", "text": "x"}])
        )
        result = await brain_with_client._agentic_loop()

        # end_turn takes priority — no tool execution
        assert result == "No need to run tools"


# =========================================================================
# _execute_tool unit tests
# =========================================================================

class TestExecuteToolMethod:
    """Direct tests for the _execute_tool method."""

    @pytest.mark.asyncio
    async def test_execute_tool_success(self, brain_with_client):
        brain_with_client.execute_tool = MagicMock(return_value="result text")
        result = await brain_with_client._execute_tool("my_tool", {"a": 1}, "id_1")

        assert result["tool_use_id"] == "id_1"
        assert result["content"] == "result text"
        assert "is_error" not in result

    @pytest.mark.asyncio
    async def test_execute_tool_failure(self, brain_with_client):
        brain_with_client.execute_tool = MagicMock(
            side_effect=OSError("disk full")
        )
        result = await brain_with_client._execute_tool("my_tool", {}, "id_2")

        assert result["is_error"] is True
        assert "OSError" in result["content"]
        assert "disk full" in result["content"]
        assert result["tool_use_id"] == "id_2"

    @pytest.mark.asyncio
    async def test_execute_tool_none_result(self, brain_with_client):
        brain_with_client.execute_tool = MagicMock(return_value=None)
        result = await brain_with_client._execute_tool("my_tool", {}, "id_3")

        assert result["content"] == "Done (no output)."

    @pytest.mark.asyncio
    async def test_execute_tool_dict_result(self, brain_with_client):
        brain_with_client.execute_tool = MagicMock(return_value={"status": "ok"})
        result = await brain_with_client._execute_tool("my_tool", {}, "id_4")

        parsed = json.loads(result["content"])
        assert parsed["status"] == "ok"
