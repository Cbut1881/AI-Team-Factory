"""
Tests for AIBrain context windowing and message building.

Covers _build_messages, _summarize_and_trim, _create_summary, hard trim
at MAX_CONVERSATION_MESSAGES, and the first-message-must-be-user guarantee.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from brain.ai_brain import (
    AIBrain,
    ConversationMessage,
    CONTEXT_WINDOW_KEEP_RECENT,
    MAX_CONVERSATION_MESSAGES,
    MessageRole,
    SUMMARY_TRIGGER_COUNT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _user_msg(text: str) -> ConversationMessage:
    return ConversationMessage(
        role=MessageRole.USER,
        content=[{"type": "text", "text": text}],
    )


def _assistant_msg(text: str) -> ConversationMessage:
    return ConversationMessage(
        role=MessageRole.ASSISTANT,
        content=[{"type": "text", "text": text}],
    )


def _fill_conversation(brain: AIBrain, count: int) -> None:
    """Add alternating user/assistant messages to the brain's conversation."""
    for i in range(count):
        if i % 2 == 0:
            brain.conversation.append(_user_msg(f"User message {i}"))
        else:
            brain.conversation.append(_assistant_msg(f"Assistant message {i}"))


# =========================================================================
# _build_messages — few messages
# =========================================================================

class TestBuildMessagesFew:
    """When total messages fit within CONTEXT_WINDOW_KEEP_RECENT."""

    def test_empty_conversation(self, brain_with_client):
        msgs = brain_with_client._build_messages()
        assert msgs == []

    def test_single_user_message(self, brain_with_client):
        brain_with_client.conversation.append(_user_msg("Hello"))
        msgs = brain_with_client._build_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_few_messages_all_included(self, brain_with_client):
        _fill_conversation(brain_with_client, 6)
        msgs = brain_with_client._build_messages()
        assert len(msgs) == 6

    def test_exactly_at_keep_recent_limit(self, brain_with_client):
        _fill_conversation(brain_with_client, CONTEXT_WINDOW_KEEP_RECENT)
        msgs = brain_with_client._build_messages()
        assert len(msgs) == CONTEXT_WINDOW_KEEP_RECENT

    def test_messages_preserve_order(self, brain_with_client):
        brain_with_client.conversation.append(_user_msg("first"))
        brain_with_client.conversation.append(_assistant_msg("second"))
        brain_with_client.conversation.append(_user_msg("third"))

        msgs = brain_with_client._build_messages()
        roles = [m["role"] for m in msgs]
        assert roles == ["user", "assistant", "user"]


# =========================================================================
# _build_messages — many messages (context windowing with summary)
# =========================================================================

class TestBuildMessagesWithSummary:
    """When total messages exceed CONTEXT_WINDOW_KEEP_RECENT."""

    def test_summary_prepended_when_exceeding_limit(self, brain_with_client):
        count = CONTEXT_WINDOW_KEEP_RECENT + 10
        _fill_conversation(brain_with_client, count)

        msgs = brain_with_client._build_messages()

        # Should have: summary(user) + ack(assistant) + CONTEXT_WINDOW_KEEP_RECENT recent
        assert len(msgs) == CONTEXT_WINDOW_KEEP_RECENT + 2
        assert msgs[0]["role"] == "user"
        assert "context summary" in msgs[0]["content"].lower()

    def test_summary_ack_message_present(self, brain_with_client):
        _fill_conversation(brain_with_client, CONTEXT_WINDOW_KEEP_RECENT + 5)
        msgs = brain_with_client._build_messages()

        assert msgs[1]["role"] == "assistant"
        assert "context" in msgs[1]["content"].lower()

    def test_recent_messages_preserved_verbatim(self, brain_with_client):
        _fill_conversation(brain_with_client, CONTEXT_WINDOW_KEEP_RECENT + 4)

        last_msg = brain_with_client.conversation[-1]
        msgs = brain_with_client._build_messages()

        assert msgs[-1] == last_msg.to_api_format()

    def test_summary_cache_set(self, brain_with_client):
        _fill_conversation(brain_with_client, CONTEXT_WINDOW_KEEP_RECENT + 5)
        brain_with_client._build_messages()
        assert brain_with_client._summary_cache is not None

    def test_summary_includes_old_message_text(self, brain_with_client):
        brain_with_client.conversation.append(_user_msg("unique_old_phrase"))
        _fill_conversation(brain_with_client, CONTEXT_WINDOW_KEEP_RECENT + 5)

        brain_with_client._build_messages()
        assert "unique_old_phrase" in brain_with_client._summary_cache


# =========================================================================
# _summarize_and_trim
# =========================================================================

class TestSummarizeAndTrim:
    """Tests for _summarize_and_trim."""

    def test_trim_removes_oldest(self, brain_with_client):
        _fill_conversation(brain_with_client, 10)
        original_last = brain_with_client.conversation[-1]

        brain_with_client._summarize_and_trim(3)

        assert len(brain_with_client.conversation) == 7
        assert brain_with_client.conversation[-1] is original_last

    def test_trim_zero_does_nothing(self, brain_with_client):
        _fill_conversation(brain_with_client, 5)
        brain_with_client._summarize_and_trim(0)
        assert len(brain_with_client.conversation) == 5

    def test_trim_negative_does_nothing(self, brain_with_client):
        _fill_conversation(brain_with_client, 5)
        brain_with_client._summarize_and_trim(-3)
        assert len(brain_with_client.conversation) == 5

    def test_trim_updates_summary_cache(self, brain_with_client):
        _fill_conversation(brain_with_client, 10)
        brain_with_client._summarize_and_trim(4)
        assert brain_with_client._summary_cache is not None

    def test_trim_appends_to_existing_summary(self, brain_with_client):
        brain_with_client._summary_cache = "Previous context"
        _fill_conversation(brain_with_client, 10)
        brain_with_client._summarize_and_trim(3)
        assert "Previous context" in brain_with_client._summary_cache

    def test_trim_all_messages(self, brain_with_client):
        _fill_conversation(brain_with_client, 5)
        brain_with_client._summarize_and_trim(5)
        assert len(brain_with_client.conversation) == 0
        assert brain_with_client._summary_cache is not None


# =========================================================================
# _create_summary
# =========================================================================

class TestCreateSummary:
    """Tests for the static _create_summary method."""

    def test_empty_list(self):
        result = AIBrain._create_summary([])
        assert result == ""

    def test_string_content_message(self):
        msgs = [ConversationMessage(role="user", content="Hello world")]
        result = AIBrain._create_summary(msgs)
        assert "User" in result
        assert "Hello world" in result

    def test_assistant_label(self):
        msgs = [ConversationMessage(role="assistant", content="Hi there")]
        result = AIBrain._create_summary(msgs)
        assert "Nova" in result

    def test_list_content_with_text(self):
        msgs = [
            ConversationMessage(
                role="user",
                content=[{"type": "text", "text": "What is this?"}],
            )
        ]
        result = AIBrain._create_summary(msgs)
        assert "What is this?" in result

    def test_list_content_with_tool_use(self):
        msgs = [
            ConversationMessage(
                role="assistant",
                content=[
                    {"type": "tool_use", "name": "run_shell_command", "id": "t1", "input": {}},
                ],
            )
        ]
        result = AIBrain._create_summary(msgs)
        assert "run_shell_command" in result

    def test_list_content_with_tool_result(self):
        msgs = [
            ConversationMessage(
                role="user",
                content=[
                    {"type": "tool_result", "tool_use_id": "t1", "content": "Success output"},
                ],
            )
        ]
        result = AIBrain._create_summary(msgs)
        assert "Success output" in result

    def test_list_content_with_image(self):
        msgs = [
            ConversationMessage(
                role="user",
                content=[{"type": "image", "source": {"type": "base64", "data": "AAAA"}}],
            )
        ]
        result = AIBrain._create_summary(msgs)
        assert "Image" in result

    def test_long_text_truncated(self):
        long_text = "A" * 500
        msgs = [ConversationMessage(role="user", content=long_text)]
        result = AIBrain._create_summary(msgs)
        # Content should be truncated to 200 chars
        assert len(result) < 500

    def test_tool_result_non_string_content(self):
        msgs = [
            ConversationMessage(
                role="user",
                content=[
                    {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "image"}]},
                ],
            )
        ]
        result = AIBrain._create_summary(msgs)
        assert "Tool returned result" in result

    def test_mixed_blocks(self):
        msgs = [
            ConversationMessage(
                role="user",
                content=[
                    {"type": "text", "text": "Look at this"},
                    {"type": "image", "source": {"type": "base64", "data": "X"}},
                ],
            )
        ]
        result = AIBrain._create_summary(msgs)
        assert "Look at this" in result
        assert "Image" in result


# =========================================================================
# Hard trim at MAX_CONVERSATION_MESSAGES
# =========================================================================

class TestHardTrim:
    """_build_messages enforces the MAX_CONVERSATION_MESSAGES ceiling."""

    def test_hard_trim_triggered(self, brain_with_client):
        _fill_conversation(brain_with_client, MAX_CONVERSATION_MESSAGES + 10)
        brain_with_client._build_messages()
        assert len(brain_with_client.conversation) <= MAX_CONVERSATION_MESSAGES

    def test_hard_trim_preserves_recent(self, brain_with_client):
        _fill_conversation(brain_with_client, MAX_CONVERSATION_MESSAGES + 5)
        last_content = brain_with_client.conversation[-1].content
        brain_with_client._build_messages()
        assert brain_with_client.conversation[-1].content == last_content


# =========================================================================
# First message must be user role
# =========================================================================

class TestFirstMessageUserRole:
    """API requires messages[0] to be user role."""

    def test_first_message_is_user_when_starts_with_assistant(self, brain_with_client):
        """If windowed messages start with assistant, a user preamble is inserted."""
        # Force a scenario where recent messages start with assistant
        # Add enough messages that windowing kicks in, with assistant first in recent
        total = CONTEXT_WINDOW_KEEP_RECENT + 5
        for i in range(total):
            if i % 2 == 0:
                brain_with_client.conversation.append(_user_msg(f"u{i}"))
            else:
                brain_with_client.conversation.append(_assistant_msg(f"a{i}"))

        # Manually adjust so the first recent message is assistant
        split_idx = total - CONTEXT_WINDOW_KEEP_RECENT
        brain_with_client.conversation[split_idx] = _assistant_msg("starts with assistant")

        msgs = brain_with_client._build_messages()
        assert msgs[0]["role"] == "user"

    def test_normal_conversation_starts_with_user(self, brain_with_client):
        brain_with_client.conversation.append(_user_msg("Hello"))
        brain_with_client.conversation.append(_assistant_msg("Hi"))
        msgs = brain_with_client._build_messages()
        assert msgs[0]["role"] == "user"
