"""
Shared fixtures for brain module tests.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from brain.ai_brain import AIBrain
from config import AssistantConfig, ClaudeModelConfig


# ---------------------------------------------------------------------------
# Helpers to build mock Anthropic Message objects
# ---------------------------------------------------------------------------

def _make_text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_tool_use_block(
    tool_id: str = "toolu_01",
    name: str = "run_shell_command",
    tool_input: dict | None = None,
) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = tool_input or {}
    return block


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def _base_config() -> AssistantConfig:
    """Minimal AssistantConfig with a dummy API key."""
    cfg = AssistantConfig()
    cfg.anthropic_api_key = "sk-ant-test-key"
    cfg.claude = ClaudeModelConfig(model="claude-test", max_tokens=1024)
    cfg.ollama_base_url = "http://localhost:11434"
    return cfg


@pytest.fixture
def _no_key_config() -> AssistantConfig:
    """AssistantConfig with no API key."""
    cfg = AssistantConfig()
    cfg.anthropic_api_key = ""
    cfg.claude = ClaudeModelConfig(model="claude-test", max_tokens=1024)
    cfg.ollama_base_url = "http://localhost:11434"
    return cfg


@pytest.fixture
def brain_with_client(_base_config: AssistantConfig) -> AIBrain:
    """AIBrain initialised with a mocked Anthropic client."""
    with patch("brain.ai_brain.anthropic.Anthropic") as MockClient:
        mock_client_instance = MagicMock()
        MockClient.return_value = mock_client_instance
        brain = AIBrain(_base_config)
        # Ensure the mock instance is the one assigned
        assert brain.client is mock_client_instance
    return brain


@pytest.fixture
def brain_without_client(_no_key_config: AssistantConfig) -> AIBrain:
    """AIBrain initialised without an API key (client is None)."""
    brain = AIBrain(_no_key_config)
    assert brain.client is None
    return brain


@pytest.fixture
def mock_claude_response():
    """Factory fixture that creates mock anthropic.types.Message objects.

    Usage::

        resp = mock_claude_response(text="Hello", stop_reason="end_turn")
        resp = mock_claude_response(
            tool_calls=[("toolu_01", "run_shell_command", {"command": "ls"})],
        )
    """

    def _factory(
        text: str | None = None,
        stop_reason: str = "end_turn",
        tool_calls: list[tuple[str, str, dict]] | None = None,
    ) -> MagicMock:
        msg = MagicMock()
        blocks = []

        if text is not None:
            blocks.append(_make_text_block(text))

        if tool_calls:
            for tool_id, name, inp in tool_calls:
                blocks.append(_make_tool_use_block(tool_id, name, inp))
            if stop_reason == "end_turn":
                stop_reason = "tool_use"

        msg.content = blocks
        msg.stop_reason = stop_reason
        return msg

    return _factory


@pytest.fixture
def mock_tool_executor():
    """Patch brain.ai_brain.execute_tool and yield the mock."""
    with patch("brain.ai_brain.execute_tool") as mock_exec:
        mock_exec.return_value = "tool result text"
        yield mock_exec
