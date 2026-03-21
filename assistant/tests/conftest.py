"""
Root conftest for the Nova AI Desktop Assistant test suite.

Provides shared fixtures used across all test modules.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

# Ensure the assistant package root is importable via absolute imports.
_ASSISTANT_ROOT = Path(__file__).resolve().parent.parent
if str(_ASSISTANT_ROOT) not in sys.path:
    sys.path.insert(0, str(_ASSISTANT_ROOT))

from config import AssistantConfig

# ---------------------------------------------------------------------------
# Qt application singleton
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def qapp():
    """Return a single QApplication instance for the whole test session.

    If one already exists (e.g. from an IDE runner) it is reused.
    """
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def test_config() -> AssistantConfig:
    """Return an ``AssistantConfig`` with sensible test defaults."""
    return AssistantConfig(anthropic_api_key="test-key-123")


@pytest.fixture()
def tmp_settings_file(tmp_path):
    """Create a temporary ``settings.json`` and return its path.

    Usage::

        def test_something(tmp_settings_file):
            path = tmp_settings_file({"debug": True})
            config = AssistantConfig.load(path)
    """

    def _factory(data: dict) -> Path:
        p = tmp_path / "settings.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        return p

    return _factory


# ---------------------------------------------------------------------------
# Anthropic client / response mocks
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_anthropic_client() -> MagicMock:
    """Return a ``MagicMock`` standing in for ``anthropic.Anthropic``."""
    client = MagicMock(name="Anthropic")
    client.messages = MagicMock(name="messages")
    client.messages.create = MagicMock(name="messages.create")
    return client


@pytest.fixture()
def mock_response_factory():
    """Factory fixture that builds mock ``Message`` objects.

    Parameters
    ----------
    content : list[dict]
        Each dict describes a content block.  Supported shapes:

        * ``{"type": "text", "text": "hello"}``
        * ``{"type": "tool_use", "id": "...", "name": "...", "input": {...}}``

    stop_reason : str
        One of ``"end_turn"``, ``"tool_use"``, ``"max_tokens"``, etc.

    Example::

        msg = mock_response_factory(
            content=[{"type": "text", "text": "Hi!"}],
            stop_reason="end_turn",
        )
    """

    def _build(
        content: Optional[List[Dict[str, Any]]] = None,
        stop_reason: str = "end_turn",
    ) -> MagicMock:
        if content is None:
            content = [{"type": "text", "text": "Hello from mock"}]

        blocks: list[MagicMock] = []
        for block_data in content:
            block = MagicMock()
            block.type = block_data["type"]
            if block.type == "text":
                block.text = block_data.get("text", "")
            elif block.type == "tool_use":
                block.id = block_data.get("id", "tool_call_001")
                block.name = block_data.get("name", "unknown_tool")
                block.input = block_data.get("input", {})
            # Copy all explicit keys as attributes for flexibility.
            for k, v in block_data.items():
                setattr(block, k, v)
            blocks.append(block)

        message = MagicMock(name="Message")
        message.content = blocks
        message.stop_reason = stop_reason
        message.model = "claude-sonnet-4-20250514"
        message.usage.input_tokens = 10
        message.usage.output_tokens = 20
        return message

    return _build


# ---------------------------------------------------------------------------
# Safety: neuter pyautogui during tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def no_pyautogui(monkeypatch):
    """Replace dangerous ``pyautogui`` functions with no-ops.

    This prevents accidental mouse moves / key presses during test runs.
    """
    try:
        import pyautogui
    except ImportError:
        yield
        return

    _noop = lambda *a, **kw: None

    for func_name in (
        "click",
        "doubleClick",
        "rightClick",
        "moveTo",
        "moveRel",
        "dragTo",
        "dragRel",
        "press",
        "hotkey",
        "typewrite",
        "write",
        "scroll",
        "screenshot",
        "keyDown",
        "keyUp",
    ):
        monkeypatch.setattr(pyautogui, func_name, _noop, raising=False)

    yield
