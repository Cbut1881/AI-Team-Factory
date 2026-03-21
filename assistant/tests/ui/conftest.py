"""Shared fixtures for UI tests."""

from __future__ import annotations

import pytest

from ui.avatar_widget import AvatarWidget
from ui.chat_panel import ChatPanel


@pytest.fixture()
def avatar(qtbot):
    """Create an AvatarWidget and register it with qtbot."""
    widget = AvatarWidget()
    qtbot.addWidget(widget)
    # Stop the animation timer to avoid interference during tests
    widget._timer.stop()
    return widget


@pytest.fixture()
def chat_panel(qtbot):
    """Create a ChatPanel and register it with qtbot."""
    panel = ChatPanel()
    qtbot.addWidget(panel)
    return panel
