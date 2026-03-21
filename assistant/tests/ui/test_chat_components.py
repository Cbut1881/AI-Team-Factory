"""Tests for internal chat panel components: _MessageBubble, _ToolLogWidget, _TypingIndicator, _TitleBar."""

from __future__ import annotations

import pytest
from PyQt6.QtCore import Qt

from ui.chat_panel import (
    COL_STATUS_OFFLINE,
    COL_STATUS_ONLINE,
    _MessageBubble,
    _ToolLogWidget,
    _TitleBar,
    _TypingIndicator,
)


# ======================================================================
# _MessageBubble
# ======================================================================


class TestMessageBubble:

    def test_user_alignment_right(self, qtbot):
        bubble = _MessageBubble("You", "hello", "user")
        qtbot.addWidget(bubble)
        # The body label (second widget in layout for user) should be right-aligned
        layout = bubble.layout()
        # item 0 = sender label, item 1 = body, item 2 = timestamp
        body_item = layout.itemAt(1)
        alignment = body_item.alignment()
        assert alignment & Qt.AlignmentFlag.AlignRight

    def test_assistant_alignment_left(self, qtbot):
        bubble = _MessageBubble("Nova", "hi", "assistant")
        qtbot.addWidget(bubble)
        layout = bubble.layout()
        body_item = layout.itemAt(1)
        alignment = body_item.alignment()
        assert alignment & Qt.AlignmentFlag.AlignLeft

    def test_system_alignment_center(self, qtbot):
        bubble = _MessageBubble("System", "connected", "system")
        qtbot.addWidget(bubble)
        layout = bubble.layout()
        # system has no sender label, so body is item 0
        body_item = layout.itemAt(0)
        alignment = body_item.alignment()
        assert alignment & Qt.AlignmentFlag.AlignHCenter

    def test_system_no_sender_label(self, qtbot):
        bubble = _MessageBubble("System", "info", "system")
        qtbot.addWidget(bubble)
        layout = bubble.layout()
        # system: body + timestamp = 2 items
        assert layout.count() == 2

    def test_user_has_sender_label(self, qtbot):
        bubble = _MessageBubble("You", "hello", "user")
        qtbot.addWidget(bubble)
        layout = bubble.layout()
        # user: sender + body + timestamp = 3 items
        assert layout.count() == 3

    def test_unknown_type_fallback(self, qtbot):
        bubble = _MessageBubble("???", "test", "unknown_type")
        qtbot.addWidget(bubble)
        layout = bubble.layout()
        # unknown gets AlignLeft (same as assistant fallback)
        body_item = layout.itemAt(1)
        alignment = body_item.alignment()
        assert alignment & Qt.AlignmentFlag.AlignLeft


# ======================================================================
# _ToolLogWidget
# ======================================================================


class TestToolLogWidget:

    def test_starts_collapsed(self, qtbot):
        widget = _ToolLogWidget("search", "results here")
        qtbot.addWidget(widget)
        assert not widget._expanded
        assert not widget._content.isVisible()

    def test_toggle_expands(self, qtbot):
        widget = _ToolLogWidget("search", "results here")
        qtbot.addWidget(widget)
        widget._header.click()
        assert widget._expanded
        # Use isVisibleTo(parent) because the top-level widget is not shown
        assert widget._content.isVisibleTo(widget)

    def test_toggle_collapses(self, qtbot):
        widget = _ToolLogWidget("search", "results here")
        qtbot.addWidget(widget)
        widget._header.click()  # expand
        widget._header.click()  # collapse
        assert not widget._expanded
        assert not widget._content.isVisibleTo(widget)

    def test_header_arrow_changes(self, qtbot):
        widget = _ToolLogWidget("my_tool", "output")
        qtbot.addWidget(widget)
        assert "\u25b6" in widget._header.text()  # right arrow (collapsed)
        widget._header.click()
        assert "\u25bc" in widget._header.text()  # down arrow (expanded)

    def test_tool_name_stored(self, qtbot):
        widget = _ToolLogWidget("read_file", "/tmp/test.py")
        qtbot.addWidget(widget)
        assert widget._tool_name == "read_file"


# ======================================================================
# _TypingIndicator
# ======================================================================


class TestTypingIndicator:

    def test_initially_hidden(self, qtbot):
        indicator = _TypingIndicator()
        qtbot.addWidget(indicator)
        assert not indicator.isVisible()

    def test_text_content(self, qtbot):
        indicator = _TypingIndicator()
        qtbot.addWidget(indicator)
        assert "Nova" in indicator.text()


# ======================================================================
# _TitleBar
# ======================================================================


class TestTitleBar:

    def test_fixed_height_40(self, qtbot):
        bar = _TitleBar()
        qtbot.addWidget(bar)
        assert bar.height() == 40

    def test_set_online_true(self, qtbot):
        bar = _TitleBar()
        qtbot.addWidget(bar)
        bar.set_online(True)
        style = bar._status_dot.styleSheet()
        assert COL_STATUS_ONLINE in style

    def test_set_online_false(self, qtbot):
        bar = _TitleBar()
        qtbot.addWidget(bar)
        bar.set_online(False)
        style = bar._status_dot.styleSheet()
        assert COL_STATUS_OFFLINE in style
