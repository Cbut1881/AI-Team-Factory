"""Tests for ui.chat_panel — ChatPanel."""

from __future__ import annotations

import pytest
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QApplication

from ui.chat_panel import (
    PANEL_WIDTH,
    ChatPanel,
    _format_markdown_lite,
    _MessageBubble,
    _ToolLogWidget,
)


# ======================================================================
# Construction
# ======================================================================


class TestConstruction:

    def test_window_flags_frameless(self, chat_panel):
        flags = chat_panel.windowFlags()
        assert flags & Qt.WindowType.FramelessWindowHint

    def test_window_flags_stays_on_top(self, chat_panel):
        flags = chat_panel.windowFlags()
        assert flags & Qt.WindowType.WindowStaysOnTopHint

    def test_window_flags_tool(self, chat_panel):
        flags = chat_panel.windowFlags()
        assert flags & Qt.WindowType.Tool

    def test_fixed_width(self, chat_panel):
        assert chat_panel.width() == PANEL_WIDTH

    def test_window_title(self, chat_panel):
        assert chat_panel.windowTitle() == "Nova Assistant"


# ======================================================================
# add_message
# ======================================================================


class TestAddMessage:

    def test_add_user_message(self, chat_panel):
        chat_panel.add_message("You", "hello", "user")
        # Check that a widget was added to the messages layout
        count = chat_panel._messages_layout.count()
        # Layout has: stretch (1) + new bubble (1) = 2
        assert count >= 2

    def test_add_assistant_message(self, chat_panel):
        chat_panel.add_message("Nova", "hi", "assistant")
        count = chat_panel._messages_layout.count()
        assert count >= 2

    def test_add_system_message(self, chat_panel):
        chat_panel.add_message("System", "connected", "system")
        count = chat_panel._messages_layout.count()
        assert count >= 2

    def test_multiple_messages(self, chat_panel):
        initial = chat_panel._messages_layout.count()
        chat_panel.add_message("You", "msg1", "user")
        chat_panel.add_message("Nova", "msg2", "assistant")
        chat_panel.add_message("System", "msg3", "system")
        assert chat_panel._messages_layout.count() == initial + 3

    def test_add_message_creates_bubble(self, chat_panel):
        chat_panel.add_message("You", "test", "user")
        count = chat_panel._messages_layout.count()
        # The item before the stretch should be a _MessageBubble
        item = chat_panel._messages_layout.itemAt(count - 2)
        assert isinstance(item.widget(), _MessageBubble)


# ======================================================================
# add_tool_log
# ======================================================================


class TestAddToolLog:

    def test_add_tool_log_creates_widget(self, chat_panel):
        chat_panel.add_tool_log("search_files", "Found 3 matches")
        count = chat_panel._messages_layout.count()
        item = chat_panel._messages_layout.itemAt(count - 2)
        assert isinstance(item.widget(), _ToolLogWidget)

    def test_add_tool_log_content(self, chat_panel):
        chat_panel.add_tool_log("run_code", "OK")
        count = chat_panel._messages_layout.count()
        widget = chat_panel._messages_layout.itemAt(count - 2).widget()
        assert isinstance(widget, _ToolLogWidget)
        assert widget._tool_name == "run_code"


# ======================================================================
# message_sent signal
# ======================================================================


class TestMessageSent:

    def test_send_button_emits_signal(self, chat_panel, qtbot):
        chat_panel._input_box.setPlainText("hello world")
        with qtbot.waitSignal(chat_panel.message_sent, timeout=1000) as sig:
            chat_panel._send_btn.click()
        assert sig.args == ["hello world"]

    def test_send_clears_input(self, chat_panel, qtbot):
        chat_panel._input_box.setPlainText("test message")
        chat_panel._send_btn.click()
        assert chat_panel._input_box.toPlainText() == ""

    def test_empty_send_does_nothing(self, chat_panel, qtbot):
        chat_panel._input_box.setPlainText("")
        initial_count = chat_panel._messages_layout.count()
        chat_panel._send_btn.click()
        # No message added, count unchanged
        assert chat_panel._messages_layout.count() == initial_count

    def test_whitespace_only_send_does_nothing(self, chat_panel, qtbot):
        chat_panel._input_box.setPlainText("   \n  ")
        initial_count = chat_panel._messages_layout.count()
        chat_panel._send_btn.click()
        assert chat_panel._messages_layout.count() == initial_count

    def test_enter_key_sends(self, chat_panel, qtbot):
        chat_panel._input_box.setPlainText("enter test")
        with qtbot.waitSignal(chat_panel.message_sent, timeout=1000):
            key_event = QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_Return,
                Qt.KeyboardModifier.NoModifier,
            )
            chat_panel.eventFilter(chat_panel._input_box, key_event)

    def test_shift_enter_does_not_send(self, chat_panel, qtbot):
        chat_panel._input_box.setPlainText("no send")
        key_event = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_Return,
            Qt.KeyboardModifier.ShiftModifier,
        )
        result = chat_panel.eventFilter(chat_panel._input_box, key_event)
        # Should not be consumed (returns False), text stays
        assert result is False
        assert chat_panel._input_box.toPlainText() == "no send"


# ======================================================================
# voice_toggled signal
# ======================================================================


class TestVoiceToggled:

    def test_voice_toggle_emits_signal(self, chat_panel, qtbot):
        with qtbot.waitSignal(chat_panel.voice_toggled, timeout=1000) as sig:
            chat_panel._voice_btn.click()
        assert sig.args == [True]

    def test_voice_toggle_off(self, chat_panel, qtbot):
        chat_panel._voice_btn.click()  # on
        with qtbot.waitSignal(chat_panel.voice_toggled, timeout=1000) as sig:
            chat_panel._voice_btn.click()  # off
        assert sig.args == [False]


# ======================================================================
# language_changed signal
# ======================================================================


class TestLanguageChanged:

    def test_language_toggle_to_en(self, chat_panel, qtbot):
        # Default is th-TH
        with qtbot.waitSignal(chat_panel.language_changed, timeout=1000) as sig:
            chat_panel._lang_btn.click()
        assert sig.args == ["en-US"]

    def test_language_toggle_back_to_th(self, chat_panel, qtbot):
        chat_panel._lang_btn.click()  # -> en-US
        with qtbot.waitSignal(chat_panel.language_changed, timeout=1000) as sig:
            chat_panel._lang_btn.click()  # -> th-TH
        assert sig.args == ["th-TH"]

    def test_language_button_text_changes(self, chat_panel):
        assert chat_panel._lang_btn.text() == "TH"
        chat_panel._lang_btn.click()
        assert chat_panel._lang_btn.text() == "EN"
        chat_panel._lang_btn.click()
        assert chat_panel._lang_btn.text() == "TH"


# ======================================================================
# Typing indicator
# ======================================================================


class TestTypingIndicator:

    def test_show_typing_indicator(self, chat_panel):
        chat_panel.show_typing_indicator()
        # Use isVisibleTo because the parent window may not be shown
        assert chat_panel._typing_indicator.isVisibleTo(chat_panel)

    def test_hide_typing_indicator(self, chat_panel):
        chat_panel.show_typing_indicator()
        chat_panel.hide_typing_indicator()
        assert not chat_panel._typing_indicator.isVisible()


# ======================================================================
# Status bar
# ======================================================================


class TestStatusBar:

    def test_set_status(self, chat_panel):
        chat_panel.set_status("Thinking...")
        assert chat_panel._status_bar.text() == "Thinking..."

    def test_default_status(self, chat_panel):
        assert chat_panel._status_bar.text() == "Ready"


# ======================================================================
# toggle_visibility
# ======================================================================


class TestToggleVisibility:

    def test_toggle_show(self, chat_panel):
        chat_panel.hide()
        chat_panel.toggle_visibility()
        assert chat_panel.isVisible()

    def test_toggle_hide(self, chat_panel):
        chat_panel.show()
        chat_panel.toggle_visibility()
        assert not chat_panel.isVisible()


# ======================================================================
# Connection status
# ======================================================================


class TestConnectionStatus:

    def test_set_connection_online(self, chat_panel):
        chat_panel.set_connection_status(True)
        style = chat_panel._title_bar._status_dot.styleSheet()
        assert "#3fb950" in style  # COL_STATUS_ONLINE

    def test_set_connection_offline(self, chat_panel):
        chat_panel.set_connection_status(False)
        style = chat_panel._title_bar._status_dot.styleSheet()
        assert "#f85149" in style  # COL_STATUS_OFFLINE


# ======================================================================
# Title bar actions
# ======================================================================


class TestTitleBar:

    def test_close_hides_panel(self, chat_panel):
        chat_panel.show()
        chat_panel._title_bar.close_requested.emit()
        assert not chat_panel.isVisible()

    def test_minimize(self, chat_panel):
        chat_panel.show()
        # Just verify the signal is connected (minimize behaviour is OS-dependent)
        chat_panel._title_bar.minimize_requested.emit()


# ======================================================================
# _format_markdown_lite
# ======================================================================


class TestFormatMarkdownLite:

    def test_bold(self):
        result = _format_markdown_lite("**hello**")
        assert "<b>hello</b>" in result

    def test_italic(self):
        result = _format_markdown_lite("*world*")
        assert "<i>world</i>" in result

    def test_inline_code(self):
        result = _format_markdown_lite("`print()`")
        assert "<code" in result
        assert "print()" in result

    def test_code_block(self):
        result = _format_markdown_lite("```\ndef foo():\n    pass\n```")
        assert "<pre" in result
        assert "def foo():" in result

    def test_html_escaping(self):
        result = _format_markdown_lite("<script>alert(1)</script>")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_newlines(self):
        result = _format_markdown_lite("line1\nline2")
        assert "<br>" in result

    def test_plain_text_passthrough(self):
        result = _format_markdown_lite("just plain text")
        assert "just plain text" in result
