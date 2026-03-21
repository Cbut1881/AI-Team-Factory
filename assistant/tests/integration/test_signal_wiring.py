"""
Tests for signal/slot wiring in the Nova AI Desktop Assistant.

Verifies that all Qt signal connections are established correctly
during NovaAssistant initialization.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

_ASSISTANT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ASSISTANT_ROOT) not in sys.path:
    sys.path.insert(0, str(_ASSISTANT_ROOT))

# Import the assistant's main module (not the root project's main.py)
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "assistant_main",
    Path(__file__).resolve().parent.parent.parent / "main.py",
)
_nova_main = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_nova_main)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_nova_with_signal_tracking(qapp):
    """Create a NovaAssistant and track all signal.connect calls."""
    with patch.object(_nova_main, "AssistantConfig") as MockConfig, \
         patch.object(_nova_main, "AvatarWidget") as MockAvatar, \
         patch.object(_nova_main, "ChatPanel") as MockChat, \
         patch.object(_nova_main, "AIBrain") as MockBrain, \
         patch.object(_nova_main, "VoiceListener") as MockListener, \
         patch.object(_nova_main, "VoiceSpeaker") as MockSpeaker, \
         patch.object(_nova_main, "QSystemTrayIcon") as MockTray, \
         patch.object(_nova_main, "QMenu"), \
         patch.object(_nova_main, "QAction"), \
         patch.object(_nova_main, "QTimer"):

        MockConfig.load.return_value = MagicMock()

        mock_avatar = MockAvatar.return_value
        mock_avatar.clicked_signal = MagicMock()
        mock_avatar.double_clicked_signal = MagicMock()
        mock_avatar.show = MagicMock()
        mock_avatar.hide = MagicMock()
        mock_avatar.isVisible = MagicMock(return_value=True)
        mock_avatar.set_state = MagicMock()
        mock_avatar.reset_position = MagicMock()

        mock_chat = MockChat.return_value
        mock_chat.message_sent = MagicMock()
        mock_chat.voice_toggled = MagicMock()
        mock_chat.add_message = MagicMock()
        mock_chat.show_typing_indicator = MagicMock()
        mock_chat.hide_typing_indicator = MagicMock()
        mock_chat.toggle_visibility = MagicMock()

        mock_listener = MockListener.return_value
        mock_listener.text_recognized = MagicMock()
        mock_listener.listening_started = MagicMock()
        mock_listener.listening_stopped = MagicMock()
        mock_listener.isRunning = MagicMock(return_value=False)
        mock_listener.start_listening = MagicMock()
        mock_listener.stop_listening = MagicMock()

        mock_speaker = MockSpeaker.return_value
        mock_speaker.speaking_started = MagicMock()
        mock_speaker.speaking_finished = MagicMock()
        mock_speaker.speak = MagicMock()
        mock_speaker.shutdown = MagicMock()

        mock_brain = MockBrain.return_value

        mock_tray = MockTray.return_value
        mock_tray.setContextMenu = MagicMock()
        mock_tray.setToolTip = MagicMock()
        mock_tray.show = MagicMock()
        mock_tray.hide = MagicMock()

        nova = _nova_main.NovaAssistant(qapp)

        return nova


# ======================================================================
# Signal connections
# ======================================================================

class TestSignalConnections:
    """All signal/slot connections are established."""

    def test_avatar_clicked_connected(self, qapp):
        nova = _create_nova_with_signal_tracking(qapp)
        nova.avatar.clicked_signal.connect.assert_called()

    def test_avatar_double_clicked_connected(self, qapp):
        nova = _create_nova_with_signal_tracking(qapp)
        nova.avatar.double_clicked_signal.connect.assert_called()

    def test_chat_message_sent_connected(self, qapp):
        nova = _create_nova_with_signal_tracking(qapp)
        nova.chat.message_sent.connect.assert_called()

    def test_chat_voice_toggled_connected(self, qapp):
        nova = _create_nova_with_signal_tracking(qapp)
        nova.chat.voice_toggled.connect.assert_called()

    def test_listener_text_recognized_connected(self, qapp):
        nova = _create_nova_with_signal_tracking(qapp)
        nova.listener.text_recognized.connect.assert_called()

    def test_listener_listening_started_connected(self, qapp):
        nova = _create_nova_with_signal_tracking(qapp)
        nova.listener.listening_started.connect.assert_called()

    def test_listener_listening_stopped_connected(self, qapp):
        nova = _create_nova_with_signal_tracking(qapp)
        nova.listener.listening_stopped.connect.assert_called()

    def test_speaker_speaking_started_connected(self, qapp):
        nova = _create_nova_with_signal_tracking(qapp)
        nova.speaker.speaking_started.connect.assert_called()

    def test_speaker_speaking_finished_connected(self, qapp):
        nova = _create_nova_with_signal_tracking(qapp)
        nova.speaker.speaking_finished.connect.assert_called()


# ======================================================================
# Avatar click handler
# ======================================================================

class TestAvatarClickHandler:
    """_on_avatar_clicked toggles voice listener."""

    def test_click_starts_listener_when_not_running(self, qapp):
        nova = _create_nova_with_signal_tracking(qapp)
        nova.listener.isRunning.return_value = False

        nova._on_avatar_clicked()

        nova.listener.start_listening.assert_called_once()

    def test_click_stops_listener_when_running(self, qapp):
        nova = _create_nova_with_signal_tracking(qapp)
        nova.listener.isRunning.return_value = True

        nova._on_avatar_clicked()

        nova.listener.stop_listening.assert_called_once()


# ======================================================================
# Avatar double-click handler
# ======================================================================

class TestAvatarDoubleClickHandler:
    """_on_avatar_double_clicked toggles chat panel."""

    def test_double_click_toggles_chat(self, qapp):
        nova = _create_nova_with_signal_tracking(qapp)

        nova._on_avatar_double_clicked()

        nova.chat.toggle_visibility.assert_called_once()


# ======================================================================
# Tray menu wiring
# ======================================================================

class TestTrayMenu:
    """System tray menu actions."""

    def test_tray_icon_created(self, qapp):
        nova = _create_nova_with_signal_tracking(qapp)
        assert nova.tray is not None

    def test_tray_has_context_menu(self, qapp):
        nova = _create_nova_with_signal_tracking(qapp)
        nova.tray.setContextMenu.assert_called_once()

    def test_tray_tooltip_set(self, qapp):
        nova = _create_nova_with_signal_tracking(qapp)
        nova.tray.setToolTip.assert_called_once_with("Nova — AI Desktop Assistant")

    def test_tray_shown(self, qapp):
        nova = _create_nova_with_signal_tracking(qapp)
        nova.tray.show.assert_called_once()
