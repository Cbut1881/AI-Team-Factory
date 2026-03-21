"""
Tests for thread safety in the Nova AI Desktop Assistant.

Validates that async event loops, thread-safe signal delivery,
processing guards, and clean shutdown work correctly.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

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

def _create_nova(qapp):
    """Create a NovaAssistant with all dependencies mocked."""
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
        mock_brain.process = AsyncMock(return_value="response")

        mock_tray = MockTray.return_value
        mock_tray.setContextMenu = MagicMock()
        mock_tray.setToolTip = MagicMock()
        mock_tray.show = MagicMock()
        mock_tray.hide = MagicMock()

        nova = _nova_main.NovaAssistant(qapp)

        return nova


# ======================================================================
# Async event loop in background thread
# ======================================================================

class TestAsyncEventLoop:
    """Background asyncio event loop setup."""

    def test_event_loop_created(self, qapp):
        nova = _create_nova(qapp)
        assert nova.loop is not None
        assert isinstance(nova.loop, asyncio.AbstractEventLoop)

    def test_loop_thread_is_daemon(self, qapp):
        nova = _create_nova(qapp)
        assert nova._loop_thread.daemon is True

    def test_loop_thread_started(self, qapp):
        nova = _create_nova(qapp)
        assert nova._loop_thread.is_alive()

    def test_loop_is_running(self, qapp):
        nova = _create_nova(qapp)
        # Give the loop a moment to start
        time.sleep(0.1)
        assert nova.loop.is_running()


# ======================================================================
# _process_input dispatches to async loop
# ======================================================================

class TestProcessInputDispatch:
    """_process_input dispatches work to the background async loop."""

    def test_process_input_sets_processing_flag(self, qapp):
        nova = _create_nova(qapp)

        nova._process_input("test")

        assert nova._processing is True

    def test_process_input_sets_thinking_state(self, qapp):
        nova = _create_nova(qapp)
        nova.avatar.set_state.reset_mock()

        nova._process_input("test")

        nova.avatar.set_state.assert_called()

    def test_process_input_shows_typing_indicator(self, qapp):
        nova = _create_nova(qapp)
        nova.chat.show_typing_indicator.reset_mock()

        nova._process_input("test")

        nova.chat.show_typing_indicator.assert_called_once()

    def test_process_input_dispatches_coroutine(self, qapp):
        nova = _create_nova(qapp)

        with patch("asyncio.run_coroutine_threadsafe") as mock_dispatch:
            nova._process_input("test")

            mock_dispatch.assert_called_once()
            # Second arg should be the event loop
            assert mock_dispatch.call_args[0][1] == nova.loop


# ======================================================================
# response_ready crosses thread boundary
# ======================================================================

class TestResponseReady:
    """response_ready signal delivers responses to the main thread."""

    def test_deliver_response_updates_chat(self, qapp):
        nova = _create_nova(qapp)

        nova._deliver_response("hello from bg thread")

        nova.chat.add_message.assert_called_with(
            "Nova", "hello from bg thread", "assistant"
        )

    def test_deliver_response_speaks(self, qapp):
        nova = _create_nova(qapp)

        nova._deliver_response("hello")

        nova.speaker.speak.assert_called_with("hello")

    def test_deliver_response_clears_processing(self, qapp):
        nova = _create_nova(qapp)
        nova._processing = True

        nova._deliver_response("done")

        assert nova._processing is False

    def test_deliver_response_hides_typing(self, qapp):
        nova = _create_nova(qapp)

        nova._deliver_response("done")

        nova.chat.hide_typing_indicator.assert_called_once()


# ======================================================================
# _processing flag prevents double-processing
# ======================================================================

class TestProcessingGuard:
    """Double-processing prevention."""

    def test_user_message_blocked_when_processing(self, qapp):
        nova = _create_nova(qapp)
        nova._processing = True

        nova._on_user_message("blocked")

        nova.chat.show_typing_indicator.assert_not_called()

    def test_voice_input_blocked_when_processing(self, qapp):
        nova = _create_nova(qapp)
        nova._processing = True

        nova._on_voice_input("blocked")

        nova.chat.show_typing_indicator.assert_not_called()

    def test_sequential_processing_allowed(self, qapp):
        nova = _create_nova(qapp)

        # First message
        nova._on_user_message("first")
        assert nova._processing is True

        # Simulate response delivery
        nova._deliver_response("response")
        assert nova._processing is False

        # Second message should now be allowed
        nova.chat.show_typing_indicator.reset_mock()
        nova._on_user_message("second")
        nova.chat.show_typing_indicator.assert_called_once()


# ======================================================================
# _quit() clean shutdown
# ======================================================================

class TestCleanShutdown:
    """_quit() performs orderly teardown."""

    def test_quit_stops_listener(self, qapp):
        nova = _create_nova(qapp)

        nova._quit()

        nova.listener.stop_listening.assert_called_once()

    def test_quit_shuts_down_speaker(self, qapp):
        nova = _create_nova(qapp)

        nova._quit()

        nova.speaker.shutdown.assert_called_once()

    def test_quit_stops_event_loop(self, qapp):
        nova = _create_nova(qapp)

        with patch.object(nova.loop, "call_soon_threadsafe") as mock_call:
            nova._quit()
            mock_call.assert_called_once()

    def test_quit_hides_tray(self, qapp):
        nova = _create_nova(qapp)

        nova._quit()

        nova.tray.hide.assert_called_once()

    def test_quit_calls_app_quit(self, qapp):
        nova = _create_nova(qapp)
        nova.app = MagicMock()

        nova._quit()

        nova.app.quit.assert_called_once()
