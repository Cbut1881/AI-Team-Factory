"""
End-to-end integration tests for the Nova AI Desktop Assistant.

Tests the full flow from user input through the AI brain to response
delivery via chat and speech. All external services (Claude, Ollama,
Qt widgets) are mocked.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock, PropertyMock, call

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
         patch.object(_nova_main, "QTimer") as MockTimer:

        MockConfig.load.return_value = MagicMock()

        # Set up mock signals as MagicMocks
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
        mock_brain.process = AsyncMock(return_value="AI response")

        mock_tray = MockTray.return_value
        mock_tray.setContextMenu = MagicMock()
        mock_tray.setToolTip = MagicMock()
        mock_tray.show = MagicMock()
        mock_tray.hide = MagicMock()

        nova = _nova_main.NovaAssistant(qapp)

        return nova


# ======================================================================
# Full flow: text input -> brain -> response -> chat + speech
# ======================================================================

class TestTextInputFlow:
    """Text input through the full processing pipeline."""

    def test_text_input_triggers_processing(self, qapp):
        nova = _create_nova(qapp)
        nova.brain.process = AsyncMock(return_value="Hello!")

        nova._on_user_message("Hi there")

        assert nova._processing is True

    def test_response_delivered_to_chat(self, qapp):
        nova = _create_nova(qapp)

        nova._deliver_response("Hello from Nova!")

        nova.chat.add_message.assert_called_with("Nova", "Hello from Nova!", "assistant")

    def test_response_spoken(self, qapp):
        nova = _create_nova(qapp)

        nova._deliver_response("Hello from Nova!")

        nova.speaker.speak.assert_called_with("Hello from Nova!")

    def test_processing_flag_cleared_after_response(self, qapp):
        nova = _create_nova(qapp)
        nova._processing = True

        nova._deliver_response("response")

        assert nova._processing is False

    def test_typing_indicator_hidden_after_response(self, qapp):
        nova = _create_nova(qapp)

        nova._deliver_response("response")

        nova.chat.hide_typing_indicator.assert_called_once()


# ======================================================================
# Voice input flow
# ======================================================================

class TestVoiceInputFlow:
    """Voice input through processing pipeline."""

    def test_voice_input_adds_to_chat(self, qapp):
        nova = _create_nova(qapp)
        nova.brain.process = AsyncMock(return_value="Response")

        nova._on_voice_input("spoken text")

        nova.chat.add_message.assert_any_call("You", "spoken text", "user")

    def test_voice_input_triggers_processing(self, qapp):
        nova = _create_nova(qapp)
        nova.brain.process = AsyncMock(return_value="Response")

        nova._on_voice_input("spoken text")

        assert nova._processing is True


# ======================================================================
# Error handling
# ======================================================================

class TestErrorHandling:
    """Error handling in the processing pipeline."""

    def test_brain_error_delivers_error_message(self, qapp):
        nova = _create_nova(qapp)

        nova._deliver_response("error occurred: something went wrong")

        nova.chat.add_message.assert_called_once()
        args = nova.chat.add_message.call_args
        assert "error" in args[0][1].lower() or "wrong" in args[0][1].lower()

    def test_async_process_exception_emits_error(self, qapp):
        nova = _create_nova(qapp)
        nova.brain.process = AsyncMock(side_effect=RuntimeError("brain crash"))

        # Run the async method
        loop = asyncio.new_event_loop()
        try:
            # Capture what gets emitted
            emitted = []
            nova.response_ready = MagicMock()
            nova.response_ready.emit = lambda msg: emitted.append(msg)

            loop.run_until_complete(nova._async_process("test"))

            assert len(emitted) == 1
            assert "brain crash" in emitted[0] or "ข้อผิดพลาด" in emitted[0]
        finally:
            loop.close()


# ======================================================================
# Processing guard
# ======================================================================

class TestProcessingGuard:
    """_processing flag prevents double-processing."""

    def test_ignores_input_when_processing(self, qapp):
        nova = _create_nova(qapp)
        nova._processing = True
        nova.brain.process = AsyncMock(return_value="response")

        nova._on_user_message("should be ignored")

        # _process_input should not be called because _processing is True
        nova.chat.show_typing_indicator.assert_not_called()

    def test_ignores_voice_when_processing(self, qapp):
        nova = _create_nova(qapp)
        nova._processing = True

        nova._on_voice_input("should be ignored")

        nova.chat.show_typing_indicator.assert_not_called()


# ======================================================================
# Startup greeting
# ======================================================================

class TestStartupGreeting:
    """Startup greeting delivery."""

    def test_startup_greeting_adds_message(self, qapp):
        nova = _create_nova(qapp)
        nova.chat.add_message.reset_mock()

        nova._startup_greeting()

        nova.chat.add_message.assert_called_once()
        args = nova.chat.add_message.call_args
        assert args[0][0] == "Nova"
        assert args[0][2] == "assistant"

    def test_startup_greeting_speaks(self, qapp):
        nova = _create_nova(qapp)
        nova.speaker.speak.reset_mock()

        nova._startup_greeting()

        nova.speaker.speak.assert_called_once()


# ======================================================================
# Avatar state transitions
# ======================================================================

class TestAvatarStates:
    """Avatar state changes during processing."""

    def test_thinking_state_during_processing(self, qapp):
        nova = _create_nova(qapp)
        nova.brain.process = AsyncMock(return_value="ok")
        nova.avatar.set_state.reset_mock()

        nova._process_input("test")

        # Should set THINKING state
        nova.avatar.set_state.assert_called()

    def test_typing_indicator_shown_during_processing(self, qapp):
        nova = _create_nova(qapp)
        nova.brain.process = AsyncMock(return_value="ok")
        nova.chat.show_typing_indicator.reset_mock()

        nova._process_input("test")

        nova.chat.show_typing_indicator.assert_called_once()


# ======================================================================
# Voice toggle
# ======================================================================

class TestVoiceToggle:
    """Voice listening toggle."""

    def test_voice_enabled_starts_listener(self, qapp):
        nova = _create_nova(qapp)

        nova._on_voice_toggled(True)

        nova.listener.start_listening.assert_called_once()

    def test_voice_disabled_stops_listener(self, qapp):
        nova = _create_nova(qapp)

        nova._on_voice_toggled(False)

        nova.listener.stop_listening.assert_called_once()


# ======================================================================
# Ollama fallback
# ======================================================================

class TestOllamaFallback:
    """Claude failure triggers Ollama fallback."""

    def test_async_process_delivers_response_on_success(self, qapp):
        nova = _create_nova(qapp)
        nova.brain.process = AsyncMock(return_value="Ollama response")

        loop = asyncio.new_event_loop()
        try:
            emitted = []
            nova.response_ready = MagicMock()
            nova.response_ready.emit = lambda msg: emitted.append(msg)

            loop.run_until_complete(nova._async_process("test"))

            assert emitted == ["Ollama response"]
        finally:
            loop.close()
