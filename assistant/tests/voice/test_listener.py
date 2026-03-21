"""
Tests for voice.listener — VoiceListener, ListenMode, ListenerStatus.

All speech_recognition and keyboard dependencies are mocked so that tests
never access real microphones or system-wide hotkeys.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock, call

import pytest

_ASSISTANT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ASSISTANT_ROOT) not in sys.path:
    sys.path.insert(0, str(_ASSISTANT_ROOT))

from voice.listener import VoiceListener, ListenMode, ListenerStatus


# ======================================================================
# Construction defaults
# ======================================================================

class TestConstruction:
    """VoiceListener construction and default values."""

    def test_default_mode_is_push_to_talk(self, qapp):
        listener = VoiceListener()
        assert listener.mode == ListenMode.PUSH_TO_TALK

    def test_default_language_is_thai(self, qapp):
        listener = VoiceListener()
        assert listener.language == "th-TH"

    def test_default_status_is_idle(self, qapp):
        listener = VoiceListener()
        assert listener.status == ListenerStatus.IDLE

    def test_custom_mode(self, qapp):
        listener = VoiceListener(mode=ListenMode.CONTINUOUS)
        assert listener.mode == ListenMode.CONTINUOUS

    def test_custom_language(self, qapp):
        listener = VoiceListener(language="en-US")
        assert listener.language == "en-US"

    def test_custom_hotkey(self, qapp):
        listener = VoiceListener(hotkey="ctrl+shift+m")
        assert listener._hotkey == "ctrl+shift+m"


# ======================================================================
# Property get / set
# ======================================================================

class TestProperties:
    """Language and mode property access."""

    def test_language_getter(self, qapp):
        listener = VoiceListener(language="ja-JP")
        assert listener.language == "ja-JP"

    def test_language_setter(self, qapp):
        listener = VoiceListener()
        listener.language = "en-US"
        assert listener.language == "en-US"

    def test_mode_getter(self, qapp):
        listener = VoiceListener(mode=ListenMode.CONTINUOUS)
        assert listener.mode == ListenMode.CONTINUOUS

    def test_mode_setter(self, qapp):
        listener = VoiceListener()
        listener.mode = ListenMode.CONTINUOUS
        assert listener.mode == ListenMode.CONTINUOUS


# ======================================================================
# start_listening / stop_listening
# ======================================================================

class TestStartStop:
    """Starting and stopping the listener thread."""

    def test_start_listening_starts_thread(self, qapp):
        listener = VoiceListener()
        with patch.object(listener, "isRunning", return_value=False), \
             patch.object(listener, "start") as mock_start:
            listener.start_listening()
            mock_start.assert_called_once()

    def test_start_listening_clears_stop_event(self, qapp):
        listener = VoiceListener()
        listener._stop_event.set()
        with patch.object(listener, "isRunning", return_value=False), \
             patch.object(listener, "start"):
            listener.start_listening()
            assert not listener._stop_event.is_set()

    def test_start_listening_noop_when_running(self, qapp):
        listener = VoiceListener()
        with patch.object(listener, "isRunning", return_value=True), \
             patch.object(listener, "start") as mock_start:
            listener.start_listening()
            mock_start.assert_not_called()

    def test_stop_listening_sets_stop_event(self, qapp):
        listener = VoiceListener()
        with patch.object(listener, "quit"), \
             patch.object(listener, "wait"):
            listener.stop_listening()
            assert listener._stop_event.is_set()

    def test_stop_listening_sets_ptt_active(self, qapp):
        """PTT event is set to unblock any wait."""
        listener = VoiceListener()
        with patch.object(listener, "quit"), \
             patch.object(listener, "wait"):
            listener.stop_listening()
            assert listener._ptt_active.is_set()


# ======================================================================
# run() — microphone init / calibration failures
# ======================================================================

class TestRunErrors:
    """run() error paths: mic init and calibration failures."""

    def test_microphone_init_oserror_emits_error(self, qapp):
        listener = VoiceListener()
        signals = []
        listener.error_occurred.connect(lambda msg: signals.append(msg))

        with patch("voice.listener.sr") as mock_sr:
            mock_sr.Microphone = MagicMock(side_effect=OSError("No mic"))
            listener.run()

        assert len(signals) == 1
        assert "Microphone not available" in signals[0]

    def test_microphone_init_attribute_error(self, qapp):
        listener = VoiceListener()
        signals = []
        listener.error_occurred.connect(lambda msg: signals.append(msg))

        with patch("voice.listener.sr") as mock_sr:
            mock_sr.Microphone = MagicMock(side_effect=AttributeError("bad"))
            listener.run()

        assert len(signals) == 1

    def test_calibration_failure_emits_error(self, qapp):
        listener = VoiceListener()
        signals = []
        listener.error_occurred.connect(lambda msg: signals.append(msg))

        mic_instance = MagicMock()
        mic_instance.__enter__ = MagicMock(side_effect=OSError("calibration fail"))
        mic_instance.__exit__ = MagicMock(return_value=False)

        with patch("voice.listener.sr") as mock_sr:
            mock_sr.Microphone = MagicMock(return_value=mic_instance)
            listener._recognizer.adjust_for_ambient_noise = MagicMock()
            listener.run()

        assert len(signals) == 1
        assert "calibration" in signals[0].lower() or "Microphone" in signals[0]


# ======================================================================
# Recognition results
# ======================================================================

class TestRecognition:
    """Text recognition, empty results, and error handling."""

    def _make_listener_with_mic(self, qapp):
        """Helper to create a listener with a working mock mic."""
        listener = VoiceListener()
        mic = MagicMock()
        mic.__enter__ = MagicMock(return_value=mic)
        mic.__exit__ = MagicMock(return_value=False)
        listener._microphone = mic
        return listener

    def test_text_recognized_signal_on_success(self, qapp):
        listener = self._make_listener_with_mic(qapp)
        signals = []
        listener.text_recognized.connect(lambda t: signals.append(t))

        audio_mock = MagicMock()
        listener._recognizer.listen = MagicMock(return_value=audio_mock)
        listener._recognizer.recognize_google = MagicMock(return_value="hello world")

        listener._capture_and_recognise()

        assert signals == ["hello world"]

    def test_empty_recognition_no_signal(self, qapp):
        listener = self._make_listener_with_mic(qapp)
        signals = []
        listener.text_recognized.connect(lambda t: signals.append(t))

        audio_mock = MagicMock()
        listener._recognizer.listen = MagicMock(return_value=audio_mock)
        listener._recognizer.recognize_google = MagicMock(return_value="")

        listener._capture_and_recognise()

        assert signals == []

    def test_whitespace_only_recognition_no_signal(self, qapp):
        listener = self._make_listener_with_mic(qapp)
        signals = []
        listener.text_recognized.connect(lambda t: signals.append(t))

        audio_mock = MagicMock()
        listener._recognizer.listen = MagicMock(return_value=audio_mock)
        listener._recognizer.recognize_google = MagicMock(return_value="   ")

        listener._capture_and_recognise()

        assert signals == []

    def test_unknown_value_error_silent(self, qapp):
        import speech_recognition as sr

        listener = self._make_listener_with_mic(qapp)
        errors = []
        listener.error_occurred.connect(lambda m: errors.append(m))

        audio_mock = MagicMock()
        listener._recognizer.listen = MagicMock(return_value=audio_mock)
        listener._recognizer.recognize_google = MagicMock(
            side_effect=sr.UnknownValueError()
        )

        listener._capture_and_recognise()

        assert errors == []

    def test_request_error_emits_error(self, qapp):
        import speech_recognition as sr

        listener = self._make_listener_with_mic(qapp)
        errors = []
        listener.error_occurred.connect(lambda m: errors.append(m))

        audio_mock = MagicMock()
        listener._recognizer.listen = MagicMock(return_value=audio_mock)
        listener._recognizer.recognize_google = MagicMock(
            side_effect=sr.RequestError("service down")
        )

        listener._capture_and_recognise()

        assert len(errors) == 1
        assert "service" in errors[0].lower() or "error" in errors[0].lower()

    def test_wait_timeout_no_error(self, qapp):
        import speech_recognition as sr

        listener = self._make_listener_with_mic(qapp)
        errors = []
        listener.error_occurred.connect(lambda m: errors.append(m))

        mic = MagicMock()
        mic.__enter__ = MagicMock(side_effect=sr.WaitTimeoutError("timeout"))
        mic.__exit__ = MagicMock(return_value=False)
        listener._microphone = mic

        # WaitTimeoutError is raised by listen(), let's simulate it properly
        listener._microphone = MagicMock()
        listener._microphone.__enter__ = MagicMock(return_value=listener._microphone)
        listener._microphone.__exit__ = MagicMock(return_value=False)
        listener._recognizer.listen = MagicMock(side_effect=sr.WaitTimeoutError())

        listener._capture_and_recognise()

        assert errors == []

    def test_listening_started_signal(self, qapp):
        listener = self._make_listener_with_mic(qapp)
        started = []
        listener.listening_started.connect(lambda: started.append(True))

        audio_mock = MagicMock()
        listener._recognizer.listen = MagicMock(return_value=audio_mock)
        listener._recognizer.recognize_google = MagicMock(return_value="hi")

        listener._capture_and_recognise()

        assert len(started) == 1

    def test_listening_stopped_signal(self, qapp):
        listener = self._make_listener_with_mic(qapp)
        stopped = []
        listener.listening_stopped.connect(lambda: stopped.append(True))

        audio_mock = MagicMock()
        listener._recognizer.listen = MagicMock(return_value=audio_mock)
        listener._recognizer.recognize_google = MagicMock(return_value="hi")

        listener._capture_and_recognise()

        # listening_stopped is emitted after listen returns (processing phase)
        assert len(stopped) >= 1


# ======================================================================
# Push-to-talk mode
# ======================================================================

class TestPushToTalk:
    """Push-to-talk hotkey registration and workflow."""

    def test_register_hotkey_success(self, qapp):
        listener = VoiceListener()
        mock_kb = MagicMock()
        mock_kb.add_hotkey = MagicMock(return_value="hook_ref")

        with patch.dict("sys.modules", {"keyboard": mock_kb}):
            listener._register_hotkey()

        assert listener._hotkey_hook is not None

    def test_register_hotkey_import_failure(self, qapp):
        listener = VoiceListener()
        errors = []
        listener.error_occurred.connect(lambda m: errors.append(m))

        with patch("builtins.__import__", side_effect=ImportError("no keyboard")):
            listener._register_hotkey()

        assert len(errors) == 1
        assert "keyboard" in errors[0].lower()

    def test_unregister_hotkey_clears_hook(self, qapp):
        listener = VoiceListener()
        listener._hotkey_hook = "some_hook"

        mock_kb = MagicMock()
        with patch.dict("sys.modules", {"keyboard": mock_kb}):
            listener._unregister_hotkey()

        assert listener._hotkey_hook is None

    def test_unregister_hotkey_noop_when_none(self, qapp):
        listener = VoiceListener()
        listener._hotkey_hook = None
        # Should not raise
        listener._unregister_hotkey()
        assert listener._hotkey_hook is None

    def test_ptt_waits_for_hotkey(self, qapp):
        """Push-to-talk loop blocks until _ptt_active is set."""
        listener = VoiceListener()
        listener._microphone = MagicMock()

        with patch.object(listener, "_register_hotkey"), \
             patch.object(listener, "_capture_and_recognise") as mock_capture:
            # Stop after one iteration
            call_count = 0

            def stop_after_capture():
                nonlocal call_count
                call_count += 1
                listener._stop_event.set()

            mock_capture.side_effect = stop_after_capture

            # Simulate hotkey press after a short delay
            def press_hotkey():
                time.sleep(0.05)
                listener._ptt_active.set()

            t = threading.Thread(target=press_hotkey)
            t.start()
            listener._run_push_to_talk()
            t.join()

            assert call_count == 1


# ======================================================================
# Continuous mode
# ======================================================================

class TestContinuousMode:
    """Continuous listening loop."""

    def test_continuous_loops_until_stop(self, qapp):
        listener = VoiceListener(mode=ListenMode.CONTINUOUS)
        listener._microphone = MagicMock()

        iterations = 0

        def capture_side_effect():
            nonlocal iterations
            iterations += 1
            if iterations >= 3:
                listener._stop_event.set()

        with patch.object(listener, "_capture_and_recognise", side_effect=capture_side_effect):
            listener._run_continuous()

        assert iterations == 3


# ======================================================================
# Status transitions
# ======================================================================

class TestStatusTransitions:
    """Listener status changes during capture and recognition."""

    def test_status_becomes_listening_during_capture(self, qapp):
        listener = VoiceListener()
        mic = MagicMock()
        mic.__enter__ = MagicMock(return_value=mic)
        mic.__exit__ = MagicMock(return_value=False)
        listener._microphone = mic

        statuses = []

        original_listen = listener._recognizer.listen

        def capture_status(*args, **kwargs):
            statuses.append(listener.status)
            return MagicMock()

        listener._recognizer.listen = capture_status
        listener._recognizer.recognize_google = MagicMock(return_value="hi")

        listener._capture_and_recognise()

        assert ListenerStatus.LISTENING in statuses

    def test_status_returns_to_idle_after_capture(self, qapp):
        listener = VoiceListener()
        mic = MagicMock()
        mic.__enter__ = MagicMock(return_value=mic)
        mic.__exit__ = MagicMock(return_value=False)
        listener._microphone = mic

        listener._recognizer.listen = MagicMock(return_value=MagicMock())
        listener._recognizer.recognize_google = MagicMock(return_value="hi")

        listener._capture_and_recognise()

        assert listener.status == ListenerStatus.IDLE

    def test_capture_with_none_microphone_returns_early(self, qapp):
        listener = VoiceListener()
        listener._microphone = None

        # Should return without error
        listener._capture_and_recognise()
        assert listener.status == ListenerStatus.IDLE
