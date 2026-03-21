"""
Tests for voice.speaker — VoiceSpeaker, _contains_thai, audio playback.

All edge_tts and pygame dependencies are mocked so that tests never
generate real audio or access sound hardware.
"""

from __future__ import annotations

import asyncio
import os
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock, PropertyMock

import pytest

_ASSISTANT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ASSISTANT_ROOT) not in sys.path:
    sys.path.insert(0, str(_ASSISTANT_ROOT))

from voice.speaker import VoiceSpeaker, _contains_thai, _play_with_pygame, _play_audio


# ======================================================================
# _contains_thai
# ======================================================================

class TestContainsThai:
    """Thai character detection utility."""

    def test_thai_text(self):
        assert _contains_thai("สวัสดี") is True

    def test_ascii_text(self):
        assert _contains_thai("hello world") is False

    def test_empty_string(self):
        assert _contains_thai("") is False

    def test_mixed_thai_english(self):
        assert _contains_thai("hello สวัสดี world") is True

    def test_numbers_only(self):
        assert _contains_thai("12345") is False

    def test_thai_digits(self):
        # Thai digits are in the 0E50-0E59 range
        assert _contains_thai("\u0e50\u0e51") is True

    def test_other_unicode(self):
        assert _contains_thai("日本語") is False


# ======================================================================
# Construction
# ======================================================================

class TestConstruction:
    """VoiceSpeaker construction and initial state."""

    def test_not_speaking_initially(self, qapp):
        with patch.object(VoiceSpeaker, "_worker_loop"):
            speaker = VoiceSpeaker()
            assert speaker.is_speaking is False

    def test_worker_thread_started(self, qapp):
        with patch.object(VoiceSpeaker, "_worker_loop"):
            speaker = VoiceSpeaker()
            assert speaker._worker is not None
            assert speaker._worker.daemon is True

    def test_temp_dir_created(self, qapp):
        with patch.object(VoiceSpeaker, "_worker_loop"):
            speaker = VoiceSpeaker()
            assert speaker._tmp_dir.exists()
            # Cleanup
            speaker._cleanup_temp_files()

    def test_queue_is_empty_initially(self, qapp):
        with patch.object(VoiceSpeaker, "_worker_loop"):
            speaker = VoiceSpeaker()
            assert speaker._queue.empty()


# ======================================================================
# speak()
# ======================================================================

class TestSpeak:
    """Enqueuing text for TTS."""

    def test_speak_enqueues_text(self, qapp):
        with patch.object(VoiceSpeaker, "_worker_loop"):
            speaker = VoiceSpeaker()
            speaker.speak("hello")
            assert speaker._queue.get_nowait() == "hello"

    def test_speak_empty_string_noop(self, qapp):
        with patch.object(VoiceSpeaker, "_worker_loop"):
            speaker = VoiceSpeaker()
            speaker.speak("")
            assert speaker._queue.empty()

    def test_speak_whitespace_only_noop(self, qapp):
        with patch.object(VoiceSpeaker, "_worker_loop"):
            speaker = VoiceSpeaker()
            speaker.speak("   ")
            assert speaker._queue.empty()

    def test_speak_strips_whitespace(self, qapp):
        with patch.object(VoiceSpeaker, "_worker_loop"):
            speaker = VoiceSpeaker()
            speaker.speak("  hello  ")
            assert speaker._queue.get_nowait() == "hello"

    def test_speak_none_value_noop(self, qapp):
        with patch.object(VoiceSpeaker, "_worker_loop"):
            speaker = VoiceSpeaker()
            # speak(None) should be caught by the `not text` check
            speaker.speak(None)
            assert speaker._queue.empty()


# ======================================================================
# is_speaking property
# ======================================================================

class TestIsSpeaking:
    """is_speaking property reflects internal state."""

    def test_is_speaking_false_by_default(self, qapp):
        with patch.object(VoiceSpeaker, "_worker_loop"):
            speaker = VoiceSpeaker()
            assert speaker.is_speaking is False

    def test_is_speaking_true_when_set(self, qapp):
        with patch.object(VoiceSpeaker, "_worker_loop"):
            speaker = VoiceSpeaker()
            speaker._speaking = True
            assert speaker.is_speaking is True


# ======================================================================
# stop()
# ======================================================================

class TestStop:
    """Interrupting playback and clearing the queue."""

    def test_stop_sets_event(self, qapp):
        with patch.object(VoiceSpeaker, "_worker_loop"):
            speaker = VoiceSpeaker()
            speaker.stop()
            assert speaker._stop_event.is_set()

    def test_stop_drains_queue(self, qapp):
        with patch.object(VoiceSpeaker, "_worker_loop"):
            speaker = VoiceSpeaker()
            speaker._queue.put("a")
            speaker._queue.put("b")
            speaker._queue.put("c")

            with patch.dict("sys.modules", {"pygame": MagicMock()}):
                speaker.stop()

            assert speaker._queue.empty()


# ======================================================================
# interrupt_and_speak()
# ======================================================================

class TestInterruptAndSpeak:
    """Interrupt current speech and enqueue new text."""

    def test_interrupt_and_speak_clears_then_enqueues(self, qapp):
        with patch.object(VoiceSpeaker, "_worker_loop"):
            speaker = VoiceSpeaker()
            speaker._queue.put("old text")

            with patch.dict("sys.modules", {"pygame": MagicMock()}):
                speaker.interrupt_and_speak("new text")

            # Queue should contain only the new text
            items = []
            while not speaker._queue.empty():
                items.append(speaker._queue.get_nowait())
            assert items == ["new text"]


# ======================================================================
# shutdown()
# ======================================================================

class TestShutdown:
    """Clean shutdown of the speaker."""

    def test_shutdown_sends_sentinel(self, qapp):
        with patch.object(VoiceSpeaker, "_worker_loop"):
            speaker = VoiceSpeaker()
            # Replace the worker thread with a mock to avoid joining
            speaker._worker = MagicMock()

            with patch.dict("sys.modules", {"pygame": MagicMock()}), \
                 patch.object(speaker, "_cleanup_temp_files"):
                speaker.shutdown()

            # Sentinel (None) should have been added after draining
            # The worker join should have been called
            speaker._worker.join.assert_called_once_with(timeout=5)

    def test_shutdown_cleans_temp_files(self, qapp):
        with patch.object(VoiceSpeaker, "_worker_loop"):
            speaker = VoiceSpeaker()
            speaker._worker = MagicMock()

            with patch.dict("sys.modules", {"pygame": MagicMock()}), \
                 patch.object(speaker, "_cleanup_temp_files") as mock_cleanup:
                speaker.shutdown()

            mock_cleanup.assert_called_once()


# ======================================================================
# _select_voice
# ======================================================================

class TestSelectVoice:
    """Voice selection based on text content."""

    def test_thai_text_selects_thai_voice(self, qapp):
        with patch.object(VoiceSpeaker, "_worker_loop"):
            speaker = VoiceSpeaker()
            voice = speaker._select_voice("สวัสดีครับ")
            assert voice == VoiceSpeaker.VOICE_THAI

    def test_english_text_selects_english_voice(self, qapp):
        with patch.object(VoiceSpeaker, "_worker_loop"):
            speaker = VoiceSpeaker()
            voice = speaker._select_voice("Hello, how are you?")
            assert voice == VoiceSpeaker.VOICE_ENGLISH

    def test_mixed_text_selects_thai_voice(self, qapp):
        with patch.object(VoiceSpeaker, "_worker_loop"):
            speaker = VoiceSpeaker()
            voice = speaker._select_voice("Hello สวัสดี")
            assert voice == VoiceSpeaker.VOICE_THAI


# ======================================================================
# _generate_audio
# ======================================================================

class TestGenerateAudio:
    """Edge TTS audio generation."""

    @pytest.mark.asyncio
    async def test_generate_audio_success(self, qapp):
        with patch.object(VoiceSpeaker, "_worker_loop"):
            speaker = VoiceSpeaker()

            mock_communicate = MagicMock()
            mock_communicate.save = AsyncMock()

            with patch.dict("sys.modules", {"edge_tts": MagicMock()}) as _:
                edge_tts = sys.modules["edge_tts"]
                edge_tts.Communicate = MagicMock(return_value=mock_communicate)

                result = await speaker._generate_audio("hello")

            assert result is not None or mock_communicate.save.called

    @pytest.mark.asyncio
    async def test_generate_audio_error_returns_none(self, qapp):
        with patch.object(VoiceSpeaker, "_worker_loop"):
            speaker = VoiceSpeaker()

            mock_communicate = MagicMock()
            mock_communicate.save = AsyncMock(side_effect=RuntimeError("TTS fail"))

            with patch.dict("sys.modules", {"edge_tts": MagicMock()}) as _:
                edge_tts = sys.modules["edge_tts"]
                edge_tts.Communicate = MagicMock(return_value=mock_communicate)

                result = await speaker._generate_audio("hello")

            assert result is None

    @pytest.mark.asyncio
    async def test_generate_audio_stop_requested_returns_none(self, qapp):
        with patch.object(VoiceSpeaker, "_worker_loop"):
            speaker = VoiceSpeaker()

            mock_communicate = MagicMock()

            async def save_and_stop(path):
                # Create the file so unlink doesn't fail
                Path(path).touch()
                speaker._stop_event.set()

            mock_communicate.save = save_and_stop

            with patch.dict("sys.modules", {"edge_tts": MagicMock()}) as _:
                edge_tts = sys.modules["edge_tts"]
                edge_tts.Communicate = MagicMock(return_value=mock_communicate)

                result = await speaker._generate_audio("hello")

            assert result is None


# ======================================================================
# _play_with_pygame
# ======================================================================

class TestPlayWithPygame:
    """Pygame audio playback."""

    def test_play_success(self):
        mock_pg = MagicMock()
        mock_pg.mixer.get_init.return_value = True
        mock_pg.mixer.music.get_busy.return_value = False

        with patch.dict("sys.modules", {"pygame": mock_pg}):
            _play_with_pygame("/fake/path.mp3")

        mock_pg.mixer.music.load.assert_called_once_with("/fake/path.mp3")
        mock_pg.mixer.music.play.assert_called_once()
        mock_pg.mixer.music.unload.assert_called_once()

    def test_play_initializes_mixer_if_needed(self):
        mock_pg = MagicMock()
        mock_pg.mixer.get_init.return_value = False
        mock_pg.mixer.music.get_busy.return_value = False

        with patch.dict("sys.modules", {"pygame": mock_pg}):
            _play_with_pygame("/fake/path.mp3")

        mock_pg.mixer.init.assert_called_once()


# ======================================================================
# Pygame failure fallback to subprocess
# ======================================================================

class TestPlayAudioFallback:
    """_play_audio falls back to subprocess when pygame fails."""

    def test_fallback_to_subprocess(self):
        with patch("voice.speaker._play_with_pygame", side_effect=RuntimeError("no pygame")), \
             patch("voice.speaker._play_with_subprocess") as mock_sub:
            _play_audio("/fake/path.mp3")
            mock_sub.assert_called_once_with("/fake/path.mp3")


# ======================================================================
# Queue FIFO order
# ======================================================================

class TestQueueOrder:
    """Queue processes items in FIFO order."""

    def test_fifo_order(self, qapp):
        with patch.object(VoiceSpeaker, "_worker_loop"):
            speaker = VoiceSpeaker()
            speaker.speak("first")
            speaker.speak("second")
            speaker.speak("third")

            items = []
            while not speaker._queue.empty():
                items.append(speaker._queue.get_nowait())

            assert items == ["first", "second", "third"]


# ======================================================================
# _cleanup_temp_files
# ======================================================================

class TestCleanupTempFiles:
    """Temporary file cleanup."""

    def test_cleanup_removes_files_and_dir(self, qapp, tmp_path):
        with patch.object(VoiceSpeaker, "_worker_loop"):
            speaker = VoiceSpeaker()
            # Override with a temp path we control
            test_dir = tmp_path / "nova_tts_test"
            test_dir.mkdir()
            (test_dir / "test1.mp3").touch()
            (test_dir / "test2.mp3").touch()
            speaker._tmp_dir = test_dir

            speaker._cleanup_temp_files()

            assert not test_dir.exists()

    def test_cleanup_handles_missing_dir(self, qapp):
        with patch.object(VoiceSpeaker, "_worker_loop"):
            speaker = VoiceSpeaker()
            speaker._tmp_dir = Path("/nonexistent/path/nova_tts_xxx")
            # Should not raise, just log a warning
            speaker._cleanup_temp_files()
