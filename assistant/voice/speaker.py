"""Text-to-Speech voice speaker using Edge TTS.

Generates high-quality speech via Microsoft Edge TTS and plays it back
using *pygame.mixer* (preferred) or a subprocess fallback (mpv / ffplay).
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import shutil
import tempfile
import threading
import time
import unicodedata
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thai character detection
# ---------------------------------------------------------------------------

_THAI_RANGE_START = 0x0E00
_THAI_RANGE_END = 0x0E7F


def _contains_thai(text: str) -> bool:
    """Return ``True`` if *text* contains any Thai Unicode characters."""
    for ch in text:
        cp = ord(ch)
        if _THAI_RANGE_START <= cp <= _THAI_RANGE_END:
            return True
    return False


# ---------------------------------------------------------------------------
# Audio playback helpers
# ---------------------------------------------------------------------------

def _play_with_pygame(path: str) -> None:
    """Play an audio file using *pygame.mixer*."""
    import pygame  # type: ignore[import-untyped]

    if not pygame.mixer.get_init():
        pygame.mixer.init()

    pygame.mixer.music.load(path)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(0.05)

    pygame.mixer.music.unload()


def _play_with_subprocess(path: str) -> None:
    """Play an audio file using an external player (mpv or ffplay)."""
    import subprocess

    player: Optional[str] = None
    for candidate in ("mpv", "ffplay"):
        if shutil.which(candidate):
            player = candidate
            break

    if player is None:
        raise RuntimeError(
            "No suitable audio player found. Install pygame, mpv, or ffplay."
        )

    cmd = (
        [player, "--no-video", "--really-quiet", path]
        if player == "mpv"
        else [player, "-nodisp", "-autoexit", "-loglevel", "quiet", path]
    )
    subprocess.run(cmd, check=True)  # noqa: S603


def _play_audio(path: str) -> None:
    """Play an audio file, trying pygame first then subprocess fallback."""
    try:
        _play_with_pygame(path)
    except Exception:  # noqa: BLE001
        _play_with_subprocess(path)


# ---------------------------------------------------------------------------
# VoiceSpeaker
# ---------------------------------------------------------------------------

class VoiceSpeaker(QObject):
    """Generates speech from text and plays it back.

    Signals
    -------
    speaking_started : (no payload)
        Emitted when audio playback begins.
    speaking_finished : (no payload)
        Emitted when audio playback ends (or is interrupted).
    """

    speaking_started = pyqtSignal()
    speaking_finished = pyqtSignal()

    # Voice mapping
    VOICE_THAI = "th-TH-PremwadeeNeural"
    VOICE_ENGLISH = "en-US-AriaNeural"

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)

        self._speaking = False
        self._stop_event = threading.Event()
        self._queue: queue.Queue[str | None] = queue.Queue()

        # Background worker thread
        self._worker = threading.Thread(
            target=self._worker_loop, daemon=True, name="VoiceSpeaker-worker"
        )
        self._worker.start()

        # Temp directory for generated audio files
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="nova_tts_"))
        logger.debug("TTS temp directory: %s", self._tmp_dir)

    # -- Public API ------------------------------------------------------

    @property
    def is_speaking(self) -> bool:
        """Return ``True`` if audio is currently being played."""
        return self._speaking

    def speak(self, text: str) -> None:
        """Enqueue *text* for TTS generation and playback.

        If the speaker is currently busy, the text is added to the queue
        and will be spoken after the current utterance finishes.
        """
        if not text or not text.strip():
            return
        self._queue.put(text.strip())

    def interrupt_and_speak(self, text: str) -> None:
        """Stop any current speech, clear the queue, and speak *text*."""
        self.stop()
        self.speak(text)

    def stop(self) -> None:
        """Interrupt current speech and clear the queue."""
        self._stop_event.set()

        # Drain the queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        # Stop pygame playback if active
        try:
            import pygame  # type: ignore[import-untyped]

            if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
        except Exception:  # noqa: BLE001
            pass

    def shutdown(self) -> None:
        """Stop the worker and clean up temp files."""
        self.stop()
        self._queue.put(None)  # Sentinel to exit worker loop
        self._worker.join(timeout=5)
        self._cleanup_temp_files()

    # -- Voice selection -------------------------------------------------

    def _select_voice(self, text: str) -> str:
        """Return the appropriate Edge TTS voice for *text*."""
        if _contains_thai(text):
            return self.VOICE_THAI
        return self.VOICE_ENGLISH

    # -- Background worker -----------------------------------------------

    def _worker_loop(self) -> None:
        """Process queued TTS requests sequentially."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            while True:
                text = self._queue.get()
                if text is None:
                    break  # Shutdown sentinel

                self._stop_event.clear()
                self._speaking = True
                self.speaking_started.emit()

                try:
                    audio_path = loop.run_until_complete(
                        self._generate_audio(text)
                    )
                    if audio_path and not self._stop_event.is_set():
                        _play_audio(str(audio_path))
                except Exception as exc:  # noqa: BLE001
                    logger.error("TTS error: %s", exc)
                finally:
                    self._speaking = False
                    self.speaking_finished.emit()
        finally:
            loop.close()

    async def _generate_audio(self, text: str) -> Optional[Path]:
        """Generate an MP3 file from *text* using Edge TTS.

        Returns the path to the generated file, or ``None`` on failure.
        """
        import edge_tts  # type: ignore[import-untyped]

        voice = self._select_voice(text)
        filename = f"tts_{int(time.time() * 1000)}.mp3"
        output_path = self._tmp_dir / filename

        logger.debug("Generating TTS: voice=%s, file=%s", voice, output_path)

        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(str(output_path))
        except Exception as exc:  # noqa: BLE001
            logger.error("Edge TTS generation failed: %s", exc)
            return None

        if self._stop_event.is_set():
            # Generation finished but stop was requested – discard
            output_path.unlink(missing_ok=True)
            return None

        return output_path

    # -- Temp file cleanup -----------------------------------------------

    def _cleanup_temp_files(self) -> None:
        """Remove all temporary audio files."""
        try:
            for f in self._tmp_dir.iterdir():
                f.unlink(missing_ok=True)
            self._tmp_dir.rmdir()
            logger.debug("Cleaned up TTS temp directory")
        except OSError as exc:
            logger.warning("Failed to clean up temp files: %s", exc)

    def __del__(self) -> None:
        try:
            self._cleanup_temp_files()
        except Exception:  # noqa: BLE001
            pass
