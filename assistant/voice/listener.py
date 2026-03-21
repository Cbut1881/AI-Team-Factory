"""Speech-to-Text voice listener running as a QThread.

Uses ``sounddevice`` for audio capture (no PyAudio dependency) and
Google Speech Recognition for transcription.  Supports push-to-talk
and continuous listening modes.
"""

from __future__ import annotations

import enum
import io
import json
import logging
import struct
import threading
import wave
from typing import Optional

import numpy as np
import sounddevice as sd  # type: ignore[import-untyped]
from PyQt6.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Google Speech Recognition (no pyaudio needed)
# ---------------------------------------------------------------------------

def _recognize_google(audio_data: bytes, sample_rate: int, language: str) -> str | None:
    """Send WAV audio to Google Speech Recognition and return text."""
    import urllib.request
    import urllib.parse

    # Build WAV in memory
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)
    wav_data = buf.getvalue()

    # Google Speech API (free tier, same as speech_recognition uses)
    url = (
        "http://www.google.com/speech-api/v2/recognize"
        f"?client=chromium&lang={language}&key=AIzaSyBOti4mM-6x9WDnZIjIeyEU21OpBXqWBgw"
    )

    request = urllib.request.Request(
        url,
        data=wav_data,
        headers={
            "Content-Type": f"audio/l16; rate={sample_rate};",
        },
    )

    try:
        response = urllib.request.urlopen(request, timeout=10)
        response_text = response.read().decode("utf-8")
    except Exception as exc:
        logger.error("Google Speech API error: %s", exc)
        return None

    # Parse response (one JSON per line, first is empty)
    for line in response_text.strip().split("\n"):
        if not line.strip():
            continue
        try:
            result = json.loads(line)
            if "result" in result and result["result"]:
                for r in result["result"]:
                    if "alternative" in r and r["alternative"]:
                        return r["alternative"][0].get("transcript")
        except json.JSONDecodeError:
            continue

    return None


class ListenMode(enum.Enum):
    """Supported listening modes."""

    PUSH_TO_TALK = "push_to_talk"
    CONTINUOUS = "continuous"


class ListenerStatus(enum.Enum):
    """Current status of the voice listener."""

    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"


class VoiceListener(QThread):
    """Background thread that captures microphone input and converts it to text.

    Uses ``sounddevice`` for audio capture (works on Python 3.14+).

    Signals
    -------
    text_recognized : str
        Emitted when a speech segment has been successfully transcribed.
    listening_started : (no payload)
        Emitted when the listener begins capturing audio.
    listening_stopped : (no payload)
        Emitted when the listener stops capturing audio.
    error_occurred : str
        Emitted when an error occurs (missing mic, permission denied, etc.).
    """

    text_recognized = pyqtSignal(str)
    listening_started = pyqtSignal()
    listening_stopped = pyqtSignal()
    error_occurred = pyqtSignal(str)

    # Defaults -----------------------------------------------------------
    _DEFAULT_LANGUAGE = "th-TH"
    _DEFAULT_HOTKEY = "ctrl+space"
    _SAMPLE_RATE = 48000  # Use WASAPI native rate
    _CHANNELS = 2  # WASAPI stereo
    _DEVICE_INDEX: int | None = None  # Auto-detect best device
    _SOFTWARE_GAIN = 80  # Amplify weak mic signal
    _PHRASE_TIME_LIMIT = 15  # seconds per phrase
    _SILENCE_THRESHOLD = 30  # RMS threshold (low for weak mics)
    _SILENCE_DURATION = 1.5  # seconds of silence to end phrase
    _MIN_PHRASE_DURATION = 0.3  # minimum seconds for a valid phrase

    def __init__(
        self,
        mode: ListenMode = ListenMode.PUSH_TO_TALK,
        language: str | None = None,
        hotkey: str | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)

        self._mode = mode
        self._language: str = language or self._DEFAULT_LANGUAGE
        self._hotkey: str = hotkey or self._DEFAULT_HOTKEY
        self._status = ListenerStatus.IDLE

        # Threading / control
        self._stop_event = threading.Event()
        self._ptt_active = threading.Event()

        # Keyboard hook reference (so we can unhook later)
        self._hotkey_hook: object | None = None

        # Calibrated noise level
        self._ambient_energy = 300.0

    # -- Public properties -----------------------------------------------

    @property
    def status(self) -> ListenerStatus:
        """Return the current listener status."""
        return self._status

    @property
    def language(self) -> str:
        return self._language

    @language.setter
    def language(self, value: str) -> None:
        """Switch the recognition language on the fly."""
        self._language = value
        logger.info("Recognition language changed to %s", value)

    @property
    def mode(self) -> ListenMode:
        return self._mode

    @mode.setter
    def mode(self, value: ListenMode) -> None:
        self._mode = value
        logger.info("Listen mode changed to %s", value.value)

    # -- Public control --------------------------------------------------

    def start_listening(self) -> None:  # noqa: D401
        """Convenience wrapper – starts the QThread."""
        if not self.isRunning():
            self._stop_event.clear()
            self.start()

    def stop_listening(self) -> None:
        """Request the listener to stop gracefully."""
        self._stop_event.set()
        self._ptt_active.set()  # unblock any wait
        self._unregister_hotkey()
        self.quit()
        self.wait(3000)

    # -- QThread entry point ---------------------------------------------

    def run(self) -> None:  # noqa: D401 – Qt override
        """Main loop executed in the background thread."""
        # Auto-detect best mic device
        if not self._detect_best_device():
            self.error_occurred.emit("No working microphone found")
            return

        # Calibrate ambient noise
        self._calibrate_noise()

        if self._mode == ListenMode.PUSH_TO_TALK:
            self._run_push_to_talk()
        else:
            self._run_continuous()

        self._status = ListenerStatus.IDLE
        self.listening_stopped.emit()

    # -- Device detection ------------------------------------------------

    def _detect_best_device(self) -> bool:
        """Find the best working input device. Prefer WASAPI."""
        try:
            all_devices = sd.query_devices()
        except Exception as exc:
            logger.error("Cannot query audio devices: %s", exc)
            return False

        # Rank: WASAPI > DirectSound > MME
        candidates = []
        for i, d in enumerate(all_devices):
            if d["max_input_channels"] <= 0:
                continue
            hostapi = sd.query_hostapis(d["hostapi"])["name"]
            if "WASAPI" in hostapi:
                priority = 0
            elif "DirectSound" in hostapi:
                priority = 1
            else:
                priority = 2
            candidates.append((priority, i, d))

        candidates.sort(key=lambda x: x[0])

        for _prio, idx, dev in candidates:
            rate = int(dev["default_samplerate"])
            ch = min(dev["max_input_channels"], 2)
            try:
                test = sd.rec(int(0.3 * rate), samplerate=rate, channels=ch, dtype="int16", device=idx)
                sd.wait()
                peak = int(np.max(np.abs(test)))
                logger.info("Device [%d] %s: rate=%d ch=%d peak=%d", idx, dev["name"], rate, ch, peak)
                self._DEVICE_INDEX = idx
                self._SAMPLE_RATE = rate
                self._CHANNELS = ch
                logger.info("Selected mic: [%d] %s @ %dHz", idx, dev["name"], rate)
                return True
            except Exception as exc:
                logger.debug("Device [%d] failed: %s", idx, exc)
                continue

        return False

    # -- Noise calibration -----------------------------------------------

    def _calibrate_noise(self) -> None:
        """Record a short sample to set the ambient noise level."""
        try:
            logger.debug("Calibrating ambient noise...")
            audio = sd.rec(
                int(0.5 * self._SAMPLE_RATE),
                samplerate=self._SAMPLE_RATE,
                channels=self._CHANNELS,
                dtype="int16",
                device=self._DEVICE_INDEX,
            )
            sd.wait()
            # Convert to mono
            if audio.ndim > 1:
                audio = audio[:, 0]
            boosted = np.clip(audio.astype(np.float64) * self._SOFTWARE_GAIN, -32768, 32767)
            rms = np.sqrt(np.mean(boosted ** 2))
            self._ambient_energy = rms
            # Set threshold just above ambient noise
            self._SILENCE_THRESHOLD = max(int(rms * 2.0), 20)
            logger.info("Ambient RMS: %.0f (gain x%d), threshold: %d",
                        rms, self._SOFTWARE_GAIN, self._SILENCE_THRESHOLD)
        except Exception as exc:
            logger.warning("Noise calibration failed: %s", exc)

    # -- Push-to-talk ----------------------------------------------------

    def _run_push_to_talk(self) -> None:
        """Listen only while the hotkey is held down."""
        self._register_hotkey()
        logger.info("Push-to-talk mode active (hotkey: %s)", self._hotkey)

        while not self._stop_event.is_set():
            # Block until hotkey pressed (or stop requested)
            self._ptt_active.wait(timeout=0.5)
            if self._stop_event.is_set():
                break
            if not self._ptt_active.is_set():
                continue

            self._capture_and_recognise()
            self._ptt_active.clear()

    def _register_hotkey(self) -> None:
        """Register the push-to-talk keyboard hotkey."""
        try:
            import keyboard  # type: ignore[import-untyped]

            def _on_press(_event: object) -> None:
                self._ptt_active.set()

            self._hotkey_hook = keyboard.add_hotkey(
                self._hotkey, _on_press, suppress=False
            )
            logger.debug("Hotkey '%s' registered", self._hotkey)
        except ImportError:
            self.error_occurred.emit(
                "The 'keyboard' library is required for push-to-talk mode. "
                "Install it with: pip install keyboard"
            )
        except Exception as exc:  # noqa: BLE001
            self.error_occurred.emit(f"Failed to register hotkey: {exc}")

    def _unregister_hotkey(self) -> None:
        """Remove the keyboard hotkey hook if registered."""
        if self._hotkey_hook is not None:
            try:
                import keyboard  # type: ignore[import-untyped]

                keyboard.remove_hotkey(self._hotkey_hook)
            except Exception:  # noqa: BLE001
                pass
            self._hotkey_hook = None

    # -- Continuous mode -------------------------------------------------

    def _run_continuous(self) -> None:
        """Continuously listen and transcribe."""
        logger.info("Continuous listening mode active")

        while not self._stop_event.is_set():
            self._capture_and_recognise()

    # -- Core capture / recognition --------------------------------------

    def _capture_and_recognise(self) -> None:
        """Record a single phrase from the microphone and recognise it."""
        self._status = ListenerStatus.LISTENING
        self.listening_started.emit()

        try:
            audio_data = self._record_phrase()
        except Exception as exc:
            self.error_occurred.emit(f"Microphone error: {exc}")
            self._status = ListenerStatus.IDLE
            self.listening_stopped.emit()
            return

        if audio_data is None or len(audio_data) < int(self._MIN_PHRASE_DURATION * self._SAMPLE_RATE * 2):
            self._status = ListenerStatus.IDLE
            self.listening_stopped.emit()
            return

        # -- Recognition phase --
        self._status = ListenerStatus.PROCESSING
        self.listening_stopped.emit()

        try:
            text = _recognize_google(audio_data, self._SAMPLE_RATE, self._language)
            if text and text.strip():
                logger.info("Recognised: %s", text)
                self.text_recognized.emit(text.strip())
        except Exception as exc:  # noqa: BLE001
            self.error_occurred.emit(f"Recognition error: {exc}")
            logger.exception("Unexpected error during recognition")
        finally:
            self._status = ListenerStatus.IDLE

    def _record_phrase(self) -> bytes | None:
        """Record audio until silence is detected or time limit reached.

        Returns raw PCM int16 mono bytes at 16kHz (resampled for Google API).
        """
        chunk_duration = 0.1  # seconds per chunk
        chunk_samples = int(self._SAMPLE_RATE * chunk_duration)
        max_chunks = int(self._PHRASE_TIME_LIMIT / chunk_duration)
        silence_chunks = int(self._SILENCE_DURATION / chunk_duration)

        frames: list[np.ndarray] = []
        silent_count = 0
        speech_detected = False

        for _ in range(max_chunks):
            if self._stop_event.is_set():
                return None

            try:
                chunk = sd.rec(
                    chunk_samples,
                    samplerate=self._SAMPLE_RATE,
                    channels=self._CHANNELS,
                    dtype="int16",
                    device=self._DEVICE_INDEX,
                )
                sd.wait()
            except Exception:
                return None

            # Convert to mono
            if chunk.ndim > 1:
                mono = chunk[:, 0]
            else:
                mono = chunk.flatten()

            # Apply software gain
            boosted = np.clip(
                mono.astype(np.float64) * self._SOFTWARE_GAIN, -32768, 32767
            ).astype(np.int16)

            rms = np.sqrt(np.mean(boosted.astype(np.float64) ** 2))

            if rms > self._SILENCE_THRESHOLD:
                speech_detected = True
                silent_count = 0
                frames.append(boosted)
            elif speech_detected:
                silent_count += 1
                frames.append(boosted)
                if silent_count >= silence_chunks:
                    break
            # else: waiting for speech to start

        if not speech_detected:
            return None

        # Concatenate and resample to 16kHz for Google Speech API
        full_audio = np.concatenate(frames)
        if self._SAMPLE_RATE != 16000:
            ratio = 16000 / self._SAMPLE_RATE
            new_len = int(len(full_audio) * ratio)
            indices = np.round(np.linspace(0, len(full_audio) - 1, new_len)).astype(int)
            full_audio = full_audio[indices]

        return full_audio.tobytes()
