"""Speech-to-Text voice listener running as a QThread.

Uses the ``speech_recognition`` library with Google Speech Recognition and
supports both push-to-talk and continuous listening modes.
"""

from __future__ import annotations

import enum
import logging
import threading
from typing import Optional

import speech_recognition as sr  # type: ignore[import-untyped]
from PyQt6.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)


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
    _PHRASE_TIME_LIMIT = 15  # seconds per phrase
    _PAUSE_THRESHOLD = 1.0  # seconds of silence to consider phrase complete
    _ENERGY_ADJUST_DURATION = 0.5  # seconds for ambient energy calibration

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

        # Speech recognizer
        self._recognizer = sr.Recognizer()
        self._recognizer.pause_threshold = self._PAUSE_THRESHOLD
        self._recognizer.dynamic_energy_threshold = True

        # Microphone (lazily initialised in run())
        self._microphone: Optional[sr.Microphone] = None

        # Keyboard hook reference (so we can unhook later)
        self._hotkey_hook: object | None = None

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
        try:
            self._microphone = sr.Microphone()
        except (OSError, AttributeError) as exc:
            self.error_occurred.emit(f"Microphone not available: {exc}")
            logger.error("Failed to initialise microphone: %s", exc)
            return

        # Initial ambient noise calibration
        try:
            with self._microphone as source:
                self._recognizer.adjust_for_ambient_noise(
                    source, duration=self._ENERGY_ADJUST_DURATION
                )
        except OSError as exc:
            self.error_occurred.emit(f"Microphone error during calibration: {exc}")
            return

        if self._mode == ListenMode.PUSH_TO_TALK:
            self._run_push_to_talk()
        else:
            self._run_continuous()

        self._status = ListenerStatus.IDLE
        self.listening_stopped.emit()

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
        if self._microphone is None:
            return

        self._status = ListenerStatus.LISTENING
        self.listening_started.emit()

        try:
            with self._microphone as source:
                # Re-calibrate energy threshold periodically in continuous mode
                if self._mode == ListenMode.CONTINUOUS:
                    self._recognizer.adjust_for_ambient_noise(
                        source, duration=self._ENERGY_ADJUST_DURATION
                    )

                audio = self._recognizer.listen(
                    source,
                    timeout=5,
                    phrase_time_limit=self._PHRASE_TIME_LIMIT,
                )
        except sr.WaitTimeoutError:
            # No speech detected within timeout – not an error in continuous mode
            self._status = ListenerStatus.IDLE
            self.listening_stopped.emit()
            return
        except OSError as exc:
            self.error_occurred.emit(f"Microphone error: {exc}")
            self._status = ListenerStatus.IDLE
            self.listening_stopped.emit()
            return

        # -- Recognition phase --
        self._status = ListenerStatus.PROCESSING
        self.listening_stopped.emit()

        try:
            text = self._recognizer.recognize_google(
                audio, language=self._language
            )
            if text and text.strip():
                logger.info("Recognised: %s", text)
                self.text_recognized.emit(text.strip())
        except sr.UnknownValueError:
            logger.debug("Speech not understood")
        except sr.RequestError as exc:
            self.error_occurred.emit(f"Speech recognition service error: {exc}")
            logger.error("Google API request error: %s", exc)
        except Exception as exc:  # noqa: BLE001
            self.error_occurred.emit(f"Unexpected recognition error: {exc}")
            logger.exception("Unexpected error during recognition")
        finally:
            self._status = ListenerStatus.IDLE
