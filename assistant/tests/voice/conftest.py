"""
Shared fixtures for voice module tests.

Provides mock objects for speech_recognition, keyboard, edge_tts, and pygame
so that tests never touch real hardware or network services.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Ensure the assistant package root is importable.
_ASSISTANT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ASSISTANT_ROOT) not in sys.path:
    sys.path.insert(0, str(_ASSISTANT_ROOT))


# ---------------------------------------------------------------------------
# speech_recognition mocks
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_recognizer():
    """Return a ``MagicMock`` standing in for ``sr.Recognizer``."""
    recognizer = MagicMock(name="Recognizer")
    recognizer.pause_threshold = 1.0
    recognizer.dynamic_energy_threshold = True
    recognizer.adjust_for_ambient_noise = MagicMock()
    recognizer.listen = MagicMock()
    recognizer.recognize_google = MagicMock(return_value="hello world")
    return recognizer


@pytest.fixture()
def mock_microphone():
    """Return a ``MagicMock`` standing in for ``sr.Microphone``.

    The mock supports the context-manager protocol (``with mic as source``).
    """
    mic_instance = MagicMock(name="MicrophoneInstance")
    mic_class = MagicMock(name="Microphone", return_value=mic_instance)
    mic_instance.__enter__ = MagicMock(return_value=mic_instance)
    mic_instance.__exit__ = MagicMock(return_value=False)
    return mic_class


@pytest.fixture()
def mock_sr(mock_recognizer, mock_microphone):
    """Patch the ``speech_recognition`` module with mock Recognizer and Microphone."""
    with patch.dict("sys.modules", {"speech_recognition": MagicMock()}) as modules:
        sr_mod = sys.modules["speech_recognition"]
        sr_mod.Recognizer = MagicMock(return_value=mock_recognizer)
        sr_mod.Microphone = mock_microphone
        sr_mod.UnknownValueError = type("UnknownValueError", (Exception,), {})
        sr_mod.RequestError = type("RequestError", (Exception,), {})
        sr_mod.WaitTimeoutError = type("WaitTimeoutError", (Exception,), {})
        yield sr_mod


# ---------------------------------------------------------------------------
# keyboard mock
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_keyboard():
    """Return a ``MagicMock`` standing in for the ``keyboard`` library."""
    kb = MagicMock(name="keyboard")
    kb.add_hotkey = MagicMock(return_value="hook_ref")
    kb.remove_hotkey = MagicMock()
    return kb


# ---------------------------------------------------------------------------
# edge_tts mock
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_edge_tts():
    """Return a ``MagicMock`` standing in for ``edge_tts``."""
    edge = MagicMock(name="edge_tts")
    communicate = MagicMock(name="Communicate")
    communicate.save = MagicMock()
    edge.Communicate = MagicMock(return_value=communicate)
    return edge


# ---------------------------------------------------------------------------
# pygame mock
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_pygame():
    """Return a ``MagicMock`` standing in for ``pygame`` and ``pygame.mixer``."""
    pg = MagicMock(name="pygame")
    pg.mixer.get_init = MagicMock(return_value=True)
    pg.mixer.init = MagicMock()
    pg.mixer.music.load = MagicMock()
    pg.mixer.music.play = MagicMock()
    pg.mixer.music.stop = MagicMock()
    pg.mixer.music.unload = MagicMock()
    # get_busy returns False immediately so _play_with_pygame doesn't loop
    pg.mixer.music.get_busy = MagicMock(return_value=False)
    return pg
