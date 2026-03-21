"""
Tests for vision.screen — screenshot capture and image utilities.

pyautogui is fully mocked so that no actual screenshots are taken.
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

_ASSISTANT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ASSISTANT_ROOT) not in sys.path:
    sys.path.insert(0, str(_ASSISTANT_ROOT))

from vision.screen import (
    capture_full_screen,
    capture_region,
    get_screen_size,
    _resize_if_needed,
    _image_to_base64,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(width: int = 1920, height: int = 1080) -> Image.Image:
    """Create a simple PIL Image of the given size."""
    return Image.new("RGB", (width, height), color=(100, 150, 200))


def _make_size_mock(width: int = 1920, height: int = 1080):
    """Return a mock that mimics pyautogui.size()."""
    mock = MagicMock()
    mock.width = width
    mock.height = height
    return mock


# ======================================================================
# capture_full_screen
# ======================================================================

class TestCaptureFullScreen:
    """Full-screen screenshot capture."""

    def test_success_returns_base64(self):
        img = _make_image(1920, 1080)

        with patch("vision.screen.pyautogui") as mock_pag:
            mock_pag.screenshot.return_value = img
            mock_pag.size.return_value = _make_size_mock(1920, 1080)

            result = capture_full_screen()

        assert isinstance(result, str)
        # Verify it's valid base64
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_downscaling_applied(self):
        img = _make_image(2560, 1440)

        with patch("vision.screen.pyautogui") as mock_pag:
            mock_pag.screenshot.return_value = img
            mock_pag.size.return_value = _make_size_mock(2560, 1440)

            result = capture_full_screen(max_width=1280)

        # Decode and check the size
        decoded = base64.b64decode(result)
        result_img = Image.open(io.BytesIO(decoded))
        assert result_img.width <= 1280

    def test_no_downscaling_for_small_image(self):
        img = _make_image(800, 600)

        with patch("vision.screen.pyautogui") as mock_pag:
            mock_pag.screenshot.return_value = img
            mock_pag.size.return_value = _make_size_mock(800, 600)

            result = capture_full_screen(max_width=1280)

        decoded = base64.b64decode(result)
        result_img = Image.open(io.BytesIO(decoded))
        assert result_img.width == 800

    def test_error_raises_runtime_error(self):
        with patch("vision.screen.pyautogui") as mock_pag:
            mock_pag.screenshot.side_effect = RuntimeError("no display")

            with pytest.raises(RuntimeError, match="Screenshot capture failed"):
                capture_full_screen()


# ======================================================================
# capture_region
# ======================================================================

class TestCaptureRegion:
    """Regional screenshot capture."""

    def test_success_returns_base64(self):
        img = _make_image(400, 300)

        with patch("vision.screen.pyautogui") as mock_pag:
            mock_pag.screenshot.return_value = img

            result = capture_region(100, 100, 400, 300)

        assert isinstance(result, str)
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_zero_width_raises_value_error(self):
        with pytest.raises(ValueError, match="positive"):
            capture_region(0, 0, 0, 100)

    def test_negative_height_raises_value_error(self):
        with pytest.raises(ValueError, match="positive"):
            capture_region(0, 0, 100, -5)

    def test_zero_height_raises_value_error(self):
        with pytest.raises(ValueError, match="positive"):
            capture_region(0, 0, 100, 0)

    def test_error_raises_runtime_error(self):
        with patch("vision.screen.pyautogui") as mock_pag:
            mock_pag.screenshot.side_effect = OSError("capture failed")

            with pytest.raises(RuntimeError, match="Region capture failed"):
                capture_region(0, 0, 100, 100)


# ======================================================================
# get_screen_size
# ======================================================================

class TestGetScreenSize:
    """Screen size retrieval."""

    def test_returns_tuple(self):
        with patch("vision.screen.pyautogui") as mock_pag:
            mock_pag.size.return_value = _make_size_mock(1920, 1080)
            result = get_screen_size()

        assert result == (1920, 1080)

    def test_returns_correct_dimensions(self):
        with patch("vision.screen.pyautogui") as mock_pag:
            mock_pag.size.return_value = _make_size_mock(3840, 2160)
            result = get_screen_size()

        assert result == (3840, 2160)


# ======================================================================
# _resize_if_needed
# ======================================================================

class TestResizeIfNeeded:
    """Image resize utility."""

    def test_no_resize_when_small_enough(self):
        img = _make_image(800, 600)
        result = _resize_if_needed(img, 1280)
        assert result is img  # Same object, no copy

    def test_resize_when_too_wide(self):
        img = _make_image(2560, 1440)
        result = _resize_if_needed(img, 1280)
        assert result.width == 1280
        # Aspect ratio preserved
        expected_height = int(1440 * (1280 / 2560))
        assert result.height == expected_height

    def test_exact_width_no_resize(self):
        img = _make_image(1280, 720)
        result = _resize_if_needed(img, 1280)
        assert result is img


# ======================================================================
# _image_to_base64
# ======================================================================

class TestImageToBase64:
    """Base64 encoding utility."""

    def test_returns_valid_base64(self):
        img = _make_image(100, 100)
        result = _image_to_base64(img)
        assert isinstance(result, str)
        # Must be decodable
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_encoded_image_is_png(self):
        img = _make_image(100, 100)
        result = _image_to_base64(img)
        decoded = base64.b64decode(result)
        # PNG magic bytes
        assert decoded[:4] == b"\x89PNG"
