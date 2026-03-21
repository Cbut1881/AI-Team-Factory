"""
Tests for brain.tools -- Screen / Vision functions (Section B).

Mocks pyautogui.screenshot and cv2 so no real capture occurs.
"""

from __future__ import annotations

import base64
import io
import sys
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from brain.tools import screenshot, webcam_capture


def _make_pil_image(width: int = 800, height: int = 600) -> Image.Image:
    """Create a small in-memory PIL Image for testing."""
    return Image.new("RGB", (width, height), color=(0, 128, 255))


# ---------------------------------------------------------------------------
# screenshot()
# ---------------------------------------------------------------------------

class TestScreenshot:
    @patch("brain.tools.pyautogui.screenshot")
    def test_full_screen(self, mock_ss):
        mock_ss.return_value = _make_pil_image(800, 600)
        result = screenshot()
        assert result["status"] == "ok"
        assert "image_base64" in result
        assert result["width"] == 800
        assert result["height"] == 600
        assert result["media_type"] == "image/png"
        # Verify valid base64
        raw = base64.b64decode(result["image_base64"])
        img = Image.open(io.BytesIO(raw))
        assert img.format == "PNG"

    @patch("brain.tools.pyautogui.screenshot")
    def test_with_region(self, mock_ss):
        mock_ss.return_value = _make_pil_image(200, 150)
        region = {"x": 10, "y": 20, "width": 200, "height": 150}
        result = screenshot(region=region)
        mock_ss.assert_called_once_with(region=(10, 20, 200, 150))
        assert result["status"] == "ok"
        assert result["width"] == 200
        assert result["height"] == 150

    @patch("brain.tools.pyautogui.screenshot")
    def test_downscaling(self, mock_ss):
        """Images wider than max_width should be resized."""
        mock_ss.return_value = _make_pil_image(2560, 1440)
        result = screenshot(max_width=1280)
        assert result["status"] == "ok"
        assert result["width"] == 1280
        # Height should preserve aspect ratio: 1440 * (1280/2560) = 720
        assert result["height"] == 720

    @patch("brain.tools.pyautogui.screenshot")
    def test_no_downscaling_when_small(self, mock_ss):
        mock_ss.return_value = _make_pil_image(640, 480)
        result = screenshot(max_width=1280)
        assert result["status"] == "ok"
        assert result["width"] == 640
        assert result["height"] == 480

    @patch("brain.tools.pyautogui.screenshot")
    def test_custom_max_width(self, mock_ss):
        mock_ss.return_value = _make_pil_image(1920, 1080)
        result = screenshot(max_width=960)
        assert result["status"] == "ok"
        assert result["width"] == 960

    @patch("brain.tools.pyautogui.screenshot", side_effect=OSError("no display"))
    def test_screenshot_error(self, mock_ss):
        result = screenshot()
        assert result["status"] == "error"
        assert "no display" in result["error"]

    @patch("brain.tools.pyautogui.screenshot")
    def test_full_screen_no_region_call(self, mock_ss):
        """When no region is given, screenshot() is called without region kwarg."""
        mock_ss.return_value = _make_pil_image(800, 600)
        screenshot()
        mock_ss.assert_called_once_with()


# ---------------------------------------------------------------------------
# webcam_capture()
# ---------------------------------------------------------------------------

class TestWebcamCapture:
    @patch.dict("sys.modules", {"cv2": MagicMock()})
    def test_webcam_success(self):
        import importlib
        cv2_mock = sys.modules["cv2"]
        cap_mock = MagicMock()
        cv2_mock.VideoCapture.return_value = cap_mock
        cap_mock.isOpened.return_value = True

        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cap_mock.read.return_value = (True, frame)
        cv2_mock.COLOR_BGR2RGB = 4
        cv2_mock.cvtColor.return_value = frame

        result = webcam_capture(camera_index=0)
        assert result["status"] == "ok"
        assert "image_base64" in result
        assert result["width"] == 640
        assert result["height"] == 480
        cap_mock.release.assert_called_once()

    @patch.dict("sys.modules", {"cv2": MagicMock()})
    def test_webcam_camera_not_opened(self):
        cv2_mock = sys.modules["cv2"]
        cap_mock = MagicMock()
        cv2_mock.VideoCapture.return_value = cap_mock
        cap_mock.isOpened.return_value = False

        result = webcam_capture(camera_index=1)
        assert result["status"] == "error"
        assert "Cannot open camera" in result["error"]
        cap_mock.release.assert_called_once()

    @patch.dict("sys.modules", {"cv2": MagicMock()})
    def test_webcam_frame_read_fails(self):
        cv2_mock = sys.modules["cv2"]
        cap_mock = MagicMock()
        cv2_mock.VideoCapture.return_value = cap_mock
        cap_mock.isOpened.return_value = True
        cap_mock.read.return_value = (False, None)

        result = webcam_capture()
        assert result["status"] == "error"
        assert "Failed to capture frame" in result["error"]
        cap_mock.release.assert_called_once()

    def test_webcam_cv2_not_installed(self):
        """When cv2 is not importable, webcam_capture returns an error."""
        with patch.dict("sys.modules", {"cv2": None}):
            # Reload the function context -- but since webcam_capture does
            # `import cv2` inside the function, we can simulate ImportError
            # by removing cv2 from sys.modules entirely.
            pass

        # The actual function tries `import cv2` -- mock that to raise ImportError
        with patch("builtins.__import__", side_effect=_import_blocker("cv2")):
            result = webcam_capture()
        assert result["status"] == "error"
        assert "cv2" in result["error"].lower()


def _import_blocker(blocked: str):
    """Return a side_effect function that blocks a specific import."""
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def _blocker(name, *args, **kwargs):
        if name == blocked:
            raise ImportError(f"No module named '{blocked}'")
        return real_import(name, *args, **kwargs)

    return _blocker
