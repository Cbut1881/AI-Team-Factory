"""
Tests for vision.camera — WebcamCapture.

OpenCV (cv2) is fully mocked so that no actual camera is accessed.
"""

from __future__ import annotations

import base64
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import numpy as np

_ASSISTANT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ASSISTANT_ROOT) not in sys.path:
    sys.path.insert(0, str(_ASSISTANT_ROOT))


# We need to handle the lazy cv2 import in camera.py
def _make_mock_cv2():
    """Create a mock cv2 module with required constants and functions."""
    cv2 = MagicMock(name="cv2")
    cv2.COLOR_BGR2RGB = 4  # Actual OpenCV constant
    cv2.VideoCapture = MagicMock()
    cv2.cvtColor = MagicMock(side_effect=lambda frame, code: frame)
    return cv2


def _make_fake_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a fake BGR frame (numpy array)."""
    return np.zeros((height, width, 3), dtype=np.uint8)


# ======================================================================
# Module-level import guard
# ======================================================================

class TestCv2ImportGuard:
    """cv2 not installed raises ImportError."""

    def test_ensure_cv2_raises_import_error(self):
        import vision.camera as cam_module
        # Reset the cached cv2 reference
        original = cam_module._cv2
        cam_module._cv2 = None

        try:
            with patch.dict("sys.modules", {"cv2": None}), \
                 patch("builtins.__import__", side_effect=ImportError("no cv2")):
                with pytest.raises(ImportError, match="opencv-python"):
                    cam_module._ensure_cv2()
        finally:
            cam_module._cv2 = original


# ======================================================================
# WebcamCapture init
# ======================================================================

class TestInit:
    """WebcamCapture construction."""

    def test_default_device_index(self):
        from vision.camera import WebcamCapture
        cam = WebcamCapture()
        assert cam._device_index == 0

    def test_custom_device_index(self):
        from vision.camera import WebcamCapture
        cam = WebcamCapture(device_index=2)
        assert cam._device_index == 2

    def test_not_started_initially(self):
        from vision.camera import WebcamCapture
        cam = WebcamCapture()
        assert cam._started is False
        assert cam._cap is None


# ======================================================================
# start()
# ======================================================================

class TestStart:
    """Camera start lifecycle."""

    def test_start_success(self):
        mock_cv2 = _make_mock_cv2()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cv2.VideoCapture.return_value = mock_cap

        import vision.camera as cam_module
        original = cam_module._cv2
        cam_module._cv2 = mock_cv2

        try:
            from vision.camera import WebcamCapture
            cam = WebcamCapture()
            result = cam.start()
            assert result is True
            assert cam._started is True
        finally:
            cam_module._cv2 = original

    def test_start_failure_camera_not_opened(self):
        mock_cv2 = _make_mock_cv2()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = mock_cap

        import vision.camera as cam_module
        original = cam_module._cv2
        cam_module._cv2 = mock_cv2

        try:
            from vision.camera import WebcamCapture
            cam = WebcamCapture()
            result = cam.start()
            assert result is False
            assert cam._started is False
        finally:
            cam_module._cv2 = original

    def test_start_already_running_returns_true(self):
        mock_cv2 = _make_mock_cv2()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cv2.VideoCapture.return_value = mock_cap

        import vision.camera as cam_module
        original = cam_module._cv2
        cam_module._cv2 = mock_cv2

        try:
            from vision.camera import WebcamCapture
            cam = WebcamCapture()
            cam.start()
            # Start again
            result = cam.start()
            assert result is True
        finally:
            cam_module._cv2 = original

    def test_start_exception_returns_false(self):
        mock_cv2 = _make_mock_cv2()
        mock_cv2.VideoCapture.side_effect = RuntimeError("driver error")

        import vision.camera as cam_module
        original = cam_module._cv2
        cam_module._cv2 = mock_cv2

        try:
            from vision.camera import WebcamCapture
            cam = WebcamCapture()
            result = cam.start()
            assert result is False
            assert cam._started is False
        finally:
            cam_module._cv2 = original


# ======================================================================
# stop()
# ======================================================================

class TestStop:
    """Camera stop lifecycle."""

    def test_stop_releases_camera(self):
        mock_cv2 = _make_mock_cv2()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cv2.VideoCapture.return_value = mock_cap

        import vision.camera as cam_module
        original = cam_module._cv2
        cam_module._cv2 = mock_cv2

        try:
            from vision.camera import WebcamCapture
            cam = WebcamCapture()
            cam.start()
            cam.stop()

            mock_cap.release.assert_called_once()
            assert cam._started is False
            assert cam._cap is None
        finally:
            cam_module._cv2 = original

    def test_stop_when_not_started_is_noop(self):
        from vision.camera import WebcamCapture
        cam = WebcamCapture()
        # Should not raise
        cam.stop()
        assert cam._started is False


# ======================================================================
# is_running property
# ======================================================================

class TestIsRunning:
    """is_running property reflects camera state."""

    def test_not_running_initially(self):
        from vision.camera import WebcamCapture
        cam = WebcamCapture()
        assert cam.is_running is False

    def test_running_after_start(self):
        mock_cv2 = _make_mock_cv2()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cv2.VideoCapture.return_value = mock_cap

        import vision.camera as cam_module
        original = cam_module._cv2
        cam_module._cv2 = mock_cv2

        try:
            from vision.camera import WebcamCapture
            cam = WebcamCapture()
            cam.start()
            assert cam.is_running is True
        finally:
            cam_module._cv2 = original

    def test_not_running_after_stop(self):
        mock_cv2 = _make_mock_cv2()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cv2.VideoCapture.return_value = mock_cap

        import vision.camera as cam_module
        original = cam_module._cv2
        cam_module._cv2 = mock_cv2

        try:
            from vision.camera import WebcamCapture
            cam = WebcamCapture()
            cam.start()
            cam.stop()
            assert cam.is_running is False
        finally:
            cam_module._cv2 = original


# ======================================================================
# capture_frame
# ======================================================================

class TestCaptureFrame:
    """Frame capture from webcam."""

    def _setup_cam(self, mock_cv2, frame=None, read_ret=True):
        """Helper to set up a camera module with mock cv2."""
        if frame is None:
            frame = _make_fake_frame()

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (read_ret, frame if read_ret else None)
        mock_cv2.VideoCapture.return_value = mock_cap

        import vision.camera as cam_module
        cam_module._cv2 = mock_cv2
        return mock_cap

    def test_capture_success_returns_base64(self):
        mock_cv2 = _make_mock_cv2()
        self._setup_cam(mock_cv2)

        from vision.camera import WebcamCapture
        cam = WebcamCapture()
        cam.start()
        result = cam.capture_frame()

        assert result is not None
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_capture_auto_starts(self):
        mock_cv2 = _make_mock_cv2()
        mock_cap = self._setup_cam(mock_cv2)

        from vision.camera import WebcamCapture
        cam = WebcamCapture()
        # Don't call start() explicitly
        result = cam.capture_frame()

        assert result is not None

    def test_capture_auto_start_fails_returns_none(self):
        mock_cv2 = _make_mock_cv2()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = mock_cap

        import vision.camera as cam_module
        cam_module._cv2 = mock_cv2

        from vision.camera import WebcamCapture
        cam = WebcamCapture()
        result = cam.capture_frame()

        assert result is None

    def test_capture_read_fails_returns_none(self):
        mock_cv2 = _make_mock_cv2()
        self._setup_cam(mock_cv2, read_ret=False)

        from vision.camera import WebcamCapture
        cam = WebcamCapture()
        cam.start()
        result = cam.capture_frame()

        assert result is None

    def test_capture_encoding_error_returns_none(self):
        mock_cv2 = _make_mock_cv2()
        mock_cv2.cvtColor = MagicMock(side_effect=RuntimeError("encoding error"))
        self._setup_cam(mock_cv2)

        from vision.camera import WebcamCapture
        cam = WebcamCapture()
        cam.start()
        result = cam.capture_frame()

        assert result is None


# ======================================================================
# Context manager
# ======================================================================

class TestContextManager:
    """WebcamCapture as a context manager."""

    def test_enter_starts_camera(self):
        mock_cv2 = _make_mock_cv2()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cv2.VideoCapture.return_value = mock_cap

        import vision.camera as cam_module
        original = cam_module._cv2
        cam_module._cv2 = mock_cv2

        try:
            from vision.camera import WebcamCapture
            with WebcamCapture() as cam:
                assert cam._started is True
        finally:
            cam_module._cv2 = original

    def test_exit_stops_camera(self):
        mock_cv2 = _make_mock_cv2()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cv2.VideoCapture.return_value = mock_cap

        import vision.camera as cam_module
        original = cam_module._cv2
        cam_module._cv2 = mock_cv2

        try:
            from vision.camera import WebcamCapture
            cam = WebcamCapture()
            cam.__enter__()
            cam.__exit__(None, None, None)
            mock_cap.release.assert_called_once()
        finally:
            cam_module._cv2 = original


# ======================================================================
# Thread safety
# ======================================================================

class TestThreadSafety:
    """Lock-based thread safety."""

    def test_has_lock(self):
        from vision.camera import WebcamCapture
        cam = WebcamCapture()
        assert isinstance(cam._lock, type(threading.Lock()))

    def test_concurrent_capture_does_not_crash(self):
        mock_cv2 = _make_mock_cv2()
        frame = _make_fake_frame()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, frame)
        mock_cv2.VideoCapture.return_value = mock_cap

        import vision.camera as cam_module
        original = cam_module._cv2
        cam_module._cv2 = mock_cv2

        try:
            from vision.camera import WebcamCapture
            cam = WebcamCapture()
            cam.start()

            results = []
            errors = []

            def capture():
                try:
                    r = cam.capture_frame()
                    results.append(r)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=capture) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)

            assert len(errors) == 0
            assert len(results) == 5
        finally:
            cam_module._cv2 = original


# ======================================================================
# __repr__
# ======================================================================

class TestRepr:
    """String representation."""

    def test_repr_stopped(self):
        from vision.camera import WebcamCapture
        cam = WebcamCapture(device_index=1)
        r = repr(cam)
        assert "device=1" in r
        assert "stopped" in r

    def test_repr_running(self):
        mock_cv2 = _make_mock_cv2()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cv2.VideoCapture.return_value = mock_cap

        import vision.camera as cam_module
        original = cam_module._cv2
        cam_module._cv2 = mock_cv2

        try:
            from vision.camera import WebcamCapture
            cam = WebcamCapture(device_index=0)
            cam.start()
            r = repr(cam)
            assert "running" in r
        finally:
            cam_module._cv2 = original
