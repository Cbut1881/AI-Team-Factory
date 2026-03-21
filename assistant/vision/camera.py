"""
Webcam capture module for the AI Desktop Assistant.

Provides the :class:`WebcamCapture` class which wraps OpenCV's
``cv2.VideoCapture`` with a clean start/stop lifecycle and a
``capture_frame()`` method that returns a base64-encoded PNG.

Usage
-----
::

    cam = WebcamCapture(device_index=0)
    cam.start()
    try:
        b64_png = cam.capture_frame()
    finally:
        cam.stop()

The class is also a context manager::

    with WebcamCapture() as cam:
        b64_png = cam.capture_frame()

Error handling
--------------
If the camera cannot be opened (missing hardware, driver issue, or
already in use) the module logs a warning and returns ``None`` from
``capture_frame()`` rather than raising.
"""

from __future__ import annotations

import base64
import io
import logging
import threading
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

# cv2 is an optional dependency -- we defer the import so that the
# rest of the assistant can load even when OpenCV is not installed.
_cv2 = None


def _ensure_cv2():
    """Lazily import ``cv2`` and cache the module reference."""
    global _cv2
    if _cv2 is None:
        try:
            import cv2
            _cv2 = cv2
        except ImportError:
            raise ImportError(
                "opencv-python is required for webcam support. "
                "Install it with: pip install opencv-python"
            )
    return _cv2


class WebcamCapture:
    """Thread-safe webcam capture backed by OpenCV.

    Parameters
    ----------
    device_index : int
        OpenCV camera device index (default ``0`` = first camera).
    """

    def __init__(self, device_index: int = 0) -> None:
        self._device_index = device_index
        self._cap = None  # cv2.VideoCapture instance
        self._lock = threading.Lock()
        self._started = False

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def start(self) -> bool:
        """Open the camera device.

        Returns
        -------
        bool
            ``True`` if the camera was opened successfully, ``False``
            otherwise.
        """
        cv2 = _ensure_cv2()

        with self._lock:
            if self._started and self._cap is not None and self._cap.isOpened():
                logger.debug("Camera %d already started", self._device_index)
                return True

            try:
                self._cap = cv2.VideoCapture(self._device_index)
                if not self._cap.isOpened():
                    logger.warning(
                        "Could not open camera at index %d. "
                        "Check that a camera is connected and not in use.",
                        self._device_index,
                    )
                    self._cap = None
                    self._started = False
                    return False

                self._started = True
                logger.info("Camera %d opened successfully", self._device_index)
                return True

            except Exception:
                logger.exception("Failed to start camera %d", self._device_index)
                self._cap = None
                self._started = False
                return False

    def stop(self) -> None:
        """Release the camera device."""
        with self._lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    logger.exception("Error releasing camera %d", self._device_index)
                finally:
                    self._cap = None
                    self._started = False
                    logger.info("Camera %d released", self._device_index)

    @property
    def is_running(self) -> bool:
        """Whether the camera is currently opened and readable."""
        with self._lock:
            return (
                self._started
                and self._cap is not None
                and self._cap.isOpened()
            )

    # -----------------------------------------------------------------
    # Capture
    # -----------------------------------------------------------------

    def capture_frame(self) -> Optional[str]:
        """Capture a single frame and return it as a base64-encoded PNG.

        If the camera is not started this method will attempt to start it
        automatically.

        Returns
        -------
        str or None
            Base64-encoded PNG image data, or ``None`` if capture failed.
        """
        cv2 = _ensure_cv2()

        # Auto-start if needed
        if not self.is_running:
            if not self.start():
                return None

        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                logger.warning("Camera not available for capture")
                return None

            ret, frame = self._cap.read()

        if not ret or frame is None:
            logger.warning("Failed to read frame from camera %d", self._device_index)
            return None

        try:
            # OpenCV returns BGR; convert to RGB for PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")

            logger.debug(
                "Captured frame from camera %d: %dx%d",
                self._device_index,
                img.width,
                img.height,
            )
            return b64

        except Exception:
            logger.exception("Failed to encode webcam frame")
            return None

    # -----------------------------------------------------------------
    # Context manager
    # -----------------------------------------------------------------

    def __enter__(self) -> "WebcamCapture":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    # -----------------------------------------------------------------
    # Repr
    # -----------------------------------------------------------------

    def __repr__(self) -> str:
        state = "running" if self.is_running else "stopped"
        return f"<WebcamCapture(device={self._device_index}, {state})>"
