"""
Live Camera Widget
==================
A PyQt6 widget that displays a live webcam feed using
:class:`~assistant.vision.camera.WebcamCapture`.

The widget renders frames at ~30 fps into a QLabel, shows a red "LIVE"
overlay indicator while active, and falls back to a dark placeholder
with a camera-off message when the feed is stopped or unavailable.
"""

from __future__ import annotations

import base64
import logging
from typing import Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from vision.camera import WebcamCapture

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dark-theme palette
# ---------------------------------------------------------------------------
COL_BG = "#0d1117"
COL_BORDER = "#30363d"
COL_TEXT = "#e6edf3"
COL_TEXT_DIM = "#8b949e"
COL_ACCENT_CYAN = "#58a6ff"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_WIDTH = 360
MAX_HEIGHT = 270
FRAME_INTERVAL_MS = 33  # ~30 fps


class LiveCameraWidget(QWidget):
    """Compact live-camera preview widget.

    Parameters
    ----------
    device_index : int
        Camera device index forwarded to :class:`WebcamCapture`.
    parent : QWidget or None
        Optional parent widget.
    """

    def __init__(
        self,
        device_index: int = 0,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._camera: Optional[WebcamCapture] = None
        self._device_index = device_index
        self._active = False

        self.setMaximumSize(MAX_WIDTH, MAX_HEIGHT)
        self.setMinimumSize(160, 120)

        self._setup_ui()
        self._show_placeholder()

        # Timer drives frame capture while active.
        self._timer = QTimer(self)
        self._timer.setInterval(FRAME_INTERVAL_MS)
        self._timer.timeout.connect(self._grab_frame)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        """Build the internal widget tree."""
        self.setStyleSheet(
            f"background-color: {COL_BG}; "
            f"border: 1px solid {COL_BORDER}; "
            f"border-radius: 6px;"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Main video label
        self._video_label = QLabel(self)
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setStyleSheet(
            "border: none; "
            f"background-color: {COL_BG};"
        )
        layout.addWidget(self._video_label)

        # "LIVE" overlay indicator (top-left corner)
        self._live_indicator = QLabel("LIVE", self)
        self._live_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._live_indicator.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        self._live_indicator.setFixedSize(42, 18)
        self._live_indicator.setStyleSheet(
            "background-color: #d32f2f; "
            "color: #ffffff; "
            "border-radius: 4px; "
            "border: none; "
            "padding: 0px 4px;"
        )
        self._live_indicator.move(8, 8)
        self._live_indicator.raise_()
        self._live_indicator.hide()

    # ------------------------------------------------------------------
    # Placeholder
    # ------------------------------------------------------------------

    def _show_placeholder(self) -> None:
        """Display a dark placeholder with a camera-off message."""
        self._video_label.setText("\U0001f4f7 Camera Off")
        self._video_label.setStyleSheet(
            "border: none; "
            f"background-color: {COL_BG}; "
            f"color: {COL_TEXT_DIM}; "
            "font-size: 14px;"
        )
        self._video_label.setPixmap(QPixmap())  # clear any pixmap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Start the live camera feed.

        Returns ``True`` if the camera opened successfully.
        """
        if self._active:
            return True

        try:
            self._camera = WebcamCapture(device_index=self._device_index)
            if not self._camera.start():
                logger.warning("LiveCameraWidget: camera failed to start")
                self._camera = None
                return False
        except Exception:
            logger.exception("LiveCameraWidget: error starting camera")
            self._camera = None
            return False

        self._active = True
        self._video_label.setText("")
        self._video_label.setStyleSheet(
            "border: none; "
            f"background-color: {COL_BG};"
        )
        self._live_indicator.show()
        self._timer.start()
        logger.info("LiveCameraWidget: feed started")
        return True

    def stop(self) -> None:
        """Stop the live camera feed and show the placeholder."""
        self._timer.stop()
        self._active = False
        self._live_indicator.hide()

        if self._camera is not None:
            try:
                self._camera.stop()
            except Exception:
                logger.exception("LiveCameraWidget: error stopping camera")
            finally:
                self._camera = None

        self._show_placeholder()
        logger.info("LiveCameraWidget: feed stopped")

    @property
    def is_active(self) -> bool:
        """Whether the camera feed is currently running."""
        return self._active

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------

    def _grab_frame(self) -> None:
        """Capture a single frame and display it in the label."""
        if not self._active or self._camera is None:
            return

        try:
            b64_png = self._camera.capture_frame()
        except Exception:
            logger.exception("LiveCameraWidget: error capturing frame")
            return

        if b64_png is None:
            return

        try:
            raw = base64.b64decode(b64_png)
            image = QImage()
            if not image.loadFromData(raw):
                return

            pixmap = QPixmap.fromImage(image)
            # Scale while keeping aspect ratio
            scaled = pixmap.scaled(
                self._video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._video_label.setPixmap(scaled)
        except Exception:
            logger.exception("LiveCameraWidget: error rendering frame")

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------

    def resizeEvent(self, event) -> None:  # noqa: N802
        """Reposition the LIVE indicator on resize."""
        super().resizeEvent(event)
        self._live_indicator.move(8, 8)

    def closeEvent(self, event) -> None:  # noqa: N802
        """Ensure the camera is released when the widget is closed."""
        self.stop()
        super().closeEvent(event)
