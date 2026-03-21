"""
Screenshot capture utilities for the AI Desktop Assistant.

Provides functions to capture the full screen or a sub-region, returning
the image as a base64-encoded PNG string suitable for sending to the
Claude vision API.

All images are automatically down-scaled so that the longest axis does
not exceed a configurable maximum (default 1280 px) in order to keep
API payloads reasonable.
"""

from __future__ import annotations

import base64
import ctypes
import io
import logging
import sys
from typing import Optional, Tuple

import pyautogui
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DPI awareness (Windows)
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            logger.debug("Could not set DPI awareness")

# ---------------------------------------------------------------------------
# Default maximum width for screenshots sent to the vision model
# ---------------------------------------------------------------------------
_DEFAULT_MAX_WIDTH = 1280


def _resize_if_needed(img: Image.Image, max_width: int) -> Image.Image:
    """Down-scale *img* so its width does not exceed *max_width*.

    The aspect ratio is preserved.  If the image is already small enough
    it is returned unchanged (no copy).
    """
    if img.width <= max_width:
        return img
    ratio = max_width / img.width
    new_size = (max_width, int(img.height * ratio))
    return img.resize(new_size, Image.LANCZOS)


def _image_to_base64(img: Image.Image) -> str:
    """Encode a PIL Image as a base64 PNG string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def capture_full_screen(max_width: int = _DEFAULT_MAX_WIDTH) -> str:
    """Capture the entire screen and return a base64-encoded PNG.

    Parameters
    ----------
    max_width : int
        Maximum pixel width of the returned image.  Images wider than
        this are resized (preserving aspect ratio) using Lanczos
        resampling.  Default ``1280``.

    Returns
    -------
    str
        Base64-encoded PNG image data.

    Raises
    ------
    RuntimeError
        If the screenshot could not be captured (e.g. no display).
    """
    try:
        img = pyautogui.screenshot()
    except Exception as exc:
        logger.exception("Failed to capture full screen")
        raise RuntimeError(f"Screenshot capture failed: {exc}") from exc

    img = _resize_if_needed(img, max_width)
    b64 = _image_to_base64(img)

    logger.info(
        "Captured full screen: %dx%d (resized to %dx%d)",
        pyautogui.size().width,
        pyautogui.size().height,
        img.width,
        img.height,
    )
    return b64


def capture_region(
    x: int,
    y: int,
    width: int,
    height: int,
    max_width: int = _DEFAULT_MAX_WIDTH,
) -> str:
    """Capture a rectangular region of the screen.

    Parameters
    ----------
    x : int
        Left edge of the capture region (pixels).
    y : int
        Top edge of the capture region (pixels).
    width : int
        Width of the capture region (pixels).
    height : int
        Height of the capture region (pixels).
    max_width : int
        Maximum pixel width of the returned image (default ``1280``).

    Returns
    -------
    str
        Base64-encoded PNG image data.

    Raises
    ------
    ValueError
        If the region dimensions are non-positive.
    RuntimeError
        If the screenshot could not be captured.
    """
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Region dimensions must be positive, got {width}x{height}"
        )

    try:
        img = pyautogui.screenshot(region=(x, y, width, height))
    except Exception as exc:
        logger.exception("Failed to capture region (%d,%d,%d,%d)", x, y, width, height)
        raise RuntimeError(f"Region capture failed: {exc}") from exc

    img = _resize_if_needed(img, max_width)
    b64 = _image_to_base64(img)

    logger.info(
        "Captured region (%d, %d, %d, %d) -> %dx%d",
        x, y, width, height, img.width, img.height,
    )
    return b64


def get_screen_size() -> Tuple[int, int]:
    """Return the primary screen resolution as ``(width, height)``."""
    size = pyautogui.size()
    return (size.width, size.height)
