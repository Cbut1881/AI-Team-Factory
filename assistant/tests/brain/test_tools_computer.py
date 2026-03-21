"""
Tests for brain.tools -- Computer Control functions (Section A).

Every test mocks pyautogui so that no real mouse / keyboard actions occur.
"""

from __future__ import annotations

import threading
from collections import namedtuple
from unittest.mock import MagicMock, patch

import pyautogui
import pytest

from brain.tools import (
    click,
    double_click,
    get_mouse_position,
    hotkey,
    mouse_move,
    open_application,
    right_click,
    scroll,
    type_text,
    _gui_lock,
)


# ---------------------------------------------------------------------------
# click()
# ---------------------------------------------------------------------------

class TestClick:
    @patch("brain.tools.pyautogui.click")
    def test_click_left_default(self, mock_click):
        result = click(100, 200)
        mock_click.assert_called_once_with(100, 200, button="left")
        assert result["status"] == "ok"
        assert "(100, 200)" in result["result"]

    @patch("brain.tools.pyautogui.click")
    def test_click_right(self, mock_click):
        result = click(50, 75, button="right")
        mock_click.assert_called_once_with(50, 75, button="right")
        assert result["status"] == "ok"
        assert "right" in result["result"]

    @patch("brain.tools.pyautogui.click")
    def test_click_middle(self, mock_click):
        result = click(0, 0, button="middle")
        mock_click.assert_called_once_with(0, 0, button="middle")
        assert result["status"] == "ok"

    @patch("brain.tools.pyautogui.click", side_effect=pyautogui.FailSafeException("fail-safe"))
    def test_click_failsafe(self, mock_click):
        result = click(0, 0)
        assert result["status"] == "error"
        assert "fail-safe" in result["error"].lower() or "FailSafe" in result["error"]

    @patch("brain.tools.pyautogui.click", side_effect=OSError("display error"))
    def test_click_generic_error(self, mock_click):
        result = click(10, 20)
        assert result["status"] == "error"
        assert "display error" in result["error"]


# ---------------------------------------------------------------------------
# double_click()
# ---------------------------------------------------------------------------

class TestDoubleClick:
    @patch("brain.tools.pyautogui.doubleClick")
    def test_double_click_success(self, mock_dc):
        result = double_click(300, 400)
        mock_dc.assert_called_once_with(300, 400)
        assert result["status"] == "ok"
        assert "Double-clicked" in result["result"]

    @patch("brain.tools.pyautogui.doubleClick", side_effect=RuntimeError("boom"))
    def test_double_click_error(self, mock_dc):
        result = double_click(1, 2)
        assert result["status"] == "error"
        assert "boom" in result["error"]


# ---------------------------------------------------------------------------
# right_click()
# ---------------------------------------------------------------------------

class TestRightClick:
    @patch("brain.tools.pyautogui.rightClick")
    def test_right_click_success(self, mock_rc):
        result = right_click(500, 600)
        mock_rc.assert_called_once_with(500, 600)
        assert result["status"] == "ok"
        assert "Right-clicked" in result["result"]

    @patch("brain.tools.pyautogui.rightClick", side_effect=Exception("err"))
    def test_right_click_error(self, mock_rc):
        result = right_click(0, 0)
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# type_text()
# ---------------------------------------------------------------------------

class TestTypeText:
    @patch("brain.tools.pyautogui.typewrite")
    def test_type_ascii(self, mock_tw):
        result = type_text("hello world")
        mock_tw.assert_called_once_with("hello world", interval=0.02)
        assert result["status"] == "ok"
        assert "11" in result["result"]  # 11 characters

    @patch("brain.tools.pyautogui.write")
    def test_type_non_ascii_thai(self, mock_write):
        thai = "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35"  # Thai text
        result = type_text(thai)
        mock_write.assert_called_once_with(thai)
        assert result["status"] == "ok"

    @patch("brain.tools.pyautogui.typewrite")
    def test_type_custom_interval(self, mock_tw):
        result = type_text("abc", interval=0.1)
        mock_tw.assert_called_once_with("abc", interval=0.1)
        assert result["status"] == "ok"

    @patch("brain.tools.pyautogui.typewrite", side_effect=Exception("keyboard error"))
    def test_type_text_error(self, mock_tw):
        result = type_text("fail")
        assert result["status"] == "error"
        assert "keyboard error" in result["error"]

    @patch("brain.tools.pyautogui.typewrite")
    def test_type_empty_string(self, mock_tw):
        result = type_text("")
        # Empty string is ASCII, so typewrite should be called
        mock_tw.assert_called_once_with("", interval=0.02)
        assert result["status"] == "ok"
        assert "0" in result["result"]


# ---------------------------------------------------------------------------
# hotkey()
# ---------------------------------------------------------------------------

class TestHotkey:
    @patch("brain.tools.pyautogui.hotkey")
    def test_hotkey_ctrl_s(self, mock_hk):
        result = hotkey("ctrl", "s")
        mock_hk.assert_called_once_with("ctrl", "s")
        assert result["status"] == "ok"
        assert "ctrl+s" in result["result"]

    @patch("brain.tools.pyautogui.hotkey")
    def test_hotkey_single_key(self, mock_hk):
        result = hotkey("enter")
        mock_hk.assert_called_once_with("enter")
        assert result["status"] == "ok"

    @patch("brain.tools.pyautogui.hotkey")
    def test_hotkey_three_keys(self, mock_hk):
        result = hotkey("ctrl", "shift", "esc")
        mock_hk.assert_called_once_with("ctrl", "shift", "esc")
        assert result["status"] == "ok"
        assert "ctrl+shift+esc" in result["result"]

    @patch("brain.tools.pyautogui.hotkey", side_effect=Exception("key error"))
    def test_hotkey_error(self, mock_hk):
        result = hotkey("invalid")
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# mouse_move()
# ---------------------------------------------------------------------------

class TestMouseMove:
    @patch("brain.tools.pyautogui.moveTo")
    def test_mouse_move_default_duration(self, mock_move):
        result = mouse_move(100, 200)
        mock_move.assert_called_once_with(100, 200, duration=0.3)
        assert result["status"] == "ok"
        assert "(100, 200)" in result["result"]

    @patch("brain.tools.pyautogui.moveTo")
    def test_mouse_move_custom_duration(self, mock_move):
        result = mouse_move(50, 60, duration=1.5)
        mock_move.assert_called_once_with(50, 60, duration=1.5)
        assert result["status"] == "ok"

    @patch("brain.tools.pyautogui.moveTo", side_effect=Exception("move error"))
    def test_mouse_move_error(self, mock_move):
        result = mouse_move(0, 0)
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# scroll()
# ---------------------------------------------------------------------------

class TestScroll:
    @patch("brain.tools.pyautogui.scroll")
    def test_scroll_up(self, mock_scroll):
        result = scroll(5)
        mock_scroll.assert_called_once_with(5)
        assert result["status"] == "ok"
        assert "5" in result["result"]

    @patch("brain.tools.pyautogui.scroll")
    def test_scroll_down(self, mock_scroll):
        result = scroll(-3)
        mock_scroll.assert_called_once_with(-3)
        assert result["status"] == "ok"

    @patch("brain.tools.pyautogui.scroll")
    @patch("brain.tools.pyautogui.moveTo")
    def test_scroll_with_coordinates(self, mock_move, mock_scroll):
        result = scroll(2, x=100, y=200)
        mock_move.assert_called_once_with(100, 200)
        mock_scroll.assert_called_once_with(2)
        assert result["status"] == "ok"

    @patch("brain.tools.pyautogui.scroll")
    @patch("brain.tools.pyautogui.moveTo")
    def test_scroll_without_coordinates_no_move(self, mock_move, mock_scroll):
        result = scroll(1)
        mock_move.assert_not_called()
        mock_scroll.assert_called_once_with(1)
        assert result["status"] == "ok"

    @patch("brain.tools.pyautogui.scroll", side_effect=Exception("scroll err"))
    def test_scroll_error(self, mock_scroll):
        result = scroll(1)
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# open_application()
# ---------------------------------------------------------------------------

class TestOpenApplication:
    @patch("brain.tools.sys")
    @patch("brain.tools.os.startfile")
    def test_open_application_win32(self, mock_startfile, mock_sys):
        mock_sys.platform = "win32"
        result = open_application("notepad")
        mock_startfile.assert_called_once_with("notepad")
        assert result["status"] == "ok"
        assert "notepad" in result["result"]

    @patch("brain.tools.subprocess.Popen")
    @patch("brain.tools.sys")
    def test_open_application_non_win32(self, mock_sys, mock_popen):
        mock_sys.platform = "linux"
        result = open_application("gedit")
        mock_popen.assert_called_once()
        assert result["status"] == "ok"
        assert "gedit" in result["result"]

    @patch("brain.tools.sys")
    @patch("brain.tools.os.startfile", side_effect=FileNotFoundError("not found"))
    def test_open_application_not_found(self, mock_startfile, mock_sys):
        mock_sys.platform = "win32"
        result = open_application("nonexistent_app")
        assert result["status"] == "error"
        assert "not found" in result["error"]


# ---------------------------------------------------------------------------
# get_mouse_position()
# ---------------------------------------------------------------------------

class TestGetMousePosition:
    @patch("brain.tools.pyautogui.position")
    def test_get_mouse_position_success(self, mock_pos):
        Point = namedtuple("Point", ["x", "y"])
        mock_pos.return_value = Point(x=123, y=456)
        result = get_mouse_position()
        assert result["status"] == "ok"
        assert result["result"]["x"] == 123
        assert result["result"]["y"] == 456

    @patch("brain.tools.pyautogui.position", side_effect=Exception("pos error"))
    def test_get_mouse_position_error(self, mock_pos):
        result = get_mouse_position()
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# Thread safety (_gui_lock)
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_gui_lock_is_a_lock(self):
        assert isinstance(_gui_lock, type(threading.Lock()))

    @patch("brain.tools.pyautogui.click")
    def test_concurrent_clicks_serialized(self, mock_click):
        """Two concurrent click calls should not interleave (both succeed)."""
        results = []

        def do_click(x, y):
            results.append(click(x, y))

        t1 = threading.Thread(target=do_click, args=(10, 20))
        t2 = threading.Thread(target=do_click, args=(30, 40))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(results) == 2
        assert all(r["status"] == "ok" for r in results)
        assert mock_click.call_count == 2
