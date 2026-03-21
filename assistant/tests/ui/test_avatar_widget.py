"""Tests for ui.avatar_widget — AvatarWidget and helpers."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QColor, QMouseEvent
from PyQt6.QtWidgets import QApplication

from ui.avatar_widget import (
    AvatarState,
    AvatarWidget,
    _STATE_COLORS,
    _lerp,
    _lerp_color,
)


# ======================================================================
# Construction
# ======================================================================


class TestConstruction:
    """Verify widget initialisation."""

    def test_window_flags_frameless(self, avatar):
        flags = avatar.windowFlags()
        assert flags & Qt.WindowType.FramelessWindowHint

    def test_window_flags_stays_on_top(self, avatar):
        flags = avatar.windowFlags()
        assert flags & Qt.WindowType.WindowStaysOnTopHint

    def test_window_flags_tool(self, avatar):
        flags = avatar.windowFlags()
        assert flags & Qt.WindowType.Tool

    def test_fixed_size(self, avatar):
        assert avatar.width() == AvatarWidget.WIDTH
        assert avatar.height() == AvatarWidget.HEIGHT

    def test_default_position_is_set(self, avatar):
        """Widget should be positioned on screen (not at 0,0)."""
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            expected_x = geo.right() - AvatarWidget.WIDTH - 20
            expected_y = geo.bottom() - AvatarWidget.HEIGHT - 20
            assert avatar.x() == expected_x
            assert avatar.y() == expected_y


# ======================================================================
# Default state
# ======================================================================


class TestDefaultState:
    """Confirm initial state values."""

    def test_state_is_idle(self, avatar):
        assert avatar._state == AvatarState.IDLE

    def test_target_state_is_idle(self, avatar):
        assert avatar._target_state == AvatarState.IDLE

    def test_state_blend_is_one(self, avatar):
        assert avatar._state_blend == 1.0

    def test_status_text_empty(self, avatar):
        assert avatar._status_text == ""

    def test_drag_pos_none(self, avatar):
        assert avatar._drag_pos is None


# ======================================================================
# set_state
# ======================================================================


class TestSetState:
    """Test state transition logic."""

    def test_set_state_changes_target(self, avatar):
        avatar.set_state(AvatarState.THINKING)
        assert avatar._target_state == AvatarState.THINKING

    def test_set_state_resets_blend(self, avatar):
        avatar.set_state(AvatarState.SPEAKING)
        assert avatar._state_blend == 0.0

    def test_set_state_same_state_no_change(self, avatar):
        """Setting the same state should not reset blend."""
        avatar._state_blend = 0.8
        avatar._target_state = AvatarState.IDLE
        avatar.set_state(AvatarState.IDLE)
        assert avatar._state_blend == 0.8

    def test_set_state_error(self, avatar):
        avatar.set_state(AvatarState.ERROR)
        assert avatar._target_state == AvatarState.ERROR
        assert avatar._state_blend == 0.0


# ======================================================================
# set_speaking / set_listening
# ======================================================================


class TestSpeakingListening:
    """Test convenience state wrappers."""

    def test_set_speaking_true(self, avatar):
        avatar.set_speaking(True)
        assert avatar._target_state == AvatarState.SPEAKING

    def test_set_speaking_false(self, avatar):
        avatar.set_state(AvatarState.SPEAKING)
        avatar.set_speaking(False)
        assert avatar._target_state == AvatarState.IDLE

    def test_set_listening_true(self, avatar):
        avatar.set_listening(True)
        assert avatar._target_state == AvatarState.LISTENING

    def test_set_listening_false(self, avatar):
        avatar.set_state(AvatarState.LISTENING)
        avatar.set_listening(False)
        assert avatar._target_state == AvatarState.IDLE


# ======================================================================
# set_status_text
# ======================================================================


class TestStatusText:

    def test_set_status_text(self, avatar):
        avatar.set_status_text("Processing...")
        assert avatar._status_text == "Processing..."

    def test_set_status_text_empty(self, avatar):
        avatar.set_status_text("hello")
        avatar.set_status_text("")
        assert avatar._status_text == ""


# ======================================================================
# reset_position
# ======================================================================


class TestResetPosition:

    def test_reset_position_calls_default(self, avatar):
        with patch.object(avatar, "_move_to_default_position") as mock_move:
            avatar.reset_position()
            mock_move.assert_called_once()


# ======================================================================
# Signals
# ======================================================================


class TestSignals:

    def test_clicked_signal_on_left_release(self, avatar, qtbot):
        with qtbot.waitSignal(avatar.clicked_signal, timeout=1000):
            # Simulate left button release
            from PyQt6.QtCore import QPointF
            from PyQt6.QtGui import QMouseEvent

            release = QMouseEvent(
                QMouseEvent.Type.MouseButtonRelease,
                QPointF(50, 50),
                QPointF(50, 50),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
            )
            avatar.mouseReleaseEvent(release)

    def test_double_clicked_signal(self, avatar, qtbot):
        with qtbot.waitSignal(avatar.double_clicked_signal, timeout=1000):
            from PyQt6.QtCore import QPointF
            from PyQt6.QtGui import QMouseEvent

            dbl = QMouseEvent(
                QMouseEvent.Type.MouseButtonDblClick,
                QPointF(50, 50),
                QPointF(50, 50),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
            avatar.mouseDoubleClickEvent(dbl)


# ======================================================================
# Animation tick
# ======================================================================


class TestTick:

    def test_tick_increments_blend(self, avatar):
        avatar.set_state(AvatarState.THINKING)
        assert avatar._state_blend == 0.0
        avatar._tick()
        assert avatar._state_blend > 0.0

    def test_tick_blend_caps_at_one(self, avatar):
        avatar.set_state(AvatarState.THINKING)
        # Tick many times to reach 1.0
        for _ in range(200):
            avatar._tick()
        assert avatar._state_blend == 1.0

    def test_tick_completes_state_transition(self, avatar):
        avatar.set_state(AvatarState.ERROR)
        for _ in range(200):
            avatar._tick()
        assert avatar._state == AvatarState.ERROR

    def test_blink_cycle(self, avatar):
        """Force a blink and check progress advances."""
        avatar._blinking = True
        avatar._blink_progress = 0.0
        avatar._tick()
        assert avatar._blink_progress > 0.0

    def test_speaking_mouth_oscillation(self, avatar):
        avatar.set_state(AvatarState.SPEAKING)
        avatar._state_blend = 1.0
        avatar._state = AvatarState.SPEAKING
        avatar._tick()
        assert avatar._mouth_open > 0.0

    def test_timer_interval_33ms(self, avatar):
        assert avatar._timer.interval() == 33


# ======================================================================
# paintEvent (smoke tests)
# ======================================================================


class TestPaintEvent:
    """Ensure paintEvent doesn't crash for each state."""

    @pytest.mark.parametrize("state", list(AvatarState))
    def test_paint_per_state(self, avatar, state):
        avatar._state = state
        avatar._target_state = state
        avatar._state_blend = 1.0
        avatar.repaint()  # triggers paintEvent synchronously


# ======================================================================
# Context menu
# ======================================================================


class TestContextMenu:

    def test_menu_action_reset_position(self, avatar):
        with patch.object(avatar, "_move_to_default_position") as mock_move:
            avatar._on_menu_action("Reset Position")
            mock_move.assert_called_once()

    def test_menu_action_quit(self, avatar):
        with patch.object(QApplication, "quit") as mock_quit:
            avatar._on_menu_action("Quit")
            mock_quit.assert_called_once()

    def test_menu_action_emits_signal(self, avatar, qtbot):
        with qtbot.waitSignal(avatar.right_click_menu_action, timeout=1000):
            avatar._on_menu_action("Chat")


# ======================================================================
# _lerp helper
# ======================================================================


class TestLerp:

    def test_lerp_at_zero(self):
        assert _lerp(10.0, 20.0, 0.0) == 10.0

    def test_lerp_at_half(self):
        assert _lerp(10.0, 20.0, 0.5) == 15.0

    def test_lerp_at_one(self):
        assert _lerp(10.0, 20.0, 1.0) == 20.0

    def test_lerp_clamps_negative(self):
        assert _lerp(10.0, 20.0, -0.5) == 10.0

    def test_lerp_clamps_above_one(self):
        assert _lerp(10.0, 20.0, 1.5) == 20.0


# ======================================================================
# _lerp_color
# ======================================================================


class TestLerpColor:

    def test_lerp_color_at_zero(self):
        c1 = QColor(0, 0, 0, 255)
        c2 = QColor(255, 255, 255, 255)
        result = _lerp_color(c1, c2, 0.0)
        assert result.red() == 0
        assert result.green() == 0
        assert result.blue() == 0

    def test_lerp_color_at_one(self):
        c1 = QColor(0, 0, 0, 255)
        c2 = QColor(255, 255, 255, 255)
        result = _lerp_color(c1, c2, 1.0)
        assert result.red() == 255

    def test_lerp_color_at_half(self):
        c1 = QColor(0, 0, 0, 255)
        c2 = QColor(200, 100, 50, 255)
        result = _lerp_color(c1, c2, 0.5)
        assert result.red() == 100
        assert result.green() == 50
        assert result.blue() == 25


# ======================================================================
# State colours
# ======================================================================


class TestStateColors:

    def test_all_states_have_color(self):
        for state in AvatarState:
            assert state in _STATE_COLORS

    def test_idle_is_cyan(self):
        c = _STATE_COLORS[AvatarState.IDLE]
        # The colour has red=0, blue channel dominant (cyan)
        assert c.red() == 0
        assert c.blue() == 255

    def test_error_is_red(self):
        c = _STATE_COLORS[AvatarState.ERROR]
        assert c.red() == 255


# ======================================================================
# Drag behaviour
# ======================================================================


class TestDragBehaviour:

    def test_press_sets_drag_pos(self, avatar):
        from PyQt6.QtCore import QPointF

        press = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPointF(50, 50),
            avatar.mapToGlobal(QPoint(50, 50)).toPointF(),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        avatar.mousePressEvent(press)
        assert avatar._drag_pos is not None

    def test_release_clears_drag_pos(self, avatar):
        from PyQt6.QtCore import QPointF

        avatar._drag_pos = QPoint(10, 10)
        release = QMouseEvent(
            QMouseEvent.Type.MouseButtonRelease,
            QPointF(50, 50),
            QPointF(50, 50),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        avatar.mouseReleaseEvent(release)
        assert avatar._drag_pos is None
