"""
Animated Robot Avatar Widget
============================
A beautiful, animated robot character rendered with QPainter.
Visual centerpiece of the AI Team Factory assistant.
"""

import enum
import math
import random
import time

from PyQt6.QtCore import (
    QPoint,
    QPointF,
    QRectF,
    QSize,
    QTimer,
    Qt,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QCursor,
    QFont,
    QFontMetrics,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)
from PyQt6.QtWidgets import (
    QApplication,
    QMenu,
    QWidget,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation clamped to [0, 1]."""
    t = max(0.0, min(1.0, t))
    return a + (b - a) * t


def _lerp_color(c1: QColor, c2: QColor, t: float) -> QColor:
    t = max(0.0, min(1.0, t))
    return QColor(
        int(_lerp(c1.red(), c2.red(), t)),
        int(_lerp(c1.green(), c2.green(), t)),
        int(_lerp(c1.blue(), c2.blue(), t)),
        int(_lerp(c1.alpha(), c2.alpha(), t)),
    )


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------

class AvatarState(enum.IntEnum):
    IDLE = 0
    LISTENING = 1
    THINKING = 2
    SPEAKING = 3
    ERROR = 4


# Per-state colour palettes
_STATE_COLORS = {
    AvatarState.IDLE:      QColor(0, 220, 255, 120),   # cyan
    AvatarState.LISTENING: QColor(0, 255, 140, 140),    # green
    AvatarState.THINKING:  QColor(180, 100, 255, 130),  # purple
    AvatarState.SPEAKING:  QColor(0, 200, 255, 140),    # bright cyan
    AvatarState.ERROR:     QColor(255, 80, 80, 130),    # red
}

_ANTENNA_COLORS = {
    AvatarState.IDLE:      QColor(0, 220, 255),
    AvatarState.LISTENING: QColor(0, 255, 120),
    AvatarState.THINKING:  QColor(255, 220, 60),
    AvatarState.SPEAKING:  QColor(0, 220, 255),
    AvatarState.ERROR:     QColor(255, 80, 80),
}


# ---------------------------------------------------------------------------
# AvatarWidget
# ---------------------------------------------------------------------------

class AvatarWidget(QWidget):
    """Frameless, transparent, always-on-top animated robot avatar."""

    # Signals
    clicked_signal = pyqtSignal()
    double_clicked_signal = pyqtSignal()
    right_click_menu_action = pyqtSignal(str)

    # Geometry constants
    WIDTH = 180
    HEIGHT = 200

    def __init__(self, parent=None):
        super().__init__(parent)

        # -- Window setup -----------------------------------------------------
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(self.WIDTH, self.HEIGHT)
        self._move_to_default_position()

        # -- State ------------------------------------------------------------
        self._state = AvatarState.IDLE
        self._target_state = AvatarState.IDLE
        self._state_blend = 1.0  # 1.0 = fully at _state

        self._status_text: str = ""
        self._start_time = time.monotonic()

        # Eye tracking
        self._eye_angle_x = 0.0
        self._eye_angle_y = 0.0
        self._target_eye_x = 0.0
        self._target_eye_y = 0.0

        # Blink
        self._blink_progress = 0.0  # 0 = open, 1 = closed
        self._blinking = False
        self._next_blink = self._schedule_blink()

        # Speaking mouth
        self._mouth_open = 0.0

        # Thinking dots
        self._thinking_dot_phase = 0.0

        # Transition smoothing
        self._glow_color = QColor(_STATE_COLORS[AvatarState.IDLE])

        # Dragging
        self._drag_pos: QPoint | None = None

        # -- Animation timer (30 fps) ----------------------------------------
        self._timer = QTimer(self)
        self._timer.setInterval(33)  # ~30 fps
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # --------------------------------------------------------------------- #
    # Public API                                                              #
    # --------------------------------------------------------------------- #

    def set_state(self, state: AvatarState):
        if state != self._target_state:
            self._target_state = state
            self._state_blend = 0.0

    def set_speaking(self, is_speaking: bool):
        self.set_state(AvatarState.SPEAKING if is_speaking else AvatarState.IDLE)

    def set_listening(self, is_listening: bool):
        self.set_state(AvatarState.LISTENING if is_listening else AvatarState.IDLE)

    def set_status_text(self, text: str):
        self._status_text = text
        self.update()

    def reset_position(self):
        self._move_to_default_position()

    # --------------------------------------------------------------------- #
    # Internal helpers                                                        #
    # --------------------------------------------------------------------- #

    def _move_to_default_position(self):
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            self.move(geo.right() - self.WIDTH - 20,
                      geo.bottom() - self.HEIGHT - 20)

    @staticmethod
    def _schedule_blink() -> float:
        return time.monotonic() + random.uniform(3.0, 5.0)

    def _elapsed(self) -> float:
        return time.monotonic() - self._start_time

    # --------------------------------------------------------------------- #
    # Animation tick                                                          #
    # --------------------------------------------------------------------- #

    def _tick(self):
        dt = 0.033  # ~30 fps
        now = time.monotonic()
        t = self._elapsed()

        # -- State blend transition -------------------------------------------
        if self._state_blend < 1.0:
            self._state_blend = min(1.0, self._state_blend + dt * 3.0)
            if self._state_blend >= 1.0:
                self._state = self._target_state

        active = self._target_state

        # -- Glow colour lerp -------------------------------------------------
        target_glow = _STATE_COLORS[active]
        self._glow_color = _lerp_color(self._glow_color, target_glow, dt * 4.0)

        # -- Eye tracking -----------------------------------------------------
        cursor = QCursor.pos()
        center = self.mapToGlobal(QPoint(self.WIDTH // 2, 70))
        dx = cursor.x() - center.x()
        dy = cursor.y() - center.y()
        dist = math.hypot(dx, dy) + 1e-6
        max_offset = 5.0
        self._target_eye_x = (dx / dist) * min(max_offset, dist * 0.02)
        self._target_eye_y = (dy / dist) * min(max_offset, dist * 0.02)

        # Thinking override: look up-right
        if active == AvatarState.THINKING:
            self._target_eye_x = 3.5
            self._target_eye_y = -3.5

        self._eye_angle_x = _lerp(self._eye_angle_x, self._target_eye_x, dt * 8.0)
        self._eye_angle_y = _lerp(self._eye_angle_y, self._target_eye_y, dt * 8.0)

        # -- Blink ------------------------------------------------------------
        if self._blinking:
            self._blink_progress += dt / 0.075  # 150ms total (75ms each way)
            if self._blink_progress >= 2.0:
                self._blinking = False
                self._blink_progress = 0.0
                self._next_blink = self._schedule_blink()
        elif now >= self._next_blink:
            self._blinking = True
            self._blink_progress = 0.0

        # -- Speaking mouth oscillation ---------------------------------------
        if active == AvatarState.SPEAKING:
            self._mouth_open = 0.5 + 0.5 * math.sin(t * 12.0)
        else:
            self._mouth_open = _lerp(self._mouth_open, 0.0, dt * 10.0)

        # -- Thinking dots phase ----------------------------------------------
        if active == AvatarState.THINKING:
            self._thinking_dot_phase = (self._thinking_dot_phase + dt * 3.0) % 4.0

        self.update()

    # --------------------------------------------------------------------- #
    # Painting                                                                #
    # --------------------------------------------------------------------- #

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        t = self._elapsed()
        active = self._target_state

        # -- Global transforms: floating bob + breathing ----------------------
        bob_y = 3.0 * math.sin(2.0 * math.pi * t / 3.0)
        breath = 1.0 + 0.02 * math.sin(2.0 * math.pi * t / 4.0)

        cx, cy = self.WIDTH / 2.0, self.HEIGHT / 2.0 - 5.0
        p.translate(cx, cy + bob_y)
        p.scale(breath, breath)
        p.translate(-cx, -cy)

        self._draw_outer_glow(p)
        self._draw_body(p)
        self._draw_antenna(p, t, active)
        self._draw_head(p)
        self._draw_eyes(p, active)
        self._draw_mouth(p, active, t)
        self._draw_status_text(p)

        p.end()

    # -- Sub-painters ---------------------------------------------------------

    def _draw_outer_glow(self, p: QPainter):
        """Radial glow around the character, colour driven by state."""
        center = QPointF(self.WIDTH / 2.0, 85.0)
        grad = QRadialGradient(center, 100.0)
        gc = QColor(self._glow_color)
        gc.setAlpha(60)
        grad.setColorAt(0.0, gc)
        gc2 = QColor(gc)
        gc2.setAlpha(0)
        grad.setColorAt(1.0, gc2)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(grad))
        p.drawEllipse(center, 100.0, 100.0)

    def _draw_body(self, p: QPainter):
        """Small trapezoid below head suggesting shoulders."""
        grad = QLinearGradient(60, 140, 120, 165)
        grad.setColorAt(0.0, QColor(30, 35, 80))
        grad.setColorAt(1.0, QColor(60, 30, 100))
        p.setPen(QPen(QColor(0, 200, 255, 40), 1.5))
        p.setBrush(QBrush(grad))
        path = QPainterPath()
        path.moveTo(58, 138)
        path.lineTo(122, 138)
        path.lineTo(132, 165)
        path.lineTo(48, 165)
        path.closeSubpath()
        p.drawPath(path)

    def _draw_antenna(self, p: QPainter, t: float, state: AvatarState):
        """Antenna stalk with pulsing orb."""
        # Stalk
        p.setPen(QPen(QColor(80, 90, 140), 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(90, 35), QPointF(90, 12))

        # Orb pulse
        base_r = 7.0
        pulse = 1.0 + 0.2 * math.sin(t * 4.0)
        r = base_r * pulse

        orb_color = QColor(_ANTENNA_COLORS[state])
        # Thinking: blink effect
        if state == AvatarState.THINKING:
            alpha = int(140 + 115 * math.sin(t * 6.0))
            orb_color.setAlpha(max(0, min(255, alpha)))
        else:
            orb_color.setAlpha(220)

        center = QPointF(90, 10)
        grad = QRadialGradient(center, r * 1.5)
        grad.setColorAt(0.0, QColor(255, 255, 255, 200))
        grad.setColorAt(0.35, orb_color)
        glow_outer = QColor(orb_color)
        glow_outer.setAlpha(0)
        grad.setColorAt(1.0, glow_outer)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(grad))
        p.drawEllipse(center, r * 1.5, r * 1.5)

        # Solid orb core
        p.setBrush(QBrush(orb_color))
        p.drawEllipse(center, r, r)

    def _draw_head(self, p: QPainter):
        """Rounded rectangle head with metallic gradient and cyan border glow."""
        head_rect = QRectF(30, 35, 120, 105)

        # -- Metallic gradient fill -------------------------------------------
        grad = QLinearGradient(head_rect.topLeft(), head_rect.bottomRight())
        grad.setColorAt(0.0, QColor(25, 30, 72))
        grad.setColorAt(0.45, QColor(40, 35, 95))
        grad.setColorAt(1.0, QColor(55, 25, 110))

        # Border glow (outer)
        glow_pen = QPen(QColor(0, 220, 255, 90), 3)
        p.setPen(glow_pen)
        p.setBrush(QBrush(grad))
        p.drawRoundedRect(head_rect, 20, 20)

        # Inner shadow/glow overlay
        inner = QRadialGradient(QPointF(90, 75), 80)
        inner.setColorAt(0.0, QColor(100, 140, 255, 18))
        inner.setColorAt(0.6, QColor(60, 40, 120, 10))
        inner.setColorAt(1.0, QColor(0, 0, 0, 40))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(inner))
        p.drawRoundedRect(head_rect.adjusted(2, 2, -2, -2), 18, 18)

    def _draw_eyes(self, p: QPainter, state: AvatarState):
        """Two animated eyes with irises that track the cursor."""
        left_center = QPointF(65, 75)
        right_center = QPointF(115, 75)
        eye_radius = 16.0

        # Blink squish factor: 1 = open, 0 = line
        if self._blinking:
            if self._blink_progress < 1.0:
                squish = 1.0 - self._blink_progress
            else:
                squish = self._blink_progress - 1.0
        else:
            squish = 1.0

        # Thinking squint
        if state == AvatarState.THINKING:
            squish = min(squish, 0.7)

        for center in (left_center, right_center):
            # Dark eye socket
            socket_rect = QRectF(
                center.x() - eye_radius,
                center.y() - eye_radius * squish,
                eye_radius * 2,
                eye_radius * 2 * squish,
            )
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(10, 12, 30))
            p.drawEllipse(socket_rect)

            if squish > 0.15:
                # Iris
                iris_radius = 8.0
                iris_cx = center.x() + self._eye_angle_x
                iris_cy = center.y() + self._eye_angle_y * squish

                iris_color = QColor(0, 230, 200)
                # Listening: brighter pulsing glow
                if state == AvatarState.LISTENING:
                    pulse = 0.5 + 0.5 * math.sin(self._elapsed() * 5.0)
                    iris_color = _lerp_color(
                        QColor(0, 230, 200), QColor(100, 255, 220), pulse
                    )

                iris_grad = QRadialGradient(
                    QPointF(iris_cx, iris_cy), iris_radius
                )
                iris_grad.setColorAt(0.0, QColor(200, 255, 255, 255))
                iris_grad.setColorAt(0.3, iris_color)
                iris_grad.setColorAt(1.0, QColor(0, 80, 100, 200))

                p.setBrush(QBrush(iris_grad))
                p.drawEllipse(
                    QPointF(iris_cx, iris_cy),
                    iris_radius,
                    iris_radius * squish,
                )

                # Pupil
                p.setBrush(QColor(5, 8, 20))
                pupil_r = 3.5
                p.drawEllipse(
                    QPointF(iris_cx, iris_cy),
                    pupil_r,
                    pupil_r * squish,
                )

                # Specular highlight
                p.setBrush(QColor(255, 255, 255, 210))
                spec_x = iris_cx - 2.5
                spec_y = iris_cy - 3.0 * squish
                p.drawEllipse(QPointF(spec_x, spec_y), 2.2, 1.8 * squish)

            # Listening glow ring around eyes
            if state == AvatarState.LISTENING:
                pulse_a = int(40 + 35 * math.sin(self._elapsed() * 5.0))
                p.setPen(QPen(QColor(0, 255, 180, pulse_a), 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(center, eye_radius + 3, (eye_radius + 3) * squish)

    def _draw_mouth(self, p: QPainter, state: AvatarState, t: float):
        """Mouth changes shape based on state."""
        mouth_cx = 90.0
        mouth_cy = 112.0

        if state == AvatarState.ERROR:
            # Frown
            path = QPainterPath()
            path.moveTo(mouth_cx - 18, mouth_cy + 4)
            path.cubicTo(
                mouth_cx - 8, mouth_cy - 6,
                mouth_cx + 8, mouth_cy - 6,
                mouth_cx + 18, mouth_cy + 4,
            )
            p.setPen(QPen(QColor(255, 100, 100, 220), 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)

        elif state == AvatarState.THINKING:
            # Small 'o' + animated dots
            p.setPen(QPen(QColor(180, 160, 255, 200), 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.setBrush(Qt.BrushStyle.NoBrush)
            o_r = 5.0
            p.drawEllipse(QPointF(mouth_cx, mouth_cy), o_r, o_r * 0.8)

            # Dots
            for i in range(3):
                phase = self._thinking_dot_phase - i * 0.7
                if 0.0 < phase < 1.5:
                    alpha = int(255 * math.sin(phase / 1.5 * math.pi))
                else:
                    alpha = 40
                p.setBrush(QColor(180, 160, 255, max(30, alpha)))
                p.setPen(Qt.PenStyle.NoPen)
                dot_x = mouth_cx + 16 + i * 8
                p.drawEllipse(QPointF(dot_x, mouth_cy), 2.2, 2.2)

        elif self._mouth_open > 0.05:
            # Speaking: open/close rounded rect
            open_h = 6.0 * self._mouth_open
            rect = QRectF(mouth_cx - 14, mouth_cy - open_h, 28, open_h * 2)
            p.setPen(QPen(QColor(0, 200, 255, 180), 1.5))
            p.setBrush(QColor(15, 10, 40, 200))
            p.drawRoundedRect(rect, 6, 6)
            # Tongue / inner glow
            if open_h > 2.0:
                inner_grad = QRadialGradient(QPointF(mouth_cx, mouth_cy), 12)
                inner_grad.setColorAt(0.0, QColor(255, 100, 140, int(80 * self._mouth_open)))
                inner_grad.setColorAt(1.0, QColor(15, 10, 40, 0))
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(inner_grad))
                p.drawRoundedRect(rect.adjusted(2, 2, -2, -2), 4, 4)
        else:
            # Idle smile (bezier curve)
            path = QPainterPath()
            path.moveTo(mouth_cx - 16, mouth_cy - 2)
            path.cubicTo(
                mouth_cx - 6, mouth_cy + 10,
                mouth_cx + 6, mouth_cy + 10,
                mouth_cx + 16, mouth_cy - 2,
            )
            p.setPen(QPen(QColor(0, 220, 255, 180), 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)

    def _draw_status_text(self, p: QPainter):
        if not self._status_text:
            return
        font = QFont("Segoe UI", 8)
        p.setFont(font)
        fm = QFontMetrics(font)
        text_w = fm.horizontalAdvance(self._status_text)
        x = (self.WIDTH - text_w) / 2.0
        y = 188.0
        # Background pill
        pill = QRectF(x - 6, y - 11, text_w + 12, 16)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(15, 12, 35, 180))
        p.drawRoundedRect(pill, 8, 8)
        # Text
        p.setPen(QColor(200, 220, 255, 220))
        p.drawText(QPointF(x, y), self._status_text)

    # --------------------------------------------------------------------- #
    # Mouse / input events                                                    #
    # --------------------------------------------------------------------- #

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
        elif event.button() == Qt.MouseButton.RightButton:
            self._show_context_menu(event.globalPosition().toPoint())
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = None
            self.clicked_signal.emit()
            event.accept()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.double_clicked_signal.emit()
            event.accept()

    def _show_context_menu(self, pos: QPoint):
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu {"
            "  background: #1a1530;"
            "  color: #d0daf0;"
            "  border: 1px solid #3a3560;"
            "  border-radius: 6px;"
            "  padding: 4px;"
            "}"
            "QMenu::item:selected {"
            "  background: #302860;"
            "}"
        )
        for label in ("Chat", "Settings", "Reset Position", "Quit"):
            action = QAction(label, self)
            action.triggered.connect(
                lambda checked, lbl=label: self._on_menu_action(lbl)
            )
            menu.addAction(action)
        menu.exec(pos)

    def _on_menu_action(self, label: str):
        if label == "Reset Position":
            self._move_to_default_position()
        elif label == "Quit":
            QApplication.quit()
        self.right_click_menu_action.emit(label)


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    avatar = AvatarWidget()
    avatar.set_status_text("Hello, human!")
    avatar.show()

    # Cycle through states for demo purposes
    _demo_states = [
        AvatarState.IDLE,
        AvatarState.LISTENING,
        AvatarState.THINKING,
        AvatarState.SPEAKING,
        AvatarState.ERROR,
    ]
    _demo_idx = [0]

    def _cycle():
        _demo_idx[0] = (_demo_idx[0] + 1) % len(_demo_states)
        s = _demo_states[_demo_idx[0]]
        avatar.set_state(s)
        avatar.set_status_text(s.name.capitalize())

    cycle_timer = QTimer()
    cycle_timer.setInterval(3000)
    cycle_timer.timeout.connect(_cycle)
    cycle_timer.start()

    sys.exit(app.exec())
