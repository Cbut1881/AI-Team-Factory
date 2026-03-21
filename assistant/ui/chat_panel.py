"""Chat sidebar panel for Nova Assistant.

A frameless, dark-themed chat window that anchors to the right edge of the
screen.  Provides message bubbles, a multi-line input area, voice controls,
and status information.
"""

from __future__ import annotations

import html
import logging
import re
from datetime import datetime
from typing import Optional

from PyQt6.QtCore import (
    QPoint,
    QPropertyAnimation,
    QSize,
    Qt,
    pyqtSignal,
)
from PyQt6.QtGui import QColor, QFont, QGuiApplication, QIcon, QMouseEvent
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PANEL_WIDTH = 400

# Colour palette
COL_BG = "#0d1117"
COL_BG_SECONDARY = "#161b22"
COL_TITLE_BAR = "#0d1117"
COL_BORDER = "#30363d"
COL_TEXT = "#e6edf3"
COL_TEXT_DIM = "#8b949e"
COL_ACCENT_CYAN = "#58a6ff"
COL_ACCENT_PURPLE = "#bc8cff"
COL_USER_BUBBLE = "#0e3a5c"
COL_ASSISTANT_BUBBLE = "#2a1a4e"
COL_SYSTEM_BUBBLE = "#21262d"
COL_TOOL_BG = "#161b22"
COL_INPUT_BG = "#0d1117"
COL_BUTTON_HOVER = "#1f6feb"
COL_STATUS_ONLINE = "#3fb950"
COL_STATUS_OFFLINE = "#f85149"

# ---------------------------------------------------------------------------
# Global stylesheet
# ---------------------------------------------------------------------------

GLOBAL_QSS = f"""
QMainWindow {{
    background-color: {COL_BG};
    border-left: 1px solid {COL_BORDER};
}}
QScrollArea {{
    background-color: transparent;
    border: none;
}}
QScrollBar:vertical {{
    background: {COL_BG_SECONDARY};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {COL_BORDER};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {COL_TEXT_DIM};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: none;
}}
"""


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _format_markdown_lite(text: str) -> str:
    """Very lightweight Markdown-ish formatting to HTML.

    Handles: **bold**, *italic*, `inline code`, and ```code blocks```.
    """
    escaped = html.escape(text)

    # Code blocks (triple backtick)
    escaped = re.sub(
        r"```([\s\S]*?)```",
        lambda m: (
            f'<pre style="background:{COL_TOOL_BG}; padding:6px; '
            f'border-radius:4px; font-family:Consolas,monospace; '
            f'font-size:12px; color:{COL_TEXT};">'
            f"{m.group(1).strip()}</pre>"
        ),
        escaped,
    )
    # Inline code
    escaped = re.sub(
        r"`([^`]+)`",
        lambda m: (
            f'<code style="background:{COL_TOOL_BG}; padding:1px 4px; '
            f'border-radius:3px; font-family:Consolas,monospace; '
            f'font-size:12px;">{m.group(1)}</code>'
        ),
        escaped,
    )
    # Bold
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    # Italic
    escaped = re.sub(r"\*(.+?)\*", r"<i>\1</i>", escaped)
    # Newlines
    escaped = escaped.replace("\n", "<br>")

    return escaped


# ═══════════════════════════════════════════════════════════════════════════
# Title bar
# ═══════════════════════════════════════════════════════════════════════════

class _TitleBar(QWidget):
    """Custom frameless title bar with drag support."""

    close_requested = pyqtSignal()
    minimize_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(40)
        self._drag_pos: Optional[QPoint] = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 8, 0)
        layout.setSpacing(8)

        # Status dot
        self._status_dot = QLabel()
        self._status_dot.setFixedSize(10, 10)
        self._set_dot_color(COL_STATUS_ONLINE)
        layout.addWidget(self._status_dot)

        # Title
        title = QLabel("Nova Assistant")
        title.setStyleSheet(
            f"color: {COL_TEXT}; font-size: 14px; font-weight: 600;"
        )
        layout.addWidget(title)
        layout.addStretch()

        # Minimize button
        btn_min = QPushButton("\u2500")
        btn_min.setFixedSize(28, 28)
        btn_min.setStyleSheet(self._button_style())
        btn_min.clicked.connect(self.minimize_requested.emit)
        layout.addWidget(btn_min)

        # Close button
        btn_close = QPushButton("\u2715")
        btn_close.setFixedSize(28, 28)
        btn_close.setStyleSheet(self._button_style(hover_color="#f85149"))
        btn_close.clicked.connect(self.close_requested.emit)
        layout.addWidget(btn_close)

        self.setStyleSheet(
            f"background-color: {COL_TITLE_BAR}; "
            f"border-bottom: 1px solid {COL_BORDER};"
        )

    # -- Status dot colour -----------------------------------------------

    def set_online(self, online: bool) -> None:
        self._set_dot_color(COL_STATUS_ONLINE if online else COL_STATUS_OFFLINE)

    def _set_dot_color(self, color: str) -> None:
        self._status_dot.setStyleSheet(
            f"background-color: {color}; border-radius: 5px;"
        )

    # -- Button style helper ---------------------------------------------

    @staticmethod
    def _button_style(hover_color: str = COL_BUTTON_HOVER) -> str:
        return (
            f"QPushButton {{ color: {COL_TEXT_DIM}; background: transparent; "
            f"border: none; border-radius: 4px; font-size: 14px; }}"
            f"QPushButton:hover {{ background: {hover_color}; color: {COL_TEXT}; }}"
        )

    # -- Window dragging -------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_pos is not None:
            window = self.window()
            if window is not None:
                delta = event.globalPosition().toPoint() - self._drag_pos
                window.move(window.pos() + delta)
                self._drag_pos = event.globalPosition().toPoint()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_pos = None


# ═══════════════════════════════════════════════════════════════════════════
# Message bubble
# ═══════════════════════════════════════════════════════════════════════════

class _MessageBubble(QFrame):
    """A single chat-message bubble."""

    def __init__(
        self,
        sender: str,
        text: str,
        msg_type: str = "user",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(2)

        # -- Bubble colours / alignment --
        if msg_type == "user":
            bg = COL_USER_BUBBLE
            alignment = Qt.AlignmentFlag.AlignRight
        elif msg_type == "assistant":
            bg = COL_ASSISTANT_BUBBLE
            alignment = Qt.AlignmentFlag.AlignLeft
        elif msg_type == "system":
            bg = COL_SYSTEM_BUBBLE
            alignment = Qt.AlignmentFlag.AlignHCenter
        else:
            bg = COL_BG_SECONDARY
            alignment = Qt.AlignmentFlag.AlignLeft

        # Sender label
        if msg_type != "system":
            sender_label = QLabel(sender)
            sender_label.setStyleSheet(
                f"color: {COL_TEXT_DIM}; font-size: 11px; "
                f"background: transparent; padding: 0;"
            )
            sender_label.setAlignment(alignment)
            layout.addWidget(sender_label)

        # Body
        body = QLabel()
        body.setWordWrap(True)
        body.setTextFormat(Qt.TextFormat.RichText)
        body.setText(_format_markdown_lite(text))
        body.setOpenExternalLinks(True)
        body.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        body.setStyleSheet(
            f"color: {COL_TEXT}; background-color: {bg}; "
            f"border-radius: 10px; padding: 10px 14px; font-size: 13px;"
        )
        body.setMaximumWidth(PANEL_WIDTH - 80)
        layout.addWidget(body, alignment=alignment)

        # Timestamp
        ts = QLabel(datetime.now().strftime("%H:%M"))
        ts.setStyleSheet(
            f"color: {COL_TEXT_DIM}; font-size: 10px; "
            f"background: transparent; padding: 0;"
        )
        ts.setAlignment(alignment)
        layout.addWidget(ts)


# ═══════════════════════════════════════════════════════════════════════════
# Tool log (collapsible)
# ═══════════════════════════════════════════════════════════════════════════

class _ToolLogWidget(QFrame):
    """Collapsible tool execution log."""

    def __init__(
        self,
        tool_name: str,
        result: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)

        self._expanded = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 2, 12, 2)
        outer.setSpacing(0)

        # Header (clickable)
        self._header = QPushButton(f"\u25b6  {tool_name}")
        self._header.setStyleSheet(
            f"QPushButton {{ text-align: left; color: {COL_TEXT_DIM}; "
            f"background: {COL_TOOL_BG}; border: 1px solid {COL_BORDER}; "
            f"border-radius: 6px; padding: 6px 10px; font-size: 12px; "
            f"font-family: Consolas, monospace; }}"
            f"QPushButton:hover {{ background: {COL_BG_SECONDARY}; }}"
        )
        self._header.clicked.connect(self._toggle)
        outer.addWidget(self._header)

        # Content (hidden by default)
        self._content = QLabel(result)
        self._content.setWordWrap(True)
        self._content.setTextFormat(Qt.TextFormat.PlainText)
        self._content.setStyleSheet(
            f"color: {COL_TEXT_DIM}; background: {COL_TOOL_BG}; "
            f"border: 1px solid {COL_BORDER}; border-top: none; "
            f"border-bottom-left-radius: 6px; border-bottom-right-radius: 6px; "
            f"padding: 8px 10px; font-size: 11px; "
            f"font-family: Consolas, monospace;"
        )
        self._content.setVisible(False)
        outer.addWidget(self._content)

        self._tool_name = tool_name

    def _toggle(self) -> None:
        self._expanded = not self._expanded
        self._content.setVisible(self._expanded)
        arrow = "\u25bc" if self._expanded else "\u25b6"
        self._header.setText(f"{arrow}  {self._tool_name}")


# ═══════════════════════════════════════════════════════════════════════════
# Typing indicator
# ═══════════════════════════════════════════════════════════════════════════

class _TypingIndicator(QLabel):
    """Animated "Assistant is typing..." indicator."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Nova is thinking\u2026", parent)
        self.setStyleSheet(
            f"color: {COL_ACCENT_PURPLE}; font-size: 12px; "
            f"font-style: italic; background: transparent; "
            f"padding: 4px 16px;"
        )
        self.setVisible(False)


# ═══════════════════════════════════════════════════════════════════════════
# ChatPanel (main window)
# ═══════════════════════════════════════════════════════════════════════════

class ChatPanel(QMainWindow):
    """Dark-themed chat sidebar anchored to the right edge of the screen.

    Signals
    -------
    message_sent : str
        Emitted when the user submits a message.
    voice_toggled : bool
        Emitted when the voice button is toggled (True = on).
    language_changed : str
        Emitted when the user switches language ("th-TH" or "en-US").
    """

    message_sent = pyqtSignal(str)
    voice_toggled = pyqtSignal(bool)
    language_changed = pyqtSignal(str)
    live_mode_toggled = pyqtSignal(bool)
    camera_toggled = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Nova Assistant")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setFixedWidth(PANEL_WIDTH)
        self.setStyleSheet(GLOBAL_QSS)

        self._voice_on = False
        self._live_on = False
        self._current_lang = "th-TH"

        self._build_ui()
        self._position_on_screen()

    # -- UI construction -------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Title bar
        self._title_bar = _TitleBar()
        self._title_bar.close_requested.connect(self.hide)
        self._title_bar.minimize_requested.connect(self.showMinimized)
        root.addWidget(self._title_bar)

        # Camera preview (hidden by default)
        from ui.live_camera_widget import LiveCameraWidget

        self._camera_widget = LiveCameraWidget()
        self._camera_widget.setVisible(False)
        root.addWidget(self._camera_widget)

        # Message area
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        self._messages_container = QWidget()
        self._messages_layout = QVBoxLayout(self._messages_container)
        self._messages_layout.setContentsMargins(0, 8, 0, 8)
        self._messages_layout.setSpacing(4)
        self._messages_layout.addStretch()

        self._scroll_area.setWidget(self._messages_container)
        root.addWidget(self._scroll_area, stretch=1)

        # Typing indicator
        self._typing_indicator = _TypingIndicator()
        root.addWidget(self._typing_indicator)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color: {COL_BORDER};")
        root.addWidget(sep)

        # Input area
        input_container = QWidget()
        input_container.setStyleSheet(f"background-color: {COL_BG_SECONDARY};")
        input_layout = QVBoxLayout(input_container)
        input_layout.setContentsMargins(10, 8, 10, 8)
        input_layout.setSpacing(6)

        # Text input
        self._input_box = QTextEdit()
        self._input_box.setPlaceholderText("Type a message...")
        self._input_box.setFixedHeight(64)
        self._input_box.setStyleSheet(
            f"QTextEdit {{ color: {COL_TEXT}; background-color: {COL_INPUT_BG}; "
            f"border: 1px solid {COL_BORDER}; border-radius: 8px; "
            f"padding: 8px 10px; font-size: 13px; }}"
            f"QTextEdit:focus {{ border-color: {COL_ACCENT_CYAN}; }}"
        )
        self._input_box.installEventFilter(self)
        input_layout.addWidget(self._input_box)

        # Buttons row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        # Live button
        self._live_btn = QPushButton("\u26a1 Live")
        self._live_btn.setCheckable(True)
        self._live_btn.setFixedHeight(32)
        self._live_btn.setStyleSheet(self._toggle_btn_style(False))
        self._live_btn.clicked.connect(self._on_live_toggled)
        btn_row.addWidget(self._live_btn)

        # Camera toggle button
        self._camera_btn = QPushButton("\U0001f4f7")
        self._camera_btn.setCheckable(True)
        self._camera_btn.setFixedSize(40, 32)
        self._camera_btn.setStyleSheet(self._toggle_btn_style(False))
        self._camera_btn.clicked.connect(self._on_camera_toggled)
        btn_row.addWidget(self._camera_btn)

        # Voice button
        self._voice_btn = QPushButton("\U0001f399 Voice")
        self._voice_btn.setCheckable(True)
        self._voice_btn.setFixedHeight(32)
        self._voice_btn.setStyleSheet(self._toggle_btn_style(False))
        self._voice_btn.clicked.connect(self._on_voice_toggled)
        btn_row.addWidget(self._voice_btn)

        # Language toggle
        self._lang_btn = QPushButton("TH")
        self._lang_btn.setFixedSize(40, 32)
        self._lang_btn.setStyleSheet(
            f"QPushButton {{ color: {COL_TEXT}; background: {COL_BG}; "
            f"border: 1px solid {COL_BORDER}; border-radius: 6px; "
            f"font-size: 12px; font-weight: bold; }}"
            f"QPushButton:hover {{ border-color: {COL_ACCENT_CYAN}; }}"
        )
        self._lang_btn.clicked.connect(self._on_language_toggled)
        btn_row.addWidget(self._lang_btn)

        btn_row.addStretch()

        # Send button
        self._send_btn = QPushButton("Send")
        self._send_btn.setFixedHeight(32)
        self._send_btn.setStyleSheet(
            f"QPushButton {{ color: {COL_BG}; background-color: {COL_ACCENT_CYAN}; "
            f"border: none; border-radius: 6px; padding: 0 20px; "
            f"font-size: 13px; font-weight: 600; }}"
            f"QPushButton:hover {{ background-color: {COL_BUTTON_HOVER}; "
            f"color: {COL_TEXT}; }}"
            f"QPushButton:pressed {{ background-color: #1158c7; }}"
        )
        self._send_btn.clicked.connect(self._on_send)
        btn_row.addWidget(self._send_btn)

        input_layout.addLayout(btn_row)
        root.addWidget(input_container)

        # Status bar
        self._status_bar = QLabel("Ready")
        self._status_bar.setFixedHeight(24)
        self._status_bar.setStyleSheet(
            f"color: {COL_TEXT_DIM}; background-color: {COL_BG}; "
            f"font-size: 11px; padding: 0 12px; "
            f"border-top: 1px solid {COL_BORDER};"
        )
        root.addWidget(self._status_bar)

    # -- Positioning -----------------------------------------------------

    def _position_on_screen(self) -> None:
        """Position the panel on the right edge of the primary screen."""
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        geom = screen.availableGeometry()
        self.setGeometry(
            geom.right() - PANEL_WIDTH + 1,
            geom.top(),
            PANEL_WIDTH,
            geom.height(),
        )

    # -- Event filter (Enter to send) ------------------------------------

    def eventFilter(self, obj, event) -> bool:  # type: ignore[override]
        from PyQt6.QtCore import QEvent
        from PyQt6.QtGui import QKeyEvent

        if obj is self._input_box and event.type() == QEvent.Type.KeyPress:
            key_event: QKeyEvent = event  # type: ignore[assignment]
            if (
                key_event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
                and not key_event.modifiers() & Qt.KeyboardModifier.ShiftModifier
            ):
                self._on_send()
                return True
        return super().eventFilter(obj, event)

    # -- Slots -----------------------------------------------------------

    def _on_send(self) -> None:
        text = self._input_box.toPlainText().strip()
        if not text:
            return
        self._input_box.clear()
        self.add_message("You", text, "user")
        self.message_sent.emit(text)

    def _on_voice_toggled(self) -> None:
        self._voice_on = self._voice_btn.isChecked()
        self._voice_btn.setStyleSheet(self._toggle_btn_style(self._voice_on))
        self.voice_toggled.emit(self._voice_on)
        # If voice is turned off while live mode is on, deactivate live mode
        if not self._voice_on and self._live_on:
            self._live_btn.setChecked(False)
            self._on_live_toggled()

    def _on_live_toggled(self) -> None:
        self._live_on = self._live_btn.isChecked()
        self._live_btn.setStyleSheet(self._live_btn_style(self._live_on))
        self.live_mode_toggled.emit(self._live_on)
        if self._live_on:
            # Activate voice and show camera when entering live mode
            if not self._voice_on:
                self._voice_btn.setChecked(True)
                self._voice_on = True
                self._voice_btn.setStyleSheet(self._toggle_btn_style(True))
                self.voice_toggled.emit(True)
            self.show_camera()
            self._camera_btn.setChecked(True)
        else:
            # Hide camera when leaving live mode
            self.hide_camera()
            self._camera_btn.setChecked(False)

    def _on_camera_toggled(self) -> None:
        active = self._camera_btn.isChecked()
        self._camera_btn.setStyleSheet(self._toggle_btn_style(active))
        self.camera_toggled.emit(active)
        if active:
            self.show_camera()
        else:
            self.hide_camera()

    def _on_language_toggled(self) -> None:
        if self._current_lang == "th-TH":
            self._current_lang = "en-US"
            self._lang_btn.setText("EN")
        else:
            self._current_lang = "th-TH"
            self._lang_btn.setText("TH")
        self.language_changed.emit(self._current_lang)

    # -- Public methods --------------------------------------------------

    def add_message(
        self,
        sender: str,
        text: str,
        msg_type: str = "assistant",
    ) -> None:
        """Add a message bubble to the chat area.

        Parameters
        ----------
        sender:
            Display name (e.g. "You", "Nova", "System").
        text:
            The message body (supports lightweight Markdown).
        msg_type:
            One of ``"user"``, ``"assistant"``, ``"system"``.
        """
        bubble = _MessageBubble(sender, text, msg_type)
        # Insert before the stretch
        count = self._messages_layout.count()
        self._messages_layout.insertWidget(count - 1, bubble)
        self._scroll_to_bottom()

    def add_tool_log(self, tool_name: str, result: str) -> None:
        """Add a collapsible tool execution log entry."""
        widget = _ToolLogWidget(tool_name, result)
        count = self._messages_layout.count()
        self._messages_layout.insertWidget(count - 1, widget)
        self._scroll_to_bottom()

    def set_status(self, text: str) -> None:
        """Update the status bar text."""
        self._status_bar.setText(text)

    def show_typing_indicator(self) -> None:
        """Show the 'typing' animation."""
        self._typing_indicator.setVisible(True)
        self._scroll_to_bottom()

    def hide_typing_indicator(self) -> None:
        """Hide the typing animation."""
        self._typing_indicator.setVisible(False)

    def toggle_visibility(self) -> None:
        """Toggle the panel between visible and hidden."""
        if self.isVisible():
            self.hide()
        else:
            self._position_on_screen()
            self.show()
            self.raise_()
            self.activateWindow()

    def set_connection_status(self, online: bool) -> None:
        """Update the title bar status dot."""
        self._title_bar.set_online(online)

    def show_camera(self) -> None:
        """Show the camera preview widget."""
        self._camera_widget.setVisible(True)

    def hide_camera(self) -> None:
        """Hide the camera preview widget."""
        self._camera_widget.setVisible(False)

    def get_camera_widget(self):
        """Return the LiveCameraWidget instance."""
        return self._camera_widget

    def set_live_mode(self, active: bool) -> None:
        """Programmatically set live mode on or off."""
        if active != self._live_on:
            self._live_btn.setChecked(active)
            self._on_live_toggled()

    # -- Internal helpers ------------------------------------------------

    def _scroll_to_bottom(self) -> None:
        """Scroll the message area to the bottom."""
        vsb = self._scroll_area.verticalScrollBar()
        if vsb is not None:
            # Defer so the layout has time to update
            from PyQt6.QtCore import QTimer

            QTimer.singleShot(50, lambda: vsb.setValue(vsb.maximum()))

    @staticmethod
    def _live_btn_style(active: bool) -> str:
        """Style for the Live button with red/orange accent when active."""
        if active:
            return (
                f"QPushButton {{ color: #ffffff; "
                f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
                f"stop:0 #e63946, stop:1 #f77f00); "
                f"border: none; border-radius: 6px; padding: 0 12px; "
                f"font-size: 12px; font-weight: bold; }}"
                f"QPushButton:hover {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
                f"stop:0 #ff4d5a, stop:1 #ff9a1f); }}"
            )
        return (
            f"QPushButton {{ color: {COL_TEXT_DIM}; background: {COL_BG}; "
            f"border: 1px solid {COL_BORDER}; border-radius: 6px; "
            f"padding: 0 12px; font-size: 12px; }}"
            f"QPushButton:hover {{ border-color: #f77f00; "
            f"color: {COL_TEXT}; }}"
        )

    @staticmethod
    def _toggle_btn_style(active: bool) -> str:
        if active:
            return (
                f"QPushButton {{ color: {COL_BG}; "
                f"background-color: {COL_ACCENT_CYAN}; "
                f"border: none; border-radius: 6px; padding: 0 12px; "
                f"font-size: 12px; }}"
                f"QPushButton:hover {{ background-color: {COL_BUTTON_HOVER}; "
                f"color: {COL_TEXT}; }}"
            )
        return (
            f"QPushButton {{ color: {COL_TEXT_DIM}; background: {COL_BG}; "
            f"border: 1px solid {COL_BORDER}; border-radius: 6px; "
            f"padding: 0 12px; font-size: 12px; }}"
            f"QPushButton:hover {{ border-color: {COL_ACCENT_CYAN}; "
            f"color: {COL_TEXT}; }}"
        )
