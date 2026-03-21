"""Nova — AI Desktop Assistant
Entry point for the AI Team Factory's intelligent desktop companion.

Nova is a fully-featured AI assistant that:
- Lives on your screen as an animated avatar
- Sees you via webcam & sees your screen
- Listens and speaks in Thai and English
- Controls your computer on command
- Manages AI teams via the Factory dashboard

Usage:
    python -m assistant.main
    # or
    python assistant/main.py
"""

import sys
import os
import asyncio
import threading
import signal
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Windows DPI awareness
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass

from PyQt6.QtWidgets import QApplication, QSystemTrayIcon, QMenu
from PyQt6.QtGui import QIcon, QAction, QPixmap, QPainter, QColor, QRadialGradient
from PyQt6.QtCore import Qt, QObject, pyqtSignal, pyqtSlot, QTimer

from config import AssistantConfig
from ui.avatar_widget import AvatarWidget, AvatarState
from ui.chat_panel import ChatPanel
from brain.ai_brain import AIBrain
from voice.listener import VoiceListener
from voice.speaker import VoiceSpeaker


class NovaAssistant(QObject):
    """Main coordinator — wires all modules together via Qt signals."""

    response_ready = pyqtSignal(str)

    def __init__(self, app: QApplication):
        super().__init__()
        self.app = app
        self.config = AssistantConfig.load()
        self._processing = False

        # ── Async event loop in background thread ──
        self.loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_loop, daemon=True
        )
        self._loop_thread.start()

        # ── Initialize modules ──
        self._init_brain()
        self._init_ui()
        self._init_voice()
        self._connect_signals()

        # ── System tray ──
        self._init_tray()

        # ── Startup greeting ──
        QTimer.singleShot(1500, self._startup_greeting)

    # ════════════════════════════════════════════════════════════
    #  Initialization
    # ════════════════════════════════════════════════════════════

    def _init_brain(self):
        self.brain = AIBrain(self.config)

    def _init_ui(self):
        self.avatar = AvatarWidget()
        self.chat = ChatPanel()
        self.avatar.show()

    def _init_voice(self):
        self.listener = VoiceListener()
        self.speaker = VoiceSpeaker()

    def _connect_signals(self):
        # Avatar interactions
        self.avatar.clicked_signal.connect(self._on_avatar_clicked)
        self.avatar.double_clicked_signal.connect(self._on_avatar_double_clicked)

        # Chat panel
        self.chat.message_sent.connect(self._on_user_message)
        self.chat.voice_toggled.connect(self._on_voice_toggled)

        # Voice listener
        self.listener.text_recognized.connect(self._on_voice_input)
        self.listener.listening_started.connect(
            lambda: self.avatar.set_state(AvatarState.LISTENING)
        )
        self.listener.listening_stopped.connect(
            lambda: self.avatar.set_state(AvatarState.IDLE)
        )

        # Voice speaker
        self.speaker.speaking_started.connect(
            lambda: self.avatar.set_state(AvatarState.SPEAKING)
        )
        self.speaker.speaking_finished.connect(
            lambda: self.avatar.set_state(AvatarState.IDLE)
        )

        # Response signal (thread-safe bridge)
        self.response_ready.connect(self._deliver_response)

    def _init_tray(self):
        # Create tray icon programmatically
        icon = self._create_tray_icon()
        self.tray = QSystemTrayIcon(icon, self.app)

        menu = QMenu()
        menu.setStyleSheet("""
            QMenu {
                background: #1a1a2e;
                color: #e0e0f0;
                border: 1px solid #2a2a4a;
                padding: 4px;
            }
            QMenu::item:selected {
                background: rgba(0, 240, 255, 0.2);
            }
        """)

        show_avatar = QAction("Show/Hide Avatar", self.app)
        show_avatar.triggered.connect(self._toggle_avatar)
        menu.addAction(show_avatar)

        show_chat = QAction("Open Chat", self.app)
        show_chat.triggered.connect(self.chat.toggle_visibility)
        menu.addAction(show_chat)

        menu.addSeparator()

        reset_pos = QAction("Reset Position", self.app)
        reset_pos.triggered.connect(self.avatar.reset_position)
        menu.addAction(reset_pos)

        menu.addSeparator()

        quit_action = QAction("Quit Nova", self.app)
        quit_action.triggered.connect(self._quit)
        menu.addAction(quit_action)

        self.tray.setContextMenu(menu)
        self.tray.setToolTip("Nova — AI Desktop Assistant")
        self.tray.show()

    def _create_tray_icon(self) -> QIcon:
        """Create a simple tray icon programmatically."""
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background circle
        gradient = QRadialGradient(16, 16, 16)
        gradient.setColorAt(0, QColor(0, 240, 255))
        gradient.setColorAt(1, QColor(123, 47, 247))
        painter.setBrush(gradient)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(2, 2, 28, 28)

        # "N" letter
        painter.setPen(QColor(255, 255, 255))
        font = painter.font()
        font.setPixelSize(18)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "N")
        painter.end()

        return QIcon(pixmap)

    # ════════════════════════════════════════════════════════════
    #  Event Handlers
    # ════════════════════════════════════════════════════════════

    @pyqtSlot()
    def _on_avatar_clicked(self):
        """Toggle voice listening on avatar click."""
        if self.listener.isRunning():
            self.listener.stop_listening()
        else:
            self.listener.start_listening()

    @pyqtSlot()
    def _on_avatar_double_clicked(self):
        """Open chat panel on double click."""
        self.chat.toggle_visibility()

    @pyqtSlot(str)
    def _on_user_message(self, text: str):
        """Handle text message from chat panel."""
        if self._processing:
            return
        self._process_input(text)

    @pyqtSlot(str)
    def _on_voice_input(self, text: str):
        """Handle voice input."""
        if self._processing:
            return
        self.chat.add_message("You", text, "user")
        self._process_input(text)

    @pyqtSlot(bool)
    def _on_voice_toggled(self, enabled: bool):
        """Toggle voice listening."""
        if enabled:
            self.listener.start_listening()
        else:
            self.listener.stop_listening()

    # ════════════════════════════════════════════════════════════
    #  Processing
    # ════════════════════════════════════════════════════════════

    def _process_input(self, text: str):
        """Process user input through AI brain (async in background)."""
        self._processing = True
        self.avatar.set_state(AvatarState.THINKING)
        self.chat.show_typing_indicator()

        # Run in background async loop
        asyncio.run_coroutine_threadsafe(
            self._async_process(text), self.loop
        )

    async def _async_process(self, text: str):
        """Async processing in background thread."""
        try:
            # Process with AI brain
            response = await self.brain.process(text)
            # Emit signal to deliver response on main thread
            self.response_ready.emit(response)
        except Exception as e:
            self.response_ready.emit(f"เกิดข้อผิดพลาด: {str(e)}")

    @pyqtSlot(str)
    def _deliver_response(self, response: str):
        """Deliver AI response to chat and voice (runs on main thread)."""
        self._processing = False
        self.chat.hide_typing_indicator()
        self.chat.add_message("Nova", response, "assistant")

        # Speak the response
        self.speaker.speak(response)

    # ════════════════════════════════════════════════════════════
    #  Helpers
    # ════════════════════════════════════════════════════════════

    def _startup_greeting(self):
        """Show greeting on startup."""
        greeting = "สวัสดีครับ! ผม Nova ผู้ช่วย AI ของคุณ พร้อมทำงานแล้วครับ"
        self.chat.add_message("Nova", greeting, "assistant")
        self.speaker.speak(greeting)

    def _toggle_avatar(self):
        if self.avatar.isVisible():
            self.avatar.hide()
        else:
            self.avatar.show()

    def _run_loop(self):
        """Run asyncio event loop in background thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _quit(self):
        """Clean shutdown."""
        self.listener.stop_listening()
        self.speaker.shutdown()
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.tray.hide()
        self.app.quit()


# ════════════════════════════════════════════════════════════════
#  Entry Point
# ════════════════════════════════════════════════════════════════

def main():
    print("""
    ╔══════════════════════════════════════════════════╗
    ║           NOVA — AI Desktop Assistant             ║
    ║         AI Team Factory Companion                 ║
    ╠══════════════════════════════════════════════════╣
    ║  Click avatar    → Toggle voice listening         ║
    ║  Double-click    → Open chat panel                ║
    ║  Right-click     → Menu                           ║
    ║  Ctrl+Space      → Push-to-talk                   ║
    ╚══════════════════════════════════════════════════╝
    """)

    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)
    app.setApplicationName("Nova")
    app.setQuitOnLastWindowClosed(False)  # Keep running in tray

    # Global dark theme
    app.setStyleSheet("""
        QToolTip {
            background: #1a1a2e;
            color: #e0e0f0;
            border: 1px solid #00f0ff;
            padding: 4px;
            font-size: 12px;
        }
    """)

    nova = NovaAssistant(app)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
