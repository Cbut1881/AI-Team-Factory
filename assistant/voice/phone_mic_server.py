"""Phone Microphone Server — Use your phone as a wireless microphone.

Runs a small HTTP + WebSocket server. Open the page on your phone's
browser (same WiFi) and it streams mic audio to Nova for speech recognition.

Usage:
    python phone_mic_server.py          # starts on port 8877
    # Then open http://<PC-IP>:8877 on your phone
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import socket
import threading
import wave
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

PORT = 8877
_HTML_PAGE = """<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<title>Nova Mic</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    background: #0d1117;
    color: #e6edf3;
    font-family: 'Segoe UI', sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    overflow: hidden;
}
.container {
    text-align: center;
    padding: 20px;
}
h1 {
    font-size: 24px;
    margin-bottom: 8px;
    background: linear-gradient(135deg, #00f0ff, #7b2ff7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle { color: #8b949e; font-size: 14px; margin-bottom: 30px; }
.mic-btn {
    width: 120px;
    height: 120px;
    border-radius: 50%;
    border: 3px solid #30363d;
    background: radial-gradient(circle, #161b22, #0d1117);
    color: #8b949e;
    font-size: 48px;
    cursor: pointer;
    transition: all 0.3s;
    margin-bottom: 20px;
}
.mic-btn.active {
    border-color: #f85149;
    background: radial-gradient(circle, #2a1a1a, #1a0505);
    color: #f85149;
    animation: pulse 1.5s infinite;
}
.mic-btn.connected {
    border-color: #3fb950;
}
@keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(248,81,73,0.4); }
    50% { box-shadow: 0 0 0 20px rgba(248,81,73,0); }
}
.status {
    font-size: 16px;
    margin-bottom: 10px;
    min-height: 24px;
}
.status.error { color: #f85149; }
.status.ok { color: #3fb950; }
.status.listening { color: #f0883e; }
.level-bar {
    width: 200px;
    height: 6px;
    background: #21262d;
    border-radius: 3px;
    margin: 10px auto;
    overflow: hidden;
}
.level-fill {
    height: 100%;
    width: 0%;
    background: linear-gradient(90deg, #3fb950, #f0883e, #f85149);
    border-radius: 3px;
    transition: width 0.1s;
}
.info { color: #8b949e; font-size: 12px; margin-top: 20px; }
</style>
</head>
<body>
<div class="container">
    <h1>Nova Mic</h1>
    <p class="subtitle">Wireless Microphone</p>
    <button class="mic-btn" id="micBtn" onclick="toggleMic()">&#127908;</button>
    <div class="status" id="status">Tap to start</div>
    <div class="level-bar"><div class="level-fill" id="level"></div></div>
    <div class="info" id="info"></div>
</div>
<script>
let ws = null;
let mediaStream = null;
let audioContext = null;
let processor = null;
let isRecording = false;

const btn = document.getElementById('micBtn');
const status = document.getElementById('status');
const level = document.getElementById('level');
const info = document.getElementById('info');

function toggleMic() {
    if (isRecording) stopMic();
    else startMic();
}

async function startMic() {
    status.textContent = 'Connecting...';
    status.className = 'status';

    // Connect WebSocket
    const wsUrl = `ws://${location.host}/ws`;
    ws = new WebSocket(wsUrl);

    ws.onopen = async () => {
        status.textContent = 'Getting mic...';
        try {
            mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true }
            });

            audioContext = new AudioContext({ sampleRate: 16000 });
            const source = audioContext.createMediaStreamSource(mediaStream);

            // Use ScriptProcessor for compatibility
            processor = audioContext.createScriptProcessor(4096, 1, 1);
            processor.onaudioprocess = (e) => {
                if (!isRecording || ws.readyState !== WebSocket.OPEN) return;
                const data = e.inputBuffer.getChannelData(0);

                // Calculate level
                let sum = 0;
                for (let i = 0; i < data.length; i++) sum += data[i] * data[i];
                const rms = Math.sqrt(sum / data.length);
                level.style.width = Math.min(rms * 500, 100) + '%';

                // Convert to 16-bit PCM
                const pcm = new Int16Array(data.length);
                for (let i = 0; i < data.length; i++) {
                    pcm[i] = Math.max(-32768, Math.min(32767, data[i] * 32768));
                }

                // Send as base64
                const bytes = new Uint8Array(pcm.buffer);
                let binary = '';
                for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
                ws.send(JSON.stringify({ type: 'audio', data: btoa(binary) }));
            };

            source.connect(processor);
            processor.connect(audioContext.destination);

            isRecording = true;
            btn.classList.add('active');
            status.textContent = 'Listening...';
            status.className = 'status listening';
            info.textContent = 'Speak into your phone';
        } catch (err) {
            status.textContent = 'Mic error: ' + err.message;
            status.className = 'status error';
            ws.close();
        }
    };

    ws.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        if (msg.type === 'text') {
            info.textContent = 'Recognized: ' + msg.text;
        } else if (msg.type === 'status') {
            info.textContent = msg.message;
        }
    };

    ws.onerror = () => {
        status.textContent = 'Connection error';
        status.className = 'status error';
    };

    ws.onclose = () => {
        if (isRecording) stopMic();
    };
}

function stopMic() {
    isRecording = false;
    btn.classList.remove('active');
    status.textContent = 'Tap to start';
    status.className = 'status';
    level.style.width = '0%';
    info.textContent = '';

    if (processor) { processor.disconnect(); processor = null; }
    if (audioContext) { audioContext.close(); audioContext = null; }
    if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
    if (ws) { ws.close(); ws = null; }
}
</script>
</body>
</html>"""


class PhoneMicServer:
    """HTTP + WebSocket server that receives audio from a phone browser.

    Parameters
    ----------
    on_audio_text : callable
        Called with recognized text ``(str)`` when speech is detected.
    port : int
        Port to listen on (default 8877).
    language : str
        Language for Google Speech Recognition (default "th-TH").
    """

    def __init__(
        self,
        on_audio_text: Callable[[str], None],
        port: int = PORT,
        language: str = "th-TH",
    ) -> None:
        self._on_text = on_audio_text
        self._port = port
        self._language = language
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._audio_buffer = bytearray()
        self._silence_count = 0
        self._speech_detected = False

    @property
    def url(self) -> str:
        ip = _get_local_ip()
        return f"http://{ip}:{self._port}"

    def start(self) -> None:
        """Start the server in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="PhoneMicServer")
        self._thread.start()
        logger.info("Phone mic server started at %s", self.url)

    def stop(self) -> None:
        """Stop the server."""
        self._running = False

    def _run(self) -> None:
        """Run the async server."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._serve())
        finally:
            loop.close()

    async def _serve(self) -> None:
        """Simple HTTP + WebSocket server using asyncio."""
        import struct

        server = await asyncio.start_server(
            self._handle_connection, "0.0.0.0", self._port
        )
        logger.info("Phone mic listening on port %d", self._port)

        while self._running:
            await asyncio.sleep(0.1)

        server.close()

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle an incoming HTTP or WebSocket connection."""
        try:
            request_line = await asyncio.wait_for(reader.readline(), timeout=5)
            request = request_line.decode("utf-8", errors="replace")
            headers: dict[str, str] = {}

            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=5)
                decoded = line.decode("utf-8", errors="replace").strip()
                if not decoded:
                    break
                if ":" in decoded:
                    key, val = decoded.split(":", 1)
                    headers[key.strip().lower()] = val.strip()

            if "upgrade" in headers.get("connection", "").lower() and \
               headers.get("upgrade", "").lower() == "websocket":
                await self._handle_websocket(reader, writer, headers)
            elif "GET / " in request or "GET /index" in request:
                self._send_html(writer)
            else:
                self._send_html(writer)

        except Exception as exc:
            logger.debug("Connection error: %s", exc)
        finally:
            try:
                writer.close()
            except Exception:
                pass

    def _send_html(self, writer: asyncio.StreamWriter) -> None:
        """Send the HTML page."""
        body = _HTML_PAGE.encode("utf-8")
        header = (
            f"HTTP/1.1 200 OK\r\n"
            f"Content-Type: text/html; charset=utf-8\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        )
        writer.write(header.encode() + body)

    async def _handle_websocket(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        headers: dict[str, str],
    ) -> None:
        """Handle WebSocket upgrade and audio streaming."""
        import hashlib

        # WebSocket handshake
        ws_key = headers.get("sec-websocket-key", "")
        magic = "258EAFA5-E914-47DA-95CA-5AB9FDF5B324"
        accept = base64.b64encode(
            hashlib.sha1((ws_key + magic).encode()).digest()
        ).decode()

        response = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n"
            "\r\n"
        )
        writer.write(response.encode())
        await writer.drain()

        logger.info("Phone mic connected via WebSocket")

        # Send welcome
        await self._ws_send(writer, json.dumps({
            "type": "status", "message": "Connected to Nova!"
        }))

        self._audio_buffer.clear()
        self._silence_count = 0
        self._speech_detected = False

        try:
            while self._running:
                frame = await self._ws_recv(reader)
                if frame is None:
                    break

                try:
                    msg = json.loads(frame)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

                if msg.get("type") == "audio":
                    pcm_bytes = base64.b64decode(msg["data"])
                    self._process_audio_chunk(pcm_bytes, writer)

        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            logger.info("Phone mic disconnected")

    def _process_audio_chunk(
        self, pcm_bytes: bytes, writer: asyncio.StreamWriter
    ) -> None:
        """Process a chunk of PCM audio, detect speech, and recognize."""
        import struct
        import numpy as np

        # Convert to numpy
        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))

        threshold = 800  # Adjust as needed

        if rms > threshold:
            self._speech_detected = True
            self._silence_count = 0
            self._audio_buffer.extend(pcm_bytes)
        elif self._speech_detected:
            self._silence_count += 1
            self._audio_buffer.extend(pcm_bytes)

            # ~1.5s of silence (4096 samples @ 16kHz ≈ 0.256s per chunk)
            if self._silence_count >= 6:
                # Recognize the buffered audio
                if len(self._audio_buffer) > 16000:  # at least 0.5s
                    self._recognize_and_emit(bytes(self._audio_buffer), writer)
                self._audio_buffer.clear()
                self._silence_count = 0
                self._speech_detected = False

    def _recognize_and_emit(
        self, audio_data: bytes, writer: asyncio.StreamWriter
    ) -> None:
        """Send audio to Google Speech Recognition and emit result."""
        import urllib.request

        # Build WAV
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_data)
        wav_data = buf.getvalue()

        url = (
            "http://www.google.com/speech-api/v2/recognize"
            f"?client=chromium&lang={self._language}"
            "&key=AIzaSyBOti4mM-6x9WDnZIjIeyEU21OpBXqWBgw"
        )

        try:
            request = urllib.request.Request(
                url,
                data=wav_data,
                headers={"Content-Type": "audio/l16; rate=16000;"},
            )
            response = urllib.request.urlopen(request, timeout=10)
            response_text = response.read().decode("utf-8")

            for line in response_text.strip().split("\n"):
                if not line.strip():
                    continue
                try:
                    result = json.loads(line)
                    if "result" in result and result["result"]:
                        for r in result["result"]:
                            if "alternative" in r and r["alternative"]:
                                text = r["alternative"][0].get("transcript", "")
                                if text.strip():
                                    logger.info("Phone mic recognized: %s", text)
                                    self._on_text(text.strip())
                                    # Send back to phone
                                    try:
                                        msg = json.dumps({"type": "text", "text": text})
                                        asyncio.get_event_loop().call_soon_threadsafe(
                                            lambda m=msg: writer.write(
                                                self._ws_encode(m)
                                            )
                                        )
                                    except Exception:
                                        pass
                                    return
                except json.JSONDecodeError:
                    continue

        except Exception as exc:
            logger.error("Phone mic recognition error: %s", exc)

    # -- WebSocket helpers -----------------------------------------------

    async def _ws_recv(self, reader: asyncio.StreamReader) -> Optional[str]:
        """Read a single WebSocket text frame."""
        try:
            b1, b2 = await asyncio.wait_for(reader.readexactly(2), timeout=30)
        except (asyncio.TimeoutError, asyncio.IncompleteReadError):
            return None

        opcode = b1 & 0x0F
        if opcode == 0x8:  # Close
            return None

        masked = b2 & 0x80
        length = b2 & 0x7F

        if length == 126:
            data = await reader.readexactly(2)
            length = int.from_bytes(data, "big")
        elif length == 127:
            data = await reader.readexactly(8)
            length = int.from_bytes(data, "big")

        if masked:
            mask = await reader.readexactly(4)
            payload = bytearray(await reader.readexactly(length))
            for i in range(length):
                payload[i] ^= mask[i % 4]
        else:
            payload = await reader.readexactly(length)

        return bytes(payload).decode("utf-8", errors="replace")

    async def _ws_send(self, writer: asyncio.StreamWriter, text: str) -> None:
        """Send a WebSocket text frame."""
        writer.write(self._ws_encode(text))
        await writer.drain()

    @staticmethod
    def _ws_encode(text: str) -> bytes:
        """Encode a string as a WebSocket text frame."""
        data = text.encode("utf-8")
        length = len(data)

        if length < 126:
            header = bytes([0x81, length])
        elif length < 65536:
            header = bytes([0x81, 126]) + length.to_bytes(2, "big")
        else:
            header = bytes([0x81, 127]) + length.to_bytes(8, "big")

        return header + data


def _get_local_ip() -> str:
    """Get this machine's local network IP."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# -- Standalone entry point ----------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    def _on_text(text: str) -> None:
        print(f"  >> {text}")

    server = PhoneMicServer(on_audio_text=_on_text)
    print(f"\n  Nova Phone Mic Server")
    print(f"  Open on your phone: {server.url}")
    print(f"  Press Ctrl+C to stop\n")
    server.start()

    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
        print("\nStopped.")
