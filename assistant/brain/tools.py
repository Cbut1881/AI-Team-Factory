"""
Tool Execution Engine for the AI Desktop Assistant.

Provides every tool that the Claude API can invoke via ``tool_use`` blocks.
Tools are grouped into four categories:

    A) Computer Control  (pyautogui)
    B) Screen / Vision   (screenshots, webcam)
    C) Dashboard API     (requests to localhost:5555)
    D) System            (shell, clipboard, files, sysinfo)

Thread safety
-------------
All pyautogui calls are serialised through ``_gui_lock`` so that concurrent
tool invocations cannot interleave mouse / keyboard actions.

DPI awareness
-------------
On Windows we call ``SetProcessDpiAwareness(2)`` (per-monitor V2) at import
time so that pixel coordinates match the physical display.

Fail-safe
---------
``pyautogui.FAILSAFE`` is always ``True``; moving the mouse to the top-left
corner will raise ``pyautogui.FailSafeException`` and abort any action.
"""

from __future__ import annotations

import base64
import ctypes
import io
import json
import logging
import os
import platform
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import psutil
import pyautogui
import requests
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Platform setup
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            logger.debug("Could not set DPI awareness")

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05  # small pause between pyautogui calls

_gui_lock = threading.Lock()

# Dashboard base URL (overridden at runtime via config)
_DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "http://localhost:5555")

# Timeout for dashboard HTTP calls (seconds)
_DASHBOARD_TIMEOUT = 30

# Maximum screenshot dimension before down-scaling
_SCREENSHOT_MAX_WIDTH = 1280


def _ok(result: Any = None, **kwargs: Any) -> Dict[str, Any]:
    """Return a success envelope."""
    payload: Dict[str, Any] = {"status": "ok"}
    if result is not None:
        payload["result"] = result
    payload.update(kwargs)
    return payload


def _err(message: str, **kwargs: Any) -> Dict[str, Any]:
    """Return an error envelope."""
    payload: Dict[str, Any] = {"status": "error", "error": message}
    payload.update(kwargs)
    return payload


# =========================================================================
# A) COMPUTER CONTROL  (pyautogui)
# =========================================================================

def click(x: int, y: int, button: str = "left") -> Dict[str, Any]:
    """Click the mouse at screen coordinates (*x*, *y*).

    Parameters
    ----------
    x, y : int
        Pixel coordinates on screen.
    button : str
        ``"left"`` (default), ``"right"``, or ``"middle"``.
    """
    try:
        with _gui_lock:
            pyautogui.click(x, y, button=button)
        logger.info("click(%d, %d, button=%s)", x, y, button)
        return _ok(f"Clicked {button} at ({x}, {y})")
    except Exception as exc:
        logger.exception("click failed")
        return _err(str(exc))


def double_click(x: int, y: int) -> Dict[str, Any]:
    """Double-click the left mouse button at (*x*, *y*)."""
    try:
        with _gui_lock:
            pyautogui.doubleClick(x, y)
        logger.info("double_click(%d, %d)", x, y)
        return _ok(f"Double-clicked at ({x}, {y})")
    except Exception as exc:
        logger.exception("double_click failed")
        return _err(str(exc))


def right_click(x: int, y: int) -> Dict[str, Any]:
    """Right-click at (*x*, *y*)."""
    try:
        with _gui_lock:
            pyautogui.rightClick(x, y)
        logger.info("right_click(%d, %d)", x, y)
        return _ok(f"Right-clicked at ({x}, {y})")
    except Exception as exc:
        logger.exception("right_click failed")
        return _err(str(exc))


def type_text(text: str, interval: float = 0.02) -> Dict[str, Any]:
    """Type *text* character-by-character using the keyboard.

    Parameters
    ----------
    text : str
        The string to type.
    interval : float
        Seconds between each keystroke (default 0.02).
    """
    try:
        with _gui_lock:
            pyautogui.typewrite(text, interval=interval) if text.isascii() else pyautogui.write(text)
        logger.info("type_text(len=%d)", len(text))
        return _ok(f"Typed {len(text)} characters")
    except Exception as exc:
        logger.exception("type_text failed")
        return _err(str(exc))


def hotkey(*keys: str) -> Dict[str, Any]:
    """Press a keyboard shortcut (e.g. ``hotkey('ctrl', 's')``).

    Parameters
    ----------
    keys : str
        One or more key names such as ``'ctrl'``, ``'alt'``, ``'shift'``,
        ``'enter'``, ``'tab'``, letter keys, etc.
    """
    try:
        with _gui_lock:
            pyautogui.hotkey(*keys)
        combo = "+".join(keys)
        logger.info("hotkey(%s)", combo)
        return _ok(f"Pressed {combo}")
    except Exception as exc:
        logger.exception("hotkey failed")
        return _err(str(exc))


def mouse_move(x: int, y: int, duration: float = 0.3) -> Dict[str, Any]:
    """Move the mouse cursor to (*x*, *y*) over *duration* seconds."""
    try:
        with _gui_lock:
            pyautogui.moveTo(x, y, duration=duration)
        logger.info("mouse_move(%d, %d)", x, y)
        return _ok(f"Moved mouse to ({x}, {y})")
    except Exception as exc:
        logger.exception("mouse_move failed")
        return _err(str(exc))


def scroll(clicks: int, x: Optional[int] = None, y: Optional[int] = None) -> Dict[str, Any]:
    """Scroll the mouse wheel.

    Parameters
    ----------
    clicks : int
        Positive = scroll up, negative = scroll down.
    x, y : int | None
        If given, move the cursor here before scrolling.
    """
    try:
        with _gui_lock:
            if x is not None and y is not None:
                pyautogui.moveTo(x, y)
            pyautogui.scroll(clicks)
        logger.info("scroll(%d, x=%s, y=%s)", clicks, x, y)
        return _ok(f"Scrolled {clicks} clicks")
    except Exception as exc:
        logger.exception("scroll failed")
        return _err(str(exc))


def open_application(path_or_name: str) -> Dict[str, Any]:
    """Launch an application by file path or executable name.

    Parameters
    ----------
    path_or_name : str
        Full path (``C:\\Windows\\notepad.exe``) or a command available
        on ``PATH`` (``notepad``).
    """
    try:
        if sys.platform == "win32":
            os.startfile(path_or_name)
        else:
            subprocess.Popen(
                [path_or_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        logger.info("open_application(%s)", path_or_name)
        return _ok(f"Opened {path_or_name}")
    except Exception as exc:
        logger.exception("open_application failed")
        return _err(str(exc))


def get_mouse_position() -> Dict[str, Any]:
    """Return the current mouse cursor position as ``{x, y}``."""
    try:
        with _gui_lock:
            pos = pyautogui.position()
        return _ok({"x": pos.x, "y": pos.y})
    except Exception as exc:
        logger.exception("get_mouse_position failed")
        return _err(str(exc))


# =========================================================================
# B) SCREEN / VISION
# =========================================================================

def screenshot(
    region: Optional[Dict[str, int]] = None,
    max_width: int = _SCREENSHOT_MAX_WIDTH,
) -> Dict[str, Any]:
    """Capture a screenshot and return it as a base64-encoded PNG.

    Parameters
    ----------
    region : dict | None
        If provided, must contain ``x``, ``y``, ``width``, ``height`` keys
        to capture a sub-region of the screen.
    max_width : int
        Resize the image so its width does not exceed this value (preserving
        aspect ratio).  Default 1280.
    """
    try:
        if region:
            img = pyautogui.screenshot(
                region=(region["x"], region["y"], region["width"], region["height"]),
            )
        else:
            img = pyautogui.screenshot()

        # Down-scale if needed
        if img.width > max_width:
            ratio = max_width / img.width
            new_size = (max_width, int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        logger.info(
            "screenshot(region=%s, size=%dx%d)", region, img.width, img.height
        )
        return _ok(
            image_base64=b64,
            width=img.width,
            height=img.height,
            media_type="image/png",
        )
    except Exception as exc:
        logger.exception("screenshot failed")
        return _err(str(exc))


def webcam_capture(camera_index: int = 0) -> Dict[str, Any]:
    """Capture a single frame from the webcam and return it as base64 PNG.

    Parameters
    ----------
    camera_index : int
        OpenCV camera device index (default ``0``).
    """
    try:
        import cv2
    except ImportError:
        return _err("opencv-python (cv2) is not installed")

    cap = None
    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return _err(f"Cannot open camera at index {camera_index}")

        ret, frame = cap.read()
        if not ret or frame is None:
            return _err("Failed to capture frame from webcam")

        # Convert BGR -> RGB -> PIL -> PNG bytes
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        logger.info("webcam_capture(index=%d, size=%dx%d)", camera_index, img.width, img.height)
        return _ok(
            image_base64=b64,
            width=img.width,
            height=img.height,
            media_type="image/png",
        )
    except Exception as exc:
        logger.exception("webcam_capture failed")
        return _err(str(exc))
    finally:
        if cap is not None:
            cap.release()


# =========================================================================
# C) DASHBOARD API  (requests to localhost:5555)
# =========================================================================

def _dashboard_get(endpoint: str, params: Optional[dict] = None) -> Dict[str, Any]:
    """Helper: GET request to the dashboard."""
    url = f"{_DASHBOARD_URL}{endpoint}"
    try:
        resp = requests.get(url, params=params, timeout=_DASHBOARD_TIMEOUT)
        resp.raise_for_status()
        return _ok(resp.json())
    except requests.ConnectionError:
        return _err(f"Cannot connect to dashboard at {_DASHBOARD_URL}")
    except Exception as exc:
        logger.exception("Dashboard GET %s failed", endpoint)
        return _err(str(exc))


def _dashboard_post(endpoint: str, payload: Optional[dict] = None) -> Dict[str, Any]:
    """Helper: POST request to the dashboard."""
    url = f"{_DASHBOARD_URL}{endpoint}"
    try:
        resp = requests.post(url, json=payload or {}, timeout=_DASHBOARD_TIMEOUT)
        resp.raise_for_status()
        return _ok(resp.json())
    except requests.ConnectionError:
        return _err(f"Cannot connect to dashboard at {_DASHBOARD_URL}")
    except Exception as exc:
        logger.exception("Dashboard POST %s failed", endpoint)
        return _err(str(exc))


def _dashboard_delete(endpoint: str) -> Dict[str, Any]:
    """Helper: DELETE request to the dashboard."""
    url = f"{_DASHBOARD_URL}{endpoint}"
    try:
        resp = requests.delete(url, timeout=_DASHBOARD_TIMEOUT)
        resp.raise_for_status()
        return _ok(resp.json())
    except requests.ConnectionError:
        return _err(f"Cannot connect to dashboard at {_DASHBOARD_URL}")
    except Exception as exc:
        logger.exception("Dashboard DELETE %s failed", endpoint)
        return _err(str(exc))


# --- Agent management ---

def list_agents() -> Dict[str, Any]:
    """List all agents registered in the AI Team Factory dashboard."""
    return _dashboard_get("/api/agents")


def create_agent(
    name: str,
    role: str,
    model: str,
    system_prompt: str = "",
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """Create a new agent in the dashboard.

    Parameters
    ----------
    name : str
        Display name for the agent.
    role : str
        Role description (e.g. ``"researcher"``, ``"coder"``).
    model : str
        LLM model identifier.
    system_prompt : str
        Custom system prompt for the agent.
    temperature : float
        Sampling temperature.
    """
    return _dashboard_post("/api/agents", {
        "name": name,
        "role": role,
        "model": model,
        "system_prompt": system_prompt,
        "temperature": temperature,
    })


def delete_agent(agent_name: str) -> Dict[str, Any]:
    """Delete an agent by name."""
    return _dashboard_delete(f"/api/agents/{agent_name}")


# --- Team management ---

def list_teams() -> Dict[str, Any]:
    """List all teams registered in the dashboard."""
    return _dashboard_get("/api/teams")


def create_team(name: str, agents: List[str], workflow: str = "sequential") -> Dict[str, Any]:
    """Create a new team of agents.

    Parameters
    ----------
    name : str
        Team name.
    agents : list[str]
        List of agent names to include.
    workflow : str
        Workflow type: ``"sequential"``, ``"parallel"``, ``"debate"``.
    """
    return _dashboard_post("/api/teams", {
        "name": name,
        "agents": agents,
        "workflow": workflow,
    })


# --- Model management ---

def list_models() -> Dict[str, Any]:
    """List available LLM models (Ollama + cloud)."""
    return _dashboard_get("/api/models")


# --- Run modes ---

def run_ask(agent_name: str, prompt: str) -> Dict[str, Any]:
    """Send a prompt to a single agent and return the response.

    Parameters
    ----------
    agent_name : str
        Name of the agent to query.
    prompt : str
        User prompt / question.
    """
    return _dashboard_post("/api/run/ask", {
        "agent": agent_name,
        "prompt": prompt,
    })


def run_pipeline(team_name: str, prompt: str) -> Dict[str, Any]:
    """Run a sequential pipeline through a team.

    Parameters
    ----------
    team_name : str
        Name of the team.
    prompt : str
        Initial prompt fed to the first agent.
    """
    return _dashboard_post("/api/run/pipeline", {
        "team": team_name,
        "prompt": prompt,
    })


def run_parallel(team_name: str, prompt: str) -> Dict[str, Any]:
    """Run all agents in a team in parallel on the same prompt.

    Parameters
    ----------
    team_name : str
        Name of the team.
    prompt : str
        Prompt sent to every agent simultaneously.
    """
    return _dashboard_post("/api/run/parallel", {
        "team": team_name,
        "prompt": prompt,
    })


def run_debate(team_name: str, prompt: str, rounds: int = 3) -> Dict[str, Any]:
    """Run a multi-round debate among team agents.

    Parameters
    ----------
    team_name : str
        Name of the team.
    prompt : str
        Debate topic / question.
    rounds : int
        Number of debate rounds (default 3).
    """
    return _dashboard_post("/api/run/debate", {
        "team": team_name,
        "prompt": prompt,
        "rounds": rounds,
    })


# --- Training ---

def train_distill(
    teacher_model: str,
    student_model: str,
    dataset: str,
    epochs: int = 3,
) -> Dict[str, Any]:
    """Start a knowledge-distillation training run.

    Parameters
    ----------
    teacher_model : str
        Model name of the teacher.
    student_model : str
        Model name of the student to fine-tune.
    dataset : str
        Dataset identifier.
    epochs : int
        Number of training epochs.
    """
    return _dashboard_post("/api/train/distill", {
        "teacher_model": teacher_model,
        "student_model": student_model,
        "dataset": dataset,
        "epochs": epochs,
    })


def train_agent(agent_name: str, dataset: str, epochs: int = 3) -> Dict[str, Any]:
    """Fine-tune a specific agent on a dataset.

    Parameters
    ----------
    agent_name : str
        Agent to train.
    dataset : str
        Dataset identifier.
    epochs : int
        Number of training epochs.
    """
    return _dashboard_post("/api/train/agent", {
        "agent": agent_name,
        "dataset": dataset,
        "epochs": epochs,
    })


def train_exam(agent_name: str, exam: str) -> Dict[str, Any]:
    """Run an evaluation exam on an agent.

    Parameters
    ----------
    agent_name : str
        Agent to evaluate.
    exam : str
        Exam identifier.
    """
    return _dashboard_post("/api/train/exam", {
        "agent": agent_name,
        "exam": exam,
    })


def train_full(
    agent_name: str,
    dataset: str,
    exam: str,
    epochs: int = 3,
) -> Dict[str, Any]:
    """Run a full training pipeline: train then evaluate.

    Parameters
    ----------
    agent_name : str
        Agent to train and evaluate.
    dataset : str
        Dataset identifier.
    exam : str
        Exam identifier.
    epochs : int
        Number of training epochs.
    """
    return _dashboard_post("/api/train/full", {
        "agent": agent_name,
        "dataset": dataset,
        "exam": exam,
        "epochs": epochs,
    })


# --- Dashboard info ---

def get_dashboard_system_info() -> Dict[str, Any]:
    """Retrieve system information from the dashboard API."""
    return _dashboard_get("/api/system_info")


def get_datasets() -> Dict[str, Any]:
    """List available training datasets."""
    return _dashboard_get("/api/datasets")


def get_exams() -> Dict[str, Any]:
    """List available evaluation exams."""
    return _dashboard_get("/api/exams")


# =========================================================================
# D) SYSTEM
# =========================================================================

def run_command(command: str, timeout: int = 60) -> Dict[str, Any]:
    """Execute a shell command and return its output.

    Parameters
    ----------
    command : str
        Shell command string.
    timeout : int
        Maximum seconds to wait (default 60).
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        logger.info("run_command(%r) -> returncode=%d", command, result.returncode)
        return _ok({
            "stdout": result.stdout[-10_000:] if result.stdout else "",
            "stderr": result.stderr[-5_000:] if result.stderr else "",
            "returncode": result.returncode,
        })
    except subprocess.TimeoutExpired:
        return _err(f"Command timed out after {timeout}s")
    except Exception as exc:
        logger.exception("run_command failed")
        return _err(str(exc))


def open_url(url: str) -> Dict[str, Any]:
    """Open a URL in the default web browser.

    Parameters
    ----------
    url : str
        The URL to open.
    """
    try:
        webbrowser.open(url)
        logger.info("open_url(%s)", url)
        return _ok(f"Opened {url}")
    except Exception as exc:
        logger.exception("open_url failed")
        return _err(str(exc))


def get_clipboard() -> Dict[str, Any]:
    """Read the current clipboard text content."""
    try:
        import pyperclip
        text = pyperclip.paste()
        return _ok(text)
    except ImportError:
        # Fallback: pyautogui does not support clipboard directly on all OS
        try:
            if sys.platform == "win32":
                import ctypes
                from ctypes import wintypes
                CF_UNICODETEXT = 13
                user32 = ctypes.windll.user32
                kernel32 = ctypes.windll.kernel32
                user32.OpenClipboard(0)
                try:
                    handle = user32.GetClipboardData(CF_UNICODETEXT)
                    if handle:
                        kernel32.GlobalLock.restype = ctypes.c_wchar_p
                        text = kernel32.GlobalLock(handle)
                        kernel32.GlobalUnlock(handle)
                        return _ok(text or "")
                    return _ok("")
                finally:
                    user32.CloseClipboard()
            else:
                result = subprocess.run(
                    ["xclip", "-selection", "clipboard", "-o"],
                    capture_output=True, text=True, timeout=5,
                )
                return _ok(result.stdout)
        except Exception as exc:
            return _err(f"Clipboard read failed: {exc}")
    except Exception as exc:
        logger.exception("get_clipboard failed")
        return _err(str(exc))


def set_clipboard(text: str) -> Dict[str, Any]:
    """Write text to the system clipboard.

    Parameters
    ----------
    text : str
        Content to place on the clipboard.
    """
    try:
        import pyperclip
        pyperclip.copy(text)
        logger.info("set_clipboard(len=%d)", len(text))
        return _ok(f"Copied {len(text)} characters to clipboard")
    except ImportError:
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["clip"],
                    input=text,
                    text=True,
                    timeout=5,
                    check=True,
                )
            else:
                subprocess.run(
                    ["xclip", "-selection", "clipboard"],
                    input=text,
                    text=True,
                    timeout=5,
                    check=True,
                )
            return _ok(f"Copied {len(text)} characters to clipboard")
        except Exception as exc:
            return _err(f"Clipboard write failed: {exc}")
    except Exception as exc:
        logger.exception("set_clipboard failed")
        return _err(str(exc))


def file_read(path: str, max_chars: int = 100_000) -> Dict[str, Any]:
    """Read a text file and return its contents.

    Parameters
    ----------
    path : str
        Absolute or relative file path.
    max_chars : int
        Maximum number of characters to return (default 100 000).
    """
    try:
        p = Path(path).resolve()
        if not p.exists():
            return _err(f"File not found: {p}")
        if not p.is_file():
            return _err(f"Not a file: {p}")
        text = p.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n... [truncated at {max_chars} chars]"
        logger.info("file_read(%s, len=%d)", p, len(text))
        return _ok({"path": str(p), "content": text, "size": p.stat().st_size})
    except Exception as exc:
        logger.exception("file_read failed")
        return _err(str(exc))


def file_write(path: str, content: str, create_dirs: bool = True) -> Dict[str, Any]:
    """Write text content to a file.

    Parameters
    ----------
    path : str
        Destination file path.
    content : str
        Text to write.
    create_dirs : bool
        Create parent directories if they do not exist (default True).
    """
    try:
        p = Path(path).resolve()
        if create_dirs:
            p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        logger.info("file_write(%s, len=%d)", p, len(content))
        return _ok(f"Wrote {len(content)} characters to {p}")
    except Exception as exc:
        logger.exception("file_write failed")
        return _err(str(exc))


def get_system_info() -> Dict[str, Any]:
    """Return local system information (CPU, RAM, disk, platform)."""
    try:
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/") if sys.platform != "win32" else psutil.disk_usage("C:\\")
        info = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": psutil.cpu_count(logical=True),
            "cpu_percent": psutil.cpu_percent(interval=0.5),
            "ram_total_gb": round(mem.total / (1024 ** 3), 2),
            "ram_used_gb": round(mem.used / (1024 ** 3), 2),
            "ram_percent": mem.percent,
            "disk_total_gb": round(disk.total / (1024 ** 3), 2),
            "disk_used_gb": round(disk.used / (1024 ** 3), 2),
            "disk_percent": disk.percent,
        }
        return _ok(info)
    except Exception as exc:
        logger.exception("get_system_info failed")
        return _err(str(exc))


# =========================================================================
# TOOL DISPATCH TABLE
# =========================================================================

TOOL_FUNCTIONS: Dict[str, callable] = {
    # A) Computer Control
    "click": click,
    "double_click": double_click,
    "right_click": right_click,
    "type_text": type_text,
    "hotkey": hotkey,
    "mouse_move": mouse_move,
    "scroll": scroll,
    "open_application": open_application,
    "get_mouse_position": get_mouse_position,
    # B) Screen / Vision
    "screenshot": screenshot,
    "webcam_capture": webcam_capture,
    # C) Dashboard API
    "list_agents": list_agents,
    "list_teams": list_teams,
    "list_models": list_models,
    "create_agent": create_agent,
    "create_team": create_team,
    "delete_agent": delete_agent,
    "run_ask": run_ask,
    "run_pipeline": run_pipeline,
    "run_parallel": run_parallel,
    "run_debate": run_debate,
    "train_distill": train_distill,
    "train_agent": train_agent,
    "train_exam": train_exam,
    "train_full": train_full,
    "get_dashboard_system_info": get_dashboard_system_info,
    "get_datasets": get_datasets,
    "get_exams": get_exams,
    # D) System
    "run_command": run_command,
    "open_url": open_url,
    "get_clipboard": get_clipboard,
    "set_clipboard": set_clipboard,
    "file_read": file_read,
    "file_write": file_write,
    "get_system_info": get_system_info,
}


def execute_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Look up a tool by *name* and call it with *arguments*.

    This is the single entry-point that the Claude API response handler
    should invoke when processing ``tool_use`` content blocks.
    """
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        logger.error("Unknown tool requested: %s", name)
        return _err(f"Unknown tool: {name}")
    try:
        return func(**arguments)
    except TypeError as exc:
        logger.exception("Bad arguments for tool %s", name)
        return _err(f"Invalid arguments for {name}: {exc}")
    except Exception as exc:
        logger.exception("Tool %s raised an exception", name)
        return _err(f"Tool {name} failed: {exc}")


# =========================================================================
# TOOL DEFINITIONS  (Claude API tool_use schema)
# =========================================================================

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    # ------------------------------------------------------------------
    # A) Computer Control
    # ------------------------------------------------------------------
    {
        "name": "click",
        "description": (
            "Click the mouse at the given screen coordinates. "
            "Use this to interact with buttons, links, and UI elements."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {
                    "type": "integer",
                    "description": "X pixel coordinate on screen.",
                },
                "y": {
                    "type": "integer",
                    "description": "Y pixel coordinate on screen.",
                },
                "button": {
                    "type": "string",
                    "enum": ["left", "right", "middle"],
                    "description": "Mouse button to click (default: left).",
                },
            },
            "required": ["x", "y"],
        },
    },
    {
        "name": "double_click",
        "description": "Double-click the left mouse button at the given screen coordinates.",
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "X pixel coordinate."},
                "y": {"type": "integer", "description": "Y pixel coordinate."},
            },
            "required": ["x", "y"],
        },
    },
    {
        "name": "right_click",
        "description": "Right-click at the given screen coordinates to open a context menu.",
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "X pixel coordinate."},
                "y": {"type": "integer", "description": "Y pixel coordinate."},
            },
            "required": ["x", "y"],
        },
    },
    {
        "name": "type_text",
        "description": (
            "Type text using the keyboard. The text is typed character by "
            "character as if the user were pressing keys."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to type.",
                },
                "interval": {
                    "type": "number",
                    "description": "Seconds between each keystroke (default 0.02).",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "hotkey",
        "description": (
            "Press a keyboard shortcut combination. Pass each key as a "
            "separate element in the 'keys' array. Examples: "
            "[\"ctrl\", \"s\"] for save, [\"alt\", \"tab\"] to switch windows, "
            "[\"ctrl\", \"shift\", \"esc\"] for task manager."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of key names to press simultaneously.",
                },
            },
            "required": ["keys"],
        },
    },
    {
        "name": "mouse_move",
        "description": "Move the mouse cursor to the specified screen coordinates.",
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "Target X coordinate."},
                "y": {"type": "integer", "description": "Target Y coordinate."},
                "duration": {
                    "type": "number",
                    "description": "Duration in seconds for the movement (default 0.3).",
                },
            },
            "required": ["x", "y"],
        },
    },
    {
        "name": "scroll",
        "description": (
            "Scroll the mouse wheel. Positive clicks scroll up, negative "
            "scroll down. Optionally move the cursor first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "clicks": {
                    "type": "integer",
                    "description": "Number of scroll clicks (positive=up, negative=down).",
                },
                "x": {
                    "type": "integer",
                    "description": "Optional X coordinate to move cursor to before scrolling.",
                },
                "y": {
                    "type": "integer",
                    "description": "Optional Y coordinate to move cursor to before scrolling.",
                },
            },
            "required": ["clicks"],
        },
    },
    {
        "name": "open_application",
        "description": (
            "Launch an application by its executable path or name. "
            "Examples: 'notepad', 'C:\\\\Program Files\\\\app.exe', 'calc'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path_or_name": {
                    "type": "string",
                    "description": "Full path to the executable or a command name on PATH.",
                },
            },
            "required": ["path_or_name"],
        },
    },
    {
        "name": "get_mouse_position",
        "description": "Get the current mouse cursor position on screen.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    # ------------------------------------------------------------------
    # B) Screen / Vision
    # ------------------------------------------------------------------
    {
        "name": "screenshot",
        "description": (
            "Capture a screenshot of the entire screen or a specific region. "
            "Returns a base64-encoded PNG image. Use this to see what is on "
            "screen before taking actions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer", "description": "Left edge X."},
                        "y": {"type": "integer", "description": "Top edge Y."},
                        "width": {"type": "integer", "description": "Width in pixels."},
                        "height": {"type": "integer", "description": "Height in pixels."},
                    },
                    "required": ["x", "y", "width", "height"],
                    "description": "Optional sub-region to capture.",
                },
                "max_width": {
                    "type": "integer",
                    "description": "Maximum image width in pixels (default 1280). Images wider than this are resized.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "webcam_capture",
        "description": (
            "Capture a single photo from the webcam. Returns a base64-encoded "
            "PNG image."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "camera_index": {
                    "type": "integer",
                    "description": "Camera device index (default 0).",
                },
            },
            "required": [],
        },
    },
    # ------------------------------------------------------------------
    # C) Dashboard API
    # ------------------------------------------------------------------
    {
        "name": "list_agents",
        "description": "List all AI agents registered in the Team Factory dashboard.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "list_teams",
        "description": "List all teams registered in the Team Factory dashboard.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "list_models",
        "description": "List all available LLM models (local Ollama + cloud providers).",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "create_agent",
        "description": "Create a new AI agent in the dashboard with a given name, role, and model.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Display name for the agent."},
                "role": {"type": "string", "description": "Role description (e.g. 'researcher', 'coder')."},
                "model": {"type": "string", "description": "LLM model identifier."},
                "system_prompt": {"type": "string", "description": "Custom system prompt for the agent."},
                "temperature": {"type": "number", "description": "Sampling temperature (default 0.7)."},
            },
            "required": ["name", "role", "model"],
        },
    },
    {
        "name": "create_team",
        "description": "Create a new team of agents with a specified workflow type.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Team name."},
                "agents": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of agent names to include in the team.",
                },
                "workflow": {
                    "type": "string",
                    "enum": ["sequential", "parallel", "debate"],
                    "description": "Workflow type (default: sequential).",
                },
            },
            "required": ["name", "agents"],
        },
    },
    {
        "name": "delete_agent",
        "description": "Delete an agent from the dashboard by name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_name": {"type": "string", "description": "Name of the agent to delete."},
            },
            "required": ["agent_name"],
        },
    },
    {
        "name": "run_ask",
        "description": "Send a prompt to a single agent and get a response.",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_name": {"type": "string", "description": "Name of the agent to query."},
                "prompt": {"type": "string", "description": "The question or instruction to send."},
            },
            "required": ["agent_name", "prompt"],
        },
    },
    {
        "name": "run_pipeline",
        "description": (
            "Run a sequential pipeline through a team. Each agent processes "
            "the output of the previous agent."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "team_name": {"type": "string", "description": "Name of the team."},
                "prompt": {"type": "string", "description": "Initial prompt for the pipeline."},
            },
            "required": ["team_name", "prompt"],
        },
    },
    {
        "name": "run_parallel",
        "description": "Run all agents in a team simultaneously on the same prompt and collect all responses.",
        "input_schema": {
            "type": "object",
            "properties": {
                "team_name": {"type": "string", "description": "Name of the team."},
                "prompt": {"type": "string", "description": "Prompt sent to all agents."},
            },
            "required": ["team_name", "prompt"],
        },
    },
    {
        "name": "run_debate",
        "description": "Run a multi-round debate among team agents on a topic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "team_name": {"type": "string", "description": "Name of the team."},
                "prompt": {"type": "string", "description": "Debate topic or question."},
                "rounds": {"type": "integer", "description": "Number of debate rounds (default 3)."},
            },
            "required": ["team_name", "prompt"],
        },
    },
    {
        "name": "train_distill",
        "description": "Start a knowledge distillation training run from a teacher model to a student model.",
        "input_schema": {
            "type": "object",
            "properties": {
                "teacher_model": {"type": "string", "description": "Model name of the teacher."},
                "student_model": {"type": "string", "description": "Model name of the student."},
                "dataset": {"type": "string", "description": "Dataset identifier."},
                "epochs": {"type": "integer", "description": "Number of training epochs (default 3)."},
            },
            "required": ["teacher_model", "student_model", "dataset"],
        },
    },
    {
        "name": "train_agent",
        "description": "Fine-tune a specific agent on a training dataset.",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_name": {"type": "string", "description": "Agent to train."},
                "dataset": {"type": "string", "description": "Dataset identifier."},
                "epochs": {"type": "integer", "description": "Number of training epochs (default 3)."},
            },
            "required": ["agent_name", "dataset"],
        },
    },
    {
        "name": "train_exam",
        "description": "Run an evaluation exam on an agent to measure its performance.",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_name": {"type": "string", "description": "Agent to evaluate."},
                "exam": {"type": "string", "description": "Exam identifier."},
            },
            "required": ["agent_name", "exam"],
        },
    },
    {
        "name": "train_full",
        "description": "Run a full training pipeline: train an agent on a dataset then evaluate with an exam.",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_name": {"type": "string", "description": "Agent to train and evaluate."},
                "dataset": {"type": "string", "description": "Dataset identifier."},
                "exam": {"type": "string", "description": "Exam identifier."},
                "epochs": {"type": "integer", "description": "Number of training epochs (default 3)."},
            },
            "required": ["agent_name", "dataset", "exam"],
        },
    },
    {
        "name": "get_dashboard_system_info",
        "description": "Get system information from the AI Team Factory dashboard.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_datasets",
        "description": "List all available training datasets in the dashboard.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_exams",
        "description": "List all available evaluation exams in the dashboard.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    # ------------------------------------------------------------------
    # D) System
    # ------------------------------------------------------------------
    {
        "name": "run_command",
        "description": (
            "Execute a shell command and return stdout, stderr, and the exit "
            "code. Use this for file operations, git commands, pip install, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute."},
                "timeout": {
                    "type": "integer",
                    "description": "Maximum seconds to wait for the command (default 60).",
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "open_url",
        "description": "Open a URL in the user's default web browser.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to open."},
            },
            "required": ["url"],
        },
    },
    {
        "name": "get_clipboard",
        "description": "Read the current text content from the system clipboard.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "set_clipboard",
        "description": "Copy text to the system clipboard.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to copy to the clipboard."},
            },
            "required": ["text"],
        },
    },
    {
        "name": "file_read",
        "description": "Read the contents of a text file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read."},
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return (default 100000).",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "file_write",
        "description": "Write text content to a file, creating parent directories if needed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Destination file path."},
                "content": {"type": "string", "description": "Text content to write."},
                "create_dirs": {
                    "type": "boolean",
                    "description": "Create parent directories if they don't exist (default true).",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "get_system_info",
        "description": (
            "Get local system information including CPU usage, RAM, disk space, "
            "and platform details."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]
