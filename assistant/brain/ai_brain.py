"""
AI Brain - Core Reasoning Engine for Nova Assistant
====================================================

The central intelligence module that powers Nova, a bilingual (Thai/English)
desktop AI assistant. Implements a full agentic loop with Claude as primary
LLM and Ollama as local fallback.

Architecture:
    User Input -> AIBrain.process() -> Claude API (tool_use) -> ToolExecutor
    -> tool_result -> Claude API -> ... -> final text response

The agentic loop continues until Claude returns a text response without
requesting further tool calls, enabling multi-step reasoning and action.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import anthropic
import httpx

from brain.tools import TOOL_DEFINITIONS, execute_tool
from config import AssistantConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_CONVERSATION_MESSAGES: int = 50
"""Hard ceiling on conversation turns kept in memory."""

CONTEXT_WINDOW_KEEP_RECENT: int = 20
"""Number of recent messages to always preserve verbatim."""

SUMMARY_TRIGGER_COUNT: int = 30
"""When total messages exceed this, older messages get summarized."""

MAX_AGENTIC_ITERATIONS: int = 25
"""Safety limit on tool-use loop iterations to prevent runaway agents."""

DEFAULT_MODEL: str = "claude-sonnet-4-20250514"
"""Default Claude model to use for reasoning."""

MAX_TOKENS: int = 8192
"""Maximum tokens for Claude responses."""

OLLAMA_DEFAULT_MODEL: str = "llama3.1:8b"
"""Default Ollama model for local fallback."""

OLLAMA_BASE_URL: str = "http://localhost:11434"
"""Default Ollama API endpoint."""


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = """\
You are Nova — a friendly, highly capable AI assistant built into AI Team Factory, \
a desktop application for managing AI-powered development teams.

## Identity & Personality
- Your name is **Nova**. You are warm, confident, and genuinely helpful.
- You speak naturally, like a knowledgeable colleague — never robotic or overly formal.
- You have a calm, can-do attitude. Complex problems excite you; you break them down clearly.
- You use light humor when appropriate but always stay professional and respectful.

## Language Behavior
- You are **bilingual in Thai (ไทย) and English**.
- **Always match the user's language.** If they write in Thai, respond entirely in Thai. \
If they write in English, respond in English. If they mix, follow their dominant language.
- When speaking Thai, use natural conversational Thai — not stiff textbook Thai. \
Use particles like ครับ/ค่ะ, นะ, เลย where natural.

## Capabilities
You are a desktop AI assistant with powerful abilities:
1. **Screen Awareness** — You can see the user's screen and understand what's displayed.
2. **Voice Interaction** — You can hear the user speak and respond naturally.
3. **Computer Control** — You can control the mouse, keyboard, open applications, \
   manage files, and execute system commands.
4. **AI Team Management** — You can create, configure, and orchestrate teams of AI agents \
   for software development tasks through AI Team Factory.
5. **Web & Research** — You can browse the web, search for information, and gather data.
6. **Code & Development** — You can read, write, and modify code; run scripts; \
   manage git repositories; and assist with the full software development lifecycle.

## Behavioral Guidelines
- **Be proactive**: Anticipate what the user might need next and suggest it.
- **Explain your actions**: When performing tasks, briefly narrate what you're doing \
  so the user can follow along (e.g., "Let me take a look at your screen..." or \
  "กำลังเปิดไฟล์ให้นะครับ...").
- **Ask before destructive actions**: Always confirm before deleting files, closing \
  unsaved work, modifying system settings, or any irreversible operation. \
  Example: "This will delete 3 files permanently. Should I proceed?"
- **Handle errors gracefully**: If something fails, explain what happened in plain \
  language and suggest alternatives. Never show raw tracebacks to the user.
- **Respect privacy**: Never read or transmit sensitive information (passwords, tokens, \
  private keys) unless the user explicitly asks you to handle them for a specific task.

## Response Style
- Keep responses concise but complete. Avoid walls of text.
- Use markdown formatting when it aids readability (bullet points, code blocks, headers).
- For multi-step tasks, outline the plan first, then execute step by step.
- When showing code, always specify the language for syntax highlighting.

## Tool Usage
- Use the tools available to you to accomplish tasks. Prefer action over explanation.
- When multiple tools are needed, chain them logically and efficiently.
- If a tool fails, try an alternative approach before reporting failure.
- After completing a tool-based task, summarize what was done and the result.
"""


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

class MessageRole(str, Enum):
    """Valid message roles for the conversation."""
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ConversationMessage:
    """A single message in the conversation history."""
    role: str
    content: Any  # str | list[dict] for multimodal content
    timestamp: float = field(default_factory=time.time)
    token_estimate: int = 0

    def to_api_format(self) -> dict[str, Any]:
        """Convert to the format expected by the Anthropic messages API."""
        return {"role": self.role, "content": self.content}


# ---------------------------------------------------------------------------
# AIBrain
# ---------------------------------------------------------------------------

class AIBrain:
    """Core reasoning engine for the Nova assistant.

    Orchestrates conversation with Claude (primary) or Ollama (fallback),
    manages conversation history with smart context windowing, and implements
    the full agentic tool-use loop.

    Usage::

        config = Config()
        brain = AIBrain(config)
        response = await brain.process("What files are on my desktop?")
        response = await brain.process_with_screen("What am I looking at?")
    """

    def __init__(self, config: AssistantConfig) -> None:
        self.config = config
        self.model: str = config.claude.model or DEFAULT_MODEL
        self.max_tokens: int = config.claude.max_tokens or MAX_TOKENS
        self.preferred_language: Optional[str] = None
        self.conversation: list[ConversationMessage] = []
        self.execute_tool = execute_tool
        self._summary_cache: Optional[str] = None

        # Primary: Claude via Anthropic SDK
        api_key: Optional[str] = config.anthropic_api_key or None
        if api_key:
            self.client: Optional[anthropic.Anthropic] = anthropic.Anthropic(
                api_key=api_key,
            )
        else:
            self.client = None
            logger.warning(
                "No CLAUDE_API_KEY configured — Claude API unavailable, "
                "will rely on Ollama fallback."
            )

        # Fallback: Ollama
        self.ollama_model: str = OLLAMA_DEFAULT_MODEL
        self.ollama_base_url: str = config.ollama_base_url or OLLAMA_BASE_URL

        logger.info(
            "AIBrain initialized | model=%s | ollama_fallback=%s",
            self.model,
            self.ollama_model,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process(
        self,
        user_input: str,
        images: Optional[list[str]] = None,
    ) -> str:
        """Process user input through the full agentic loop.

        Args:
            user_input: The user's text message.
            images: Optional list of base64-encoded images (PNG/JPEG)
                    or file paths to include with the message.

        Returns:
            The assistant's final text response after all tool execution
            is complete.

        The agentic loop:
            1. Build user message (with optional images) and append to history.
            2. Send conversation + tool definitions to Claude.
            3. If the response contains ``tool_use`` blocks, execute each tool
               via ``ToolExecutor``, collect results.
            4. Append assistant response and tool results to conversation.
            5. Send updated conversation back to Claude.
            6. Repeat steps 3-5 until Claude responds with pure text.
            7. Return the final text.
        """
        # Build the user content block
        content: list[dict[str, Any]] = self._build_user_content(
            user_input, images
        )

        # Append user message to conversation
        self.conversation.append(
            ConversationMessage(
                role=MessageRole.USER,
                content=content,
                token_estimate=self._estimate_tokens(user_input),
            )
        )

        # Try Claude first, fall back to Ollama
        if self.client is not None:
            try:
                return await self._agentic_loop()
            except (
                anthropic.AuthenticationError,
                anthropic.PermissionDeniedError,
            ) as exc:
                logger.error("Claude auth error: %s", exc)
                return await self._fallback_ollama(user_input)
            except anthropic.RateLimitError as exc:
                logger.warning("Claude rate limited: %s", exc)
                return await self._fallback_ollama(user_input)
            except (
                anthropic.APIConnectionError,
                anthropic.APITimeoutError,
            ) as exc:
                logger.warning("Claude connection error: %s", exc)
                return await self._fallback_ollama(user_input)
            except anthropic.APIStatusError as exc:
                logger.error("Claude API error %d: %s", exc.status_code, exc)
                if exc.status_code >= 500:
                    return await self._fallback_ollama(user_input)
                raise
        else:
            return await self._fallback_ollama(user_input)

    async def process_with_screen(self, user_input: str) -> str:
        """Process input with an automatic screenshot included.

        Captures the current screen, encodes it as base64 PNG, and sends
        it alongside the user's text message for visual understanding.

        Args:
            user_input: The user's text message about what's on screen.

        Returns:
            The assistant's response with screen-aware context.
        """
        screenshot_b64: Optional[str] = await self._capture_screen()
        images: Optional[list[str]] = (
            [screenshot_b64] if screenshot_b64 else None
        )
        return await self.process(user_input, images=images)

    async def process_with_camera(self, user_input: str) -> str:
        """Process input with an automatic webcam capture included.

        Captures the current webcam frame, encodes it as base64 JPEG,
        and sends it alongside the user's text message.

        Args:
            user_input: The user's text message about what's visible.

        Returns:
            The assistant's response with camera-aware context.
        """
        camera_b64: Optional[str] = await self._capture_camera()
        images: Optional[list[str]] = (
            [camera_b64] if camera_b64 else None
        )
        return await self.process(user_input, images=images)

    def reset_conversation(self) -> None:
        """Clear all conversation history and cached summaries."""
        self.conversation.clear()
        self._summary_cache = None
        logger.info("Conversation history cleared.")

    def set_language(self, lang: str) -> None:
        """Set the preferred response language.

        Args:
            lang: Language code, e.g. ``"th"`` for Thai, ``"en"`` for English.
                  Set to ``None`` to return to auto-detection.
        """
        self.preferred_language = lang
        logger.info("Preferred language set to: %s", lang)

    # ------------------------------------------------------------------
    # Agentic Loop (Core)
    # ------------------------------------------------------------------

    async def _agentic_loop(self) -> str:
        """Execute the agentic tool-use loop until a final text response.

        Returns:
            The final text content from Claude's response.

        Raises:
            RuntimeError: If the loop exceeds MAX_AGENTIC_ITERATIONS.
        """
        for iteration in range(MAX_AGENTIC_ITERATIONS):
            logger.debug("Agentic loop iteration %d", iteration + 1)

            # Build messages with context windowing
            messages: list[dict[str, Any]] = self._build_messages()
            system_prompt = self._build_system_prompt()

            # Call Claude (sync client, run in executor to avoid blocking)
            response: anthropic.types.Message = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.messages.create(  # type: ignore[union-attr]
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=system_prompt,
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                ),
            )

            # Parse response content blocks
            assistant_content: list[dict[str, Any]] = []
            tool_use_blocks: list[dict[str, Any]] = []
            final_text_parts: list[str] = []

            for block in response.content:
                if block.type == "text":
                    assistant_content.append({
                        "type": "text",
                        "text": block.text,
                    })
                    final_text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_block = {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                    assistant_content.append(tool_block)
                    tool_use_blocks.append(tool_block)

            # Append assistant message to conversation
            self.conversation.append(
                ConversationMessage(
                    role=MessageRole.ASSISTANT,
                    content=assistant_content,
                )
            )

            # If no tool calls, we have our final response
            if response.stop_reason == "end_turn" or not tool_use_blocks:
                final_text = "\n".join(final_text_parts).strip()
                if not final_text:
                    final_text = "(Nova completed the requested actions.)"
                return final_text

            # Execute each tool and collect results
            tool_results: list[dict[str, Any]] = []
            for tool_block in tool_use_blocks:
                tool_result = await self._execute_tool(
                    tool_name=tool_block["name"],
                    tool_input=tool_block["input"],
                    tool_use_id=tool_block["id"],
                )
                tool_results.append(tool_result)

            # Append tool results as a user message
            self.conversation.append(
                ConversationMessage(
                    role=MessageRole.USER,
                    content=tool_results,
                )
            )

        # Safety: loop exhausted
        logger.error(
            "Agentic loop exceeded %d iterations — forcing stop.",
            MAX_AGENTIC_ITERATIONS,
        )
        return (
            "I've been working on this for a while and need to pause. "
            "Could you clarify what you'd like me to focus on?"
        )

    async def _execute_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_use_id: str,
    ) -> dict[str, Any]:
        """Execute a single tool call and format the result.

        Args:
            tool_name: Name of the tool to execute.
            tool_input: Input parameters for the tool.
            tool_use_id: The unique ID from Claude's tool_use block.

        Returns:
            A tool_result content block ready to send back to Claude.
        """
        logger.info("Executing tool: %s(%s)", tool_name, _truncate_repr(tool_input))

        try:
            result: Any = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.execute_tool(tool_name, tool_input),
            )
        except Exception as exc:
            logger.error(
                "Tool %s failed: %s\n%s",
                tool_name,
                exc,
                traceback.format_exc(),
            )
            return {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "is_error": True,
                "content": f"Tool execution failed: {type(exc).__name__}: {exc}",
            }

        # Format result — handle image results specially
        content = self._format_tool_result(result)

        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
        }

    def _format_tool_result(self, result: Any) -> Union[str, list[dict[str, Any]]]:
        """Format a tool execution result for the Claude API.

        Handles plain text, dicts with image data, and structured results.

        Args:
            result: Raw result from ToolExecutor.

        Returns:
            A string or list of content blocks suitable for a tool_result.
        """
        if result is None:
            return "Done (no output)."

        # If result is a dict with an image key, return multimodal content
        if isinstance(result, dict):
            if "image_base64" in result:
                blocks: list[dict[str, Any]] = []
                media_type = result.get("media_type", "image/png")
                blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": result["image_base64"],
                    },
                })
                if "text" in result:
                    blocks.append({"type": "text", "text": result["text"]})
                return blocks
            # Regular dict — serialize to JSON
            return json.dumps(result, indent=2, ensure_ascii=False, default=str)

        if isinstance(result, (list, tuple)):
            return json.dumps(result, indent=2, ensure_ascii=False, default=str)

        return str(result)

    # ------------------------------------------------------------------
    # Message Building & Context Management
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Build the system prompt, optionally with language preference."""
        prompt = SYSTEM_PROMPT
        if self.preferred_language:
            lang_map = {"th": "Thai (ภาษาไทย)", "en": "English"}
            lang_name = lang_map.get(self.preferred_language, self.preferred_language)
            prompt += (
                f"\n\n## Language Override\n"
                f"The user has set their preferred language to **{lang_name}**. "
                f"Always respond in {lang_name} unless they switch languages."
            )
        return prompt

    def _build_messages(self) -> list[dict[str, Any]]:
        """Build the messages list with smart context windowing.

        Strategy:
            - If total messages <= CONTEXT_WINDOW_KEEP_RECENT, send all.
            - Otherwise, summarize older messages into a context preamble
              and keep the most recent messages verbatim.
            - Enforce MAX_CONVERSATION_MESSAGES hard limit by trimming.

        Returns:
            List of message dicts ready for the Anthropic messages API.
        """
        # Hard trim if conversation is too long
        if len(self.conversation) > MAX_CONVERSATION_MESSAGES:
            overflow = len(self.conversation) - MAX_CONVERSATION_MESSAGES
            self._summarize_and_trim(overflow)

        total = len(self.conversation)

        if total <= CONTEXT_WINDOW_KEEP_RECENT:
            # Everything fits — send it all
            return [msg.to_api_format() for msg in self.conversation]

        # Split into old and recent
        split_idx = total - CONTEXT_WINDOW_KEEP_RECENT
        old_messages = self.conversation[:split_idx]
        recent_messages = self.conversation[split_idx:]

        # Build a summary of old messages if not cached
        if self._summary_cache is None or split_idx > 0:
            self._summary_cache = self._create_summary(old_messages)

        # Construct message list: summary as first user message, then recent
        messages: list[dict[str, Any]] = []

        if self._summary_cache:
            messages.append({
                "role": "user",
                "content": (
                    f"[Conversation context summary — earlier messages condensed]\n\n"
                    f"{self._summary_cache}"
                ),
            })
            messages.append({
                "role": "assistant",
                "content": (
                    "Understood, I have the context from our earlier conversation. "
                    "Let's continue."
                ),
            })

        for msg in recent_messages:
            messages.append(msg.to_api_format())

        # Ensure messages start with a user role (API requirement)
        if messages and messages[0]["role"] != "user":
            messages.insert(0, {
                "role": "user",
                "content": "(Continuing our conversation.)",
            })

        return messages

    def _summarize_and_trim(self, count: int) -> None:
        """Remove the oldest messages, updating the summary cache.

        Args:
            count: Number of oldest messages to remove.
        """
        if count <= 0:
            return

        removed = self.conversation[:count]
        self.conversation = self.conversation[count:]

        # Build summary of removed messages
        summary_parts: list[str] = []
        if self._summary_cache:
            summary_parts.append(self._summary_cache)

        new_summary = self._create_summary(removed)
        if new_summary:
            summary_parts.append(new_summary)

        self._summary_cache = "\n".join(summary_parts) if summary_parts else None
        logger.debug(
            "Trimmed %d old messages. Remaining: %d",
            count,
            len(self.conversation),
        )

    @staticmethod
    def _create_summary(messages: list[ConversationMessage]) -> str:
        """Create a text summary of a list of conversation messages.

        This is a lightweight extractive summary — it pulls key content
        from each message without calling an LLM (to avoid recursion).

        Args:
            messages: Messages to summarize.

        Returns:
            A condensed text summary.
        """
        summary_lines: list[str] = []

        for msg in messages:
            role_label = "User" if msg.role == "user" else "Nova"

            if isinstance(msg.content, str):
                text = msg.content[:200]
                summary_lines.append(f"- {role_label}: {text}")
            elif isinstance(msg.content, list):
                # Extract text blocks, skip images and tool internals
                texts: list[str] = []
                has_tools = False
                for block in msg.content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            texts.append(block["text"][:150])
                        elif block.get("type") == "tool_use":
                            has_tools = True
                            texts.append(
                                f"[Called tool: {block.get('name', '?')}]"
                            )
                        elif block.get("type") == "tool_result":
                            content = block.get("content", "")
                            if isinstance(content, str):
                                texts.append(f"[Tool result: {content[:100]}]")
                            else:
                                texts.append("[Tool returned result]")
                        elif block.get("type") == "image":
                            texts.append("[Image included]")

                if texts:
                    combined = " | ".join(texts)
                    summary_lines.append(f"- {role_label}: {combined}")

        return "\n".join(summary_lines)

    # ------------------------------------------------------------------
    # User Content Building
    # ------------------------------------------------------------------

    def _build_user_content(
        self,
        text: str,
        images: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Build a multimodal content block for a user message.

        Args:
            text: The user's text input.
            images: Optional list of base64-encoded images or file paths.

        Returns:
            A list of content blocks (text + images).
        """
        content: list[dict[str, Any]] = []

        # Add images first so Claude "sees" them before reading the text
        if images:
            for img in images:
                image_data = self._resolve_image(img)
                if image_data:
                    content.append(image_data)

        # Add text
        content.append({"type": "text", "text": text})

        return content

    @staticmethod
    def _resolve_image(image_source: str) -> Optional[dict[str, Any]]:
        """Resolve an image source to an API-ready image content block.

        Args:
            image_source: Either a base64 string or a file path.

        Returns:
            An image content block dict, or None on failure.
        """
        # Detect if it's a file path
        if len(image_source) < 500 and Path(image_source).suffix.lower() in (
            ".png", ".jpg", ".jpeg", ".gif", ".webp",
        ):
            try:
                path = Path(image_source)
                if path.exists():
                    data = base64.b64encode(path.read_bytes()).decode("utf-8")
                    suffix = path.suffix.lower()
                    media_types = {
                        ".png": "image/png",
                        ".jpg": "image/jpeg",
                        ".jpeg": "image/jpeg",
                        ".gif": "image/gif",
                        ".webp": "image/webp",
                    }
                    return {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_types.get(suffix, "image/png"),
                            "data": data,
                        },
                    }
            except Exception as exc:
                logger.warning("Failed to read image file %s: %s", image_source, exc)
                return None

        # Assume base64 string
        # Detect media type from data header or default to PNG
        media_type = "image/png"
        data = image_source
        if image_source.startswith("data:"):
            # Handle data URI: data:image/png;base64,XXXX
            try:
                header, data = image_source.split(",", 1)
                if "jpeg" in header or "jpg" in header:
                    media_type = "image/jpeg"
                elif "gif" in header:
                    media_type = "image/gif"
                elif "webp" in header:
                    media_type = "image/webp"
            except ValueError:
                pass
        elif image_source.startswith("/9j/"):
            media_type = "image/jpeg"

        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": data,
            },
        }

    # ------------------------------------------------------------------
    # Screen & Camera Capture
    # ------------------------------------------------------------------

    async def _capture_screen(self) -> Optional[str]:
        """Capture the current screen as a base64-encoded PNG.

        Returns:
            Base64 string of the screenshot, or None on failure.
        """
        try:
            import mss  # type: ignore[import-untyped]

            def _grab() -> str:
                with mss.mss() as sct:
                    monitor = sct.monitors[0]  # Full virtual screen
                    screenshot = sct.grab(monitor)
                    # Convert to PNG bytes
                    png_bytes = mss.tools.to_png(screenshot.rgb, screenshot.size)
                    return base64.b64encode(png_bytes).decode("utf-8")

            return await asyncio.get_event_loop().run_in_executor(None, _grab)
        except ImportError:
            logger.warning("mss not installed — screen capture unavailable.")
            return None
        except Exception as exc:
            logger.error("Screen capture failed: %s", exc)
            return None

    async def _capture_camera(self) -> Optional[str]:
        """Capture a single frame from the webcam as base64 JPEG.

        Returns:
            Base64 string of the camera frame, or None on failure.
        """
        try:
            import cv2  # type: ignore[import-untyped]

            def _grab() -> Optional[str]:
                cap = cv2.VideoCapture(0)
                try:
                    if not cap.isOpened():
                        logger.warning("Webcam not available.")
                        return None
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        logger.warning("Failed to read webcam frame.")
                        return None
                    _, buffer = cv2.imencode(".jpg", frame)
                    return base64.b64encode(buffer.tobytes()).decode("utf-8")
                finally:
                    cap.release()

            return await asyncio.get_event_loop().run_in_executor(None, _grab)
        except ImportError:
            logger.warning("opencv-python not installed — camera capture unavailable.")
            return None
        except Exception as exc:
            logger.error("Camera capture failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Ollama Fallback
    # ------------------------------------------------------------------

    async def _fallback_ollama(self, user_input: str) -> str:
        """Fall back to a local Ollama model when Claude is unavailable.

        This is a simplified path — no tool use, just text completion.

        Args:
            user_input: The user's text input.

        Returns:
            The response from Ollama, or an error message.
        """
        logger.info("Falling back to Ollama model: %s", self.ollama_model)

        # Build a simplified message list for Ollama
        ollama_messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        # Include recent conversation context (text only)
        for msg in self.conversation[-(CONTEXT_WINDOW_KEEP_RECENT):]:
            if isinstance(msg.content, str):
                ollama_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })
            elif isinstance(msg.content, list):
                # Extract text only
                text_parts = [
                    block["text"]
                    for block in msg.content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                if text_parts:
                    ollama_messages.append({
                        "role": msg.role,
                        "content": " ".join(text_parts),
                    })

        try:
            async with httpx.AsyncClient(timeout=60.0) as http_client:
                response = await http_client.post(
                    f"{self.ollama_base_url}/api/chat",
                    json={
                        "model": self.ollama_model,
                        "messages": ollama_messages,
                        "stream": False,
                    },
                )
                response.raise_for_status()
                data = response.json()
                reply = data.get("message", {}).get("content", "")

                if reply:
                    # Record in conversation
                    self.conversation.append(
                        ConversationMessage(
                            role=MessageRole.ASSISTANT,
                            content=reply,
                        )
                    )
                    return reply

                return (
                    "I'm having trouble connecting to both Claude and my "
                    "local backup. Please check your API key or Ollama setup."
                )

        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama at %s", self.ollama_base_url)
            return (
                "I'm currently unable to connect to any AI service. "
                "Please check that either your Claude API key is set or "
                "Ollama is running locally (ollama serve)."
            )
        except Exception as exc:
            logger.error("Ollama fallback failed: %s", exc)
            return (
                f"Both Claude and local Ollama are unavailable. "
                f"Error: {type(exc).__name__}: {exc}"
            )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: ~4 characters per token for English.

        Args:
            text: Input text.

        Returns:
            Estimated token count.
        """
        return max(1, len(text) // 4)

    @property
    def message_count(self) -> int:
        """Number of messages currently in conversation history."""
        return len(self.conversation)

    def __repr__(self) -> str:
        return (
            f"AIBrain(model={self.model!r}, "
            f"messages={len(self.conversation)}, "
            f"has_claude={'yes' if self.client else 'no'})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate_repr(obj: Any, max_len: int = 200) -> str:
    """Truncated repr for logging tool inputs without flooding logs."""
    r = repr(obj)
    if len(r) > max_len:
        return r[: max_len - 3] + "..."
    return r
