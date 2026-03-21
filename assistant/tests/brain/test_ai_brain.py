"""
Tests for AIBrain core functionality.

Covers initialisation, process(), image handling, conversation management,
system prompt building, and utility helpers.
"""

from __future__ import annotations

import asyncio
import base64
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from brain.ai_brain import (
    AIBrain,
    ConversationMessage,
    MessageRole,
    SYSTEM_PROMPT,
    DEFAULT_MODEL,
    MAX_TOKENS,
    _truncate_repr,
)
from config import AssistantConfig, ClaudeModelConfig


# =========================================================================
# __init__
# =========================================================================

class TestAIBrainInit:
    """Tests for AIBrain.__init__."""

    def test_init_with_api_key(self, brain_with_client: AIBrain):
        assert brain_with_client.client is not None
        assert brain_with_client.model == "claude-test"
        assert brain_with_client.max_tokens == 1024
        assert brain_with_client.conversation == []
        assert brain_with_client.preferred_language is None
        assert brain_with_client._summary_cache is None

    def test_init_without_api_key(self, brain_without_client: AIBrain):
        assert brain_without_client.client is None
        assert brain_without_client.model == "claude-test"

    def test_init_default_model_used_when_config_empty(self):
        cfg = AssistantConfig()
        cfg.anthropic_api_key = ""
        cfg.claude = ClaudeModelConfig(model="", max_tokens=0)
        brain = AIBrain(cfg)
        assert brain.model == DEFAULT_MODEL
        assert brain.max_tokens == MAX_TOKENS

    def test_init_ollama_defaults(self, brain_with_client: AIBrain):
        assert brain_with_client.ollama_model == "llama3.1:8b"
        assert brain_with_client.ollama_base_url == "http://localhost:11434"

    def test_init_custom_ollama_base_url(self):
        cfg = AssistantConfig()
        cfg.anthropic_api_key = ""
        cfg.ollama_base_url = "http://custom:9999"
        cfg.claude = ClaudeModelConfig()
        brain = AIBrain(cfg)
        assert brain.ollama_base_url == "http://custom:9999"

    def test_init_execute_tool_bound(self, brain_with_client: AIBrain):
        from brain.tools import execute_tool
        assert brain_with_client.execute_tool is execute_tool


# =========================================================================
# process() — simple text response
# =========================================================================

class TestProcessSimpleText:
    """Tests for process() returning plain text responses."""

    @pytest.mark.asyncio
    async def test_process_simple_text(self, brain_with_client, mock_claude_response):
        response_msg = mock_claude_response(text="Hello there!")
        brain_with_client.client.messages.create = MagicMock(return_value=response_msg)

        result = await brain_with_client.process("Hi")

        assert result == "Hello there!"

    @pytest.mark.asyncio
    async def test_process_appends_user_message(self, brain_with_client, mock_claude_response):
        response_msg = mock_claude_response(text="Reply")
        brain_with_client.client.messages.create = MagicMock(return_value=response_msg)

        await brain_with_client.process("Test input")

        assert len(brain_with_client.conversation) >= 1
        user_msg = brain_with_client.conversation[0]
        assert user_msg.role == MessageRole.USER

    @pytest.mark.asyncio
    async def test_process_appends_assistant_message(self, brain_with_client, mock_claude_response):
        response_msg = mock_claude_response(text="Reply")
        brain_with_client.client.messages.create = MagicMock(return_value=response_msg)

        await brain_with_client.process("Test input")

        assert len(brain_with_client.conversation) == 2
        asst_msg = brain_with_client.conversation[1]
        assert asst_msg.role == MessageRole.ASSISTANT

    @pytest.mark.asyncio
    async def test_process_user_content_is_list(self, brain_with_client, mock_claude_response):
        response_msg = mock_claude_response(text="Ok")
        brain_with_client.client.messages.create = MagicMock(return_value=response_msg)

        await brain_with_client.process("Hello")

        user_msg = brain_with_client.conversation[0]
        assert isinstance(user_msg.content, list)
        assert user_msg.content[-1] == {"type": "text", "text": "Hello"}

    @pytest.mark.asyncio
    async def test_process_returns_stripped_text(self, brain_with_client, mock_claude_response):
        response_msg = mock_claude_response(text="  trimmed  ")
        brain_with_client.client.messages.create = MagicMock(return_value=response_msg)

        result = await brain_with_client.process("x")
        assert result == "trimmed"


# =========================================================================
# process() — with images
# =========================================================================

class TestProcessWithImages:
    """Tests for process() when images are provided."""

    @pytest.mark.asyncio
    async def test_process_with_base64_image(self, brain_with_client, mock_claude_response):
        response_msg = mock_claude_response(text="I see an image")
        brain_with_client.client.messages.create = MagicMock(return_value=response_msg)

        b64_data = base64.b64encode(b"fakepng").decode()
        result = await brain_with_client.process("What is this?", images=[b64_data])

        assert result == "I see an image"
        user_content = brain_with_client.conversation[0].content
        assert any(block.get("type") == "image" for block in user_content)

    @pytest.mark.asyncio
    async def test_process_with_data_uri_jpeg(self, brain_with_client, mock_claude_response):
        response_msg = mock_claude_response(text="JPEG seen")
        brain_with_client.client.messages.create = MagicMock(return_value=response_msg)

        data_uri = "data:image/jpeg;base64,/9j/fakedata"
        await brain_with_client.process("Describe", images=[data_uri])

        user_content = brain_with_client.conversation[0].content
        img_block = [b for b in user_content if b.get("type") == "image"][0]
        assert img_block["source"]["media_type"] == "image/jpeg"
        assert img_block["source"]["data"] == "/9j/fakedata"

    @pytest.mark.asyncio
    async def test_process_with_file_path_image(self, brain_with_client, mock_claude_response):
        response_msg = mock_claude_response(text="File image")
        brain_with_client.client.messages.create = MagicMock(return_value=response_msg)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
            tmp_path = f.name

        try:
            await brain_with_client.process("See this", images=[tmp_path])
            user_content = brain_with_client.conversation[0].content
            img_block = [b for b in user_content if b.get("type") == "image"][0]
            assert img_block["source"]["media_type"] == "image/png"
            assert img_block["source"]["type"] == "base64"
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_process_with_multiple_images(self, brain_with_client, mock_claude_response):
        response_msg = mock_claude_response(text="Two images")
        brain_with_client.client.messages.create = MagicMock(return_value=response_msg)

        b64_1 = base64.b64encode(b"img1").decode()
        b64_2 = base64.b64encode(b"img2").decode()
        await brain_with_client.process("Compare", images=[b64_1, b64_2])

        user_content = brain_with_client.conversation[0].content
        img_blocks = [b for b in user_content if b.get("type") == "image"]
        assert len(img_blocks) == 2


# =========================================================================
# _resolve_image edge cases
# =========================================================================

class TestResolveImage:
    """Tests for _resolve_image static method."""

    def test_resolve_base64_defaults_to_png(self):
        result = AIBrain._resolve_image("AAAA" * 200)
        assert result is not None
        assert result["source"]["media_type"] == "image/png"

    def test_resolve_jpeg_magic_bytes(self):
        result = AIBrain._resolve_image("/9j/4AAQ" + "A" * 500)
        assert result["source"]["media_type"] == "image/jpeg"

    def test_resolve_data_uri_gif(self):
        result = AIBrain._resolve_image("data:image/gif;base64,R0lGODlh")
        assert result["source"]["media_type"] == "image/gif"
        assert result["source"]["data"] == "R0lGODlh"

    def test_resolve_data_uri_webp(self):
        result = AIBrain._resolve_image("data:image/webp;base64,RIFF")
        assert result["source"]["media_type"] == "image/webp"

    def test_resolve_data_uri_no_comma_returns_full(self):
        result = AIBrain._resolve_image("data:image/png;base64NoComma")
        assert result is not None
        # Falls through ValueError catch, data stays as full string
        assert result["source"]["data"] == "data:image/png;base64NoComma"

    def test_resolve_nonexistent_file_path_falls_through_to_base64(self):
        """A short string with image extension that doesn't exist on disk
        falls through to the base64 path (no error, treated as raw data)."""
        result = AIBrain._resolve_image("C:/no_such_dir_xyz/missing.png")
        assert result is not None
        # Treated as base64 data since the file doesn't exist
        assert result["source"]["type"] == "base64"
        assert result["source"]["data"] == "C:/no_such_dir_xyz/missing.png"

    def test_resolve_file_read_error_returns_none(self):
        """If path.exists() is True but reading fails, returns None."""
        with patch("brain.ai_brain.Path.exists", return_value=True), \
             patch("brain.ai_brain.Path.read_bytes", side_effect=PermissionError("denied")):
            result = AIBrain._resolve_image("C:/protected/file.png")
        assert result is None

    def test_resolve_file_path_jpg(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff" + b"\x00" * 50)
            path = f.name
        try:
            result = AIBrain._resolve_image(path)
            assert result is not None
            assert result["source"]["media_type"] == "image/jpeg"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_resolve_file_path_webp(self):
        with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as f:
            f.write(b"RIFF" + b"\x00" * 50)
            path = f.name
        try:
            result = AIBrain._resolve_image(path)
            assert result is not None
            assert result["source"]["media_type"] == "image/webp"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_resolve_long_base64_not_treated_as_path(self):
        """Strings longer than 500 chars should not be treated as file paths."""
        long_b64 = "A" * 600
        result = AIBrain._resolve_image(long_b64)
        assert result is not None
        assert result["source"]["media_type"] == "image/png"


# =========================================================================
# process_with_screen / process_with_camera
# =========================================================================

class TestProcessWithScreenCamera:
    """Tests for process_with_screen and process_with_camera."""

    @pytest.mark.asyncio
    async def test_process_with_screen_success(self, brain_with_client, mock_claude_response):
        response_msg = mock_claude_response(text="Screen content")
        brain_with_client.client.messages.create = MagicMock(return_value=response_msg)

        with patch.object(brain_with_client, "_capture_screen", return_value="fakebase64"):
            result = await brain_with_client.process_with_screen("What's on screen?")

        assert result == "Screen content"
        user_content = brain_with_client.conversation[0].content
        assert any(b.get("type") == "image" for b in user_content)

    @pytest.mark.asyncio
    async def test_process_with_screen_capture_fails(self, brain_with_client, mock_claude_response):
        response_msg = mock_claude_response(text="No screen")
        brain_with_client.client.messages.create = MagicMock(return_value=response_msg)

        with patch.object(brain_with_client, "_capture_screen", return_value=None):
            result = await brain_with_client.process_with_screen("What's here?")

        assert result == "No screen"
        user_content = brain_with_client.conversation[0].content
        img_blocks = [b for b in user_content if b.get("type") == "image"]
        assert len(img_blocks) == 0

    @pytest.mark.asyncio
    async def test_process_with_camera_success(self, brain_with_client, mock_claude_response):
        response_msg = mock_claude_response(text="Camera content")
        brain_with_client.client.messages.create = MagicMock(return_value=response_msg)

        with patch.object(brain_with_client, "_capture_camera", return_value="cambase64"):
            result = await brain_with_client.process_with_camera("What do you see?")

        assert result == "Camera content"

    @pytest.mark.asyncio
    async def test_process_with_camera_capture_fails(self, brain_with_client, mock_claude_response):
        response_msg = mock_claude_response(text="No camera")
        brain_with_client.client.messages.create = MagicMock(return_value=response_msg)

        with patch.object(brain_with_client, "_capture_camera", return_value=None):
            result = await brain_with_client.process_with_camera("Anything?")

        assert result == "No camera"


# =========================================================================
# reset_conversation / set_language
# =========================================================================

class TestConversationManagement:
    """Tests for conversation reset and language setting."""

    def test_reset_conversation_clears_messages(self, brain_with_client):
        brain_with_client.conversation.append(
            ConversationMessage(role="user", content="test")
        )
        brain_with_client._summary_cache = "old summary"

        brain_with_client.reset_conversation()

        assert brain_with_client.conversation == []
        assert brain_with_client._summary_cache is None

    def test_reset_conversation_when_already_empty(self, brain_with_client):
        brain_with_client.reset_conversation()
        assert brain_with_client.conversation == []

    def test_set_language_thai(self, brain_with_client):
        brain_with_client.set_language("th")
        assert brain_with_client.preferred_language == "th"

    def test_set_language_english(self, brain_with_client):
        brain_with_client.set_language("en")
        assert brain_with_client.preferred_language == "en"

    def test_set_language_arbitrary(self, brain_with_client):
        brain_with_client.set_language("ja")
        assert brain_with_client.preferred_language == "ja"


# =========================================================================
# message_count / __repr__ / _estimate_tokens
# =========================================================================

class TestUtilities:
    """Tests for utility properties and methods."""

    def test_message_count_empty(self, brain_with_client):
        assert brain_with_client.message_count == 0

    def test_message_count_with_messages(self, brain_with_client):
        brain_with_client.conversation.append(
            ConversationMessage(role="user", content="a")
        )
        brain_with_client.conversation.append(
            ConversationMessage(role="assistant", content="b")
        )
        assert brain_with_client.message_count == 2

    def test_repr_with_client(self, brain_with_client):
        r = repr(brain_with_client)
        assert "claude-test" in r
        assert "has_claude=yes" in r
        assert "messages=0" in r

    def test_repr_without_client(self, brain_without_client):
        r = repr(brain_without_client)
        assert "has_claude=no" in r

    def test_estimate_tokens_short_text(self):
        assert AIBrain._estimate_tokens("hi") == 1  # max(1, 2//4)

    def test_estimate_tokens_longer_text(self):
        assert AIBrain._estimate_tokens("a" * 100) == 25

    def test_estimate_tokens_empty(self):
        assert AIBrain._estimate_tokens("") == 1  # max(1, 0)


# =========================================================================
# _build_system_prompt
# =========================================================================

class TestBuildSystemPrompt:
    """Tests for _build_system_prompt."""

    def test_default_prompt_no_language(self, brain_with_client):
        prompt = brain_with_client._build_system_prompt()
        assert prompt == SYSTEM_PROMPT

    def test_prompt_with_thai(self, brain_with_client):
        brain_with_client.set_language("th")
        prompt = brain_with_client._build_system_prompt()
        assert "Thai" in prompt
        assert "Language Override" in prompt

    def test_prompt_with_english(self, brain_with_client):
        brain_with_client.set_language("en")
        prompt = brain_with_client._build_system_prompt()
        assert "English" in prompt

    def test_prompt_with_unknown_lang_uses_code(self, brain_with_client):
        brain_with_client.set_language("fr")
        prompt = brain_with_client._build_system_prompt()
        assert "fr" in prompt


# =========================================================================
# _build_user_content
# =========================================================================

class TestBuildUserContent:
    """Tests for _build_user_content."""

    def test_text_only(self, brain_with_client):
        content = brain_with_client._build_user_content("Hello")
        assert len(content) == 1
        assert content[0] == {"type": "text", "text": "Hello"}

    def test_text_with_one_image(self, brain_with_client):
        content = brain_with_client._build_user_content("Desc", images=["AAAA" * 200])
        assert len(content) == 2
        assert content[0]["type"] == "image"
        assert content[1]["type"] == "text"

    def test_images_come_before_text(self, brain_with_client):
        content = brain_with_client._build_user_content(
            "Desc", images=["A" * 600, "B" * 600]
        )
        types = [b["type"] for b in content]
        assert types == ["image", "image", "text"]

    def test_no_images_param(self, brain_with_client):
        content = brain_with_client._build_user_content("Hi", images=None)
        assert len(content) == 1

    def test_empty_images_list(self, brain_with_client):
        content = brain_with_client._build_user_content("Hi", images=[])
        assert len(content) == 1


# =========================================================================
# _truncate_repr helper
# =========================================================================

class TestTruncateRepr:
    """Tests for the module-level _truncate_repr helper."""

    def test_short_repr_unchanged(self):
        assert _truncate_repr({"a": 1}) == repr({"a": 1})

    def test_long_repr_truncated(self):
        big = {"data": "x" * 300}
        result = _truncate_repr(big, max_len=50)
        assert len(result) == 50
        assert result.endswith("...")

    def test_custom_max_len(self):
        result = _truncate_repr("hello", max_len=10)
        assert len(result) <= 10


# =========================================================================
# ConversationMessage
# =========================================================================

class TestConversationMessage:
    """Tests for the ConversationMessage dataclass."""

    def test_to_api_format_string_content(self):
        msg = ConversationMessage(role="user", content="hello")
        fmt = msg.to_api_format()
        assert fmt == {"role": "user", "content": "hello"}

    def test_to_api_format_list_content(self):
        content = [{"type": "text", "text": "hi"}]
        msg = ConversationMessage(role="assistant", content=content)
        fmt = msg.to_api_format()
        assert fmt["content"] == content

    def test_token_estimate_default(self):
        msg = ConversationMessage(role="user", content="test")
        assert msg.token_estimate == 0

    def test_timestamp_auto_set(self):
        msg = ConversationMessage(role="user", content="test")
        assert msg.timestamp > 0
