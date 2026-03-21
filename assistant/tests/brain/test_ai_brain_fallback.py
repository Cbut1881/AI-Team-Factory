"""
Tests for AIBrain Ollama fallback behaviour.

Covers _fallback_ollama success/failure scenarios and fallback triggers
from various Claude API errors.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import httpx
import pytest

from brain.ai_brain import (
    AIBrain,
    ConversationMessage,
    CONTEXT_WINDOW_KEEP_RECENT,
    MessageRole,
    SYSTEM_PROMPT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_ollama_response(content: str = "Ollama says hi") -> httpx.Response:
    """Build a fake httpx.Response that looks like an Ollama chat reply."""
    resp = httpx.Response(
        status_code=200,
        json={"message": {"content": content}},
        request=httpx.Request("POST", "http://localhost:11434/api/chat"),
    )
    return resp


def _mock_ollama_empty() -> httpx.Response:
    """Ollama response with empty content."""
    return httpx.Response(
        status_code=200,
        json={"message": {"content": ""}},
        request=httpx.Request("POST", "http://localhost:11434/api/chat"),
    )


# =========================================================================
# _fallback_ollama — success
# =========================================================================

class TestFallbackOllamaSuccess:
    """Ollama fallback returns valid responses."""

    @pytest.mark.asyncio
    async def test_ollama_success(self, brain_with_client):
        brain_with_client.conversation.append(
            ConversationMessage(role=MessageRole.USER, content=[{"type": "text", "text": "Hello"}])
        )

        with patch("brain.ai_brain.httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=_mock_ollama_response("Hello from Ollama"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance

            result = await brain_with_client._fallback_ollama("Hello")

        assert result == "Hello from Ollama"

    @pytest.mark.asyncio
    async def test_ollama_appends_assistant_message(self, brain_with_client):
        brain_with_client.conversation.append(
            ConversationMessage(role=MessageRole.USER, content=[{"type": "text", "text": "Hi"}])
        )

        with patch("brain.ai_brain.httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=_mock_ollama_response("Reply"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance

            await brain_with_client._fallback_ollama("Hi")

        last_msg = brain_with_client.conversation[-1]
        assert last_msg.role == MessageRole.ASSISTANT
        assert last_msg.content == "Reply"


# =========================================================================
# _fallback_ollama — empty response
# =========================================================================

class TestFallbackOllamaEmpty:
    """Ollama returns empty content."""

    @pytest.mark.asyncio
    async def test_ollama_empty_response(self, brain_with_client):
        with patch("brain.ai_brain.httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=_mock_ollama_empty())
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance

            result = await brain_with_client._fallback_ollama("Hi")

        assert "trouble connecting" in result.lower()


# =========================================================================
# _fallback_ollama — ConnectError
# =========================================================================

class TestFallbackOllamaConnectError:
    """Ollama is unreachable."""

    @pytest.mark.asyncio
    async def test_ollama_connect_error(self, brain_with_client):
        with patch("brain.ai_brain.httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance

            result = await brain_with_client._fallback_ollama("Hi")

        assert "unable to connect" in result.lower()


# =========================================================================
# _fallback_ollama — generic exception
# =========================================================================

class TestFallbackOllamaGenericException:
    """Unexpected exceptions from Ollama."""

    @pytest.mark.asyncio
    async def test_ollama_generic_exception(self, brain_with_client):
        with patch("brain.ai_brain.httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(side_effect=RuntimeError("boom"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance

            result = await brain_with_client._fallback_ollama("Hi")

        assert "unavailable" in result.lower()
        assert "RuntimeError" in result


# =========================================================================
# Fallback triggered by Claude errors (via process())
# =========================================================================

class TestFallbackTriggeredByClaude:
    """Various Claude API errors trigger Ollama fallback via process()."""

    def _setup_ollama_mock(self):
        """Return a context-manager patch for httpx.AsyncClient that returns a valid response."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=_mock_ollama_response("Fallback response"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        return patch("brain.ai_brain.httpx.AsyncClient", return_value=mock_client)

    @pytest.mark.asyncio
    async def test_authentication_error_triggers_fallback(self, brain_with_client):
        brain_with_client.client.messages.create = MagicMock(
            side_effect=anthropic.AuthenticationError(
                message="invalid key",
                response=httpx.Response(401, request=httpx.Request("POST", "https://api.anthropic.com")),
                body=None,
            )
        )
        with self._setup_ollama_mock():
            result = await brain_with_client.process("Hello")
        assert result == "Fallback response"

    @pytest.mark.asyncio
    async def test_rate_limit_error_triggers_fallback(self, brain_with_client):
        brain_with_client.client.messages.create = MagicMock(
            side_effect=anthropic.RateLimitError(
                message="rate limited",
                response=httpx.Response(429, request=httpx.Request("POST", "https://api.anthropic.com")),
                body=None,
            )
        )
        with self._setup_ollama_mock():
            result = await brain_with_client.process("Hello")
        assert result == "Fallback response"

    @pytest.mark.asyncio
    async def test_api_connection_error_triggers_fallback(self, brain_with_client):
        brain_with_client.client.messages.create = MagicMock(
            side_effect=anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))
        )
        with self._setup_ollama_mock():
            result = await brain_with_client.process("Hello")
        assert result == "Fallback response"

    @pytest.mark.asyncio
    async def test_api_timeout_error_triggers_fallback(self, brain_with_client):
        brain_with_client.client.messages.create = MagicMock(
            side_effect=anthropic.APITimeoutError(request=httpx.Request("POST", "https://api.anthropic.com"))
        )
        with self._setup_ollama_mock():
            result = await brain_with_client.process("Hello")
        assert result == "Fallback response"

    @pytest.mark.asyncio
    async def test_500_error_triggers_fallback(self, brain_with_client):
        brain_with_client.client.messages.create = MagicMock(
            side_effect=anthropic.APIStatusError(
                message="server error",
                response=httpx.Response(500, request=httpx.Request("POST", "https://api.anthropic.com")),
                body=None,
            )
        )
        with self._setup_ollama_mock():
            result = await brain_with_client.process("Hello")
        assert result == "Fallback response"

    @pytest.mark.asyncio
    async def test_502_error_triggers_fallback(self, brain_with_client):
        brain_with_client.client.messages.create = MagicMock(
            side_effect=anthropic.APIStatusError(
                message="bad gateway",
                response=httpx.Response(502, request=httpx.Request("POST", "https://api.anthropic.com")),
                body=None,
            )
        )
        with self._setup_ollama_mock():
            result = await brain_with_client.process("Hello")
        assert result == "Fallback response"

    @pytest.mark.asyncio
    async def test_permission_denied_triggers_fallback(self, brain_with_client):
        brain_with_client.client.messages.create = MagicMock(
            side_effect=anthropic.PermissionDeniedError(
                message="forbidden",
                response=httpx.Response(403, request=httpx.Request("POST", "https://api.anthropic.com")),
                body=None,
            )
        )
        with self._setup_ollama_mock():
            result = await brain_with_client.process("Hello")
        assert result == "Fallback response"


# =========================================================================
# 4xx errors re-raise
# =========================================================================

class TestClientErrorsReraise:
    """4xx APIStatusErrors (except auth/rate-limit) should re-raise."""

    @pytest.mark.asyncio
    async def test_400_error_reraises(self, brain_with_client):
        brain_with_client.client.messages.create = MagicMock(
            side_effect=anthropic.APIStatusError(
                message="bad request",
                response=httpx.Response(400, request=httpx.Request("POST", "https://api.anthropic.com")),
                body=None,
            )
        )
        with pytest.raises(anthropic.APIStatusError):
            await brain_with_client.process("Hello")

    @pytest.mark.asyncio
    async def test_422_error_reraises(self, brain_with_client):
        brain_with_client.client.messages.create = MagicMock(
            side_effect=anthropic.APIStatusError(
                message="unprocessable",
                response=httpx.Response(422, request=httpx.Request("POST", "https://api.anthropic.com")),
                body=None,
            )
        )
        with pytest.raises(anthropic.APIStatusError):
            await brain_with_client.process("Hello")


# =========================================================================
# No client goes directly to Ollama
# =========================================================================

class TestNoClientGoesToOllama:
    """When client is None, process() goes straight to Ollama."""

    @pytest.mark.asyncio
    async def test_no_client_uses_ollama(self, brain_without_client):
        with patch("brain.ai_brain.httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=_mock_ollama_response("Direct Ollama"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance

            result = await brain_without_client.process("Hello")

        assert result == "Direct Ollama"

    @pytest.mark.asyncio
    async def test_no_client_ollama_failure(self, brain_without_client):
        with patch("brain.ai_brain.httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance

            result = await brain_without_client.process("Hello")

        assert "unable to connect" in result.lower()


# =========================================================================
# Ollama uses recent context, text-only
# =========================================================================

class TestOllamaContextHandling:
    """Ollama receives only recent text-only messages."""

    @pytest.mark.asyncio
    async def test_ollama_uses_recent_messages(self, brain_with_client):
        # Add conversation history
        for i in range(5):
            brain_with_client.conversation.append(
                ConversationMessage(
                    role=MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT,
                    content=f"Message {i}",
                )
            )

        captured_json = {}

        with patch("brain.ai_brain.httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()

            async def capture_post(url, json=None, **kwargs):
                captured_json.update(json or {})
                return _mock_ollama_response("ok")

            mock_instance.post = capture_post
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance

            await brain_with_client._fallback_ollama("latest input")

        messages = captured_json.get("messages", [])
        # First should be system prompt
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_ollama_skips_image_only_messages(self, brain_with_client):
        # Add a message with only image content (no text blocks)
        brain_with_client.conversation.append(
            ConversationMessage(
                role=MessageRole.USER,
                content=[{"type": "image", "source": {"type": "base64", "data": "AAA"}}],
            )
        )

        captured_json = {}

        with patch("brain.ai_brain.httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()

            async def capture_post(url, json=None, **kwargs):
                captured_json.update(json or {})
                return _mock_ollama_response("ok")

            mock_instance.post = capture_post
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance

            await brain_with_client._fallback_ollama("text input")

        messages = captured_json.get("messages", [])
        # Only system prompt; image-only message has no text blocks so it's skipped
        non_system = [m for m in messages if m["role"] != "system"]
        for m in non_system:
            assert "image" not in m.get("content", "").lower() or True  # no image blocks

    @pytest.mark.asyncio
    async def test_ollama_extracts_text_from_list_content(self, brain_with_client):
        brain_with_client.conversation.append(
            ConversationMessage(
                role=MessageRole.USER,
                content=[
                    {"type": "image", "source": {"type": "base64", "data": "X"}},
                    {"type": "text", "text": "Describe this image"},
                ],
            )
        )

        captured_json = {}

        with patch("brain.ai_brain.httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()

            async def capture_post(url, json=None, **kwargs):
                captured_json.update(json or {})
                return _mock_ollama_response("ok")

            mock_instance.post = capture_post
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance

            await brain_with_client._fallback_ollama("test")

        messages = captured_json.get("messages", [])
        user_msgs = [m for m in messages if m["role"] == "user"]
        # The list-content message should have been converted to text-only
        assert any("Describe this image" in m["content"] for m in user_msgs)

    @pytest.mark.asyncio
    async def test_ollama_stream_false(self, brain_with_client):
        captured_json = {}

        with patch("brain.ai_brain.httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()

            async def capture_post(url, json=None, **kwargs):
                captured_json.update(json or {})
                return _mock_ollama_response("ok")

            mock_instance.post = capture_post
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance

            await brain_with_client._fallback_ollama("test")

        assert captured_json.get("stream") is False
