"""
Central configuration for the AI Desktop Assistant.

All settings are defined as a frozen-friendly dataclass. Values are resolved
in the following priority order (highest wins):

    1. Environment variables  (ANTHROPIC_API_KEY, OLLAMA_BASE_URL, ...)
    2. assistant/settings.json
    3. Hard-coded defaults in this module
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_SETTINGS_PATH = Path(__file__).resolve().parent / "settings.json"

# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class AvatarSettings:
    """Visual avatar rendered on the desktop overlay."""
    size: int = 180
    position_x: int = -1          # -1 = auto (bottom-right)
    position_y: int = -1
    animation_fps: int = 30
    idle_animation: str = "breathe"
    speaking_animation: str = "talk"
    sprite_sheet: str = "assistant/assets/avatar_sprite.png"


@dataclass
class VoiceSettings:
    """Edge-TTS voice configuration."""
    tts_voice_th: str = "th-TH-PremwadeeNeural"
    tts_voice_en: str = "en-US-JennyNeural"
    tts_rate: str = "+0%"
    tts_volume: str = "+0%"
    wake_word: str = "assistant"
    stt_model: str = "small"       # Whisper model size
    silence_threshold: float = 0.4
    silence_duration: float = 1.5  # seconds of silence before processing


@dataclass
class ClaudeModelConfig:
    """Anthropic API model configuration."""
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 8192
    temperature: float = 0.3
    system_prompt: str = (
        "You are a helpful AI desktop assistant. You can see the user's "
        "screen, control their computer, and interact with the AI Team "
        "Factory dashboard. Be concise, accurate, and proactive."
    )


# ---------------------------------------------------------------------------
# Root configuration
# ---------------------------------------------------------------------------

@dataclass
class AssistantConfig:
    """
    Root configuration object for the AI Desktop Assistant.

    Instantiate via ``AssistantConfig.load()`` to automatically merge
    settings.json values and environment-variable overrides.
    """

    # --- API keys & URLs ---
    anthropic_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    dashboard_url: str = "http://localhost:5555"

    # --- Nested settings ---
    avatar: AvatarSettings = field(default_factory=AvatarSettings)
    voice: VoiceSettings = field(default_factory=VoiceSettings)
    claude: ClaudeModelConfig = field(default_factory=ClaudeModelConfig)

    # --- Runtime flags ---
    debug: bool = False
    log_level: str = "INFO"
    screenshot_max_width: int = 1280
    webcam_index: int = 0

    # -----------------------------------------------------------------
    # Environment variable mapping  (env-var -> config attribute)
    # -----------------------------------------------------------------
    _ENV_MAP: Dict[str, str] = field(default_factory=lambda: {
        "ANTHROPIC_API_KEY": "anthropic_api_key",
        "OLLAMA_BASE_URL": "ollama_base_url",
        "DASHBOARD_URL": "dashboard_url",
        "ASSISTANT_DEBUG": "debug",
        "ASSISTANT_LOG_LEVEL": "log_level",
    }, repr=False)

    # -----------------------------------------------------------------
    # Factory
    # -----------------------------------------------------------------
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "AssistantConfig":
        """Build a config by layering defaults <- JSON file <- env vars."""
        config = cls()

        # --- Layer 1: settings.json ---
        json_path = path or _SETTINGS_PATH
        if json_path.exists():
            try:
                raw = json.loads(json_path.read_text(encoding="utf-8"))
                config = cls._apply_json(config, raw)
                logger.info("Loaded settings from %s", json_path)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to read %s: %s", json_path, exc)

        # --- Layer 2: environment variables ---
        config = cls._apply_env(config)

        # --- Validate ---
        if not config.anthropic_api_key:
            logger.warning(
                "ANTHROPIC_API_KEY is not set. Claude API calls will fail."
            )

        return config

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------
    @classmethod
    def _apply_json(cls, config: "AssistantConfig", raw: dict) -> "AssistantConfig":
        """Merge a parsed JSON dict into *config* (non-destructive)."""
        top_fields = {f.name for f in fields(cls) if not f.name.startswith("_")}

        for key, value in raw.items():
            if key == "avatar" and isinstance(value, dict):
                for k, v in value.items():
                    if hasattr(config.avatar, k):
                        setattr(config.avatar, k, v)
            elif key == "voice" and isinstance(value, dict):
                for k, v in value.items():
                    if hasattr(config.voice, k):
                        setattr(config.voice, k, v)
            elif key == "claude" and isinstance(value, dict):
                for k, v in value.items():
                    if hasattr(config.claude, k):
                        setattr(config.claude, k, v)
            elif key in top_fields:
                setattr(config, key, value)
            else:
                logger.debug("Ignoring unknown settings key: %s", key)

        return config

    @classmethod
    def _apply_env(cls, config: "AssistantConfig") -> "AssistantConfig":
        """Override config attributes from environment variables."""
        for env_var, attr in config._ENV_MAP.items():
            value = os.environ.get(env_var)
            if value is None:
                continue
            current = getattr(config, attr, None)
            # Coerce to the existing type when possible
            if isinstance(current, bool):
                setattr(config, attr, value.lower() in ("1", "true", "yes"))
            elif isinstance(current, int):
                try:
                    setattr(config, attr, int(value))
                except ValueError:
                    logger.warning("Cannot parse %s=%r as int", env_var, value)
            else:
                setattr(config, attr, value)
        return config

    def to_dict(self) -> dict:
        """Serialise config to a plain dict (for JSON export / logging)."""
        return {
            "anthropic_api_key": "***" if self.anthropic_api_key else "",
            "ollama_base_url": self.ollama_base_url,
            "dashboard_url": self.dashboard_url,
            "avatar": {
                "size": self.avatar.size,
                "position_x": self.avatar.position_x,
                "position_y": self.avatar.position_y,
                "animation_fps": self.avatar.animation_fps,
                "idle_animation": self.avatar.idle_animation,
                "speaking_animation": self.avatar.speaking_animation,
                "sprite_sheet": self.avatar.sprite_sheet,
            },
            "voice": {
                "tts_voice_th": self.voice.tts_voice_th,
                "tts_voice_en": self.voice.tts_voice_en,
                "tts_rate": self.voice.tts_rate,
                "tts_volume": self.voice.tts_volume,
                "wake_word": self.voice.wake_word,
                "stt_model": self.voice.stt_model,
                "silence_threshold": self.voice.silence_threshold,
                "silence_duration": self.voice.silence_duration,
            },
            "claude": {
                "model": self.claude.model,
                "max_tokens": self.claude.max_tokens,
                "temperature": self.claude.temperature,
                "system_prompt": self.claude.system_prompt,
            },
            "debug": self.debug,
            "log_level": self.log_level,
            "screenshot_max_width": self.screenshot_max_width,
            "webcam_index": self.webcam_index,
        }
