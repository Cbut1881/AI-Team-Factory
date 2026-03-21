"""
Tests for ``config.py`` — AssistantConfig, AvatarSettings, VoiceSettings,
ClaudeModelConfig.

Covers default values, JSON loading, environment-variable overrides,
serialisation via ``to_dict()``, and various edge cases.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure absolute imports resolve from the assistant package root.
_ASSISTANT_ROOT = Path(__file__).resolve().parent.parent
if str(_ASSISTANT_ROOT) not in sys.path:
    sys.path.insert(0, str(_ASSISTANT_ROOT))

from config import AssistantConfig, AvatarSettings, ClaudeModelConfig, VoiceSettings


# ===================================================================
# 1. Default instantiation
# ===================================================================

class TestDefaults:
    """Verify every field on a bare ``AssistantConfig()``."""

    def test_default_api_key_is_empty(self):
        cfg = AssistantConfig()
        assert cfg.anthropic_api_key == ""

    def test_default_ollama_base_url(self):
        cfg = AssistantConfig()
        assert cfg.ollama_base_url == "http://localhost:11434"

    def test_default_dashboard_url(self):
        cfg = AssistantConfig()
        assert cfg.dashboard_url == "http://localhost:5555"

    def test_default_debug_is_false(self):
        cfg = AssistantConfig()
        assert cfg.debug is False

    def test_default_log_level(self):
        cfg = AssistantConfig()
        assert cfg.log_level == "INFO"

    def test_default_screenshot_max_width(self):
        cfg = AssistantConfig()
        assert cfg.screenshot_max_width == 1280

    def test_default_webcam_index(self):
        cfg = AssistantConfig()
        assert cfg.webcam_index == 0

    def test_default_avatar_is_avatar_settings(self):
        cfg = AssistantConfig()
        assert isinstance(cfg.avatar, AvatarSettings)

    def test_default_voice_is_voice_settings(self):
        cfg = AssistantConfig()
        assert isinstance(cfg.voice, VoiceSettings)

    def test_default_claude_is_claude_model_config(self):
        cfg = AssistantConfig()
        assert isinstance(cfg.claude, ClaudeModelConfig)


# ===================================================================
# 2. Sub-config defaults
# ===================================================================

class TestSubConfigDefaults:

    def test_avatar_size(self):
        assert AvatarSettings().size == 180

    def test_avatar_position_auto(self):
        a = AvatarSettings()
        assert a.position_x == -1
        assert a.position_y == -1

    def test_avatar_animation_fps(self):
        assert AvatarSettings().animation_fps == 30

    def test_avatar_idle_animation(self):
        assert AvatarSettings().idle_animation == "breathe"

    def test_avatar_speaking_animation(self):
        assert AvatarSettings().speaking_animation == "talk"

    def test_voice_tts_voice_th(self):
        assert VoiceSettings().tts_voice_th == "th-TH-PremwadeeNeural"

    def test_voice_tts_voice_en(self):
        assert VoiceSettings().tts_voice_en == "en-US-JennyNeural"

    def test_voice_wake_word(self):
        assert VoiceSettings().wake_word == "assistant"

    def test_voice_stt_model(self):
        assert VoiceSettings().stt_model == "small"

    def test_voice_silence_threshold(self):
        assert VoiceSettings().silence_threshold == 0.4

    def test_voice_silence_duration(self):
        assert VoiceSettings().silence_duration == 1.5

    def test_claude_model(self):
        assert ClaudeModelConfig().model == "claude-sonnet-4-20250514"

    def test_claude_max_tokens(self):
        assert ClaudeModelConfig().max_tokens == 8192

    def test_claude_temperature(self):
        assert ClaudeModelConfig().temperature == 0.3

    def test_claude_system_prompt_not_empty(self):
        assert len(ClaudeModelConfig().system_prompt) > 0


# ===================================================================
# 3. AssistantConfig.load() — no settings.json
# ===================================================================

class TestLoadNoFile:

    def test_load_returns_config_when_no_file(self, tmp_path):
        missing = tmp_path / "nonexistent.json"
        cfg = AssistantConfig.load(path=missing)
        assert isinstance(cfg, AssistantConfig)
        assert cfg.anthropic_api_key == ""

    def test_load_defaults_match_bare_constructor(self, tmp_path):
        missing = tmp_path / "nonexistent.json"
        with patch.dict(os.environ, {}, clear=True):
            cfg = AssistantConfig.load(path=missing)
        bare = AssistantConfig()
        # Compare serialised form (avoids _ENV_MAP noise).
        assert cfg.to_dict() == bare.to_dict()


# ===================================================================
# 4. load() with valid settings.json — top-level keys
# ===================================================================

class TestLoadValidJson:

    def test_top_level_string(self, tmp_settings_file):
        path = tmp_settings_file({"anthropic_api_key": "sk-from-json"})
        cfg = AssistantConfig.load(path=path)
        assert cfg.anthropic_api_key == "sk-from-json"

    def test_top_level_bool(self, tmp_settings_file):
        path = tmp_settings_file({"debug": True})
        cfg = AssistantConfig.load(path=path)
        assert cfg.debug is True

    def test_top_level_int(self, tmp_settings_file):
        path = tmp_settings_file({"screenshot_max_width": 800})
        cfg = AssistantConfig.load(path=path)
        assert cfg.screenshot_max_width == 800

    def test_nested_avatar(self, tmp_settings_file):
        path = tmp_settings_file({"avatar": {"size": 256}})
        cfg = AssistantConfig.load(path=path)
        assert cfg.avatar.size == 256

    def test_nested_voice(self, tmp_settings_file):
        path = tmp_settings_file({"voice": {"wake_word": "nova"}})
        cfg = AssistantConfig.load(path=path)
        assert cfg.voice.wake_word == "nova"

    def test_nested_claude(self, tmp_settings_file):
        path = tmp_settings_file({"claude": {"max_tokens": 4096}})
        cfg = AssistantConfig.load(path=path)
        assert cfg.claude.max_tokens == 4096

    def test_multiple_keys(self, tmp_settings_file):
        path = tmp_settings_file({
            "anthropic_api_key": "sk-multi",
            "debug": True,
            "log_level": "DEBUG",
        })
        cfg = AssistantConfig.load(path=path)
        assert cfg.anthropic_api_key == "sk-multi"
        assert cfg.debug is True
        assert cfg.log_level == "DEBUG"


# ===================================================================
# 5. Partial JSON — other fields stay default
# ===================================================================

class TestLoadPartialJson:

    def test_partial_avatar_preserves_other_fields(self, tmp_settings_file):
        path = tmp_settings_file({"avatar": {"size": 64}})
        cfg = AssistantConfig.load(path=path)
        assert cfg.avatar.size == 64
        assert cfg.avatar.animation_fps == 30  # unchanged
        assert cfg.avatar.idle_animation == "breathe"  # unchanged

    def test_partial_voice_preserves_other_fields(self, tmp_settings_file):
        path = tmp_settings_file({"voice": {"tts_rate": "+10%"}})
        cfg = AssistantConfig.load(path=path)
        assert cfg.voice.tts_rate == "+10%"
        assert cfg.voice.stt_model == "small"  # unchanged

    def test_partial_claude_preserves_other_fields(self, tmp_settings_file):
        path = tmp_settings_file({"claude": {"temperature": 0.9}})
        cfg = AssistantConfig.load(path=path)
        assert cfg.claude.temperature == 0.9
        assert cfg.claude.max_tokens == 8192  # unchanged


# ===================================================================
# 6. Invalid / malformed JSON
# ===================================================================

class TestLoadInvalidJson:

    def test_malformed_json_returns_defaults(self, tmp_path):
        p = tmp_path / "settings.json"
        p.write_text("{bad json", encoding="utf-8")
        cfg = AssistantConfig.load(path=p)
        assert isinstance(cfg, AssistantConfig)
        assert cfg.anthropic_api_key == ""  # fell back to default


# ===================================================================
# 7. Unknown keys silently ignored
# ===================================================================

class TestLoadUnknownKeys:

    def test_unknown_top_level_key_ignored(self, tmp_settings_file):
        path = tmp_settings_file({"totally_unknown": 42, "debug": True})
        cfg = AssistantConfig.load(path=path)
        assert cfg.debug is True
        assert not hasattr(cfg, "totally_unknown") or cfg.to_dict().get("totally_unknown") is None

    def test_unknown_nested_key_ignored(self, tmp_settings_file):
        path = tmp_settings_file({"avatar": {"nonexistent_field": 99}})
        cfg = AssistantConfig.load(path=path)
        assert cfg.avatar.size == 180  # defaults intact


# ===================================================================
# 8. _apply_env — string override
# ===================================================================

class TestApplyEnvString:

    def test_anthropic_api_key_from_env(self, tmp_path):
        missing = tmp_path / "no.json"
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-env-key"}, clear=False):
            cfg = AssistantConfig.load(path=missing)
        assert cfg.anthropic_api_key == "sk-env-key"

    def test_ollama_base_url_from_env(self, tmp_path):
        missing = tmp_path / "no.json"
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://remote:11434"}, clear=False):
            cfg = AssistantConfig.load(path=missing)
        assert cfg.ollama_base_url == "http://remote:11434"


# ===================================================================
# 9. _apply_env — bool coercion
# ===================================================================

class TestApplyEnvBool:

    @pytest.mark.parametrize("value", ["1", "true", "yes", "True", "YES"])
    def test_debug_truthy(self, value, tmp_path):
        missing = tmp_path / "no.json"
        with patch.dict(os.environ, {"ASSISTANT_DEBUG": value}, clear=False):
            cfg = AssistantConfig.load(path=missing)
        assert cfg.debug is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "False", "NO", ""])
    def test_debug_falsy(self, value, tmp_path):
        missing = tmp_path / "no.json"
        with patch.dict(os.environ, {"ASSISTANT_DEBUG": value}, clear=False):
            cfg = AssistantConfig.load(path=missing)
        assert cfg.debug is False


# ===================================================================
# 10. _apply_env — int coercion (not in default _ENV_MAP, but test
#     the mechanism by temporarily adding one)
# ===================================================================

class TestApplyEnvInt:

    def test_int_coercion_valid(self, tmp_path):
        """Manually wire an int field into _ENV_MAP and verify coercion."""
        missing = tmp_path / "no.json"
        cfg = AssistantConfig()
        cfg._ENV_MAP["ASSISTANT_SCREENSHOT_WIDTH"] = "screenshot_max_width"
        with patch.dict(os.environ, {"ASSISTANT_SCREENSHOT_WIDTH": "640"}, clear=False):
            cfg = AssistantConfig._apply_env(cfg)
        assert cfg.screenshot_max_width == 640

    def test_int_coercion_invalid(self, tmp_path):
        """Non-numeric value logs a warning and leaves the field unchanged."""
        cfg = AssistantConfig()
        cfg._ENV_MAP["ASSISTANT_SCREENSHOT_WIDTH"] = "screenshot_max_width"
        with patch.dict(os.environ, {"ASSISTANT_SCREENSHOT_WIDTH": "not_a_number"}, clear=False):
            cfg = AssistantConfig._apply_env(cfg)
        assert cfg.screenshot_max_width == 1280  # unchanged default


# ===================================================================
# 11. Env var precedence over JSON
# ===================================================================

class TestEnvPrecedence:

    def test_env_overrides_json(self, tmp_settings_file):
        path = tmp_settings_file({"anthropic_api_key": "sk-from-json"})
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-from-env"}, clear=False):
            cfg = AssistantConfig.load(path=path)
        assert cfg.anthropic_api_key == "sk-from-env"

    def test_env_overrides_json_bool(self, tmp_settings_file):
        path = tmp_settings_file({"debug": False})
        with patch.dict(os.environ, {"ASSISTANT_DEBUG": "true"}, clear=False):
            cfg = AssistantConfig.load(path=path)
        assert cfg.debug is True


# ===================================================================
# 12. to_dict()
# ===================================================================

class TestToDict:

    def test_api_key_masked(self):
        cfg = AssistantConfig(anthropic_api_key="sk-secret")
        d = cfg.to_dict()
        assert d["anthropic_api_key"] == "***"

    def test_empty_api_key_not_masked(self):
        cfg = AssistantConfig()
        d = cfg.to_dict()
        assert d["anthropic_api_key"] == ""

    def test_to_dict_contains_all_top_level_keys(self):
        cfg = AssistantConfig()
        d = cfg.to_dict()
        expected_keys = {
            "anthropic_api_key", "ollama_base_url", "dashboard_url",
            "avatar", "voice", "claude",
            "debug", "log_level", "screenshot_max_width", "webcam_index",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_avatar_keys(self):
        d = AssistantConfig().to_dict()
        expected = {"size", "position_x", "position_y", "animation_fps",
                    "idle_animation", "speaking_animation", "sprite_sheet"}
        assert set(d["avatar"].keys()) == expected

    def test_to_dict_voice_keys(self):
        d = AssistantConfig().to_dict()
        expected = {"tts_voice_th", "tts_voice_en", "tts_rate", "tts_volume",
                    "wake_word", "stt_model", "silence_threshold", "silence_duration"}
        assert set(d["voice"].keys()) == expected

    def test_to_dict_claude_keys(self):
        d = AssistantConfig().to_dict()
        expected = {"model", "max_tokens", "temperature", "system_prompt"}
        assert set(d["claude"].keys()) == expected

    def test_to_dict_values_match_config(self):
        cfg = AssistantConfig(debug=True, log_level="DEBUG", webcam_index=2)
        d = cfg.to_dict()
        assert d["debug"] is True
        assert d["log_level"] == "DEBUG"
        assert d["webcam_index"] == 2


# ===================================================================
# 13. Edge cases
# ===================================================================

class TestEdgeCases:

    def test_empty_json_object(self, tmp_settings_file):
        path = tmp_settings_file({})
        cfg = AssistantConfig.load(path=path)
        assert cfg.to_dict() == AssistantConfig().to_dict()

    def test_unreadable_file_falls_back_to_defaults(self, tmp_path):
        """Simulate an OSError when reading the settings file."""
        p = tmp_path / "settings.json"
        p.write_text("{}", encoding="utf-8")
        # Patch Path.read_text to raise OSError for this specific path.
        with patch.object(type(p), "read_text", side_effect=OSError("Permission denied (mock)")):
            cfg = AssistantConfig.load(path=p)
        assert isinstance(cfg, AssistantConfig)

    def test_nested_avatar_merge_changes_one_preserves_rest(self, tmp_settings_file):
        path = tmp_settings_file({"avatar": {"speaking_animation": "wave"}})
        cfg = AssistantConfig.load(path=path)
        assert cfg.avatar.speaking_animation == "wave"
        assert cfg.avatar.size == 180
        assert cfg.avatar.animation_fps == 30
        assert cfg.avatar.idle_animation == "breathe"
        assert cfg.avatar.position_x == -1

    def test_nested_claude_merge_changes_one_preserves_rest(self, tmp_settings_file):
        path = tmp_settings_file({"claude": {"model": "claude-opus-4-20250514"}})
        cfg = AssistantConfig.load(path=path)
        assert cfg.claude.model == "claude-opus-4-20250514"
        assert cfg.claude.max_tokens == 8192
        assert cfg.claude.temperature == 0.3

    def test_nested_voice_merge_changes_one_preserves_rest(self, tmp_settings_file):
        path = tmp_settings_file({"voice": {"silence_duration": 2.0}})
        cfg = AssistantConfig.load(path=path)
        assert cfg.voice.silence_duration == 2.0
        assert cfg.voice.tts_voice_th == "th-TH-PremwadeeNeural"
        assert cfg.voice.wake_word == "assistant"
