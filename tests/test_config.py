"""Unit tests for configuration validation and settings."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import Settings, get_settings


def test_missing_required_keys_raise_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify missing GEMINI_API_KEY raises ValidationError."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    errors = exc_info.value.errors()
    assert any(err["loc"] == ("gemini_api_key",) for err in errors)


def test_defaults_populated_correctly() -> None:
    """Verify all default values are populated accurately."""
    settings = Settings(gemini_api_key="test_dummy_key", _env_file=None)

    assert settings.gemini_api_key.get_secret_value() == "test_dummy_key"
    assert settings.gemini_model == "gemini-3.6-flash"
    assert settings.app_env == "development"
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.log_level == "INFO"

    # Audio constants
    assert settings.sample_rate == 16000
    assert settings.frame_size == 512
    assert settings.channels == 1
    assert settings.max_buffer_chunks == 50

    # Model parameters
    import os

    assert settings.whisper_model_name == "base.en"
    assert settings.whisper_compute_type == "int8"
    assert settings.whisper_cpu_threads == min(4, os.cpu_count() or 2)
    assert settings.vad_threshold == 0.35
    assert settings.vad_silence_ms == 400

    # Cache paths
    assert settings.model_cache_dir == Path("./models")
    assert settings.vad_model_path == Path("./models/vad/silero_vad.onnx")
    assert settings.whisper_model_dir == Path("./models/stt")
    assert settings.kokoro_model_path == Path("./models/tts/kokoro-v0_19.onnx")
    assert settings.kokoro_voices_path == Path("./models/tts/voices.bin")


def test_custom_env_vars_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify custom environment variables override default settings."""
    monkeypatch.setenv("GEMINI_API_KEY", "override_secret_key")
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("HOST", "127.0.0.1")
    monkeypatch.setenv("PORT", "9000")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("SAMPLE_RATE", "24000")
    monkeypatch.setenv("FRAME_SIZE", "1024")
    monkeypatch.setenv("CHANNELS", "2")
    monkeypatch.setenv("MAX_BUFFER_CHUNKS", "100")
    monkeypatch.setenv("WHISPER_MODEL_NAME", "base.en")
    monkeypatch.setenv("WHISPER_COMPUTE_TYPE", "float32")
    monkeypatch.setenv("WHISPER_CPU_THREADS", "4")
    monkeypatch.setenv("VAD_THRESHOLD", "0.75")
    monkeypatch.setenv("VAD_SILENCE_MS", "500")
    monkeypatch.setenv("MODEL_CACHE_DIR", "/custom/models")

    settings = Settings(_env_file=None)

    assert settings.gemini_api_key.get_secret_value() == "override_secret_key"
    assert settings.app_env == "production"
    assert settings.host == "127.0.0.1"
    assert settings.port == 9000
    assert settings.log_level == "DEBUG"
    assert settings.sample_rate == 24000
    assert settings.frame_size == 1024
    assert settings.channels == 2
    assert settings.max_buffer_chunks == 100
    assert settings.whisper_model_name == "base.en"
    assert settings.whisper_compute_type == "float32"
    assert settings.whisper_cpu_threads == 4
    assert settings.vad_threshold == 0.75
    assert settings.vad_silence_ms == 500
    assert settings.model_cache_dir == Path("/custom/models")


def test_invalid_ranges_raise_validation_error() -> None:
    """Verify boundary checks on port and VAD threshold."""
    with pytest.raises(ValidationError):
        Settings(gemini_api_key="key", port=70000, _env_file=None)

    with pytest.raises(ValidationError):
        Settings(gemini_api_key="key", vad_threshold=1.5, _env_file=None)


def test_get_settings_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify get_settings returns a cached instance."""
    monkeypatch.setenv("GEMINI_API_KEY", "cached_key")
    get_settings.cache_clear()
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    get_settings.cache_clear()


def test_whisper_cpu_threads_zero_fallback() -> None:
    """Verify whisper_cpu_threads=0 defaults to min(4, cpu_count)."""
    import os

    settings = Settings(gemini_api_key="key", whisper_cpu_threads=0, _env_file=None)
    assert settings.whisper_cpu_threads == min(4, os.cpu_count() or 2)
