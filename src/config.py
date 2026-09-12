"""Runtime configuration management with Pydantic v2 validation."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Authentication & Upstream Models
    gemini_api_key: SecretStr = Field(
        ...,
        description="Google AI Studio authentication key",
    )
    gemini_model: str = Field(
        default="gemini-3.6-flash",
        description="Google Gemini model identifier",
    )

    # Server Runtime
    app_env: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Application deployment environment",
    )
    host: str = Field(
        default="0.0.0.0",
        description="Gateway binding IP address",
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Gateway listener TCP port",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging verbosity level",
    )

    # Audio Constants
    sample_rate: int = Field(
        default=16000,
        description="Inbound PCM audio sample rate in Hz",
    )
    frame_size: int = Field(
        default=512,
        description="Number of audio samples per inbound frame (32ms at 16kHz)",
    )
    channels: int = Field(
        default=1,
        description="Number of audio channels (1 = mono)",
    )
    max_buffer_chunks: int = Field(
        default=50,
        ge=1,
        description="Maximum inbound audio queue length before dropping frames",
    )

    # Model Parameters: STT (faster-whisper / CTranslate2)
    whisper_model_name: str = Field(
        default="base.en",
        description="Faster-whisper model identifier",
    )
    whisper_compute_type: str = Field(
        default="int8",
        description="CTranslate2 quantization compute type",
    )
    whisper_cpu_threads: int = Field(
        default_factory=lambda: min(4, os.cpu_count() or 2),
        ge=1,
        description="Number of worker threads for STT inference",
    )

    @field_validator("whisper_cpu_threads", mode="before")
    @classmethod
    def validate_cpu_threads(cls, v: int | None) -> int:
        """Default to min(4, cpu_count) if unset or 0."""
        if v is None or v == 0:
            return min(4, os.cpu_count() or 2)
        return int(v)

    # Model Parameters: VAD (Silero-VAD ONNX)
    # Tuning guidance: keep the silence window (vad_silence_ms) in the 400-1000ms range.
    # Shorter than the speaker's natural mid-sentence pause clips trailing words,
    # truncating transcripts; much longer delays the end-of-turn before the assistant replies.
    # vad_min_speech_ms rejects coughs/clicks shorter than the cutoff; vad_pre_padding_frames
    # retains the first ~96ms (3 frames @32ms) of pre-speech context so onset is never lost.
    vad_threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Speech probability threshold for Silero VAD",
    )
    vad_silence_ms: int = Field(
        default=400,
        ge=50,
        description="Sustained silence duration in milliseconds to trigger utterance boundary",
    )
    vad_min_speech_ms: int = Field(
        default=250,
        ge=50,
        description="Minimum contiguous speech duration in milliseconds to accept as a valid utterance",
    )
    vad_pre_padding_frames: int = Field(
        default=3,
        ge=0,
        description="Number of pre-speech frames retained as padding before detected speech (96ms at 32ms/frame)",
    )

    # Conversation Memory
    llm_memory_turns: int = Field(
        default=8,
        ge=0,
        description="Number of prior user/assistant turn pairs retained as LLM conversation memory",
    )

    # Local Model Cache Paths
    model_cache_dir: Path = Field(
        default=Path("./models"),
        description="Root directory for local model storage",
    )
    vad_model_path: Path = Field(
        default=Path("./models/vad/silero_vad.onnx"),
        description="Path to Silero VAD ONNX model file",
    )
    whisper_model_dir: Path = Field(
        default=Path("./models/stt"),
        description="Directory for faster-whisper model files",
    )
    kokoro_model_path: Path = Field(
        default=Path("./models/tts/kokoro-v0_19.onnx"),
        description="Path to Kokoro-82M ONNX model file",
    )
    kokoro_voices_path: Path = Field(
        default=Path("./models/tts/voices.bin"),
        description="Path to Kokoro voices binary embedding file",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()
