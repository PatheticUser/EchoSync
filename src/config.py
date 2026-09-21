"""Runtime configuration management with Pydantic v2 validation."""

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


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

    whisper_initial_prompt: str = Field(
        default="A conversational voice assistant discussing code, Linux, technology, and daily tasks.",
        description="Initial vocabulary and domain bias prompt passed to Whisper decoder",
    )
    stt_boost_audio: bool = Field(
        default=True,
        description="Peak-normalize quiet audio segments to improve Whisper signal-to-noise ratio",
    )
    llm_inject_tools_context: bool = Field(
        default=True,
        description="Inject live system context and clock into LLM system prompt without latency penalty",
    )

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
    # Barge-in probability gate (GAPS INTERRUPT-PROB): a SPEECH_ACTIVE frame seen while
    # the assistant is speaking only interrupts (barge-in) when its VAD probability reaches
    # this threshold; fainter background noise below it is ignored so model speech continues.
    # Tuning guidance: raise toward 0.95 to make barge-in harder to trigger (fewer false
    # cutoffs from ambient noise), lower toward 0.05 to make interruptions more responsive.
    vad_interrupt_prob: float = Field(
        default=0.75,
        ge=0.05,
        le=0.95,
        description="Minimum VAD speech probability that triggers barge-in during model speech",
    )

    # Conversation Memory
    llm_memory_turns: int = Field(
        default=8,
        ge=0,
        description="Number of prior user/assistant turn pairs retained as LLM conversation memory",
    )

    # Streaming Chunking
    llm_chunk_min_chars: int = Field(
        default=20,
        ge=1,
        description="Minimum character length of a speech clause before the streaming chunker emits it for TTS",
    )
    llm_chunk_early_first: bool = Field(
        default=True,
        description="Emit the first clause at a comma/colon/terminal boundary for low TTS latency; false enforces sentence-terminal punctuation for every chunk",
    )

    # Model Parameters: LLM Generation (Gemini)
    llm_temperature: float = Field(
        default=0.4,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for Gemini text generation, 0.0-2.0",
    )
    llm_top_p: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Nucleus top-p sampling probability for Gemini text generation",
    )
    llm_max_output_tokens: int = Field(
        default=350,
        ge=1,
        description="Maximum number of tokens in a Gemini generation reply",
    )
    llm_timeout_s: float = Field(
        default=5.0,
        gt=0.0,
        description="Per-attempt timeout in seconds for a Gemini streaming request",
    )

    # Model Parameters: TTS Engine (EdgeTTS for free-tier/low-CPU vs Kokoro-82M ONNX for offline)
    tts_engine: Literal["edge", "kokoro"] = Field(
        default="kokoro",
        description="TTS engine: 'edge' (fast zero-CPU streaming, ideal for Railway/free tier) or 'kokoro' (local ONNX)",
    )
    edge_voice: str = Field(
        default="en-US-JennyNeural",
        description="Default voice for EdgeTTS (e.g. en-US-JennyNeural, en-US-GuyNeural, en-GB-SoniaNeural)",
    )

    # Model Parameters: TTS (Kokoro-82M ONNX) — Naturalness Knobs
    # Voice list e.g. af_sarah, af_bella, am_michael, bm_george.
    # Speed 1.0 is the natural default; 0.9-1.1 is typical for conversational
    # pacing; higher values can produce audible clicks.
    tts_voice: str = Field(
        default="af_sarah",
        description="Default Kokoro voice identifier used when no per-utterance voice is requested",
    )
    tts_speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Kokoro speech rate multiplier (1.0 natural; 0.9-1.1 typical)",
    )
    tts_chunk_size: int = Field(
        default=2048,
        ge=256,
        description="Byte size of TTS audio chunks streamed to the client",
    )
    tts_sample_rate: int = Field(
        default=24000,
        description="Outbound audio synthesis sample rate in Hz",
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

    # WebSocket Gateway Access Control (T2.1, T2.2, T2.3)
    ws_allowed_origins: Annotated[list[str], NoDecode] = Field(
        default=["http://127.0.0.1:8000", "http://localhost:8000"],
        description="Allowed Origin header values for the /ws/audio WebSocket endpoint",
    )
    ws_max_concurrent: int = Field(
        default=10,
        ge=1,
        description="Maximum concurrently active WebSocket audio sessions",
    )
    ws_max_per_ip: int = Field(
        default=3,
        ge=1,
        description="Maximum simultaneously active WebSocket sessions per client IP",
    )
    ws_bearer_token: SecretStr | None = Field(
        default=None,
        description="Optional bearer token required via the Authorization header on every /ws/audio connection",
    )

    @field_validator("ws_allowed_origins", mode="before")
    @classmethod
    def parse_ws_allowed_origins(cls, v: object) -> object:
        """Parse WS_ALLOWED_ORIGINS as JSON ``[..]`` or a comma-separated string."""
        if isinstance(v, str):
            txt = v.strip()
            if txt.startswith("["):
                try:
                    parsed = json.loads(txt)
                    if isinstance(parsed, list):
                        return parsed
                except json.JSONDecodeError:
                    pass
            return [item.strip() for item in txt.split(",") if item.strip()]
        return v


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()
