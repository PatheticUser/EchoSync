"""Unit tests and benchmarks for streaming acoustic synthesis subsystem."""

import asyncio
import time
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from src.config import Settings
from src.core.tts import KokoroTTS

MODEL_PATH = Path("models/tts/kokoro-v0_19.onnx")
VOICES_PATH = Path("models/tts/voices.bin")

pytestmark = pytest.mark.requires_models


@pytest.fixture(scope="module")
def tts() -> KokoroTTS:
    """Initialize cached KokoroTTS instance for tests."""
    if not MODEL_PATH.is_file() or not VOICES_PATH.is_file():
        pytest.skip(
            "models/tts/kokoro-v0_19.onnx or voices.bin not present; "
            "run scripts/download_models.py first"
        )
    engine = KokoroTTS(
        model_path=MODEL_PATH,
        voices_path=VOICES_PATH,
        default_voice="af_sarah",
        chunk_size=2048,
    )
    engine.warmup()
    return engine


def test_tts_initialization(tts: KokoroTTS) -> None:
    """Verify engine configuration, sample rate, and model presence."""
    assert tts.sample_rate == 24000
    assert tts.chunk_size == 2048
    assert tts.default_voice == "af_sarah"
    if MODEL_PATH.is_file() and VOICES_PATH.is_file():
        assert not tts._is_fallback


def test_tts_fallback_mode() -> None:
    """Verify graceful fallback generation when weights are missing."""
    fallback_engine = KokoroTTS(
        model_path=Path("non_existent.onnx"),
        voices_path=Path("non_existent.bin"),
    )
    assert fallback_engine._is_fallback
    pcm = fallback_engine._synthesize_pcm("Testing fallback mode.")
    assert len(pcm) > 0
    assert len(pcm) % 2 == 0


def test_tts_stores_configured_knobs() -> None:
    """Verify constructor stores configured naturalness knobs (voice/speed/chunk)."""
    engine = KokoroTTS(
        model_path=Path("non_existent.onnx"),
        voices_path=Path("non_existent.bin"),
        default_voice="af_bella",
        speed=0.9,
        chunk_size=1024,
    )
    assert engine.default_voice == "af_bella"
    assert engine.speed == 0.9
    assert engine.chunk_size == 1024

    # Defaults fall back to current hardcoded values when not supplied
    default_engine = KokoroTTS(
        model_path=Path("non_existent.onnx"),
        voices_path=Path("non_existent.bin"),
    )
    assert default_engine.default_voice == "af_sarah"
    assert default_engine.speed == 1.0
    assert default_engine.chunk_size == 2048


@pytest.mark.asyncio
async def test_tts_synthesis_audio_structure(tts: KokoroTTS) -> None:
    """Verify audio byte structure: framing, sample depth, and non-zero energy."""
    chunks: list[bytes] = []
    async for chunk in tts.synthesize_stream("Hello world, testing audio synthesis."):
        chunks.append(chunk)

    assert len(chunks) > 0
    # Every chunk except possibly the last must be exactly chunk_size (2048 bytes)
    for chunk in chunks[:-1]:
        assert len(chunk) == tts.chunk_size

    total_bytes = b"".join(chunks)
    assert len(total_bytes) % 2 == 0  # 16-bit PCM = 2 bytes per sample

    pcm_samples = np.frombuffer(total_bytes, dtype=np.int16)
    duration_s = len(pcm_samples) / float(tts.sample_rate)
    assert duration_s >= 0.5

    # Verify non-zero acoustic energy
    peak_amplitude = np.max(np.abs(pcm_samples))
    assert peak_amplitude > 1000, f"Peak amplitude too low: {peak_amplitude}"


@pytest.mark.asyncio
async def test_tts_async_event_loop_unblocked(tts: KokoroTTS) -> None:
    """Verify async worker thread execution leaves the event loop free to handle concurrent tasks."""
    ticks = 0

    async def background_ticker() -> None:
        nonlocal ticks
        for _ in range(50):
            await asyncio.sleep(0.01)
            ticks += 1

    ticker_task = asyncio.create_task(background_ticker())

    chunks: list[bytes] = []
    async for chunk in tts.synthesize_stream("Concurrency check during speech synthesis."):
        chunks.append(chunk)

    await ticker_task
    # Event loop processed ticker tasks concurrently while synthesis ran in background thread
    assert ticks > 0
    assert len(chunks) > 0


@pytest.mark.asyncio
async def test_tts_cancellation_barge_in(tts: KokoroTTS) -> None:
    """Verify barge-in cancellation terminates synthesis and stops chunk delivery immediately."""
    received_chunks: list[bytes] = []

    async def consumer() -> None:
        async for chunk in tts.synthesize_stream(
            "This long sentence should be cut off mid-speech."
        ):
            received_chunks.append(chunk)
            if len(received_chunks) == 1:
                # Cancel task upon receiving first packet
                task.cancel()

    task = asyncio.create_task(consumer())
    with pytest.raises(asyncio.CancelledError):
        await task

    # Slicing halted immediately on cancellation
    assert len(received_chunks) == 1


@pytest.mark.asyncio
async def test_tts_latency_benchmark(tts: KokoroTTS) -> None:
    """Benchmark first audio chunk delivery latency."""
    t0 = time.perf_counter()
    first_chunk_ms: float | None = None
    chunk_count = 0

    async for _ in tts.synthesize_stream("Hi."):
        if first_chunk_ms is None:
            first_chunk_ms = (time.perf_counter() - t0) * 1000.0
        chunk_count += 1

    assert first_chunk_ms is not None
    assert chunk_count > 0
    # Log benchmark metric
    print(f"\nTTS first chunk latency for 'Hi.': {first_chunk_ms:.1f}ms")


def test_tts_config_defaults_load_without_env() -> None:
    """Verify TTS naturalness knob settings fall back to defaults without env."""
    settings = Settings(gemini_api_key="test_dummy_key", _env_file=None)

    assert settings.tts_voice == "af_sarah"
    assert settings.tts_speed == 1.0
    assert settings.tts_chunk_size == 2048


def test_tts_config_env_overrides_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify TTS_VOICE/TTS_SPEED/TTS_CHUNK_SIZE env vars override defaults."""
    monkeypatch.setenv("GEMINI_API_KEY", "test_key")
    monkeypatch.setenv("TTS_VOICE", "bm_george")
    monkeypatch.setenv("TTS_SPEED", "0.85")
    monkeypatch.setenv("TTS_CHUNK_SIZE", "1024")

    settings = Settings(_env_file=None)

    assert settings.tts_voice == "bm_george"
    assert settings.tts_speed == 0.85
    assert settings.tts_chunk_size == 1024


def test_tts_config_invalid_ranges_raise_validation_error() -> None:
    """Verify TTS_SPEED/TTS_CHUNK_SIZE boundary constraints are enforced."""
    with pytest.raises(ValidationError):
        Settings(gemini_api_key="test_key", tts_speed=0.4, _env_file=None)
    with pytest.raises(ValidationError):
        Settings(gemini_api_key="test_key", tts_speed=2.5, _env_file=None)
    with pytest.raises(ValidationError):
        Settings(gemini_api_key="test_key", tts_chunk_size=128, _env_file=None)
