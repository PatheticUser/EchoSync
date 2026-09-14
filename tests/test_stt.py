"""Unit tests and benchmarks for local faster-whisper STT subsystem."""

import os
import wave
from pathlib import Path

import numpy as np
import pytest

from src.core.stt import TranscriptionResult, WhisperSTT

MODEL_PATH = Path("./models/stt")
FIXTURE_WAV = Path("tests/fixtures/reference.wav")

pytestmark = pytest.mark.requires_models


@pytest.fixture(scope="module")
def stt() -> WhisperSTT:
    """Initialize cached WhisperSTT instance for tests."""
    if not MODEL_PATH.exists():
        pytest.skip("models/stt/ not present; run scripts/download_models.py first")
    return WhisperSTT(
        model_name="tiny.en",
        compute_type="int8",
        cpu_threads=min(4, os.cpu_count() or 2),
        download_root="./models/stt",
        sample_rate=16000,
    )


def test_stt_warmup(stt: WhisperSTT) -> None:
    """Verify warmup passes 0.2s zero audio to initialize engine graph."""
    stt.warmup()


def test_audio_normalization(stt: WhisperSTT) -> None:
    """Verify conversion of int16 PCM bytes and numpy arrays to float32."""
    raw_int16 = np.full(16000, 16384, dtype=np.int16)
    raw_bytes = raw_int16.tobytes()

    norm_bytes = stt.normalize_audio(raw_bytes)
    assert norm_bytes.shape == (16000,)
    assert norm_bytes.dtype == np.float32
    assert np.isclose(norm_bytes[0], 0.5, atol=1e-4)

    raw_float = np.full(8000, -0.25, dtype=np.float32)
    norm_float = stt.normalize_audio(raw_float)
    assert norm_float.shape == (8000,)
    assert np.isclose(norm_float[0], -0.25)

    with pytest.raises(ValueError):
        stt.normalize_audio(b"")

    with pytest.raises(ValueError):
        stt.normalize_audio(np.array([], dtype=np.float32))

    with pytest.raises(TypeError):
        stt.normalize_audio(12345)  # type: ignore[arg-type]


def test_transcribe_reference_wav(stt: WhisperSTT) -> None:
    """Verify transcription text accuracy and execution speed on reference WAV."""
    assert FIXTURE_WAV.is_file(), f"Fixture WAV not found at {FIXTURE_WAV}"

    with wave.open(str(FIXTURE_WAV), "rb") as wf:
        n_frames = wf.getnframes()
        pcm_bytes = wf.readframes(n_frames)

    # Pre-warm engine
    stt.warmup()

    result = stt.transcribe(pcm_bytes)
    assert isinstance(result, TranscriptionResult)
    assert len(result.text) > 0

    normalized_text = result.text.lower()
    assert "hello" in normalized_text
    assert "welcome" in normalized_text or "echo" in normalized_text

    # Verify duration and RTF metrics
    assert result.audio_duration_s > 2.0
    rtf = (result.duration_ms / 1000.0) / result.audio_duration_s
    # Verify reasonable CPU inference latency (RTF < 1.50 on CPU / constrained CI runner)
    assert rtf < 1.50, f"RTF {rtf:.3f} exceeded 1.50 limit"


@pytest.mark.asyncio
async def test_async_transcribe(stt: WhisperSTT) -> None:
    """Verify async_transcribe non-blocking execution via worker thread."""
    silence = np.zeros(16000, dtype=np.float32)
    result = await stt.async_transcribe(silence)
    assert isinstance(result, TranscriptionResult)
    assert result.audio_duration_s == 1.0
