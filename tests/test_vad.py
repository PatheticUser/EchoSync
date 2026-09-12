"""Unit tests and benchmarks for Silero VAD edge engine."""

import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.core.vad import SileroVAD, VADEvent, VADState

MODEL_PATH = Path("models/vad/silero_vad.onnx")


@pytest.fixture
def vad() -> SileroVAD:
    """Initialize SileroVAD instance with test model path."""
    return SileroVAD(
        model_path=MODEL_PATH,
        sample_rate=16000,
        frame_size=512,
        threshold=0.5,
        silence_ms=400,
        min_speech_ms=250,
        pre_speech_padding_frames=3,
    )


def test_vad_initialization_not_found() -> None:
    """Verify FileNotFoundError when model path does not exist."""
    with pytest.raises(FileNotFoundError):
        SileroVAD(model_path=Path("non_existent_model.onnx"))


def test_vad_normalization(vad: SileroVAD) -> None:
    """Verify PCM bytes and numpy array normalization."""
    # 512 int16 samples as bytes (1024 bytes)
    raw_int16 = np.full(512, 16384, dtype=np.int16)
    raw_bytes = raw_int16.tobytes()

    norm_bytes = vad.normalize_frame(raw_bytes)
    assert norm_bytes.shape == (512,)
    assert norm_bytes.dtype == np.float32
    assert np.isclose(norm_bytes[0], 0.5, atol=1e-4)

    # Already normalized float32
    raw_float = np.full(512, 0.75, dtype=np.float32)
    norm_float = vad.normalize_frame(raw_float)
    assert norm_float.shape == (512,)
    assert np.isclose(norm_float[0], 0.75)

    # Invalid frame length raises ValueError
    with pytest.raises(ValueError):
        vad.normalize_frame(np.zeros(256, dtype=np.float32))

    # Invalid type raises TypeError
    with pytest.raises(TypeError):
        vad.normalize_frame("not_audio")  # type: ignore[arg-type]


def test_vad_silence_inference(vad: SileroVAD) -> None:
    """Verify silence frames produce near-zero speech probability."""
    silence_frame = np.zeros(512, dtype=np.float32)
    prob = vad._infer_probability(silence_frame)
    assert 0.0 <= prob <= 0.05


def test_vad_inference_latency_benchmark(vad: SileroVAD) -> None:
    """Verify single-frame inference latency stays well below 5ms SLA."""
    silence_frame = np.zeros(512, dtype=np.float32)

    # Warmup
    for _ in range(10):
        vad._infer_probability(silence_frame)

    latencies_ms: list[float] = []
    for _ in range(100):
        t0 = time.perf_counter()
        vad._infer_probability(silence_frame)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    mean_latency = float(np.mean(latencies_ms))
    p95_latency = float(np.percentile(latencies_ms, 95))

    # Strict SLA assertions: frame size is 32ms, so mean < 5.0ms and p95 < 10.0ms provides 3x real-time margin
    assert mean_latency < 5.0, f"Mean latency {mean_latency:.3f}ms exceeded 5.0ms SLA"
    assert p95_latency < 10.0, f"P95 latency {p95_latency:.3f}ms exceeded 10.0ms SLA"


def test_vad_state_machine_full_utterance(vad: SileroVAD) -> None:
    """Verify full state machine cycle: SILENCE -> SPEECH_ACTIVE -> SPEECH_END."""
    frame = np.zeros(512, dtype=np.float32)

    # Phase A: Pure silence (initial state)
    with patch.object(vad, "_infer_probability", return_value=0.01):
        for _ in range(5):
            event = vad.process_frame(frame)
            assert event.state == VADState.SILENCE
            assert event.audio_buffer is None

    # Phase B: Single speech frame (insufficient to trigger active speech)
    with patch.object(vad, "_infer_probability", return_value=0.9):
        event = vad.process_frame(frame)
        assert event.state == VADState.SILENCE
        assert event.audio_buffer is None

    # Phase C: Second speech frame (triggers SPEECH_ACTIVE)
    with patch.object(vad, "_infer_probability", return_value=0.9):
        event = vad.process_frame(frame)
        assert event.state == VADState.SPEECH_ACTIVE
        assert event.audio_buffer is None

    # Phase D: Sustained speech for 10 frames (10 * 32ms = 320ms > 250ms min threshold)
    with patch.object(vad, "_infer_probability", return_value=0.9):
        for _ in range(10):
            event = vad.process_frame(frame)
            assert event.state == VADState.SPEECH_ACTIVE

    # Phase E: Sustained silence triggering SPEECH_END (silence_threshold_frames = 13)
    final_event: VADEvent | None = None
    with patch.object(vad, "_infer_probability", return_value=0.05):
        for _ in range(vad.silence_threshold_frames):
            event = vad.process_frame(frame)
            if event.state == VADState.SPEECH_END:
                final_event = event

    assert final_event is not None
    assert final_event.state == VADState.SPEECH_END
    assert final_event.audio_buffer is not None
    # Audio buffer contains pre_buffer (3 frames: 2 silence + 1st speech) + 2nd trigger frame + 10 sustained frames = 14 frames
    assert len(final_event.audio_buffer) == 14 * 512
    assert final_event.duration_ms == 14 * 32.0


def test_vad_state_machine_rejects_transient_noise(vad: SileroVAD) -> None:
    """Verify speech burst shorter than min_speech_ms (250ms) is discarded."""
    frame = np.zeros(512, dtype=np.float32)

    # Trigger speech (2 frames)
    with patch.object(vad, "_infer_probability", return_value=0.95):
        vad.process_frame(frame)
        vad.process_frame(frame)

    # Only 1 additional speech frame (total 3 frames = 96ms < 250ms)
    with patch.object(vad, "_infer_probability", return_value=0.95):
        vad.process_frame(frame)

    # Followed immediately by silence
    end_events: list[VADEvent] = []
    with patch.object(vad, "_infer_probability", return_value=0.01):
        for _ in range(vad.silence_threshold_frames):
            event = vad.process_frame(frame)
            if event.state == VADState.SPEECH_END:
                end_events.append(event)

    # Discarded noise emits no SPEECH_END, returns to SILENCE
    assert len(end_events) == 0
    assert vad._current_state == VADState.SILENCE


def test_vad_silence_threshold_frames_math() -> None:
    """Verify silence_ms converts to frames via ceil at 32ms/frame.

    Covers the tuning window 400-1000ms; 600ms -> ceil(600/32) = 19 frames.
    """
    cases = {400: 13, 600: 19, 800: 25, 1000: 32}
    for silence_ms, expected_frames in cases.items():
        vad = SileroVAD(
            model_path=MODEL_PATH,
            sample_rate=16000,
            frame_size=512,
            threshold=0.5,
            silence_ms=silence_ms,
        )
        assert vad.silence_threshold_frames == expected_frames


def test_vad_min_speech_frames_math() -> None:
    """Verify min_speech_ms converts to frames via ceil at 32ms/frame."""
    cases = {250: 8, 400: 13, 600: 19}
    for min_speech_ms, expected_frames in cases.items():
        vad = SileroVAD(
            model_path=MODEL_PATH,
            sample_rate=16000,
            frame_size=512,
            threshold=0.5,
            min_speech_ms=min_speech_ms,
        )
        assert vad.min_speech_frames == expected_frames


def test_vad_constructor_accepts_tuning_overrides() -> None:
    """Verify non-default VAD tuning knobs are accepted as constructor overrides."""
    vad = SileroVAD(
        model_path=MODEL_PATH,
        sample_rate=16000,
        frame_size=512,
        threshold=0.5,
        silence_ms=600,
        min_speech_ms=400,
        pre_speech_padding_frames=5,
    )
    assert vad.silence_ms == 600
    assert vad.min_speech_ms == 400
    assert vad.pre_speech_padding_frames == 5
    assert vad.silence_threshold_frames == 19
    assert vad.min_speech_frames == 13
    assert vad._pre_buffer.maxlen == 5


@pytest.mark.asyncio
async def test_vad_async_process_frame(vad: SileroVAD) -> None:
    """Verify async_process_frame non-blocking offload via asyncio.to_thread."""
    frame = np.zeros(512, dtype=np.float32)
    event = await vad.async_process_frame(frame)
    assert isinstance(event, VADEvent)
    assert event.state == VADState.SILENCE
