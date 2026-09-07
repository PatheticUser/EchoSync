"""Sanity and verification tests for Silero VAD v5 with context buffer on speech audio."""

import wave
from pathlib import Path

import numpy as np

from src.core.vad import SileroVAD, VADState

MODEL_PATH = Path("models/vad/silero_vad.onnx")
FIXTURE_WAV = Path("tests/fixtures/reference.wav")


def test_vad_live_speech_detection() -> None:
    """Verify Silero VAD v5 detects speech frames on reference WAV with >0.90 probability."""
    vad = SileroVAD(model_path=MODEL_PATH, threshold=0.35)

    with wave.open(str(FIXTURE_WAV), "rb") as wf:
        raw_pcm = wf.readframes(wf.getnframes())

    samples = np.frombuffer(raw_pcm, dtype=np.int16)
    speech_events: list[float] = []
    state_transitions: list[VADState] = []

    for i in range(0, len(samples) - 512, 512):
        chunk = samples[i : i + 512]
        event = vad.process_frame(chunk)
        state_transitions.append(event.state)
        if event.probability >= 0.35:
            speech_events.append(event.probability)

    # Must detect dozens of high-confidence speech frames (probability > 0.90)
    assert len(speech_events) > 30, f"Expected >30 speech frames, got {len(speech_events)}"
    assert max(speech_events) > 0.95, f"Expected max probability >0.95, got {max(speech_events)}"
    # State machine must have transitioned to SPEECH_ACTIVE
    assert VADState.SPEECH_ACTIVE in state_transitions


def test_vad_predict_method() -> None:
    """Verify SileroVAD.predict() helper on voice audio."""
    vad = SileroVAD(model_path=MODEL_PATH, threshold=0.35)

    with wave.open(str(FIXTURE_WAV), "rb") as wf:
        raw_pcm = wf.readframes(wf.getnframes())

    samples = np.frombuffer(raw_pcm, dtype=np.int16)
    # Warm up context on initial frames
    for i in range(12):
        vad.process_frame(samples[i * 512 : (i + 1) * 512])

    # Frame 15 is active speech
    active_chunk = samples[15 * 512 : 16 * 512]
    prob = vad.predict(active_chunk)
    assert prob > 0.90, f"Expected speech probability >0.90, got {prob}"
