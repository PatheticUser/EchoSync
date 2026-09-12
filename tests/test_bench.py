"""Unit tests for the latency-benchmark harness (GAPS T1.6)."""

from pathlib import Path

import numpy as np

from scripts.bench_turn import load_wav, quantile


def make_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    import wave

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes((samples * 32768.0).clip(-32768, 32767).astype("<i2").tobytes())


def test_quantile_median(tmp_path: Path) -> None:
    assert quantile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert quantile([100.0], 0.95) == 100.0
    assert np.isnan(quantile([], 0.5))


def test_load_wav_resamples_mono_to_target_rate(tmp_path: Path) -> None:
    wav = tmp_path / "tone.wav"
    tone = np.sin(2 * np.pi * 440 * np.arange(8000) / 16000).astype(np.float32)
    make_wav(wav, tone, 8000)
    out = load_wav(wav, 16000)
    assert out.dtype == np.float32
    assert len(out) > 8000  # 8k input resampled up to 16k target
    assert np.max(np.abs(out)) <= 1.0
