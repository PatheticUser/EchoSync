#!/usr/bin/env python3
"""Latency budget harness for EchoSync voice turns (GAPS T1.6 / T1.1).

Drives a live WebSocket session with a canned ``int16`` mono WAV utterance
followed by trailing silence, then parses the per-turn ``metrics`` block the
server already emits inside the final ``LISTENING`` status message. Prints a
per-stage table (STT / LLM TTFT / TTS first chunk / total RTT) across
``--runs`` and a p50/p95 summary.

Usage: python scripts/bench_turn.py --audio clap.wav --runs 3

The server under test must be running (``uv run python scripts/run.py`` works).
Whisper model under test is chosen by WHISPER_MODEL_NAME on the server side, so
the same harness doubles as the T1.1 tiny/base/small A/B rig.

Regression gate: exits non-zero when filtered p50 total RTT exceeds
``--gate-total`` (default 2.5s) or p50 TTFT exceeds ``--gate-ttft`` (1.0s).
Writes an optional JSON summary via ``--json out.json`` for CI comparison.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import get_settings


def load_wav(path: Path, sample_rate: int) -> np.ndarray:
    """Return float32 mono samples at ``sample_rate`` (-1..1) from a WAV file."""
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() not in (1, 2):
            raise ValueError(f"Unsupported channel count: {wf.getnchannels()}")
        raw = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32)
        raw /= 32768.0

        src_rate = wf.getframerate()
        if wf.getnchannels() == 2:
            raw = raw.reshape(-1, 2).mean(axis=1)
        if src_rate != sample_rate:
            # Linear resample is good enough for latency benchmarking.
            n_out = round(len(raw) * sample_rate / src_rate)
            indices = np.linspace(0, len(raw) - 1, n_out)
            raw = np.interp(indices, np.arange(len(raw)), raw)
            raw = raw.astype(np.float32)
        return raw


def quantile(values: list[float], q: float) -> float:
    """Interpolated percentile. Returns NaN for empty input."""
    if not values:
        return math.nan
    arr = np.asarray(sorted(values), dtype=np.float64)
    return float(np.quantile(arr, q))


def format_ms(value: float) -> str:
    return f"{value:.0f}" if not math.isnan(value) else "  -"


async def run_turn(ws: Any, audio: np.ndarray, frame_bytes: int, trailing_silence_s: float) -> dict:
    """Run one utterance through the session; return the server-side metrics dict."""
    metrics: dict | None = None

    # Drain the initial LISTENING status.
    init = json.loads(await ws.recv())
    assert init["data"]["state"] == "LISTENING", init

    # Feed utterance + trailing silence so the VAD state machine emits SPEECH_END.
    silence = np.zeros(int(get_settings().sample_rate * trailing_silence_s), dtype=np.float32)
    samples = np.concatenate([audio, silence])
    pcm = (samples * 32768.0).clip(-32768, 32767).astype("<i2")
    payload = pcm.tobytes()
    for i in range(0, len(payload), frame_bytes):
        await ws.send(payload[i : i + frame_bytes])

    async def drain() -> dict:
        nonlocal metrics
        while True:
            msg = await ws.recv()
            if isinstance(msg, (bytes, bytearray)):
                continue  # audio frame, irrelevant to latency measurement
            msg = json.loads(msg)
            mtype = msg.get("type")
            if mtype == "transcript":
                metrics = {"transcript": msg["data"].get("text", "")}
            elif mtype == "status":
                data = msg.get("data", {})
                if data.get("state") == "LISTENING" and "metrics" in data:
                    stage = data["metrics"]
                    stage["transcript"] = (metrics or {}).get("transcript", "")
                return stage
            # audio_header / audio bytes are irrelevant to latency measurement.

    return await drain()


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8000/ws/audio")
    parser.add_argument("--audio", required=True, type=Path, help="int16 mono WAV utterance")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--trailing-silence", type=float, default=0.8, help="seconds of silence after utterance"
    )
    parser.add_argument("--gate-total", type=float, default=2.5, help="p50 total RTT gate, seconds")
    parser.add_argument("--gate-ttft", type=float, default=1.0, help="p50 LLM TTFT gate, seconds")
    parser.add_argument("--json", type=Path, help="write p50/p95 summary JSON")
    args = parser.parse_args()

    settings = get_settings()
    frame_bytes = settings.frame_size * 2  # int16 mono
    audio = load_wav(args.audio, settings.sample_rate)
    if len(audio) == 0:
        print(f"[bench] ERROR: empty audio in {args.audio}")
        return 2

    print(
        f"[bench] target={args.url} runs={args.runs} frame={settings.frame_size} "
        f"whisper={settings.whisper_model_name} tts={settings.tts_voice}"
    )
    print(f"[bench] audio={args.audio.name} duration={len(audio) / settings.sample_rate:.2f}s\n")

    import websockets

    rows: list[dict] = []
    async with websockets.connect(args.url) as ws:
        for n in range(1, args.runs + 1):
            try:
                stage = await run_turn(ws, audio, frame_bytes, args.trailing_silence)
            except Exception as exc:  # noqa: BLE001
                print(f"[bench] run {n}: FAILED ({exc.__class__.__name__}: {exc})")
                rows.append({})
            else:
                rows.append(stage)
                t = stage.get("transcript", "")
                print(
                    f"[bench] run {n}: stt={format_ms(stage.get('stt_ms', math.nan))}ms "
                    f"ttft={format_ms(stage.get('llm_ttft_ms', math.nan))}ms "
                    f"tts={format_ms(stage.get('tts_first_chunk_ms', math.nan))}ms "
                    f"rtt={format_ms(stage.get('total_rtt_ms', math.nan))}ms "
                    f"| transcript: {t[:60] or '(empty)'}"
                )

    def series(key: str) -> list[float]:
        return [float(r.get(key)) for r in rows if r and r.get(key) is not None]

    def summary(key: str, label: str) -> tuple[float, float]:
        vals = series(key)
        return quantile(vals, 0.5), quantile(vals, 0.95)

    p50_rtt, _p95_rtt = summary("total_rtt_ms", "RTT")
    p50_ttft, _ = summary("llm_ttft_ms", "TTFT")

    print("\n[bench] summary (p50 / p95, ms):")
    for key, label in (
        ("stt_ms", "STT"),
        ("llm_ttft_ms", "LLM TTFT"),
        ("tts_first_chunk_ms", "TTS 1st chunk"),
        ("total_rtt_ms", "Total RTT"),
    ):
        p50, p95 = summary(key, label)
        print(f"[bench]   {label:<14} {format_ms(p50):>6} / {format_ms(p95):>6}")

    ok = True
    if not math.isnan(p50_rtt):
        ok &= p50_rtt <= args.gate_total * 1000
        ok &= p50_ttft <= args.gate_ttft * 1000
        print(
            f"\n[bench] gate: p50 RTT {'PASS' if p50_rtt <= args.gate_total * 1000 else 'FAIL'} "
            f"({p50_rtt / 1000:.2f}s <= {args.gate_total}s) | "
            f"p50 TTFT {'PASS' if p50_ttft <= args.gate_ttft * 1000 else 'FAIL'} "
            f"({p50_ttft / 1000:.2f}s <= {args.gate_ttft}s)"
        )
    else:
        ok = False
        print("\n[bench] gate: FAIL (no completed turns measured)")

    if args.json:
        out = {
            "runs": rows,
            "p50": {
                k: quantile(series(k), 0.5)
                for k in ("stt_ms", "llm_ttft_ms", "tts_first_chunk_ms", "total_rtt_ms")
            },
            "p95": {
                k: quantile(series(k), 0.95)
                for k in ("stt_ms", "llm_ttft_ms", "tts_first_chunk_ms", "total_rtt_ms")
            },
            "gate": {"rtt_p50_s": args.gate_total, "ttft_p50_s": args.gate_ttft, "pass": ok},
            "settings": {
                "whisper_model": settings.whisper_model_name,
                "tts_voice": settings.tts_voice,
            },
        }
        args.json.write_text(json.dumps(out, indent=2))
        print(f"[bench] wrote {args.json}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
