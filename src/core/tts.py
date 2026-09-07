import asyncio
import logging
import re
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import numpy as np

# Suppress harmless phonemizer word count mismatch warnings
logging.getLogger("phonemizer").setLevel(logging.ERROR)


class KokoroTTS:
    """Acoustic synthesis engine producing 24kHz 16-bit PCM audio streams."""

    def __init__(
        self,
        model_path: Path | str = "./models/tts/kokoro-v0_19.onnx",
        voices_path: Path | str = "./models/tts/voices.bin",
        default_voice: str = "af_sarah",
        speed: float = 1.0,
        sample_rate: int = 24000,
        chunk_size: int = 2048,
    ) -> None:
        self.model_path = Path(model_path)
        self.voices_path = Path(voices_path)
        self.default_voice = default_voice
        self.speed = speed
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

        self._kokoro: Any | None = None
        self._is_fallback = False

        if self.model_path.is_file() and self.voices_path.is_file():
            try:
                import onnxruntime as rt
                from kokoro_onnx import Kokoro

                sess_opts = rt.SessionOptions()
                sess_opts.intra_op_num_threads = 4
                sess_opts.inter_op_num_threads = 1
                sess_opts.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
                sess_opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

                session = rt.InferenceSession(
                    str(self.model_path),
                    sess_options=sess_opts,
                    providers=["CPUExecutionProvider"],
                )
                self._kokoro = Kokoro.from_session(session, str(self.voices_path))
            except (RuntimeError, OSError, ValueError, KeyError, ImportError):
                self._is_fallback = True
        else:
            self._is_fallback = True

    def warmup(self) -> None:
        """Pre-warm ONNX graph execution with a minimal phoneme string."""
        if not self._is_fallback and self._kokoro is not None:
            self._synthesize_pcm("Warmup.")

    def _synthesize_pcm(self, text: str, voice: str | None = None) -> bytes:
        """Synchronously synthesize text to 24kHz 16-bit signed Linear PCM bytes."""
        if self._is_fallback or self._kokoro is None:
            return self._fallback_synthesize(text)

        v = voice or self.default_voice
        samples, _ = self._kokoro.create(text, voice=v, speed=self.speed)
        pcm16 = np.clip(samples * 32767.0, -32768.0, 32767.0).astype(np.int16)
        return pcm16.tobytes()

    def _fallback_synthesize(self, text: str) -> bytes:
        """Generate mock synthetic 24kHz 16-bit PCM tone for offline test environments."""
        duration_s = max(0.2, min(3.0, len(text) * 0.05))
        t = np.linspace(0, duration_s, int(self.sample_rate * duration_s), endpoint=False)
        tone = (0.2 * np.sin(2 * np.pi * 440 * t) * 32767.0).astype(np.int16)
        return tone.tobytes()

    def _split_subclauses(self, text: str) -> list[str]:
        """Split text into vocalizable sub-clauses for progressive low-latency streaming."""
        if len(text) < 25:
            return [text]
        parts = re.split(r"([.!?;:,\—\–])(?:\s+)", text)
        if len(parts) <= 1:
            return [text]

        clauses: list[str] = []
        current = ""
        for i in range(0, len(parts), 2):
            clause = parts[i]
            punct = parts[i + 1] if i + 1 < len(parts) else ""
            combined = (
                (current + " " + clause + punct).strip() if current else (clause + punct).strip()
            )
            if len(combined) >= 8:
                clauses.append(combined)
                current = ""
            else:
                current = combined
        if current:
            if clauses:
                clauses[-1] = (clauses[-1] + " " + current).strip()
            else:
                clauses.append(current)
        return clauses if clauses else [text]

    async def synthesize_stream(
        self,
        text: str,
        voice: str | None = None,
    ) -> AsyncGenerator[bytes, None]:
        """Synthesize text in progressive sub-clauses and yield framed binary audio chunks."""
        cleaned_text = text.strip()
        if not cleaned_text:
            return

        clauses = self._split_subclauses(cleaned_text)
        carry_buffer = b""
        for clause in clauses:
            pcm_bytes = await asyncio.to_thread(self._synthesize_pcm, clause, voice)
            full_bytes = carry_buffer + pcm_bytes
            total_len = len(full_bytes)
            remainder = total_len % self.chunk_size
            end_offset = total_len - remainder

            for offset in range(0, end_offset, self.chunk_size):
                await asyncio.sleep(0)
                yield full_bytes[offset : offset + self.chunk_size]

            carry_buffer = full_bytes[end_offset:]

        if carry_buffer:
            await asyncio.sleep(0)
            yield carry_buffer
