"""Local Speech-to-Text (STT) subsystem using faster-whisper on CTranslate2."""

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel


@dataclass(slots=True)
class TranscriptionResult:
    """Transcription output payload with execution and token metadata."""

    text: str
    duration_ms: float
    audio_duration_s: float
    language: str = "en"
    probability: float = 1.0


class WhisperSTT:
    """Worker for local speech transcription via CTranslate2."""

    def __init__(
        self,
        model_name: str = "base.en",
        compute_type: str = "int8",
        cpu_threads: int = 4,
        download_root: Path | str = "./models/stt",
        sample_rate: int = 16000,
    ) -> None:
        self.model_name = model_name
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.download_root = Path(download_root)
        self.sample_rate = sample_rate

        self._model = WhisperModel(
            model_size_or_path=self.model_name,
            device="cpu",
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
            download_root=str(self.download_root),
        )

    def warmup(self) -> None:
        """Pass a 0.2s zero-filled float32 array through transcribe to pre-warm engine graph."""
        warmup_samples = int(self.sample_rate * 0.2)
        zero_audio = np.zeros(warmup_samples, dtype=np.float32)
        self.transcribe(zero_audio)

    def normalize_audio(self, audio: bytes | np.ndarray) -> np.ndarray:
        """Convert inbound 16-bit PCM bytes or raw array to 32-bit float array."""
        if isinstance(audio, bytes):
            if len(audio) == 0:
                raise ValueError("Audio buffer cannot be empty")
            int16_arr = np.frombuffer(audio, dtype=np.int16)
            float_arr = int16_arr.astype(np.float32) / 32768.0
        elif isinstance(audio, np.ndarray):
            if audio.size == 0:
                raise ValueError("Audio buffer cannot be empty")
            if audio.dtype == np.int16:
                float_arr = audio.astype(np.float32) / 32768.0
            elif audio.dtype in (np.float32, np.float64):
                float_arr = audio.astype(np.float32)
            else:
                raise ValueError(f"Unsupported numpy array dtype: {audio.dtype}")
        else:
            raise TypeError(f"Audio must be bytes or np.ndarray, got {type(audio)}")

        return float_arr.reshape(-1)

    def transcribe(self, audio: bytes | np.ndarray) -> TranscriptionResult:
        """Transcribe normalized audio segment using greedy deterministic decoding."""
        norm_audio = self.normalize_audio(audio)
        audio_duration_s = float(len(norm_audio)) / float(self.sample_rate)

        start_time = time.perf_counter()
        segments, info = self._model.transcribe(
            audio=norm_audio,
            language="en",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True,
        )

        # Consume segment generator and assemble full utterance text
        transcript_text = " ".join(segment.text.strip() for segment in segments).strip()
        elapsed_duration_ms = (time.perf_counter() - start_time) * 1000.0

        return TranscriptionResult(
            text=transcript_text,
            duration_ms=elapsed_duration_ms,
            audio_duration_s=audio_duration_s,
            language=info.language,
            probability=info.language_probability,
        )

    async def async_transcribe(self, audio: bytes | np.ndarray) -> TranscriptionResult:
        """Asynchronously transcribe audio in worker thread to prevent event loop blocking."""
        return await asyncio.to_thread(self.transcribe, audio)
