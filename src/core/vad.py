"""Voice Activity Detection engine using Silero-VAD under ONNX Runtime."""

import asyncio
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import onnxruntime as ort


class VADState(str, Enum):
    """Voice activity detection state indicators."""

    SILENCE = "SILENCE"
    SPEECH_ACTIVE = "SPEECH_ACTIVE"
    SPEECH_END = "SPEECH_END"


@dataclass(slots=True)
class VADEvent:
    """Event emitted by the VAD state machine per processed frame."""

    state: VADState
    probability: float
    audio_buffer: np.ndarray | None = None
    duration_ms: float = 0.0


class SileroVAD:
    """Stateful Voice Activity Detection worker running Silero-VAD ONNX."""

    def __init__(
        self,
        model_path: Path | str = Path("models/vad/silero_vad.onnx"),
        sample_rate: int = 16000,
        frame_size: int = 512,
        threshold: float = 0.35,
        silence_ms: int = 400,
        min_speech_ms: int = 250,
        pre_speech_padding_frames: int = 3,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Silero VAD model not found at: {self.model_path}")

        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.threshold = threshold
        self.silence_ms = silence_ms
        self.min_speech_ms = min_speech_ms
        self.pre_speech_padding_frames = pre_speech_padding_frames

        # Frame duration in milliseconds (32ms for 512 samples at 16kHz).
        # Tuning guidance: keep silence_ms in the 400-1000ms range — shorter than the
        # speaker's natural pause clips trailing words (truncated transcripts); much
        # longer delays SPEECH_END and thus end-of-turn latency.
        self.frame_duration_ms = (self.frame_size / self.sample_rate) * 1000.0
        self.silence_threshold_frames = max(
            1, int(np.ceil(self.silence_ms / self.frame_duration_ms))
        )
        self.min_speech_frames = max(1, int(np.ceil(self.min_speech_ms / self.frame_duration_ms)))

        # Pin ONNX inference to 1 thread to avoid event loop thread starvation
        sess_opts = ort.SessionOptions()
        sess_opts.inter_op_num_threads = 1
        sess_opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )

        input_names = [inp.name for inp in self._session.get_inputs()]
        self._is_v5 = "state" in input_names

        self._sr_tensor = np.array(self.sample_rate, dtype=np.int64)

        # Buffers and state machine tracking
        self._context_size = 64 if self.sample_rate == 16000 else 32
        self._context: np.ndarray = np.zeros(self._context_size, dtype=np.float32)
        self._pre_buffer: deque[np.ndarray] = deque(maxlen=self.pre_speech_padding_frames)
        self._speech_buffer: list[np.ndarray] = []
        self._current_state = VADState.SILENCE
        self._consecutive_speech_frames = 0
        self._consecutive_silence_frames = 0
        self._state: np.ndarray = np.zeros((2, 1, 128), dtype=np.float32)
        self._h: np.ndarray = np.zeros((2, 1, 64), dtype=np.float32)
        self._c: np.ndarray = np.zeros((2, 1, 64), dtype=np.float32)

        self.reset()

    def reset(self) -> None:
        """Reset internal recurrent context tensors and audio frame buffers."""
        if self._is_v5:
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
        else:
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)

        self._context = np.zeros(self._context_size, dtype=np.float32)
        self._pre_buffer.clear()
        self._speech_buffer.clear()
        self._current_state = VADState.SILENCE
        self._consecutive_speech_frames = 0
        self._consecutive_silence_frames = 0

    def normalize_frame(self, frame: bytes | np.ndarray) -> np.ndarray:
        """Convert inbound 16-bit PCM bytes or raw array to normalized float32."""
        if isinstance(frame, bytes):
            int16_arr = np.frombuffer(frame, dtype=np.int16)
            float_arr = int16_arr.astype(np.float32) / 32768.0
        elif isinstance(frame, np.ndarray):
            if frame.dtype == np.int16:
                float_arr = frame.astype(np.float32) / 32768.0
            elif frame.dtype in (np.float32, np.float64):
                float_arr = frame.astype(np.float32)
            else:
                raise ValueError(f"Unsupported numpy array dtype: {frame.dtype}")
        else:
            raise TypeError(f"Frame must be bytes or np.ndarray, got {type(frame)}")

        if float_arr.shape != (self.frame_size,):
            float_arr = float_arr.reshape(-1)
            if float_arr.shape != (self.frame_size,):
                raise ValueError(
                    f"Frame length mismatch: expected {self.frame_size} samples, got {len(float_arr)}"
                )

        return float_arr

    def _infer_probability(self, norm_frame: np.ndarray) -> float:
        """Run single-frame inference on ONNX model returning speech probability."""
        if self._is_v5:
            input_tensor = np.concatenate([self._context, norm_frame], axis=0).reshape(1, -1)
            self._context = norm_frame[-self._context_size :].copy()
            inputs = {
                "input": input_tensor,
                "state": self._state,
                "sr": self._sr_tensor,
            }
            output, self._state = self._session.run(None, inputs)
        else:
            input_tensor = norm_frame.reshape(1, self.frame_size)
            inputs = {
                "input": input_tensor,
                "h": self._h,
                "c": self._c,
                "sr": self._sr_tensor,
            }
            output, self._h, self._c = self._session.run(None, inputs)

        return float(output[0, 0] if output.ndim > 1 else output[0])

    def predict(self, frame: bytes | np.ndarray) -> float:
        """Run single-frame inference and return raw speech probability."""
        norm_frame = self.normalize_frame(frame)
        return self._infer_probability(norm_frame)

    def process_frame(self, frame: bytes | np.ndarray) -> VADEvent:
        """Process incoming audio frame through VAD inference and update state machine."""
        norm_frame = self.normalize_frame(frame)
        prob = self._infer_probability(norm_frame)
        is_speech = prob >= self.threshold

        if self._current_state == VADState.SILENCE:
            if is_speech:
                self._consecutive_speech_frames += 1
                # Positive speech begins when 2 consecutive frames exceed threshold
                if self._consecutive_speech_frames >= 2:
                    self._current_state = VADState.SPEECH_ACTIVE
                    self._speech_buffer.extend(self._pre_buffer)
                    self._speech_buffer.append(norm_frame)
                    self._consecutive_silence_frames = 0
                    return VADEvent(state=VADState.SPEECH_ACTIVE, probability=prob)
                else:
                    self._pre_buffer.append(norm_frame)
                    return VADEvent(state=VADState.SILENCE, probability=prob)
            else:
                self._consecutive_speech_frames = 0
                self._pre_buffer.append(norm_frame)
                return VADEvent(state=VADState.SILENCE, probability=prob)

        # self._current_state == VADState.SPEECH_ACTIVE
        self._speech_buffer.append(norm_frame)

        if is_speech:
            self._consecutive_silence_frames = 0
            return VADEvent(state=VADState.SPEECH_ACTIVE, probability=prob)

        self._consecutive_silence_frames += 1
        if self._consecutive_silence_frames >= self.silence_threshold_frames:
            # Utterance terminated. Calculate speech duration excluding trailing silence
            valid_frames = self._speech_buffer[: -self._consecutive_silence_frames]
            duration_ms = len(valid_frames) * self.frame_duration_ms

            if len(valid_frames) >= self.min_speech_frames:
                audio_array = np.concatenate(valid_frames, axis=0).astype(np.float32)
                event = VADEvent(
                    state=VADState.SPEECH_END,
                    probability=prob,
                    audio_buffer=audio_array,
                    duration_ms=duration_ms,
                )
            else:
                # Discard transient cough or click (< min_speech_ms)
                event = VADEvent(state=VADState.SILENCE, probability=prob)

            # Reset state for next utterance
            self._speech_buffer.clear()
            self._pre_buffer.clear()
            self._consecutive_speech_frames = 0
            self._consecutive_silence_frames = 0
            self._current_state = VADState.SILENCE
            return event

        return VADEvent(state=VADState.SPEECH_ACTIVE, probability=prob)

    async def async_process_frame(self, frame: bytes | np.ndarray) -> VADEvent:
        """Asynchronously process incoming audio frame in worker thread."""
        return await asyncio.to_thread(self.process_frame, frame)
