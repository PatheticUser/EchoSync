"""FastAPI WebSocket audio gateway and state orchestration router."""

import asyncio
import contextlib
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import asdict
from typing import Any

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.telemetry import PipelineMetrics, log_turn_telemetry
from src.config import get_settings
from src.core.llm import GeminiLLM
from src.core.stt import WhisperSTT
from src.core.tts import KokoroTTS
from src.core.vad import SileroVAD, VADState

router = APIRouter()
logger = logging.getLogger("echosync.router")


class SessionState:
    """Session-scoped container for pipeline workers and active generation tasks."""

    def __init__(
        self,
        websocket: WebSocket,
        session_id: str,
        vad: SileroVAD,
        stt: WhisperSTT,
        llm: GeminiLLM,
        tts: KokoroTTS,
    ) -> None:
        self.websocket = websocket
        self.session_id = session_id
        self.vad = vad
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.current_state = "LISTENING"
        self.active_turn_task: asyncio.Task[None] | None = None
        self.audio_queue: asyncio.Queue[bytes] = asyncio.Queue(
            maxsize=get_settings().max_buffer_chunks
        )
        # Rolling (role, text) conversation memory; only completed turns are retained.
        self.memory: deque[tuple[str, str]] = deque(maxlen=get_settings().llm_memory_turns * 2)

    async def send_status(
        self,
        state: str,
        turn_id: str | None = None,
        extra: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Send status update message to client."""
        self.current_state = state
        data: dict[str, Any] = {
            "state": state,
            "turn_id": turn_id,
        }
        if extra:
            data.update(extra)
        data.update(kwargs)
        payload = {
            "type": "status",
            "data": data,
        }
        await self.websocket.send_json(payload)

    def cancel_active_generation(self) -> None:
        """Cancel any running turn synthesis task without awaiting (sync / control path)."""
        if self.active_turn_task and not self.active_turn_task.done():
            self.active_turn_task.cancel()
            self.active_turn_task = None

    async def stop_active_turn(self) -> None:
        """Cancel and await any running turn task before starting a new utterance.

        Fully drains the previous task (including its CancelledError handler) so
        stale status/audio frames never race a newly spawned dialogue turn.
        """
        task = self.active_turn_task
        self.active_turn_task = None
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


@router.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket) -> None:
    """Bidirectional WebSocket streaming endpoint for low-latency conversational audio."""
    await websocket.accept()
    settings = get_settings()
    session_id = f"ses_{uuid.uuid4().hex[:8]}"

    # Resolve shared or app-level model singletons from application state if available
    app_state: Any = getattr(websocket.app, "state", None)
    vad: SileroVAD = getattr(app_state, "vad", None) or SileroVAD(
        model_path=settings.vad_model_path,
        sample_rate=settings.sample_rate,
        frame_size=settings.frame_size,
        threshold=settings.vad_threshold,
        silence_ms=settings.vad_silence_ms,
    )
    stt: WhisperSTT = getattr(app_state, "stt", None) or WhisperSTT(
        model_name=settings.whisper_model_name,
        compute_type=settings.whisper_compute_type,
        cpu_threads=settings.whisper_cpu_threads,
        download_root=settings.whisper_model_dir,
        sample_rate=settings.sample_rate,
    )
    llm: GeminiLLM = getattr(app_state, "llm", None) or GeminiLLM(
        api_key=settings.gemini_api_key,
        model_name=settings.gemini_model,
    )
    tts: KokoroTTS = getattr(app_state, "tts", None) or KokoroTTS(
        model_path=settings.kokoro_model_path,
        voices_path=settings.kokoro_voices_path,
    )

    session = SessionState(
        websocket=websocket,
        session_id=session_id,
        vad=vad,
        stt=stt,
        llm=llm,
        tts=tts,
    )
    await session.send_status("LISTENING")

    async def receive_frames_loop() -> None:
        """Demultiplex binary PCM chunks and text control frames from client."""
        while True:
            try:
                message = await websocket.receive()
            except (WebSocketDisconnect, RuntimeError):
                break

            if message.get("type") == "websocket.disconnect":
                break

            if message.get("bytes"):
                raw_chunk: bytes = message["bytes"]
                try:
                    session.audio_queue.put_nowait(raw_chunk)
                except asyncio.QueueFull:
                    logger.warning(
                        "Inbound audio queue full (%d chunks); dropping frame",
                        session.audio_queue.qsize(),
                    )
            elif message.get("text"):
                try:
                    control_data = json.loads(message["text"])
                    event_type = control_data.get("type")
                    if event_type == "user_interrupt":
                        session.cancel_active_generation()
                        session.vad.reset()
                        await session.send_status("LISTENING")
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON control frame received")

    async def execute_dialogue_turn(
        audio_buffer: Any,
        turn_id: str,
        vad_silence_duration: float,
    ) -> None:
        """Run STT -> LLM -> Sentence Chunker -> TTS pipeline for a single turn."""
        turn_t0 = time.perf_counter()
        metrics = PipelineMetrics(vad_silence_ms=vad_silence_duration)

        try:
            # 1. Speech-to-Text
            await session.send_status("PROCESSING_STT", turn_id)
            stt_t0 = time.perf_counter()
            stt_result = await session.stt.async_transcribe(audio_buffer)
            metrics.stt_ms = (time.perf_counter() - stt_t0) * 1000.0

            transcript = stt_result.text.strip()
            if not transcript:
                await session.send_status("LISTENING", turn_id)
                return

            await websocket.send_json(
                {
                    "type": "transcript",
                    "data": {
                        "text": transcript,
                        "turn_id": turn_id,
                        "duration_ms": metrics.stt_ms,
                    },
                }
            )

            # 2. LLM Streaming (with rolling conversation memory of completed turns)
            await session.send_status("STREAMING_LLM", turn_id)
            llm_t0 = time.perf_counter()
            first_clause = True
            chunk_index = 0
            assistant_text_parts: list[str] = []

            async for clause in session.llm.stream_sentence_chunks(
                transcript,
                history=list(session.memory),
                min_chars=settings.llm_chunk_min_chars,
            ):
                if first_clause:
                    metrics.llm_ttft_ms = (time.perf_counter() - llm_t0) * 1000.0
                    first_clause = False

                # 3. Speech Synthesis
                await session.send_status("SPEAKING", turn_id)
                chunk_index += 1
                await websocket.send_json(
                    {
                        "type": "audio_header",
                        "data": {
                            "format": "pcm_s16le",
                            "sample_rate": 24000,
                            "chunk_index": chunk_index,
                            "text_segment": clause,
                            "turn_id": turn_id,
                        },
                    }
                )

                tts_t0 = time.perf_counter()
                first_audio_chunk = True
                async for audio_chunk in session.tts.synthesize_stream(clause):
                    if first_audio_chunk and metrics.tts_first_chunk_ms == 0.0:
                        metrics.tts_first_chunk_ms = (time.perf_counter() - tts_t0) * 1000.0
                        metrics.total_rtt_ms = (time.perf_counter() - turn_t0) * 1000.0
                        first_audio_chunk = False

                    await websocket.send_bytes(audio_chunk)
                assistant_text_parts.append(clause)

            # Turn completed successfully: persist to memory only on full completion.
            # Cancelled/interrupted turns never append, preserving user/model alternation.
            if assistant_text_parts:
                session.memory.append(("user", transcript))
                session.memory.append(("model", " ".join(assistant_text_parts)))
            log_turn_telemetry(session.session_id, turn_id, metrics, transcript)
            await session.send_status("LISTENING", turn_id, extra={"metrics": asdict(metrics)})

        except asyncio.CancelledError:
            logger.info("Dialogue turn %s cancelled due to barge-in", turn_id)
            await session.send_status("LISTENING", turn_id, extra={"interrupted": True})
            raise

    async def pipeline_worker_loop() -> None:
        """Continuously process inbound audio frames through VAD state machine."""
        frame_count = 0
        while True:
            frame = await session.audio_queue.get()
            frame_count += 1

            # Diagnostic frame size & RMS amplitude calculation
            int16_samples = np.frombuffer(frame, dtype=np.int16)
            float_samples = int16_samples.astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(float_samples**2))) if len(float_samples) > 0 else 0.0

            vad_event = await session.vad.async_process_frame(frame)

            # Per-frame speech details at DEBUG to avoid log spam (30+ msg/sec during speech);
            # a periodic 50-frame heartbeat stays at INFO for liveness visibility.
            if frame_count % 50 == 0:
                logger.info(
                    "Heartbeat frame #%d: %d bytes, RMS: %.5f | VAD prob: %.4f, state: %s",
                    frame_count,
                    len(frame),
                    rms,
                    vad_event.probability,
                    vad_event.state.value,
                )
            elif vad_event.probability >= 0.15 or vad_event.state != VADState.SILENCE:
                logger.debug(
                    "Frame #%d: %d bytes, RMS: %.5f | VAD prob: %.4f, state: %s",
                    frame_count,
                    len(frame),
                    rms,
                    vad_event.probability,
                    vad_event.state.value,
                )

            if vad_event.state == VADState.SPEECH_ACTIVE:
                # If user speaks while system is speaking, trigger barge-in cutoff
                if session.current_state in ("SPEAKING", "STREAMING_LLM"):
                    await session.stop_active_turn()
                    await session.send_status("LISTENING")

            elif vad_event.state == VADState.SPEECH_END and vad_event.audio_buffer is not None:
                turn_id = f"trn_{uuid.uuid4().hex[:8]}"
                logger.info(
                    "VAD utterance completed (%d speech samples, %.1f ms). Launching STT turn %s",
                    len(vad_event.audio_buffer),
                    vad_event.duration_ms,
                    turn_id,
                )
                await session.stop_active_turn()
                session.active_turn_task = asyncio.create_task(
                    execute_dialogue_turn(
                        audio_buffer=vad_event.audio_buffer,
                        turn_id=turn_id,
                        vad_silence_duration=settings.vad_silence_ms,
                    )
                )

    receiver_task = asyncio.create_task(receive_frames_loop())
    worker_task = asyncio.create_task(pipeline_worker_loop())

    try:
        _done, pending = await asyncio.wait(
            [receiver_task, worker_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
    except (WebSocketDisconnect, asyncio.CancelledError):
        logger.info("WebSocket connection closed for session %s", session.session_id)
    finally:
        session.cancel_active_generation()
        receiver_task.cancel()
        worker_task.cancel()
        session.vad.reset()
