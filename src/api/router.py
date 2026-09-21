"""FastAPI WebSocket audio gateway and state orchestration router."""

import asyncio
import contextlib
import hmac
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import asdict
from typing import Any

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.metrics import active_ws_connections, ws_connections_total
from src.api.telemetry import PipelineMetrics, log_turn_telemetry
from src.config import Settings, get_settings
from src.core.llm import GeminiLLM
from src.core.stt import WhisperSTT
from src.core.tts import KokoroTTS
from src.core.vad import SileroVAD, VADState

router = APIRouter()
logger = logging.getLogger("echosync.router")

# ---------------------------------------------------------------------------
# WebSocket admission gate (T2.1 origin allowlist, T2.2 concurrency + per-IP
# caps, T2.3 optional bearer token). All checks run before ``accept()`` so a
# rejected client never enters the audio/VAD/LLM pipeline.
# ---------------------------------------------------------------------------

#: WebSocket close code used when the admission gate refuses a connection
#: (unknown Origin, or a concurrency / per-IP cap is exhausted).
WS_CLOSE_REJECTED = 4403
#: WebSocket close code used when T2.3 requires a bearer token and none (or an
#: invalid one) was presented.
WS_CLOSE_UNAUTHORIZED = 4401

#: Process-wide bound on simultaneous /ws/audio sessions (T2.2). Module-level
#: so the cap is shared across every connection on the event loop. Because the
#: endpoint only calls ``acquire()`` after a synchronous ``locked()`` check, it
#: never creates blocked waiters (garnering FIFO bookkeeping) on the reject path.
_ws_concurrency_semaphore: asyncio.Semaphore = asyncio.Semaphore(get_settings().ws_max_concurrent)

#: Active /ws/audio session count keyed by client IP (T2.2). A plain dict is
#: safe here: each check-and-increment is a single synchronous block with no
#: ``await`` in between, so it is atomic on the asyncio event loop.
_ws_active_by_ip: dict[str, int] = {}


def _is_origin_allowed(origin: str | None, settings: Settings) -> bool:
    """Return whether an inbound ``Origin`` header passes the T2.1 allowlist.

    Browser-backed deployments must present an Origin in ``ws_allowed_origins``.
    Non-browser clients (local dev workbench, CLIs) often omit the header; that
    is tolerated only when ``app_env == "development"`` and rejected everywhere
    else, so a public deployment still requires an allowlisted Origin.
    """
    if origin is None:
        return settings.app_env == "development"
    return origin in settings.ws_allowed_origins


def _client_ip(websocket: WebSocket) -> str:
    """Best-effort client IP for the T2.2 per-IP cap.

    Prefers the first ``X-Forwarded-For`` entry (written by the Railway / nginx
    edge proxy), then ``X-Real-IP``, then the direct peer address. A client that
    can reach this server directly can fake these headers, so the origin
    allowlist (T2.1) and optional bearer token (T2.3) remain the primary abuse
    controls; the per-IP cap is best-effort bookkeeping.
    """
    forwarded = websocket.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    real_ip = websocket.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    client = websocket.client
    return client.host if client else "unknown"


def _bearer_token_matches(authorization: str | None, expected: str) -> bool:
    """Constant-time T2.3 check of an ``Authorization: Bearer <token>`` header."""
    if not authorization:
        return False
    scheme, _, credentials = authorization.partition(" ")
    if scheme.lower() != "bearer" or not credentials:
        return False
    return hmac.compare_digest(credentials.strip(), expected)


def _ws_release_slot(client_ip: str) -> None:
    """Release the T2.2 admission slots held by a closing session.

    Purely synchronous so it is safe to call from a ``finally`` block even when
    the coroutine is being cancelled.
    """
    remaining = _ws_active_by_ip.get(client_ip, 1) - 1
    if remaining > 0:
        _ws_active_by_ip[client_ip] = remaining
    else:
        _ws_active_by_ip.pop(client_ip, None)
    _ws_concurrency_semaphore.release()


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
    """Bidirectional WebSocket streaming endpoint for low-latency conversational audio.

    Admission gate (T2.1/T2.2/T2.3) runs before ``accept()``: a disallowed
    Origin, a missing/invalid bearer token (when ``WS_BEARER_TOKEN`` is set), an
    exhausted concurrency semaphore, or a breached per-IP cap each close the
    socket with 4403/4401 and never enter the audio/VAD/LLM pipeline.
    """
    settings = get_settings()

    # T2.1 -- Origin allowlist. No-Origin is tolerated in development (local
    # workbench / CLI clients) and rejected in any non-development deployment.
    origin = websocket.headers.get("origin")
    if not _is_origin_allowed(origin, settings):
        logger.warning("Rejecting /ws/audio handshake: Origin %r not allowlisted", origin)
        await websocket.close(code=WS_CLOSE_REJECTED)
        return

    # T2.3 -- optional bearer token. Server-side check only: the demo frontend
    # is out of scope for T2.3 and does not yet present an Authorization header.
    if settings.ws_bearer_token is not None and not _bearer_token_matches(
        websocket.headers.get("authorization"),
        settings.ws_bearer_token.get_secret_value(),
    ):
        logger.warning("Rejecting /ws/audio handshake: missing or invalid bearer token")
        await websocket.close(code=WS_CLOSE_UNAUTHORIZED)
        return

    # T2.2 -- process-wide concurrency cap.
    if _ws_concurrency_semaphore.locked():
        logger.warning(
            "Rejecting /ws/audio handshake: concurrency cap (%d) reached",
            settings.ws_max_concurrent,
        )
        await websocket.close(code=WS_CLOSE_REJECTED)
        return

    # T2.2 -- per-IP active-session cap.
    client_ip = _client_ip(websocket)
    if _ws_active_by_ip.get(client_ip, 0) >= settings.ws_max_per_ip:
        logger.warning(
            "Rejecting /ws/audio handshake: per-IP cap (%d) reached for %s",
            settings.ws_max_per_ip,
            client_ip,
        )
        await websocket.close(code=WS_CLOSE_REJECTED)
        return

    # Admitted: hold the admission slots for the whole session lifetime. The
    # check-and-grab above is synchronous (no ``await`` between ``locked()`` and
    # ``acquire()``), so the acquire can never create a blocked waiter.
    await _ws_concurrency_semaphore.acquire()
    _ws_active_by_ip[client_ip] = _ws_active_by_ip.get(client_ip, 0) + 1
    try:
        await websocket.accept()
        ws_connections_total.inc()
        active_ws_connections.inc()
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
            initial_prompt=settings.whisper_initial_prompt,
            boost_audio=settings.stt_boost_audio,
        )
        llm: GeminiLLM = getattr(app_state, "llm", None) or GeminiLLM(
            api_key=settings.gemini_api_key,
            model_name=settings.gemini_model,
        )
        from src.core.tts import create_tts

        tts = getattr(app_state, "tts", None) or create_tts(settings)

        session = SessionState(
            websocket=websocket,
            session_id=session_id,
            vad=vad,
            stt=stt,
            llm=llm,
            tts=tts,
        )
        tts_engine_name = getattr(settings, "tts_engine", "edge")
        await session.send_status(
            "LISTENING",
            extra={
                "engine": {
                    "tts_engine": tts_engine_name,
                    "llm_model": settings.gemini_model,
                    "whisper_model": settings.whisper_model_name,
                    "profile": "cloud_free_tier" if tts_engine_name == "edge" else "edge_gpu",
                }
            },
        )
    except BaseException:
        # A handshake or setup failure must never leak an admission slot.
        _ws_release_slot(client_ip)
        raise

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
                early_first_chunk=settings.llm_chunk_early_first,
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
                # Barge-in during model speech is probability-gated: only a confident speech
                # frame (>= vad_interrupt_prob) cuts the assistant off while it is actively
                # speaking. Note: we ONLY interrupt during SPEAKING, never during STREAMING_LLM
                # (so ambient room noise while waiting for LLM doesn't cancel generation).
                if (
                    session.current_state == "SPEAKING"
                    and vad_event.probability >= settings.vad_interrupt_prob
                ):
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
        active_ws_connections.dec()
        # T2.2 -- prune the per-IP counter and free the concurrency slot.
        _ws_release_slot(client_ip)
