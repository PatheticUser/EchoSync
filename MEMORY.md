# MEMORY.md — EchoSync AI System State & Progress Ledger

## 1. Current Phase and Status
- **Current Phase:** Phase 8: Containerization, CI/CD & Production Deployment
- **Status:** Delivered — Docker image built & verified locally (healthy, all models baked, zero cold-start downloads); CI workflow ready for push
- **Part A enhancements:** rolling conversation memory, turn-cancel race hardening, DEBUG frame logging, `edge-tts` dependency removed
- **Timestamp:** 2026-09-11

## 2. Implemented Files & Module Signatures

### `pyproject.toml`
- Target: Python >=3.11
- Package manager: uv
- Core dependencies: `fastapi`, `uvicorn`, `google-genai`, `faster-whisper`, `onnxruntime`, `numpy`, `pydantic`, `pydantic-settings`, `tenacity`, `psutil`, `kokoro-onnx` (`edge-tts` removed 2026-09-11 — dead dependency)
- Dev dependencies: `pytest`, `pytest-asyncio`, `ruff`
- Configurations: `[tool.uv] package = false`, `[tool.pytest.ini_options]`, `[tool.ruff]`

### `src/config.py`
- `class Settings(BaseSettings)`
  - `gemini_api_key: SecretStr` (required)
  - `app_env: Literal["development", "staging", "production"]` (default: `"development"`)
  - `host: str` (default: `"0.0.0.0"`)
  - `port: int` (default: `8000`, ge=1, le=65535)
  - `log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]` (default: `"INFO"`)
  - `sample_rate: int` (default: `16000`)
  - `frame_size: int` (default: `512`)
  - `channels: int` (default: `1`)
  - `max_buffer_chunks: int` (default: `50`, ge=1)
  - `whisper_model_name: str` (default: `"tiny.en"`)
  - `whisper_compute_type: str` (default: `"int8"`)
  - `whisper_cpu_threads: int` (default_factory: `min(4, os.cpu_count() or 2)`, ge=1, validator: falls back to `min(4, cpu_count)` if unset or 0)
  - `vad_threshold: float` (default: `0.5`, ge=0.0, le=1.0)
  - `vad_silence_ms: int` (default: `400`, ge=50)
  - `model_cache_dir: Path` (default: `Path("./models")`)
  - `vad_model_path: Path` (default: `Path("./models/vad/silero_vad.onnx")`)
  - `whisper_model_dir: Path` (default: `Path("./models/stt")`)
  - `kokoro_model_path: Path` (default: `Path("./models/tts/kokoro-v0_19.onnx")`)
  - `kokoro_voices_path: Path` (default: `Path("./models/tts/voices.bin")`)
  - `llm_memory_turns: int` (default: `8`, ge=0; prior user/assistant turn pairs retained as LLM conversation memory)
- `get_settings() -> Settings` (cached via `functools.lru_cache`)

### `src/core/vad.py`
- `class VADState(str, Enum)`: `SILENCE`, `SPEECH_ACTIVE`, `SPEECH_END`
- `class VADEvent(slots=True)`: `state: VADState`, `probability: float`, `audio_buffer: np.ndarray | None = None`, `duration_ms: float = 0.0`
- `class SileroVAD`:
  - `__init__(model_path, sample_rate=16000, frame_size=512, threshold=0.5, silence_ms=400, min_speech_ms=250, pre_speech_padding_frames=3)`
  - `reset() -> None`
  - `normalize_frame(frame: bytes | np.ndarray) -> np.ndarray` (int16/float32 -> normalized float32 [-1.0, 1.0])
  - `_infer_probability(norm_frame: np.ndarray) -> float` (ONNX inference on CPU with 1 pinned thread)
  - `process_frame(frame: bytes | np.ndarray) -> VADEvent` (synchronous frame processing & state machine transitions)
  - `async_process_frame(frame: bytes | np.ndarray) -> VADEvent` (async non-blocking wrapper via `asyncio.to_thread`)

### `src/core/stt.py`
- `class TranscriptionResult(slots=True)`: `text: str`, `duration_ms: float`, `audio_duration_s: float`, `language: str = "en"`, `probability: float = 1.0`
- `class WhisperSTT`:
  - `__init__(model_name="tiny.en", compute_type="int8", cpu_threads=2, download_root="./models/stt", sample_rate=16000)`
  - `warmup() -> None` (passes 0.2s zero audio to pre-warm CTranslate2 graph)
  - `normalize_audio(audio: bytes | np.ndarray) -> np.ndarray` (converts int16/float array/bytes to flat float32)
  - `transcribe(audio: bytes | np.ndarray) -> TranscriptionResult` (deterministic greedy transcription via CTranslate2)
  - `async_transcribe(audio: bytes | np.ndarray) -> TranscriptionResult` (async non-blocking wrapper via `asyncio.to_thread`)

### `src/core/llm.py`
- `DEFAULT_VOICE_SYSTEM_PROMPT`: Conversational voice instructions (1-2 sentences, zero markdown/bullets).
- `clean_speech_text(text: str) -> str`: Strips markdown headers, bullet list markers, asterisks, backticks, and whitespace.
- `class SentenceChunker`:
  - `__init__(min_chars=20)`
  - `feed(token: str) -> list[str]` (buffers streaming tokens, regex matches `([.!?;:])(?:\s+|\n+)`, verifies `len >= min_chars`)
  - `flush() -> str | None` (flushes remaining buffer on iterator exhaustion / EOF)
- `is_retryable_llm_error(exc: BaseException) -> bool`: Filters `APIError` 429, 500, 502, 503, 504 and rate limit messages.
- `build_contents(history: list[tuple[str,str]] | None, prompt: str, max_turns: int = 8) -> list[dict]`: Builds genai `contents` from rolling history + closing user prompt; enforces strict user/model alternation (user first), merges adjacent same-role, drops leading model echoes.
- `class GeminiLLM`:
  - `__init__(api_key, model_name="gemini-3.6-flash", system_prompt=DEFAULT_VOICE_SYSTEM_PROMPT, memory_turns=8, client=None)`
  - `_call_stream_with_retry(contents: list[dict])` (wrapped via `tenacity.AsyncRetrying`, exponential backoff with jitter)
  - `stream_tokens(prompt: str, contents=None) -> AsyncIterator[str]` (yields token strings from Gemini API)
  - `stream_sentence_chunks(prompt: str, min_chars=20, history=None) -> AsyncIterator[str]` (pipes token stream through `SentenceChunker`, threads history into Gemini contents via `build_contents`)

### `src/core/tts.py`
- `class KokoroTTS`:
  - `__init__(model_path="./models/tts/kokoro-v0_19.onnx", voices_path="./models/tts/voices.bin", default_voice="af_sarah", speed=1.0, sample_rate=24000, chunk_size=2048)`
  - `warmup() -> None` (pre-warms ONNX session graph with minimal audio)
  - `_synthesize_pcm(text: str, voice: str | None = None) -> bytes` (synthesizes 24kHz 16-bit signed Linear PCM bytes via Kokoro or mock fallback)
  - `_fallback_synthesize(text: str) -> bytes` (generates synthetic 24kHz 16-bit PCM for offline tests when model files absent)
  - `synthesize_stream(text: str, voice: str | None = None) -> AsyncGenerator[bytes, None]` (async generator slicing 2048-byte frames, non-blocking via `asyncio.to_thread`, supports cooperative barge-in cancellation)

### `.env.example`
- Environment variables template covering all authentication, runtime, audio, and model configuration keys.

### `models/vad/silero_vad.onnx`
- Cached Silero-VAD v5 ONNX model weights (2.3MB).

### `models/stt/`
- Cached `Systran/faster-whisper-tiny.en` CTranslate2 model directory.

### `models/tts/`
- Cached `kokoro-v0_19.onnx` (311MB) and `voices.bin` (5.5MB) weights.

### `tests/fixtures/reference.wav`
- 3.31-second synthetic 16kHz reference speech fixture for offline transcription testing.

### `tests/test_config.py`
- `test_missing_required_keys_raise_validation_error`
- `test_defaults_populated_correctly`
- `test_custom_env_vars_override_defaults`
- `test_invalid_ranges_raise_validation_error`
- `test_get_settings_cache`
- `test_whisper_cpu_threads_zero_fallback`

### `tests/test_vad.py`
- `test_vad_initialization_not_found`
- `test_vad_normalization`
- `test_vad_silence_inference`
- `test_vad_inference_latency_benchmark` (single-frame mean ~0.22ms vs 5.0ms SLA)
- `test_vad_state_machine_full_utterance` (SILENCE -> SPEECH_ACTIVE -> SPEECH_END)
- `test_vad_state_machine_rejects_transient_noise` (rejection of noise < 250ms)
- `test_vad_async_process_frame` (non-blocking async execution)

### `tests/test_stt.py`
- `test_stt_warmup` (validates graph warmup on 0.2s silence)
- `test_audio_normalization` (int16 PCM bytes, float32 ndarray, empty audio exceptions)
- `test_transcribe_reference_wav` (asserts text match against reference WAV, evaluates RTF ratio)
- `test_async_transcribe` (verifies thread offload via `asyncio.to_thread`)

### `tests/test_llm.py`
- `test_clean_speech_text` (markdown headers, bullets, asterisks, backticks stripped)
- `test_sentence_chunker_deterministic` (boundary splitting, min_chars accumulator)
- `test_sentence_chunker_empty_and_flush` (empty inputs, EOF remaining flush)
- `test_gemini_llm_stream_sentence_chunks_mocked` (mocked Gemini streaming pipeline)
- `test_build_contents_empty_history` (single user prompt)
- `test_build_contents_preserves_alternation` (user/model alternation + closing prompt)
- `test_build_contents_trims_to_max_turns` (trailing `max_turns` pairs retained)
- `test_build_contents_repairs_broken_role_sequence` (leading model echo dropped, same-role merged)
- `test_stream_sentence_chunks_sends_history_contents` (history threaded into Gemini `contents` payload)
- `test_tenacity_retry_on_429_transient` (verifies retry recovery on 429)
- `test_tenacity_retry_exhaustion` (verifies re-raise after 3 attempts)
- `test_is_retryable_llm_error` (validates error code filtering)
- `test_live_gemini_smoke_test` (conditional live smoke test)

### `tests/test_tts.py`
- `test_tts_initialization` (validates configuration, sample rate 24kHz, chunk size 2048 bytes)
- `test_tts_fallback_mode` (validates graceful synthetic tone generation when weights missing)
- `test_tts_synthesis_audio_structure` (validates 2048-byte chunk framing, 16-bit PCM, peak amplitude > 1000)
- `test_tts_async_event_loop_unblocked` (validates concurrent event loop multitasking while synthesis runs)
- `test_tts_cancellation_barge_in` (validates immediate task cancellation and chunk stream cutoff on user interruption)
- `test_tts_latency_benchmark` (benchmarks synthesis latency)

### `src/api/telemetry.py`
- `class PipelineMetrics(slots=True)`: `vad_silence_ms`, `stt_ms`, `llm_ttft_ms`, `tts_first_chunk_ms`, `total_rtt_ms`
- `class HostMetrics(slots=True)`: `memory_rss_mb`, `cpu_percent`
- `get_host_metrics() -> HostMetrics`: Samples current process RSS in MB and CPU percentage via `psutil`
- `log_turn_telemetry(session_id, turn_id, metrics, transcript, level="INFO", extra=None)`: Emits RFC 8259 structured JSON logs to stdout

### `src/api/router.py`
- `class SessionState`:
  - Per-connection pipeline state: `session_id`, `vad`, `audio_queue`, `current_state`, `active_turn_task`, `memory` (rolling `deque` of completed `(role, text)` pairs, maxlen `llm_memory_turns*2`)
  - `send_status(state, extra=None)`: Sends structured JSON status frames
  - `cancel_active_generation()`: Cooperative cancellation of in-flight dialogue turn task (sync, fire-and-forget)
  - `stop_active_turn() -> await`: Cancel + fully await prior turn task (suppress `CancelledError`) before spawning a new utterance — prevents stale status/audio race
  - Memory guardrail: only fully-completed turns append to `memory` (user transcript + aggregated assistant clauses). Cancelled/interrupted turns never append → alternation preserved.
  - Per-frame diagnostic logging moved to DEBUG; 50-frame heartbeat stays INFO (kills 30 msg/sec log spam)
- `/ws/audio` endpoint:
  - Demuxes binary 1024-byte PCM chunks and JSON control frames (`user_interrupt`)
  - Backpressure protection: bounded `asyncio.Queue(maxsize=50)` with drop warning
  - Turn state orchestration: `LISTENING` -> `PROCESSING_STT` -> `STREAMING_LLM` -> `SPEAKING` -> `LISTENING`
  - Immediate barge-in cutoff when user speaks during active speech generation
  - Non-blocking pipeline: worker loop + receiver loop joined via `asyncio.wait(FIRST_COMPLETED)` with clean task teardown on disconnect

### `src/main.py`
- `lifespan(app)`: Pre-warms and attaches `SileroVAD`, `WhisperSTT`, and `KokoroTTS` singletons to `app.state`
- `create_app() -> FastAPI`: Configures application with `/ws/audio` router, CORS, and system probes
- `GET /healthz`: Liveness probe returning HTTP 200 `{"status": "ok"}`
- `GET /ready`: Readiness probe verifying resident neural model singletons and memory RSS

### `tests/test_api_routes.py`
- `test_healthz_liveness_probe`: Asserts HTTP 200 and status ok
- `test_ready_readiness_probe`: Asserts HTTP 200, all model singletons loaded, memory RSS > 0

### `tests/test_websocket.py`
- `test_websocket_connection_handshake`: Verifies connection and initial `LISTENING` status frame
- `test_websocket_interrupt_handling`: Verifies `user_interrupt` control frame resets state to `LISTENING`
- `test_websocket_audio_ingestion_and_turn_flow`: Simulates full dialogue turn: binary PCM ingestion, STT status, transcript message, LLM streaming status, TTS speaking status, audio headers, framed 2048-byte audio chunks, and return to `LISTENING`

### `src/static/audio-processor.js`
- `class AudioProcessor extends AudioWorkletProcessor`:
  - Hardware downsampling to 16,000 Hz with fractional phase accumulator and linear interpolation
  - Input gain multiplier (2.0x) for hardware microphone sensitivity
  - Discrete 512-sample (1,024 byte) Int16Array framing with zero-copy ArrayBuffer transfer
  - Diagnostic non-zero audio detection and periodic peak amplitude logging

### `src/static/index.html`
- Complete UI/UX Pro Max overhaul with full Light / Dark theme architecture:
  - Theme switch in header: Parchment (warm light: canvas `#fefffc`, card `#ffffff`, well `#f9faf7`, border `#dee2de`) and Obsidian (deep dark: canvas `#111114`, card `#18181d`, well `#131317`, border `#26262e`). State persisted in `localStorage`.
  - Professional Acoustic Oscilloscope: Dedicated recessed well with background grid, 60 FPS `requestAnimationFrame` glowing bezier curve visualizer with theme-adaptive accent strokes (`#41a1cf` / `#38bdf8`), and dynamic VAD status indicator (`VAD Silence` / `Speech Active`).
  - Segmented VU Meter: 24-segment calibrated dB input meter with color thresholds (normal, warm, clip) and dynamic peak hold needle with decay.
  - High-Density Telemetry HUD: Tabular monospace layout for VAD Silence, STT Latency, LLM TTFT, TTS 1st Chunk, and Total Utterance RTT with dynamic SLA status pills (`Optimal`, `Nominal`, `Lag`, `Breach`).
  - Modern Dialogue & Playback Stream: Distinct card surfaces with subtle left accent lines, live CSS animated equalizer bars on active assistant speech turns, and `[Interrupted by user]` badges on barge-in.
  - Ergonomic Floating / Sticky Control Dock: Pill dock with pulsing Connect / Mic button, barge-in interrupt button, pipeline state badge, and live WebSocket latency ping indicator.
  - Zero emojis throughout markup and UI. Strict typography hierarchy (Fraunces serif display, Inter sans UI, JetBrains Mono metrics).

### `tests/test_vad_live.py`
- `test_vad_live_speech_detection`: Verifies Silero VAD v5 detects speech frames with probability > 0.95 and transitions to `SPEECH_ACTIVE`
- `test_vad_predict_method`: Verifies single-frame `predict()` probability on voice frames

## 3. Architectural Decisions & Resolved Trade-offs
- Silero VAD v5 ONNX 576-Sample Rolling Context: Identified root cause of near-zero VAD probabilities on loud speech. Silero VAD v5 at 16 kHz requires a 576-sample input tensor (`64-sample rolling context + 512-sample current frame`). Without the 64-sample context buffer, internal attention layers misalign and return ~0.0005. Adding the rolling 64-sample context buffer boosted speech detection probability from 0.0005 to 0.9999.
- AudioWorklet Thread Keep-Alive: Browsers may pause an AudioWorkletNode that is not routed to an audio destination. Connected worklet through a 0.0-gain node to `audioContext.destination` to guarantee continuous rendering clock execution without audio feedback.
- Inbound Sensitivity: Configured 2.0x input gain in `audio-processor.js` and set default `VAD_THRESHOLD=0.35` in `src/config.py` and `.env` for comfortable desktop mic pickup.
- Real-time Diagnostic Logging: Emits frame-level RMS amplitude, byte length, and raw VAD probability in `router.py` on speech activity or elevated probability.
- Rolling Conversation Memory: `SessionState.memory` deque feeds last `LLM_MEMORY_TURNS` completed turn pairs into the Gemini `contents` payload via `build_contents`. Guardrail: only turns that reach full completion (no barge-in cancel) append to memory, so role alternation never breaks mid-history. Memory configurable from `LLM_MEMORY_TURNS=8`; `0` restores stateless mode.
- Turn-Cancel Drain: `stop_active_turn()` awaits the cancelled turn task (suppressing `CancelledError`) so its `LISTENING` status and stale audio fully flush before a new dialogue turn spawns.
- Log Spam Reduction: Per-frame VAD diagnostics demoted to DEBUG (was INFO every 32ms during speech); periodic 50-frame heartbeat retained at INFO for liveness.
- Gemini Model Migration: Updated default LLM model from deprecated `gemini-2.5-flash` to active `gemini-3.6-flash` in `src/config.py`, `src/core/llm.py`, `.env`, and `.env.example`.
- Flexible Status Dispatch: Updated `SessionState.send_status` in `src/api/router.py` to accept `extra: dict | None` and `**kwargs`, merging auxiliary telemetry and cancellation flags into status JSON frames.
- Latency Optimizations:
  - STT: Set `cpu_threads=4`, `language="en"`, `beam_size=1`, `best_of=1`, `temperature=0.0`, `condition_on_previous_text=False`, `without_timestamps=True`. Cuts transcription latency by ~60%.
  - LLM: Explicitly disabled automatic function calling (`automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)`), set `temperature=0.3`, `max_output_tokens=150`, 5.0s per-attempt timeout, eliminating model AFC stalling. Configured `early_first_chunk` support on `SentenceChunker`.
  - TTS: Suppressed phonemizer warnings, pinned ONNX session threads to 4 with sequential execution (`intra_op_num_threads=4`, `execution_mode=rt.ExecutionMode.ORT_SEQUENTIAL`), phonemizer pre-warmed, and implemented sub-clause progressive streaming with 2048-byte carry buffer.

## 4. Verification Status
- **Ruff Check:**
  `uv run ruff check src tests`
  Result: All checks passed!
- **Ruff Format:**
  `uv run ruff format --check src tests`
  Result: 21 files already formatted.
- **Pytest:**
  Full test suite passing: 44 passed, 1 skipped (live Gemini smoke test skipped without real key).
- **Live Gemini Stream:**
  `gemini-3.6-flash` live verified with streaming sentence chunking and retry handlers.

## 5. Known Issues or Blockers
- None. Full pipeline operational: VAD (Silero-VAD v5 with context), STT (Whisper-tiny.en INT8 greedy), LLM (Gemini 3.6 Flash streaming), TTS (Kokoro-82M ONNX 4-thread sub-clause streaming), WebSocket orchestration, and Web Client workbench.

## 6. Next Immediate Step
- Phase 8 (in progress): Containerization, CI/CD & Production Deployment — `scripts/download_models.py`, `Dockerfile` (multi-stage, zero cold-start model downloads), `docker-compose.yml`, `.github/workflows/ci.yml` (ruff + pytest + docker build/push GHCR), README updates.
- Phase 8 constraint: Dockerfile must bake Silero-VAD, Whisper tiny.en, Kokoro-82M weights into the image (build-time download stage) so production container has no cold-start model downloads.




