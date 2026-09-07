# PHASES.md — Phased Implementation & Verification Plan

## EchoSync AI: Low-Latency Edge/Cloud Hybrid Voice Agent

---

## Phase 1: Environment Baseline & Validated Configuration

### 1. Objectives

Establish the project workspace, define deterministic dependency boundaries using `uv`, and implement the runtime configuration management module with strict schema validation.

### 2. Implementation Steps

1. Initialize the project workspace and repository layout (`src/api`, `src/core`, `tests`, `models`).
2. Generate `pyproject.toml` with pinned core runtime dependencies (`fastapi`, `uvicorn`, `google-genai`, `faster-whisper`, `onnxruntime`, `numpy`, `pydantic`, `pydantic-settings`, `tenacity`, `psutil`, `kokoro-onnx`).
3. Implement `src/config.py` using Pydantic v2 `BaseSettings`:
* Environment variable parsing (`.env`).
* Default audio constants: `SAMPLE_RATE=16000`, `FRAME_SIZE=512`, `CHANNELS=1`.
* Model execution flags: `WHISPER_MODEL=tiny.en`, `WHISPER_COMPUTE_TYPE=int8`, `VAD_THRESHOLD=0.5`.
* Model directory cache paths.


4. Create `.env.example` documenting all configuration keys.

### 3. Verification & Exit Criteria

* Command: `uv run ruff check src/config.py` passes with zero warnings.
* Automated Test: `tests/test_config.py` verifies successful parsing from a mock `.env` and confirms validation errors when mandatory secrets are absent.
* User Confirmation Gate: Manual review of configuration settings before moving forward.

---

## Phase 2: Edge VAD & Audio Ingestion Pipeline

### 1. Objectives

Implement the local Voice Activity Detection engine using Silero-VAD v5 under ONNX Runtime, maintaining a non-blocking state machine for real-time speech boundary detection.

### 2. Implementation Steps

1. Create `src/core/vad.py`:
* Load `silero_vad.onnx` via `onnxruntime.InferenceSession` on `CPUExecutionProvider`.
* Implement the recurrent state manager (`h`, `c` context tensors).
* Implement frame normalization: convert raw 16-bit PCM bytes to 32-bit float NumPy arrays.
* Build the utterance state machine tracking `SPEECH_START`, `SPEECH_ACTIVE`, and `SPEECH_END` based on consecutive frame energy thresholds.


2. Build an async ring buffer / bounded accumulator to capture PCM frames during an active speech state and emit completed audio arrays on `SPEECH_END`.

### 3. Verification & Exit Criteria

* Automated Test: `tests/test_vad.py` passes using synthetic audio tensors (simulating alternating silence frames and active speech tone frames).
* Performance Check: Single-frame inference latency on CPU stays below $5\text{ ms}$.
* User Confirmation Gate: Verification of VAD state transitions before proceeding.

---

## Phase 3: Local Speech-to-Text (STT) Subsystem

### 1. Objectives

Integrate `faster-whisper` running on CTranslate2 with INT8 CPU quantization, ensuring speech decoding occurs in offloaded threads without blocking the event loop.

### 2. Implementation Steps

1. Create `src/core/stt.py`:
* Initialize `faster_whisper.WhisperModel` targeting local cache paths.
* Enforce INT8 quantization (`compute_type="int8"`, `device="cpu"`).
* Pin worker threads via configuration (`cpu_threads=2`).
* Wrap synchronous `model.transcribe()` inside `asyncio.to_thread`.


2. Format transcribed outputs: strip empty utterances, calculate transcription duration, and return raw transcript strings alongside token metrics.

### 3. Verification & Exit Criteria

* Automated Test: `tests/test_stt.py` feeds a pre-recorded reference WAV clip and asserts output text match and execution speed.
* Performance Check: Audio transcription latency ratio $\le 0.15\times$ real-time duration on CPU.
* User Confirmation Gate: Verification of transcription output and thread execution safety.

---

## Phase 4: Upstream LLM Reasoning & Sentence-Boundary Buffer

### 1. Objectives

Connect the official Google GenAI SDK for low-latency streaming text generation, backed by retry logic and an asynchronous sentence-boundary chunker.

### 2. Implementation Steps

1. Create `src/core/llm.py`:
* Initialize asynchronous Google GenAI client (`client.aio.models.generate_content_stream`).
* Wrap calls with `tenacity.AsyncRetrying` for handling rate limits (HTTP 429) with exponential backoff and jitter.
* Inject concise system instructions optimized for conversational voice agents.


2. Implement the regex sentence-boundary accumulator:
* Match terminal delimiters: `([.!?;:]\s|\n)`.
* Enforce minimum clause length ($\ge 20$ characters) to avoid fragmenting output audio.
* Flush remaining tokens upon LLM stream completion.



### 3. Verification & Exit Criteria

* Automated Test: `tests/test_llm.py` mocks the Gemini API stream and validates that tokens split into grammatically complete sentence chunks.
* Integration Test: Live streaming ping against Google AI Studio validating Time-to-First-Token (TTFT).
* User Confirmation Gate: Verification of streaming token yields and boundary splitting.

---

## Phase 5: Streaming Acoustic Synthesis (TTS) Subsystem

### 1. Objectives

Implement streaming text-to-speech synthesis using Kokoro-82M on ONNX Runtime (with a clean fallback interface) to emit raw 24 kHz audio chunks with low latency.

### 2. Implementation Steps

1. Create `src/core/tts.py`:
* Load Kokoro ONNX model weights and voice embeddings.
* Implement an async generator accepting incoming sentence strings and yielding binary audio buffers.
* Slice synthesized audio arrays into 2,048-byte packets for progressive network streaming.


2. Implement queue cancellation handling to stop in-flight synthesis when an interruption event occurs.

### 3. Verification & Exit Criteria

* Automated Test: `tests/test_tts.py` runs a benchmark sentence through the synthesizer and checks sample rate, byte length, and audio integrity.
* Performance Check: First audio chunk emitted within $150\text{ ms}$ of receiving the first text clause.
* User Confirmation Gate: Verification of synthesized audio buffers and fallback handling.

---

## Phase 6: Asynchronous WebSocket Gateway & State Orchestration

### 1. Objectives

Bind the individual pipeline components into an end-to-end FastAPI WebSocket gateway, implementing backpressure, session state tracking, and barge-in (interruption) handling.

### 2. Implementation Steps

1. Create `src/api/router.py`:
* Define the WebSocket endpoint `/ws/audio`.
* Implement concurrent producer/consumer coroutines (`receive_audio_loop`, `pipeline_worker_loop`).
* Enforce bounded queues: `asyncio.Queue(maxsize=50)` dropping stale audio frames during queue backpressure.


2. Implement interruption logic:
* Client sends `{"type": "user_interrupt"}` or VAD triggers during active playback.
* Immediately cancel active LLM generator tasks and flush downstream audio queues.


3. Implement `src/main.py`:
* Application lifespan hooks to pre-warm models into memory on server boot.
* Mount `/healthz` (liveness) and `/ready` (model readiness) probe routes.



### 3. Verification & Exit Criteria

* Integration Test: `tests/test_api.py` uses `TestClient` / WebSocket client to run an end-to-end simulated turn.
* End-to-End Latency Check: Measure round-trip time from utterance completion to first audio packet return.
* User Confirmation Gate: Verification of full dialogue turn across the WebSocket connection.

---

## Phase 7: Web Client & AudioWorklet Ingestion

### 1. Objectives

Create a lightweight, test client interface with custom AudioWorklet nodes for streaming 16 kHz PCM audio and playing incoming audio streams without stutter.

### 2. Implementation Steps

1. Create `src/static/audio-processor.js`:
* Downsample hardware microphone input to 16,000 Hz.
* Accumulate continuous 512-sample PCM chunks and post messages to the main thread.


2. Create `src/static/index.html`:
* Minimal testing dashboard with audio level meter, latency tracking indicators, and transcription display.
* WebSocket client managing binary frames and handling playback buffers via Web Audio API.



### 3. Verification & Exit Criteria

* Manual Testing: Run live browser session, speak into microphone, and verify end-to-end voice loop.
* Interruption Test: Speak over assistant response and ensure client playback cuts off immediately.
* User Confirmation Gate: Verification of real-time conversational stability.

---

## Phase 8: Containerization, CI/CD & Production Deployment

### 1. Objectives

Package the complete application into a multi-stage Docker container with pre-cached model weights and configure automated CI/CD for cloud deployment.

### 2. Implementation Steps

1. Build `Dockerfile`:
* Multi-stage build using `python:3.11-slim-bookworm`.
* Install system dependencies: `ffmpeg`, `libasound2-dev`.
* Pre-download model weights into `/app/models/` during image build to prevent cold-start downloads.


2. Create `docker-compose.yml` for local container testing.
3. Configure `.github/workflows/ci.yml`:
* Run `ruff` linting and formatting checks.
* Execute `pytest` test suite.
* Validate Docker build.


4. Deploy the container to a target cloud host (e.g., Render, Railway, or Koyeb).

### 3. Verification & Exit Criteria

* CI Pipeline: Green run on GitHub Actions across all jobs.
* Container Health: `GET /ready` returns HTTP 200 on the deployed container.
* Final Demonstration: Live conversational exchange through deployed cloud URL.