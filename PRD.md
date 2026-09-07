# Project Requirements Document (PRD)

## Project Title: EchoSync AI — Low-Latency Hybrid Voice Agent

---

## 1. Executive Summary & Problem Formulation

### 1.1 Problem Statement

Mainstream conversational voice agent architectures (e.g., proprietary managed stacks like Vapi, Retell AI, or pure cloud speech endpoints) introduce significant enterprise vulnerabilities:

* **Privacy & Compliance Exposure:** Streaming continuous raw microphone audio directly to third-party SaaS vendors exposes sensitive ambient audio and violates strict corporate data-residency mandates.
* **Prohibitive Cloud Unit Economics:** Transcribing continuous audio via proprietary STT APIs costs approximately $0.006 to $0.015 per audio minute. At scale (thousands of concurrent hours), cloud audio ingestion costs quickly outpace reasoning inference costs.
* **VRAM Bottlenecks on Edge/Self-Hosted Nodes:** Running end-to-end open-weight LLMs along with multi-gigabyte neural audio models locally demands dedicated GPU instances with 16GB+ VRAM, making affordable CPU-based deployments impossible.

### 1.2 The Solution Architecture: Edge/Cloud Hybrid Pipeline

EchoSync AI solves this trilemma through an asynchronous, decoupled, hybrid edge-to-cloud architecture:

* **Zero-Cloud Audio Ingestion:** Audio capture, Voice Activity Detection (VAD), and Speech-to-Text (STT) inference execute locally on the host CPU using quantized models. Raw audio data never leaves the host machine.
* **Lightweight Edge Compute:** By employing INT8-quantized CTranslate2 binaries and ONNX neural graphs, CPU overhead remains strictly constrained within 1.5 to 2.5 cores under peak active processing.
* **Targeted Cloud Reasoning:** Only finalized textual transcriptions are transmitted upstream to external LLM APIs (Google AI Studio / Gemini) via secure TLS, optimizing token payloads and eliminating cloud audio processing surcharges.
* **Streaming Overlapped Synthesis:** Downstream speech synthesis runs via streaming sentence-boundary chunking, overlapping TTS inference with upstream LLM token arrival.

---

## 2. System Architecture & Component Interactions

```
 +-------------------------------------------------------------------------+
 |                              CLIENT LAYER                               |
 |                                                                         |
 |  [Mic: 16kHz PCM Audio] ────(Binary WS Chunks)────► [AudioWorklet Node] |
 |                                                               │         |
 |  [Speaker Playback] ◄───(Binary Audio Buffers)──── [Web Audio API Node] |
 +───────────────────────────────────────────────────────────────┼─────────+
                                                                 │
                                                   Bidirectional WebSocket
                                                   (Raw PCM / Audio Stream)
                                                                 │
 +───────────────────────────────────────────────────────────────┼─────────+
 |                              SERVER LAYER                     │         |
 |                                                               ▼         |
 |  +-------------------------------------------------------------------+  |
 |  |                       FastAPI Gateway (Uvicorn)                   |  |
 |  |  - Connection Manager (Session Tracking & Keep-Alive Heartbeats)  |  |
 |  |  - Bounded Inbound Audio Buffer (asyncio.Queue, maxsize=50)       |  |
 |  +───────────────────────────────────┬───────────────────────────────+  |
 |                                      │                                  |
 |                                      ▼                                  |
 |  +───────────────────────────────────────────────────────────────────+  |
 |  |               Local Pipeline (Edge / CPU Isolated)                |  |
 |  |                                                                   |  |
 |  |   1. Silero-VAD Engine (ONNX Runtime)                             |  |
 |  |      - Evaluates continuous 512-sample PCM chunks (32ms frames)   |  |
 |  |      - Maintains Speech/Silence state machine                     |  |
 |  |      - Triggers Speech End on >= 400ms sustained silence          |  |
 |  |                                                                   |  |
 |  |   2. faster-whisper Engine (CTranslate2 Backend)                  |  |
 |  |      - Dynamic model loading (tiny.en / base.en)                  |  |
 |  |      - INT8 quantization via CPU SIMD vectorization               |  |
 |  |      - Emits completed transcript string to async worker loop     |  |
 |  +───────────────────────────────────┬───────────────────────────────+  |
 |                                      │ Final Text                     |
 |                                      ▼                                  |
 |  +───────────────────────────────────────────────────────────────────+  |
 |  |                       Upstream Reasoning Layer                    |  |
 |  |                                                                   |  |
 |  |   Google Gemini Flash API (`google-genai` SDK)                    |  |
 |  |   - System Persona & Context Window Injection                     |  |
 |  |   - Streaming Chunk Iterator (Async Server-Sent Events)           |  |
 |  |   - Tenacity Exponential Backoff Wrapper (HTTP 429 / 5xx)         |  |
 |  +───────────────────────────────────┬───────────────────────────────+  |
 |                                      │ Streaming Tokens                 |
 |                                      ▼                                  |
 |  +───────────────────────────────────────────────────────────────────+  |
 |  |                 Sentence Boundary Chunking Buffer                 |  |
 |  |   - Regex boundary accumulator: `([.!?;:]\s|\n)`                  |  |
 |  |   - Splits incoming stream into natural vocalization clauses      |  |
 |  +───────────────────────────────────┬───────────────────────────────+  |
 |                                      │ Sentence Chunks                  |
 |                                      ▼                                  |
 |  +───────────────────────────────────────────────────────────────────+  |
 |  |                      Downstream Synthesis Layer                   |  |
 |  |                                                                   |  |
 |  |   Kokoro-82M ONNX / Edge-TTS Streaming Worker                     |  |
 |  |   - Synthesizes sentence chunk to audio buffer                    |  |
 |  |   - Yields binary audio bytes back to client over WebSocket       |  |
 |  +───────────────────────────────────────────────────────────────────+  |
 +-------------------------------------------------------------------------+

```

---

## 3. Detailed Component Specifications

### 3.1 Network Ingestion & Audio Protocol

* **Wire Protocol:** RFC 6455 WebSocket running over TLS (`wss://`).
* **Inbound Audio Stream:** Single-channel (mono), 16,000 Hz sample rate, 16-bit signed Linear PCM (little-endian, 2 bytes per sample).
* **Chunk Framing:** The client AudioWorklet accumulates 512 samples per frame (exactly 32 ms of raw audio per chunk = 1,024 bytes) before transmitting over the WebSocket to align with Silero-VAD input tensors.
* **Control Signaling:** JSON-encoded control frames interleave binary audio:
* `{"event": "start", "session_id": "uuid4", "sample_rate": 16000}`
* `{"event": "interrupt"}` (Client-side interrupt trigger)
* `{"event": "ping", "timestamp": 1713000000.123}`



### 3.2 Voice Activity Detection (VAD) Engine

* **Inference Engine:** `onnxruntime` executing the `silero_vad.onnx` neural graph.
* **Execution Hardware:** CPU execution provider (`CPUExecutionProvider`) with thread affinity capped at 1 dedicated worker thread.
* **State Machine Parameters:**
* **Frame Size:** 512 samples ($32\text{ ms}$ at $16\text{ kHz}$).
* **Speech Probability Threshold:** $\ge 0.50$ enters `SPEECH_ACTIVE` state.
* **Silence Duration Window:** $400\text{ ms}$ (approximately 12 consecutive frames below threshold) triggers `SPEECH_TERMINATED` and flushes the accumulated audio buffer to STT.
* **Min Speech Duration:** $250\text{ ms}$ minimum threshold to reject transient clicks, coughs, or microphone pops.



### 3.3 Speech-to-Text (STT) Transcription Service

* **Core Library:** `faster-whisper` backed by `CTranslate2`.
* **Model Configurations:**
* Configurable via environment variables: `tiny.en` (fastest, ~39M params) or `base.en` (~74M params).
* Compute Type: `int8` quantization.
* CPU Threads: Configurable pool size (default: 2 threads).


* **Audio Buffer Pre-processing:**
* Converts the accumulated raw 16-bit integer PCM buffer into a normalized 32-bit floating-point NumPy array ($x \in [-1.0, 1.0]$) using $x = \frac{\text{raw\_bytes}}{32768.0}$.


* **Inference Parameters:**
* `beam_size=1` (Greedy search for minimum latency).
* `temperature=0.0` (Deterministic transcription).
* `vad_filter=False` (Silero already isolated the speech boundaries upstream).



### 3.4 Upstream LLM Reasoning Gateway

* **SDK:** Official `google-genai` Python library.
* **Default Model:** `gemini-2.5-flash` via Google AI Studio API.
* **Interaction Paradigm:** Asynchronous streaming text generation (`client.aio.models.generate_content_stream`).
* **Resilience Configuration:**
* Wrapped with `tenacity.AsyncRetrying`.
* Triggers on `google.genai.errors.APIError` and rate-limit HTTP status 429.
* Wait Strategy: Exponential backoff with random jitter ($\text{initial}=0.5\text{s}$, $\text{multiplier}=2.0$, $\text{max}=8.0\text{s}$, $\text{stop}=\text{stop\_after\_attempt}(4)$).


* **Prompt Optimization:** System instruction enforces succinct, conversational, conversational-turn outputs (max 1–3 sentences per response, strictly prohibiting markdown headers, asterisks, bullet points, or raw tables that break TTS delivery).

### 3.5 Streaming Sentence Boundary Chunking Buffer

* **Operational Need:** LLMs emit text in fine-grained chunks (sometimes 1–2 characters or partial words). Feeding these micro-tokens directly into TTS causes robotic, fragmented prosody.
* **Regex Accumulator:** Employs an asynchronous character buffer searching for natural vocal stopping markers:

$$\text{Delimiter Pattern: } [\.\!\?\;\:]\s+\vert{}\n+$$


* **Flush Conditions:**
* **Boundary Match:** When a delimiter pattern is satisfied and length exceeds a minimum threshold (e.g., $\ge 20$ characters).
* **Stream EOF:** On LLM iterator exhaustion, any remaining characters in the buffer are flushed immediately as the final synthesis chunk.



### 3.6 Text-to-Speech (TTS) Synthesis Engine

* **Engine:** `kokoro-onnx` (82M parameters, running locally via ONNX Runtime on CPU) with an automatic fallback interface to `edge-tts`.
* **Audio Format Output:** 24,000 Hz mono PCM or standard MP3 chunk streams sent downstream over WebSocket as binary envelopes.
* **Client Synchronization:** Preceded by a lightweight JSON payload denoting the sequence:
`{"event": "audio_chunk", "chunk_index": 1, "text": "Hello there."}`

---

## 4. Latency Budget & Quantitative Performance Targets

| Pipeline Stage | Metric | Target SLA | Degraded Limit |
| --- | --- | --- | --- |
| **VAD Silence Detection** | Time from last spoken phoneme to flush | $\le 400\text{ ms}$ | $600\text{ ms}$ |
| **STT Transcription** | Time to transcribe audio segment ($\approx 3\text{s}$) | $\le 180\text{ ms}$ | $350\text{ ms}$ |
| **LLM Time-to-First-Token** | Upstream TTFT from Gemini API | $\le 250\text{ ms}$ | $500\text{ ms}$ |
| **Sentence Chunking** | Time to accumulate first full clause | $\le 80\text{ ms}$ | $150\text{ ms}$ |
| **TTS First Chunk Delivery** | Time to synthesize first audio packet | $\le 150\text{ ms}$ | $300\text{ ms}$ |
| **Total Round-Trip Time** | **User stops speaking $\rightarrow$ Speaker emits audio** | **$\le 1060\text{ ms}$** | **$\le 1900\text{ ms}$** |

---

## 5. Telemetry, Observability & Production Resilience

### 5.1 Structured Logging (JSON)

Standard standard output (`stdout`) logs formatted as single-line JSON strings to enable ingestion by Datadog, Grafana Loki, or cloud log drains:

```json
{
  "timestamp": "2026-09-07T07:46:26.123Z",
  "level": "INFO",
  "session_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "event": "pipeline_turn_complete",
  "metrics": {
    "vad_silence_duration_ms": 412,
    "stt_duration_ms": 164,
    "llm_ttft_ms": 231,
    "tts_first_chunk_ms": 142,
    "total_rtt_ms": 949
  },
  "transcript": "What is the capital of France?",
  "host_metrics": {
    "cpu_percent": 18.4,
    "memory_rss_mb": 420.5
  }
}

```

### 5.2 Telemetry Instrumentation

* **OpenTelemetry Tracer:** Traces distributed spans across `[WebSocket -> VAD -> STT -> Gemini -> TTS -> Network Out]`.
* **System Resource Monitor:** Background async task running every 5 seconds checking `psutil.Process().memory_info().rss` and `psutil.cpu_percent()`. If memory exceeds 1.5 GB, trigger automatic buffer compaction and issue warnings.

### 5.3 Memory Leak Prevention & Backpressure

* **Bounded Queues:** Inbound WebSocket audio buffers run on `asyncio.Queue(maxsize=50)`. If network latency stalls worker processing and queue fills, new incoming frames are systematically dropped with warning logs rather than expanding system heap.
* **Session Teardown Hook:** Explicit WebSocket disconnection events forcefully cancel running background asyncio tasks and invoke explicit garbage collection on session-specific numpy arrays.

---

## 6. Containerization, DevOps & Deployment Strategy

### 6.1 Container Multi-Stage Build Requirements

* **Base Image:** `python:3.11-slim-bookworm`.
* **System Packages:** `ffmpeg`, `libasound2-dev`, `g++`, `cmake` (compiled in build stage; omitted or minimized in final runtime stage).
* **Model Cache Pre-download:** Dockerfile build step automatically downloads:
1. `silero_vad.onnx` weights into `/app/models/vad/`.
2. `faster-whisper-tiny.en` CTranslate2 model directory into `/app/models/stt/`.
3. `kokoro-v0_19.onnx` + `voices.bin` into `/app/models/tts/`.


* *Requirement:* The container must start cold without making outbound network requests to Hugging Face or GitHub to download core weights.

### 6.2 Target Deployment Platforms (Zero-Cost Configuration)

* **API Service:** Render Free Tier, Railway Trial, or Koyeb Docker instance (512MB–1GB RAM limits handled via strict INT8 quantization).
* **Frontend Web App:** Vercel, Cloudflare Pages, or Netlify hosting static compiled assets.
* **Automated CI/CD:** GitHub Actions workflow executing:
1. Static analysis via `ruff check` and `ruff format --check`.
2. Test suite via `pytest tests/`.
3. Container build check via `docker build --target runner`.



---

## 7. Configuration Schema Specification (`src/config.py`)

All environment variables validated through Pydantic v2 `BaseSettings`:

| Variable Name | Type | Default | Purpose |
| --- | --- | --- | --- |
| `GEMINI_API_KEY` | `SecretStr` | *Required* | Google AI Studio authentication key |
| `APP_ENV` | `str` | `"development"` | `"development"`, `"staging"`, or `"production"` |
| `HOST` | `str` | `"0.0.0.0"` | Gateway binding address |
| `PORT` | `int` | `8000` | Gateway listener port |
| `WHISPER_MODEL_NAME` | `str` | `"tiny.en"` | Model size for faster-whisper (`tiny.en`, `base.en`) |
| `WHISPER_COMPUTE_TYPE` | `str` | `"int8"` | CTranslate2 quantization type (`int8`, `float32`) |
| `WHISPER_CPU_THREADS` | `int` | `2` | Number of worker threads for transcription |
| `VAD_THRESHOLD` | `float` | `0.5` | Probability threshold for Silero VAD |
| `VAD_SILENCE_MS` | `int` | `400` | Milliseconds of silence to trigger end of utterance |
| `MAX_BUFFER_CHUNKS` | `int` | `50` | Maximum inbound audio queue length before dropping |
| `LOG_LEVEL` | `str` | `"INFO"` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`) |

---