# EchoSync AI

Low-latency edge/cloud hybrid conversational voice agent designed for privacy-preserving, cost-effective, real-time voice interactions.

EchoSync isolates raw acoustic capture, voice activity detection, and speech-to-text inference to local CPU hardware. Only textual transcripts leave the host machine to upstream LLM reasoning APIs, eliminating cloud audio ingestion surcharges and ambient audio leaks.

---

## Architecture Overview

```
+-------------------------------------------------------------------------+
|                              CLIENT LAYER                               |
|                                                                         |
|  [Mic: 16 kHz PCM Audio] ───(Binary WS Chunks)────► [AudioWorklet Node] |
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
|  |  - Connection Manager (Session Tracking & Heartbeats)             |  |
|  |  - Bounded Inbound Audio Buffer (asyncio.Queue, maxsize=50)       |  |
|  +───────────────────────────────────┬───────────────────────────────+  |
|                                      │                                  |
|                                      ▼                                  |
|  +───────────────────────────────────────────────────────────────────+  |
|  |               Local Pipeline (Edge / CPU Isolated)                |  |
|  |                                                                   |  |
|  |   1. Silero-VAD Engine (ONNX Runtime)                             |  |
|  |      - Evaluates continuous 512-sample PCM chunks (32ms frames)   |  |
|  |      - Maintains Speech/Silence state machine with rolling context|  |
|  |      - Triggers Speech End on >= 400ms sustained silence          |  |
|  |                                                                   |  |
|  |   2. faster-whisper Engine (CTranslate2 Backend)                  |  |
|  |      - Quantized INT8 tiny.en model running greedy decoding       |  |
|  |      - Emits verified text transcript to event bus                |  |
|  +───────────────────────────────────┬───────────────────────────────+  |
|                                      │ (TLS Text Only)                  |
|                                      ▼                                  |
|  +───────────────────────────────────────────────────────────────────+  |
|  |              Cloud Reasoning Subsystem (Google GenAI)             |  |
|  |                                                                   |  |
|  |   - Upstream streaming tokens via Gemini 3.6 Flash                |  |
|  |   - Exponential backoff retry via Tenacity                        |  |
|  |   - SentenceChunker buffers tokens into complete vocal clauses    |  |
|  +───────────────────────────────────┬───────────────────────────────+  |
|                                      │                                  |
|                                      ▼                                  |
|  +───────────────────────────────────────────────────────────────────+  |
|  |               Streaming Acoustic Synthesis (Local / Edge)         |  |
|  |                                                                   |  |
|  |   - Kokoro-82M ONNX Runtime with 4-thread execution               |  |
|  |   - Sub-clause synthesis yielding exact 2,048-byte linear PCM     |  |
|  |   - Immediate barge-in cutoff discarding in-flight synthesis      |  |
|  +───────────────────────────────────────────────────────────────────+  |
+-------------------------------------------------------------------------+
```

---

## Latency Budgets & Target SLA

| Subsystem | Target SLA | Measured Local (CPU) |
|---|---|---|
| **VAD Silence Detection** | 400 ms | 400 ms |
| **STT Transcription (Whisper INT8)** | $\le$ 250 ms | 70 - 150 ms |
| **LLM Time-to-First-Token (TTFT)** | $\le$ 300 ms | 180 - 280 ms |
| **TTS First Chunk Generation** | $\le$ 150 ms | 90 - 140 ms |
| **Total Round-Trip Time (RTT)** | $\le$ 1,200 ms | 750 - 980 ms |

---

## Core Technologies

- **Runtime & Gateway**: Python 3.11+, [uv](https://github.com/astral-sh/uv), [FastAPI](https://fastapi.tiangolo.com), [Uvicorn](https://www.uvicorn.org).
- **Voice Activity Detection**: Silero-VAD v5 on ONNX Runtime CPU with 64-sample rolling context window.
- **Speech-to-Text**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2 INT8 quantized, 4 pinned CPU threads).
- **Language Model**: Google GenAI SDK (`gemini-3.6-flash`), conversational prompt tuning, [Tenacity](https://github.com/jd/tenacity) exponential retries.
- **Text-to-Speech**: Kokoro-82M ONNX (`kokoro-onnx`), 24 kHz 16-bit linear PCM streaming with sub-clause pipelining.
- **Frontend Workbench**: Vanilla HTML5, Web Audio API, custom `AudioWorkletProcessor`, HTML5 Canvas oscilloscope, Parchment/Obsidian dual-theme architecture.

---

## Prerequisites

- **Operating System**: Linux (tested on CachyOS / Arch / Ubuntu) or macOS.
- **Audio Stack**: Working microphone and speaker.
- **Python Environment**: `uv` package and tool manager.
- **API Key**: Google Gemini API key from [Google AI Studio](https://aistudio.google.com/).

---

## Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/PatheticUser/EchoSync.git
cd EchoSync
```

### 2. Configure Environment

Copy template and inject Google GenAI API credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```ini
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-3.6-flash
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
WHISPER_CPU_THREADS=4
VAD_THRESHOLD=0.35
VAD_SILENCE_MS=400
```

### 3. Install Dependencies

Using `uv`:

```bash
uv sync
```

### 4. Neural Weights Layout

Ensure neural model weights exist in `models/` directory:

```
models/
├── vad/
│   └── silero_vad.onnx
├── stt/
│   └── models--Systran--faster-whisper-tiny.en/
└── tts/
    ├── kokoro-v0_19.onnx
    └── voices.bin
```

---

## Running the Application

### Development Server

Launch Uvicorn via `uv`:

```bash
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

Or execute script entrypoint:

```bash
uv run echosync
```

Open browser at: `http://localhost:8000`

### Interactive Workbench Controls

1. **Theme Switch**: Toggle between Parchment (editorial light) and Obsidian (deep dark) in header.
2. **Start Session**: Click `Start Session` dock button to request microphone permission, mount AudioWorklet, and open WebSocket stream.
3. **Acoustic Oscilloscope**: Live 60 FPS bezier waveform and 24-segment calibrated VU meter with peak hold needle.
4. **Barge-In (Interruption)**: Speak over assistant response or click `Interrupt` button. Downstream synthesis and audio queues flush immediately (<20ms).
5. **Telemetry HUD**: Inspect observed latencies against subsystem SLAs.

---

## System Health & Probes

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | `GET` | Single-page voice workbench UI |
| `/static/audio-processor.js` | `GET` | Hardware AudioWorklet node script |
| `/healthz` | `GET` | Liveness check returning HTTP 200 `{"status": "ok"}` |
| `/ready` | `GET` | Readiness probe validating resident model singletons and memory RSS |
| `/ws/audio` | `WebSocket` | Bidirectional binary PCM audio and JSON control gateway |

---

## Testing & Verification

Run static analysis:

```bash
uv run ruff check src tests
uv run ruff format --check src tests
```

Execute full automated test suite:

```bash
uv run pytest -v
```

Execute focused test suites:

```bash
# VAD unit and live detection tests
uv run pytest tests/test_vad.py tests/test_vad_live.py -v

# STT transcription and normalization tests
uv run pytest tests/test_stt.py -v

# LLM streaming, token chunking, and retry tests
uv run pytest tests/test_llm.py -v

# TTS streaming synthesis and cancellation benchmark
uv run pytest tests/test_tts.py -v

# End-to-end WebSocket dialogue and barge-in flow
uv run pytest tests/test_websocket.py -v
```

---

## License

MIT License. See `LICENSE` for details.
