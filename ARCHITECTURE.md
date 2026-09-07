# ARCHITECTURE.md — System Architecture & Implementation Blueprint

## EchoSync AI: Low-Latency Edge/Cloud Hybrid Voice Agent

---

## 1. System Overview & Topological Decomposition

EchoSync AI decomposes real-time conversational voice interaction into a hybrid asynchronous pipeline. Raw acoustic waveforms are processed, partitioned, and transcribed exclusively on the host edge node via quantized CPU models. Reasoning is delegated upstream via streaming token flows, while text synthesis uses an overlapped sentence chunking mechanism to mask downstream time-to-first-audio latency.

### 1.1 High-Level Component Topology

```
+───────────────────────────────────────────────────────────────────────────+
│                               BROWSER RUNTIME                             │
│                                                                           │
│   [Microphone Input] ──► [AudioWorkletProcessor] ──► [Linear Resampler]   │
│                                                            │              │
│                                                 16kHz 16-bit Mono PCM     │
│                                                            ▼              │
│   [Web Audio API Sink] ◄── [Chunk Deque / Jitter] ◄── [WebSocket Client]  │
+────────────────────────────────────────────────────────────┬──────────────+
                                                             │
                                          TLS WebSocket (wss://)
                                          Bidirectional Framing
                                                             │
+────────────────────────────────────────────────────────────▼──────────────+
│                           FASTAPI ASYNC GATEWAY                           │
│                                                                           │
│  +─────────────────────────────────────────────────────────────────────+  │
│  | Connection Manager & Frame Router                                   |  │
│  | - Validates origin & handshake protocols                            |  │
│  | - Demuxes binary frames (audio) from text frames (JSON control)     |  │
│  | - Enforces backpressure via bounded asyncio.Queue(maxsize=50)       |  │
│  +─────────────────────────────────┬───────────────────────────────────+  │
│                                    │                                      │
│                                    ▼                                      │
│  +─────────────────────────────────────────────────────────────────────+  │
│  | State Machine & Voice Activity Detection Engine                     |  │
│  | - Model: Silero-VAD v5 (ONNX Runtime, CPUExecutionProvider)         |  │
│  | - Chunk size: 512 samples (32ms windows @ 16kHz)                    |  │
│  | - Logic: Energy thresholding, speech trigger, silence windowing     |  │
│  +─────────────────────────────────┬───────────────────────────────────+  │
│                                    │ Raw Speech PCM Buffer                │
│                                    ▼                                      │
│  +─────────────────────────────────────────────────────────────────────+  │
│  | Edge Transcription Subsystem                                        |  │
│  | - Engine: faster-whisper (CTranslate2 backend)                      |  │
│  | - Quantization: INT8 CPU vectorization                              |  │
│  | - Parameters: beam_size=1, vad_filter=False, temperature=0.0        |  │
│  +─────────────────────────────────┬───────────────────────────────────+  │
│                                    │ Completed Text Transcript            │
│                                    ▼                                      │
│  +─────────────────────────────────────────────────────────────────────+  │
│  | Upstream Cloud Reasoning Adapter                                    |  │
│  | - Model: Google Gemini 2.5 Flash via google-genai SDK               |  │
│  | - Interface: Async streaming iterator (SSE over HTTP/2)             |  │
│  | - Resilience: Exponential backoff with jitter via Tenacity          |  │
│  +─────────────────────────────────┬───────────────────────────────────+  │
│                                    │ Streaming Tokens                     │
│                                    ▼                                      │
│  +─────────────────────────────────────────────────────────────────────+  │
│  | Streaming Sentence Boundary Buffer                                  |  │
│  | - Regex Accumulator: `([.!?;:]\s|\n)`                               |  │
│  | - Emits syntactically complete phoneme sequences                    |  │
│  +─────────────────────────────────┬───────────────────────────────────+  │
│                                    │ Sentence Chunks                      │
│                                    ▼                                      │
│  +─────────────────────────────────────────────────────────────────────+  │
│  | Streaming Acoustic Synthesis Subsystem                              |  │
│  | - Engine: Kokoro-82M (ONNX CPU) / Fallback: edge-tts               |  │
│  | - Output: 24kHz Raw PCM / MP3 frames                                │  │
│  | - Immediate transport over WebSocket downstream channel             |  │
│  +─────────────────────────────────────────────────────────────────────+  │
+───────────────────────────────────────────────────────────────────────────+

```

---

## 2. Ingestion & Network Transport Layer

### 2.1 Audio Worklet & Client Framing

The browser audio pipeline uses the standard `AudioWorkletNode` API running in an isolated execution thread to prevent UI thread jitter:

1. **Audio Capture:** The `MediaStreamAudioSourceNode` captures user voice at the browser's hardware-native sampling rate (typically $44.1\text{ kHz}$ or $48\text{ kHz}$).
2. **Downsampling & Quantization:** A dedicated `AudioWorkletProcessor` converts float32 samples to 16,000 Hz, 16-bit signed integer linear PCM buffers ($[-32768, 32767]$ little-endian) using a polyphase decimation filter.
3. **Framing Constraint:** Buffers are framed into discrete units of **512 samples** (1,024 bytes, representing exactly $32\text{ ms}$ of duration). Frames are immediately transmitted over the WebSocket channel as raw binary blobs.

### 2.2 Wire Protocol Framing Specification

The WebSocket connection handles binary audio transmission interleaved with UTF-8 JSON control messages:

* **Binary Payload (Client $\rightarrow$ Server):** Continuous raw byte chunks of length 1,024 bytes (16-bit 16kHz PCM).
* **Control Signaling (JSON Client $\leftrightarrow$ Server):**

```json
// Session Initializer (Client -> Server)
{
  "type": "session_init",
  "data": {
    "sample_rate": 16000,
    "channels": 1,
    "bit_depth": 16,
    "user_id": "usr_9b1deb4d"
  }
}

// Client Barge-In / Interruption Signal (Client -> Server)
{
  "type": "user_interrupt",
  "data": {
    "timestamp_ms": 1713000021450
  }
}

// Server Pipeline Status Event (Server -> Client)
{
  "type": "status",
  "data": {
    "state": "LISTENING" | "PROCESSING_STT" | "STREAMING_LLM" | "SPEAKING",
    "turn_id": "trn_01h7"
  }
}

// Audio Stream Header (Server -> Client preceding binary audio blocks)
{
  "type": "audio_header",
  "data": {
    "format": "pcm_s16le",
    "sample_rate": 24000,
    "chunk_index": 1,
    "text_segment": "Processing your request now."
  }
}

```

---

## 3. Edge Compute Subsystems (CPU-Bound Execution)

### 3.1 Voice Activity Detection (Silero-VAD Engine)

The VAD module acts as a gatekeeper to prevent continuous audio buffering into memory.

```
       [ Incoming 512-sample PCM Frame ]
                      │
                      ▼
   [ Normalize: x = int16_bytes / 32768.0 ]
                      │
                      ▼
       [ ONNX Runtime Inference (CPU) ]
                      │
                      ▼
         [ Speech Probability: P ]
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
  P >= threshold              P < threshold
 (e.g., P >= 0.5)            (e.g., P < 0.5)
        │                           │
        ▼                           ▼
[ State: SPEECH_ACTIVE ]    [ Increment Silence Counter ]
- Append to buffer          - If silence >= 400ms:
- Reset silence counter       Trigger SPEECH_END
                              Flush buffer to STT

```

* **Runtime Parameters:**
* Model: `silero_vad.onnx` executed via ONNX Runtime using `CPUExecutionProvider`.
* Context Window: Recurrent hidden states ($h, c$) maintained per WebSocket session and reset upon confirmed utterance boundary.
* Trigger Logic: A positive utterance begins when 2 consecutive frames exceed $P \ge 0.5$. The utterance terminates when consecutive negative frames aggregate to $\ge 400\text{ ms}$ ($12\text{ frames}$).



### 3.2 Speech-to-Text (`faster-whisper` on CTranslate2)

Once `SPEECH_END` triggers, the accumulated 16-bit PCM buffer converts into a normalized single-precision float32 array and passes directly to the local transcription engine.

* **Engine Configuration:**
* Engine: `faster-whisper.WhisperModel`.
* Compute Type: `int8` quantization (SIMD-accelerated on modern x86/ARM CPUs).
* Threading: Bound strictly via `cpu_threads=2` to prevent thread contention against the main asyncio event loop.


* **Execution Parameters:**
```python
segments, info = model.transcribe(
    audio_data,
    beam_size=1,            # Greedy decoding for minimal latency
    best_of=1,
    temperature=0.0,        # Fully deterministic
    vad_filter=False,       # Upstream Silero handles VAD
    condition_on_previous_text=False
)

```


* **Latency Profile:** Transcribing a typical 3-second speech buffer on an x86/ARM CPU takes approximately 120ms to 180ms under INT8 precision.

---

## 4. Reasoning & Synthesis Pipeline

### 4.1 Upstream LLM Gateway (Gemini 2.5 Flash)

The backend invokes the Google AI Studio Gemini API via the official asynchronous Python SDK (`google-genai`).

* **Client Session Management:** A long-lived asynchronous client instance (`genai.Client(http_options={'api_version': 'v1alpha'})`) handles connection pooling.
* **Streaming Consumption:** Requests use `client.aio.models.generate_content_stream` to ingest tokens incrementally as they are emitted from the provider cluster.
* **Fault Tolerance & Exponential Backoff:** The invocation is wrapped with `tenacity.AsyncRetrying`:
* Monitored Exceptions: Upstream rate limits (`ResourceExhausted` / HTTP 429), gateway drops (HTTP 502/503), and connection timeouts.
* Backoff Strategy: Exponential wait times ($\text{base}=0.5\text{s}$, $\text{factor}=2$, $\text{max\_wait}=6.0\text{s}$, $\text{stop}=3\text{ attempts}$) with randomized jitter to prevent thundering herds.


* **Prompt Conditioning:** System instructions strictly force short conversational cadence:
> *"You are EchoSync, an ultra-low-latency conversational AI. Respond immediately, concisely, and naturally. Limit responses to 1-2 spoken sentences unless explicitly asked for detail. Strictly never output Markdown syntax, asterisks, headers, code blocks, or bullet lists."*



### 4.2 Sentence-Boundary Chunking Pipeline

Because downstream TTS systems produce unnatural inflection when fed partial token fragments, an asynchronous regex boundary accumulator aggregates token streams into speakable linguistic clauses:

$$\text{Boundary Pattern: } \mathcal{R} = (?:[\.\!\?]+\vert{}\;\vert{}\:)\s+\vert{}\n+$$

```
LLM Token Stream ──► [Character Accumulator]
                           │
                           ├─► Matches Pattern R AND len(buf) >= min_chars (20)?
                           │      ├─► YES: Yield Clause ──► Send to TTS Queue
                           │      └─► NO:  Retain & await next token
                           │
                           └─► LLM Generator Finished (EOF)?
                                  └─► YES: Flush remaining string to TTS Queue

```

### 4.3 Streaming Acoustic Synthesis (Kokoro-82M Engine)

* **Model Profile:** Kokoro-82M running locally through ONNX Runtime on CPU.
* **Pipeline Mechanism:** As each sentence clause arrives from the boundary buffer, a background task executes inference, generating 24,000 Hz raw audio buffers.
* **Downstream Framing:** Output audio samples are sliced into 2,048-byte chunks and sent over the WebSocket immediately. The client begins audio playback on Chunk 1 while Chunk 2 is still being synthesized by Kokoro.

---

## 5. Concurrency, Memory & State Management

### 5.1 Async Event Loop Isolation & Thread Pools

CPU-bound operations (Silero-VAD tensor evaluation, CTranslate2 matrix operations, and Kokoro ONNX synthesis) must never block Python's primary `asyncio` event loop.

To maintain event loop responsiveness:

* **Thread Offloading:** CPU tasks execute via `asyncio.to_thread` or a dedicated `concurrent.futures.ThreadPoolExecutor(max_workers=3)`.
* **Bounded Queues:** Inbound audio buffers sit inside an `asyncio.Queue(maxsize=50)`. If network anomalies cause the VAD loop to stall, the queue drops incoming chunks to prevent unbounded memory growth:

```python
try:
    audio_queue.put_nowait(raw_chunk)
except asyncio.QueueFull:
    telemetry.record_dropped_frame()

```

### 5.2 Session State Machine

```
               ┌───────────────┐
               │  DISCONNECTED │
               └───────┬───────┘
                       │ WebSocket Handshake Accepted
                       ▼
               ┌───────────────┐
         ┌────►│   LISTENING   │◄────────────────┐
         │     └───────┬───────┘                 │
         │             │ Silero-VAD Triggers     │
         │             ▼                         │
         │     ┌───────────────┐                 │
         │     │ RECORDING_SPK │                 │
         │     └───────┬───────┘                 │
         │             │ Silence Window Met      │
         │             ▼                         │
         │     ┌───────────────┐                 │
         │     │ TRANSCRIBING  │                 │
         │     └───────┬───────┘                 │
User     │             │ Transcript Emitted      │
Interrupt│             ▼                         │ Synthesis
Occurs   │     ┌───────────────┐                 │ Finished
         │     │ STREAMING_LLM │                 │
         │     └───────┬───────┘                 │
         │             │ First Sentence Emitted  │
         │             ▼                         │
         │     ┌───────────────┐                 │
         └─────┤  SYNTHESIZING │─────────────────┘
               └───────────────┘

```

### 5.3 Interruption (Barge-In) Lifecycle

When the client detects speech while the server is actively synthesizing or transmitting response audio:

1. Client immediately drops its local playback audio buffer and transmits `{"type": "user_interrupt"}` over the WebSocket.
2. The server receives the interrupt signal, terminates active LLM streaming iterators via `task.cancel()`, flushes the TTS synthesis queue, and drops pending network audio frames.
3. The session state resets instantly to `LISTENING`.

---

## 6. End-to-End Latency Profile

| Stage | Execution Component | Min Latency | Expected P50 | SLA Max P95 |
| --- | --- | --- | --- | --- |
| **Acoustic Ingestion** | Browser AudioWorklet Framing | $32\text{ ms}$ | $32\text{ ms}$ | $64\text{ ms}$ |
| **Boundary Trigger** | Silero-VAD Silence Window | $400\text{ ms}$ | $400\text{ ms}$ | $450\text{ ms}$ |
| **Speech-to-Text** | `faster-whisper` INT8 on CPU | $110\text{ ms}$ | $150\text{ ms}$ | $220\text{ ms}$ |
| **Reasoning TTFT** | Gemini 2.5 Flash Streaming API | $180\text{ ms}$ | $240\text{ ms}$ | $400\text{ ms}$ |
| **Boundary Framing** | Regex Sentence Accumulator | $40\text{ ms}$ | $60\text{ ms}$ | $100\text{ ms}$ |
| **Local Synthesis** | Kokoro-82M ONNX (First Chunk) | $80\text{ ms}$ | $120\text{ ms}$ | $200\text{ ms}$ |
| **Transport & Playback** | WebSocket Transmission & Sink | $15\text{ ms}$ | $25\text{ ms}$ | $50\text{ ms}$ |
| **Total Round-Trip** | **Utterance End $\rightarrow$ Audio Out** | **$857\text{ ms}$** | **$1027\text{ ms}$** | **$1484\text{ ms}$** |

---

## 7. Observability, Telemetry & Health Verification

### 7.1 Distributed Tracing & Span Hierarchy

Every voice interaction turn instantiates a unique, context-propagated trace:

* `Span: turn_execution`
* `Span: vad_boundary_detection`
* `Span: stt_inference` (records `duration_ms`, `tokens_transcribed`, `audio_length_s`)
* `Span: llm_generation` (records `ttft_ms`, `total_tokens`, `prompt_tokens`)
* `Span: tts_synthesis` (records `chunk_index`, `audio_duration_ms`, `rtf_ratio`)



### 7.2 Structured Telemetry Payloads

All diagnostic messages emit to `stdout` formatted as RFC 8259-compliant JSON objects:

```json
{
  "timestamp": "2026-09-07T07:48:53.012Z",
  "level": "INFO",
  "logger": "echosync.pipeline",
  "trace_id": "8f3b145a9032d847",
  "session_id": "ses_43b819",
  "event": "voice_turn_completed",
  "timings": {
    "vad_detection_ms": 402.1,
    "stt_inference_ms": 142.6,
    "llm_ttft_ms": 218.4,
    "tts_first_chunk_ms": 110.2,
    "total_round_trip_ms": 873.3
  },
  "system_telemetry": {
    "process_rss_mb": 482.3,
    "cpu_percent_utilization": 24.8
  }
}

```

### 7.3 Health Probes & Circuit Breaking

The FastAPI gateway provides deterministic lifecycle and health checking endpoints:

* `GET /healthz`: Immediate liveness check. Returns HTTP 200 if the ASGI server is actively routing requests.
* `GET /ready`: Readiness probe verifying local model files (`silero_vad.onnx`, `faster-whisper-tiny.en`, and `kokoro-v0_19.onnx`) are loaded in system RAM and responsive to inference pings.

---

## 8. Directory Hierarchy & File Organization

```text
echosync-ai/
├── ARCHITECTURE.md
├── PRD.md
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── src/
│   ├── __init__.py
│   ├── config.py                 # Pydantic v2 validated settings
│   ├── main.py                   # ASGI app setup, lifespan & router mounting
│   ├── api/
│   │   ├── __init__.py
│   │   ├── router.py             # WebSocket endpoint (/ws/audio)
│   │   └── telemetry.py          # Metrics, tracing & system diagnostics
│   ├── core/
│   │   ├── __init__.py
│   │   ├── vad.py                # Silero VAD state machine
│   │   ├── stt.py                # faster-whisper CTranslate2 worker
│   │   ├── llm.py                # Google GenAI streaming client & retry logic
│   │   └── tts.py                # Kokoro-82M ONNX streaming synthesizer
│   └── static/
│       ├── index.html            # Test benchmarking client interface
│       └── audio-processor.js    # 16kHz PCM AudioWorklet processor
└── tests/
    ├── __init__.py
    ├── test_vad.py               # Unit tests for silence detection
    ├── test_sentence_buffer.py   # Regex boundary splitting verification
    └── test_api_routes.py        # Gateway initialization & probe tests

```
