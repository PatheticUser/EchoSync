# EchoSync

Low-latency edge/cloud hybrid conversational voice agent. Raw audio is captured, voice-activity-detected, and transcribed on the local CPU; only text leaves the machine to an upstream LLM. Built for privacy, cost control, and sub-second round-trip latency on commodity hardware.

---

## Quick Facts

| Category | Detail |
|---|---|
| **Runtime** | Python 3.11+, FastAPI, Uvicorn, `uv` |
| **VAD** | Silero-VAD v5 (ONNX Runtime, CPU) |
| **STT** | faster-whisper (CTranslate2 INT8, `tiny.en` / `base.en` / `small.en`) |
| **LLM** | Google GenAI SDK (`gemini-3.6-flash` default) |
| **TTS** | Kokoro-82M (ONNX, `kokoro-onnx`) |
| **Transport** | Bidirectional WebSocket (binary PCM + JSON control) |
| **Observability** | Structured JSON logs, Prometheus `/metrics`, per-turn telemetry |
| **Security** | WS origin allowlist, concurrency + per-IP caps, optional bearer token |
| **Tests** | 81 passed / 1 skipped (full pytest suite) |
| **CI/CD** | GitHub Actions → ruff + pytest + multi-stage Docker → GHCR (`latest` + `sha-`) → Railway auto-deploy (opt-in) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BROWSER (Client)                               │
│  ┌─────────────┐     16 kHz mono PCM (512-sample frames)      ┌─────────┐  │
│  │ Microphone  │ ────────────────────────────────────────────►│AudioWork│  │
│  └─────────────┘                                               │ let     │  │
│         ▲                                                     └────┬────┘  │
│         │ 24 kHz PCM stream (2,048-byte chunks)                   │       │
│         │ ◄───────────────────────────────────────────────────────┘       │
│  ┌─────────────┐                                               ┌─────────┐  │
│  │  Speaker    │ ◄──────────────────────────────────────────── │Web Audio│  │
│  └─────────────┘                                               │   API   │  │
└──────────────────────┬──────────────────────────────────────────┬──────────┘
                       │                                          │
              TLS WebSocket (wss://)          JSON control frames
                       │                                          │
┌──────────────────────▼──────────────────────────────────────────▼──────────┐
│                            FASTAPI SERVER (Edge)                          │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  WS Admission Gate: Origin allowlist • Concurrency semaphore       │  │
│  │  Per-IP cap • Optional bearer token (HMAC)                         │  │
│  └────────────────────────────────────┬────────────────────────────────┘  │
│                                       │                                   │
│                                       ▼                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  Session State Machine (per connection)                             │  │
│  │  LISTENING → PROCESSING_STT → STREAMING_LLM → SPEAKING → LISTENING  │  │
│  │  Barge-in: VAD speech prob ≥ VAD_INTERRUPT_PROB cancels turn        │  │
│  └────────────────────────────────────┬────────────────────────────────┘  │
│                                       │                                   │
│           ┌───────────────────────────┼───────────────────────────┐       │
│           ▼                           ▼                           ▼       │
│  ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐  │
│  │   Silero VAD    │       │  faster-whisper │       │  Gemini LLM     │  │
│  │  (ONNX CPU)     │       │  (CTranslate2)  │       │  (google-genai) │  │
│  │                 │       │                 │       │                 │  │
│  │ • 32 ms frames  │       │ • INT8 quant    │       │ • streaming     │  │
│  │ • silence ms    │       │ • beam_size=1   │       │ • temp/top_p    │  │
│  │ • pre-padding   │       │ • temp=0.0      │       │ • retry/backoff │  │
│  │ • interrupt prob│       │                 │       │ • 8-turn memory │  │
│  └────────┬────────┘       └────────┬────────┘       └────────┬────────┘  │
│           │                         │                           │         │
│           └─────────────────────────┼───────────────────────────┘         │
│                                     ▼                                     │
│                          ┌─────────────────────┐                          │
│                          │  SentenceChunker    │                          │
│                          │  • terminal only .!? │                          │
│                          │  • min_chars=20     │                          │
│                          │  • early_first_chunk│                          │
│                          └──────────┬──────────┘                          │
│                                     ▼                                     │
│                          ┌─────────────────────┐                          │
│                          │    Kokoro TTS       │                          │
│                          │  (ONNX, 24 kHz)     │                          │
│                          │                     │                          │
│                          │ • voice/speed knob  │                          │
│                          │ • chunk_size=2048   │                          │
│                          │ • barge-in cancel   │                          │
│                          └─────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Latency Targets (measured on local CPU)

| Stage | Target | Typical |
|---|---|---|
| VAD silence detection | ≤ 400 ms | 400 ms |
| STT transcription (3 s audio) | ≤ 250 ms | 100–200 ms |
| LLM TTFT | ≤ 300 ms | 180–280 ms |
| TTS first chunk | ≤ 150 ms | 90–140 ms |
| **Total RTT (user stop → audio out)** | **≤ 1.2 s** | **0.75–1.0 s** |

---

## Configuration (`.env`)

All settings driven by `pydantic-settings`. Copy `.env.example` → `.env` and edit.

```ini
# Authentication (REQUIRED)
GEMINI_API_KEY=your_google_ai_studio_key
GEMINI_MODEL=gemini-3.6-flash

# Server
APP_ENV=development           # production = strict WS origin check
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO

# WebSocket admission gate (production: set to your public origin)
WS_ALLOWED_ORIGINS=["http://127.0.0.1:8000","http://localhost:8000","https://your-app.up.railway.app"]
WS_MAX_CONCURRENT=10
WS_MAX_PER_IP=3
# WS_BEARER_TOKEN=optional_shared_secret

# Audio constants
SAMPLE_RATE=16000
FRAME_SIZE=512
CHANNELS=1
MAX_BUFFER_CHUNKS=50

# STT (faster-whisper)
WHISPER_MODEL_NAME=tiny.en    # tiny.en | base.en | small.en
WHISPER_COMPUTE_TYPE=int8
WHISPER_CPU_THREADS=2

# VAD (Silero)
VAD_THRESHOLD=0.35
VAD_SILENCE_MS=400
VAD_MIN_SPEECH_MS=250
VAD_PRE_PADDING_FRAMES=3
VAD_INTERRUPT_PROB=0.5        # 0.05–0.95, barge-in confidence gate

# LLM
LLM_MEMORY_TURNS=8
LLM_CHUNK_MIN_CHARS=20
LLM_CHUNK_EARLY_FIRST=true
LLM_TEMPERATURE=0.4
LLM_TOP_P=0.95
LLM_MAX_OUTPUT_TOKENS=180
LLM_TIMEOUT_S=5.0

# TTS (Kokoro)
TTS_VOICE=af_sarah
TTS_SPEED=1.0
TTS_CHUNK_SIZE=2048

# Model cache paths (baked into Docker image)
MODEL_CACHE_DIR=./models
VAD_MODEL_PATH=./models/vad/silero_vad.onnx
WHISPER_MODEL_DIR=./models/stt
KOKORO_MODEL_PATH=./models/tts/kokoro-v0_19.onnx
KOKORO_VOICES_PATH=./models/tts/voices.bin
```

**Production notes:**
- `GEMINI_API_KEY` is **required** — app will not boot without it.
- `APP_ENV=production` enforces `WS_ALLOWED_ORIGINS`; default allows only localhost.
- Railway/Render: set RAM ≥ 2 GB, healthcheck `start_period ≥ 60s` (model warm-up).

---

## Quick Start (Local)

```bash
# 1. Clone
git clone https://github.com/PatheticUser/EchoSync.git
cd EchoSync

# 2. Configure
cp .env.example .env
# edit .env → add GEMINI_API_KEY

# 3. Install & run (uv)
uv sync
uv run echosync
# → opens http://localhost:8000 in browser
```

**Workbench UI**: Parchment/Obsidian dual theme, live oscilloscope, VU meter, barge-in button, telemetry HUD.

---

## Docker (Zero Cold-Start)

Model weights are baked at build time — container starts instantly.

```bash
# Build (downloads + caches all weights)
docker build -t echosync:latest .

# Run (reads .env for GEMINI_API_KEY)
docker compose up -d

# Or direct
docker run -d \
  -p 8000:8000 \
  --env-file .env \
  --health-cmd="curl -f http://localhost:8000/healthz || exit 1" \
  --health-interval=30s --health-timeout=5s --health-start-period=60s \
  echosync:latest
```

---

## Endpoints

| Path | Method | Description |
|---|---|---|
| `/` | GET | Voice workbench UI |
| `/static/audio-processor.js` | GET | AudioWorklet node script |
| `/healthz` | GET | Liveness probe (`{"status":"ok"}`) |
| `/ready` | GET | Readiness probe (models loaded, memory OK) |
| `/ws/audio` | WebSocket | Binary PCM + JSON control gateway |
| `/metrics` | GET | Prometheus scrape endpoint |

---

## Testing

```bash
# Static analysis
uv run ruff check src tests
uv run ruff format --check src tests

# Full suite (81 passed, 1 skipped)
uv run pytest -v

# Focused suites
uv run pytest tests/test_vad.py tests/test_vad_live.py -v        # VAD
uv run pytest tests/test_stt.py -v                              # STT
uv run pytest tests/test_llm.py -v                              # LLM chunking + retry
uv run pytest tests/test_tts.py -v                              # TTS streaming
uv run pytest tests/test_websocket.py -v                        # E2E WS + barge-in
uv run pytest tests/test_bench.py -v                            # Harness unit tests
```

---

## Latency Budget Harness (GAPS T1.6)

`scripts/bench_turn.py` drives a live WS session with a canned WAV utterance, parses the server-side `metrics` block from the final `LISTENING` status, and prints per-stage p50/p95.

```bash
# Terminal 1: start server
uv run echosync

# Terminal 2: run benchmark (needs a mono WAV @ 16 kHz)
uv run python scripts/bench_turn.py \
  --audio scripts/audio/my_utterance.wav \
  --runs 5 \
  --gate-total 2.5 \
  --gate-ttft 1.0 \
  --json bench.json
```

Exit code **non-zero** if p50 RTT > 2.5 s or p50 TTFT > 1.0 s — CI-safe regression gate.
Same harness doubles as the **T1.1 Whisper A/B rig** — change `WHISPER_MODEL_NAME` on server, re-run, compare JSON outputs.

---

## Security (Tier 2 — before public exposure)

| Gap | Mitigation |
|---|---|
| WS origin spoofing | `WS_ALLOWED_ORIGINS` allowlist, enforced before `accept()` |
| Connection exhaustion | `asyncio.Semaphore(WS_MAX_CONCURRENT)` + per-IP dict cap |
| Unauthorized access | Optional `WS_BEARER_TOKEN` (HMAC-SHA256) |
| Barge-in false positives | `VAD_INTERRUPT_PROB` — only confident speech frames interrupt |
| PII in logs | JSON logs include raw transcript — apply retention/redaction policy |

---

## Observability (Tier 3)

- **JSON logs**: `src/api/json_logging.py` — single-line stdout, `echosync` logger, uvicorn untouched.
- **Prometheus `/metrics`**: `echosync_ws_connections_active`, `echosync_turns_total`, `echosync_turn_rtt_seconds` (histogram → p95 via `histogram_quantile`), plus STT/LLM/TTS histograms.
- **Per-turn telemetry**: Emitted in final `LISTENING` status — STT/LLM/TTS ms, total RTT, transcript, host RSS/CPU.

---

## Project Structure

```
EchoSync/
├── .github/workflows/ci.yml      # ruff + pytest + Docker → GHCR + Railway deploy
├── docker-compose.yml            # Local container stack
├── Dockerfile                    # Multi-stage, bakes weights, non-root, healthcheck
├── .dockerignore
├── pyproject.toml                # uv project, deps, ruff/pytest config
├── .env.example                  # Documented config template
├── GAPS.md                       # Living backlog (tiers, tickets, progress)
├── ARCHITECTURE.md               # Deep system design
├── PHASES.md                     # Historical phased implementation log
├── PRD.md                        # Product requirements
├── scripts/
│   ├── bench_turn.py             # Latency harness (T1.6)
│   ├── download_models.py        # Idempotent weight fetcher (used in Docker)
│   └── run.py                    # Unified launcher (env check → model dl → uvicorn)
├── src/
│   ├── main.py                   # FastAPI app, lifespan, probe routes, Instrumentator
│   ├── config.py                 # Pydantic-settings (all knobs)
│   ├── api/
│   │   ├── router.py             # WS endpoint, admission gate, pipeline loops
│   │   ├── telemetry.py          # Per-turn metrics dataclass + log/record
│   │   ├── metrics.py            # Prometheus collectors
│   │   └── json_logging.py       # Stdlib JSONFormatter
│   └── core/
│       ├── vad.py                # Silero VAD state machine + async frame proc
│       ├── stt.py                # faster-whisper async wrapper
│       ├── llm.py                # Gemini streaming + SentenceChunker
│       └── tts.py                # Kokoro streaming synthesis
└── tests/                        # 11 test modules, 81 passed
```

---

## CI/CD Pipeline

`.github/workflows/ci.yml` (runs on push to `main`):

1. **quality** — `ruff check`, `ruff format --check`, `pytest`
2. **docker** — Multi-stage build, BuildKit cache → push `ghcr.io/PatheticUser/EchoSync:latest` + `sha-<short>` + `branch-<name>`
3. **deploy** (conditional) — If `RAILWAY_TOKEN` + `RAILWAY_SERVICE_NAME` secrets present: `railway redeploy --service $NAME --detach` pulling the new `sha-` image

Add secrets in GitHub → Settings → Secrets → Actions to enable auto-deploy.

---

## Deployment Checklist (Staging)

- [ ] Railway account + project created
- [ ] GitHub secrets: `RAILWAY_TOKEN`, `RAILWAY_SERVICE_NAME`
- [ ] Railway env vars:
  - `GEMINI_API_KEY` (required)
  - `APP_ENV=production`
  - `WS_ALLOWED_ORIGINS=https://<your-app>.up.railway.app`
  - `LOG_LEVEL=INFO`
- [ ] RAM ≥ 2 GB, 1 vCPU
- [ ] Healthcheck `start_period ≥ 60s`
- [ ] `git push origin main` → CI builds → Railway pulls `sha-` image → live

---

## License

MIT — see `LICENSE`.