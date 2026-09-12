# EchoSync — GAPS & Production Backlog

Single source of truth for all known gaps, accuracy work, and production hardening.
Each ticket: branchable, PR-sized, CI-golden (existing gate: ruff + pytest + GHCR image build).

**Current posture**: working app, polished UI/UX, CI + Docker + GHCR pipeline live.
Accuracy below target (voice pipeline), production hardening incomplete, zero observability tooling. Not yet public.

**Deploy decision (pending)**: stage-first. Deploy to private staging env now, iterate accuracy live, go public when accuracy + Tier-2 gates pass. CI/CD makes every commit auto-build; deploy is not the risk, public exposure is.

## Progress (updated 2026-09-12)

**DONE (merged to main, full pytest green 79 passed/1 skipped):**
- T1.2 VAD window knobs · T1.3 chunker sentence-boundary · T1.4 LLM gen params · T1.5 prompt rewrite
- T2.1–T2.3 WS admission gate (origin allowlist, concurrency + per-IP caps, optional bearer token)
- T2.4 VAD interrupt probability gate (`VAD_INTERRUPT_PROB`)
- T3.1 unified JSON logging · T3.3 Prometheus /metrics
- T4.1 CI Railway deploy job
- Commit range: `194e413` → `HEAD` (10 merge commits). All landed via parallel agent branches beneath `git log --graph`.

**PENDING:**
- T1.1 whisper tiny/base/small A/B bench (needs live mic corpus) — run after T1.2+T1.3 tuning
- T1.6 latency budget harness (scripts/bench_turn.py) — gate before staging
- T3.2 log drain · T3.4 dashboards/alerts · T3.5 uptime pings · T3.6–3.7 sentry/tracing (post-staging)
- T4.2 WS keepalive ping · T4.3 instance sizing docs · T4.4 cost lock
- Staging deploy: Railway account, env vars (GEMINI_API_KEY; WS_ALLOWED_ORIGINS=https://<app>.up.railway.app; APP_ENV=production), healthcheck start_period

---

## Tier 0 — Deploy posture (prereq, one-time)

| ID | Gap | Detail | Status |
|----|-----|--------|--------|
| T0.1 | Platform choice | Railway or Fly.io full-stack, ≥2GB RAM, 1vCPU. Render only on paid Standard 2GB. No Vercel/Netlify backend. | open |
| T0.2 | Secrets on platform | `GEMINI_API_KEY` required field — app won't boot without it (`src/config.py:25`). Set `APP_ENV=production`, `LOG_LEVEL=INFO`. | open |
| T0.3 | Model id verify | `GEMINI_MODEL=gemini-3.6-flash` hardcoded default (`src/config.py:30`). Verify id exists in your Google account + quota. | open |
| T0.4 | Healthcheck grace | Platform start-period ≥60s (warmup 10–30s + image pull). | open |
| T0.5 | Deploy image tag | Pin `ghcr.io/...:sha-<hash>`, never `latest` in prod. | open |

## Tier 1 — Accuracy (voice pipeline) ★ user priority

### T1.1 Whisper model selection
**Context**: switched base.en → tiny.en for speed. Real accuracy drop — tiny WER ~2× base. Perceived inaccuracy ALSO caused by clipping + chunking, not only WER (see T1.2, T1.3). Fix pipeline first, re-benchmark, then decide model.

| option | size (int8) | STT latency | WER | note |
|---|---|---|---|---|
| tiny.en | ~75MB | fastest | highest | current. keep if pipeline fixes suffice |
| base.en | ~139MB | ~1.5–2× tiny | ~half tiny | was too slow before chunk tuning |
| small.en | ~461MB | ~3–4× tiny | best | accuracy priority; bigger image (+330MB), slower TTFT |

- Add `WHISPER_MODEL_NAME` A/B benchmark harness (reuse `tests/test_stt.py` fixture): same audio corpus, table of stt_ms + transcript diff.
- Action: tune chunker + VAD first (T1.2/T1.3) → re-test tiny vs base on real mic audio → pick model. small.en only if accuracy still fails.

### T1.2 VAD silence window + pre-padding
**Now**: `silence_ms=400` → 13 frames @32ms (`src/core/vad.py:58`), `pre_speech_padding_frames=3` (96ms), `min_speech_ms=250`.
**Symptom**: trailing words clipped if user pauses >400ms inside/rear → truncated transcript → wrong replies = "not accurate".
- Expose config: `VAD_SILENCE_MS`, `VAD_PRE_PADDING_FRAMES`, `VAD_MIN_SPEECH_MS` already env-driven (`src/config.py`) — no new plumbing, just tune + test.
- Test matrix: silence 600 / 800 / 1000ms; min_speech 200 / 400 / 600ms; pre-pad 3 / 5 / 8 frames. Measure: clipped-tail rate on fixed utterances, turn latency delta (longer window = later SPEECH_END = slower turn).
- Target: natural silent pause tolerance without near-doubling turn RTT.

### T1.3 Sentence chunker — root cause of "stops between punctuation"
**Now**: `SentenceChunker` (`src/core/llm.py:41`) emits clause on `.!?;:,` — first pattern includes `,`/`:`, standard includes `;`/`:`. TTS then speaks every comma-ish chunk. Short clauses = staccato, un-natural, feels like "stopping in between punctuation". `min_chars=12`, `early_first_chunk=True`.
- **Fix**: emit on sentence-terminal only (`.?!`) after first chunk; keep comma as boundary only when clause ≥ min_chars and next sentence incoming. Raise min_chars 12 → ~20 (2–3 words).
- Params via `stream_sentence_chunks` call-site (`src/api/router.py:207`): pass `min_chars` from settings instead of hardcode.

### T1.4 LLM generation params
**Now**: `temperature=0.3`, `max_output_tokens=150`, thinking off-by-default, `asyncio.timeout(5.0)`, mem turns 8 (`src/core/llm.py:153-178`).
- Expose `LLM_TEMPERATURE`, `LLM_TOP_P`, `LLM_MAX_OUTPUT_TOKENS`, `LLM_TIMEOUT_S` from settings → `GenerateContentConfig`.
- For "caveman-precise, normal vocabulary, complete sentences": **temperature ~0.4** (keep a little variance, avoid robotic), **top_p ~0.95**, no thinking budget (latency).
- Profile: max_output_tokens 150 may truncate mid-sentence on verbose replies → clipped TTS tail. Raise to ~180–200 if truncation observed.
- Verify `asyncio.timeout(5.0)` under slow Gemini — is it aborting partial streams? log it.

### T1.5 System prompt — desired voice
**Now**: `DEFAULT_VOICE_SYSTEM_PROMPT` (`src/core/llm.py:21`): "1-2 spoken sentences… never Markdown".
**Target behavior**: direct, to-the-point, but grammatically complete normal sentences. No dropped words, no filler ("ah", "um"), no bullet artifacts, no markdown. Professional register — demo-facing (employers/university).
- Rewrite prompt with positive constraints:
  - "Respond in 1–3 complete, grammatically correct sentences."
  - "Never omit words. Never telegraph or pause the reply with fillers."
  - "Direct and concise, but conversational and natural — not terse fragments."
  - "Never use markdown, lists, headers."
  - "If you do not know, say so plainly in one sentence."
- A/B: run 10 fixed prompts against current vs new system prompt, judge verbosity + grammar (manual).

### T1.6 Latency budget harness
- Script `scripts/bench_turn.py`: drives WS with canned audio, prints per-stage table (stt_ms, llm_ttft, tts_first_chunk, total_rtt) from `/ready`-style metrics + turn telemetry JSON.
- Gate: target p50 total turn < ~2.5s after tuning; TTFT < ~1s.
- Re-run after every Tier-1 change — objective regression check before/after.

## Tier 2 — Security / abuse before public

| ID | Gap | Why | Where |
|----|-----|-----|-------|
| T2.1 | WS origin allowlist | anyone can open `/ws/audio`, burn compute + Gemini quota = bill shock | `src/api/router.py:96` accept |
| T2.2 | WS connect rate + concurrency cap | per-IP rate limit, `asyncio.Semaphore` max concurrent | `src/api/router.py` |
| T2.3 | optional WS bearer token | cheap auth if demo needs it; header check | `router.py` + `app.html:921` |
| T2.4 | Transcript PII policy | raw user speech in JSON logs (`telemetry.py:58`); retention + redaction decision | ops |

## Tier 3 — Observability

| ID | Gap | Action |
|----|-----|--------|
| T3.1 | Unify JSON logging | `main.py:20` text `basicConfig` vs pipeline JSON. One schema everywhere: `python-json-logger` or structlog. |
| T3.2 | Log drain | stream stdout → Axiom (native Railway/Render drain) or Better Stack or Grafana Loki. JSON telemetry already enrich-ready. |
| T3.3 | `/metrics` | Prometheus endpoint (`prometheus-fastapi-instrumentator`): turns/min, WS conns, RTT p95 from telemetry, errors. |
| T3.4 | Dashboards + alerts | Grafana Cloud free (Loki+Mimir+alert). Alert: readiness 503, error rate, WS conns spike. |
| T3.5 | Uptime | Better Stack / Uptime Kuma 60s ping `/healthz` + `/ready`. |
| T3.6 | Error tracking (later) | Sentry fastapi integration — unhandled WS + LLM API failures. |
| T3.7 | Tracing (later) | OTLP → spans per turn (STT→LLM→TTS) if chasing latency. |

## Tier 4 — Infra hardening

| ID | Gap | Action |
|----|-----|--------|
| T4.1 | CI deploy step | auto-deploy `sha-` image to Railway/Render on main merge (`railway up` CLI or Render webhook) |
| T4.2 | WS keepalive | server-periodic WS ping so idle sessions survive proxy timeouts |
| T4.3 | Instance sizing docs | RAM≥2GB, 1vCPU; horizontal scale (instances), not `--workers` (each worker reloads models, RAM doubles) |
| T4.4 | Cost lock | 1 staging instance; scale down test; Gemini flash traffic cost watch |

---

## Parallel execution plan (team of Claudes — C-compiler pattern)

Blog pattern: task list → one implementer agent per ticket on its own branch → reviewer agent per PR → CI-golden merge → main auto-build.

**Roles**
- **Keeper** (main thread, this session): ticket queue, branch policy, merge orchestrator. No deep code work.
- **Implementers**: one per Tier-1/2 ticket, one branch each.
- **Reviewer**: one per merge, checks correctness + no regress on `src` + tests.
- **Tester**: runs `pytest` locally (`uv run pytest`), latency harness, reports back.

**Flow per ticket**
1. Keeper assigns branch `t-<tier>-<slug>` (e.g. `t-1-chunker`).
2. Implementer works isolated worktree, commits, opens PR.
3. CI gate auto-runs: ruff lint/format + full pytest + Docker build (existing `ci.yml`).
4. Reviewer agent reviews diff. Findings → implementer fixes → CI green.
5. Keeper merges to `main` → GHCR `latest` + `sha-` auto-publish → deploy target picks up.

**Branch map (first wave)**
| branch | ticket | scope |
|--------|--------|-------|
| `t-1-vad-window` | T1.2 | VAD silence/pad tuning + tests |
| `t-1-chunker` | T1.3 | chunker sentence-boundary fix + min_chars |
| `t-1-llm-params` | T1.4 | expose temp/top_p/max_tokens via settings |
| `t-1-prompt` | T1.5 | system prompt rewrite + A/B |
| `t-2-ws-gate` | T2.1–2.2 | origin allowlist + rate/concurrency cap |
| `t-3-logging` | T3.1 | unified JSON logging |
| `t-3-metrics` | T3.3 | `/metrics` Prometheus |
| `t-4-cideploy` | T4.1 | CI deploy step |

T1.1 (whisper A/B) ships after T1.2+T1.3 land — bench tiny vs base on the fixed pipeline.
T1.6 harness is a keeper-owned prep step before any accuracy work lands.

**Merge safety**: every PR green tests; ruff format enforced (`ruff format --check` in CI). Accuracy branches cannot regress the CI gate, so voice pipeline stays shippable during parallel work.

---

## Existing strengths — do not regress
- CI: ruff lint+format+full pytest+Docker build → GHCR push (`ci.yml`) ✓
- Multi-stage Docker bakes weights, non-root runtime, healthcheck ✓
- `/healthz` + `/ready` with per-model flags + mem/cpu ✓
- JSON turn telemetry (stage timings, transcript, host metrics) ✓
- WebSocket pipeline with barge-in, memory alternation, retry/backoff on Gemini ✓
- UI/UX polished, dark/light themes ✓