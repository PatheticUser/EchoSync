# syntax=docker/dockerfile:1.7
# EchoSync AI — multi-stage, multi-arch container (linux/amd64 + linux/arm64/aarch64).
#
# ARM64 (Oracle Cloud Ampere A1) support — every layer is architecture-clean:
#   * ghcr.io/astral-sh/uv:python3.11-bookworm-slim  — official multi-arch image (amd64/arm64)
#   * python:3.11-slim-bookworm                      — official multi-arch image (amd64/arm64)
#   * espeak-ng / ffmpeg / libasound2 / tini         — Debian bookworm arm64 native packages
#   * ctranslate2 (via faster-whisper) >= 4.4        — official manylinux aarch64 wheels
#   * onnxruntime (VAD + Kokoro)                     — official manylinux aarch64 wheels
#   * numpy, kokoro-onnx, edge-tts, google-genai     — pure-Python / universal wheels
#   * NO PyTorch anywhere: Silero VAD runs through ONNX Runtime, so there is no
#     torch/C++ extension to compile for aarch64.
#
# The builder only resolves and downloads wheels (no source compilation), so the
# same recipe builds natively on the ARM64 VM and on x86 CI runners (or via
# buildx/QEMU for multi-arch manifest pushes):
#   docker buildx build --platform linux/amd64,linux/arm64 -t ghcr.io/<org>/echosync:latest .
#
# Models are NOT baked into the image. They are downloaded at first container
# startup by scripts/download_models.py (idempotent — skips files that already
# exist). Persist them across container recreation with the `models` named
# volume defined in docker-compose.yml (/app/models).

# ---------- Stage 1: builder ----------
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=600 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Install dependencies first (cache layer: only invalidated on lock changes).
# uv picks the correct aarch64 wheels automatically when built on/for arm64.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Application source and scripts.
COPY src ./src
COPY scripts ./scripts

# ---------- Stage 2: runtime ----------
FROM python:3.11-slim-bookworm AS runtime

# Runtime system deps (all available as Debian arm64 packages):
#   espeak-ng  — phonemizer backend for Kokoro TTS
#   ffmpeg     — EdgeTTS MP3 chunk decoding fallback path
#   libasound2 — ALSA audio safety net
#   tini       — correct PID 1 signal handling / zombie reaping under Docker
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        espeak-ng \
        ffmpeg \
        libasound2 \
        tini \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy virtualenv and source from the builder stage.
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/scripts /app/scripts

# Non-root runtime user. Give write access to /app/models for first-run download.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/models \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Liveness probe via the bundled healthz endpoint (python-based: no curl needed).
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=5 \
    CMD ["python", "-c", "import os, urllib.request; p = os.environ.get('PORT', '8000'); urllib.request.urlopen(f'http://127.0.0.1:{p}/healthz', timeout=3)"]

# tini reaps processes and forwards signals; download models on first run
# (idempotent: skips existing files), then start the ASGI server.
#
# --proxy-headers --forwarded-allow-ips='*': honor Caddy's X-Forwarded-* on
# HTTP requests (client IP + scheme) so Prometheus labels and redirects are
# correct behind the TLS-terminating proxy. Safe to trust all senders because
# the app publishes NO host ports — only Caddy on the internal bridge can reach
# it, and Caddy overwrites X-Forwarded-For with the real remote address.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["sh", "-c", "python /app/scripts/download_models.py && exec uvicorn src.main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
