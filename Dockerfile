# EchoSync AI — multi-stage container.
#
# Stage 1 (builder) installs deps and bakes neural models with a cacheable
# download layer so the runtime image ships weights on-disk — zero cold-start
# model downloads in production.
#
# Stage 2 (runtime) copies the venv, source, and model cache; runs as non-root.

# ---------- Stage 1: builder ----------
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=600 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Install dependencies first (cache layer: only invalidated on lock changes).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Bake model weights into the image via the idempotent downloader.
COPY scripts ./scripts
RUN /app/.venv/bin/python /app/scripts/download_models.py

# Application source.
COPY src ./src

# ---------- Stage 2: runtime ----------
FROM python:3.11-slim-bookworm AS runtime

# Runtime system deps: espeak-ng phonemizer for Kokoro TTS, ffmpeg + asound for audio safety.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        espeak-ng \
        ffmpeg \
        libasound2 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy virtualenv, model weights, and source from the builder stage.
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/models /app/models
COPY --from=builder /app/src /app/src

# Non-root runtime user.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Liveness probe via the bundled healthz endpoint.
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"]

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]