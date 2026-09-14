# EchoSync AI — multi-stage container.
#
# Models are NOT baked into the image at build time. They are downloaded at
# first container startup via scripts/download_models.py (idempotent — skips
# if the file already exists at the expected path).
#
# To persist models across container restarts, mount a volume at /app/models:
#   docker run -v echosync_models:/app/models echosync
#
# This keeps the CI docker build fast and avoids HuggingFace rate-limit failures.

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

# Application source and scripts.
COPY src ./src
COPY scripts ./scripts

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

# Liveness probe via the bundled healthz endpoint.
HEALTHCHECK --interval=10s --timeout=3s --start-period=60s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"]

# Download models on first run (idempotent: skips existing files), then start server.
CMD ["sh", "-c", "python /app/scripts/download_models.py && uvicorn src.main:app --host 0.0.0.0 --port 8000"]