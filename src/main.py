"""Main ASGI application entrypoint and lifespan orchestration."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.json_logging import configure_app_logging
from src.api.router import router
from src.api.telemetry import get_host_metrics
from src.config import get_settings
from src.core.llm import GeminiLLM
from src.core.stt import WhisperSTT
from src.core.vad import SileroVAD

# Root logger stays a plain-text fallback for third-party libraries only.
# App loggers under "echosync." (main, router, pipeline) emit single-line JSON
# via configure_app_logging(); uvicorn's own loggers keep their default text
# config and remain LOG_LEVEL-driven as before.
logging.basicConfig(level=logging.INFO)
configure_app_logging()
logger = logging.getLogger("echosync.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Pre-warm neural models during ASGI startup and cleanly teardown on shutdown."""
    settings = get_settings()
    logger.info("Initializing and pre-warming pipeline models...")

    # 1. Silero VAD
    vad = SileroVAD(
        model_path=settings.vad_model_path,
        sample_rate=settings.sample_rate,
        frame_size=settings.frame_size,
        threshold=settings.vad_threshold,
        silence_ms=settings.vad_silence_ms,
        min_speech_ms=settings.vad_min_speech_ms,
        pre_speech_padding_frames=settings.vad_pre_padding_frames,
    )
    app.state.vad = vad

    # 2. Faster-Whisper STT
    stt = WhisperSTT(
        model_name=settings.whisper_model_name,
        compute_type=settings.whisper_compute_type,
        cpu_threads=settings.whisper_cpu_threads,
        download_root=settings.whisper_model_dir,
        sample_rate=settings.sample_rate,
        initial_prompt=settings.whisper_initial_prompt,
        boost_audio=settings.stt_boost_audio,
    )
    stt.warmup()
    app.state.stt = stt

    # 3. Gemini LLM
    llm = GeminiLLM(api_key=settings.gemini_api_key)
    app.state.llm = llm

    # 4. Acoustic TTS Engine (EdgeTTS for free-tier/low-CPU vs Kokoro for offline ONNX)
    from src.core.tts import create_tts

    tts = create_tts(settings)
    tts.warmup()
    app.state.tts = tts

    logger.info("Pipeline models successfully loaded into application state.")
    yield
    logger.info("Tearing down application state.")


def create_app() -> FastAPI:
    """Create and configure FastAPI application instance."""
    configure_app_logging()
    app = FastAPI(
        title="EchoSync AI",
        description="Low-Latency Edge/Cloud Hybrid Voice Agent",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(router)

    # Expose Prometheus /metrics: HTTP request totals/errors/durations from the
    # instrumentator plus the custom echosync_* collectors in src.api.metrics.
    Instrumentator().instrument(app).expose(app, include_in_schema=False)

    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def landing() -> FileResponse:
        """Serve marketing landing page."""
        return FileResponse(static_dir / "index.html")

    @app.get("/app", include_in_schema=False)
    async def app_index() -> FileResponse:
        """Serve interactive voice console workbench."""
        return FileResponse(static_dir / "app.html")

    @app.get("/healthz", tags=["Probes"])
    async def healthz() -> dict[str, str]:
        """Immediate liveness check returning HTTP 200."""
        return {"status": "ok"}

    @app.get("/ready", tags=["Probes"])
    async def ready() -> JSONResponse:
        """Readiness check verifying neural model singletons are resident in memory."""
        vad_loaded = hasattr(app.state, "vad") and app.state.vad is not None
        stt_loaded = hasattr(app.state, "stt") and app.state.stt is not None
        tts_loaded = hasattr(app.state, "tts") and app.state.tts is not None

        models_ready = vad_loaded and stt_loaded and tts_loaded
        status_code = status.HTTP_200_OK if models_ready else status.HTTP_503_SERVICE_UNAVAILABLE

        host_metrics = get_host_metrics()
        payload = {
            "status": "ready" if models_ready else "initializing",
            "models": {
                "vad": vad_loaded,
                "stt": stt_loaded,
                "tts": tts_loaded,
            },
            "memory_rss_mb": host_metrics.memory_rss_mb,
            "cpu_percent": host_metrics.cpu_percent,
        }
        return JSONResponse(content=payload, status_code=status_code)

    return app


app = create_app()


def main() -> None:
    """CLI execution entry point."""
    import uvicorn

    settings = get_settings()
    # Honor LOG_LEVEL for app loggers too; uvicorn gets its own log_level below.
    configure_app_logging(level=settings.log_level)
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
