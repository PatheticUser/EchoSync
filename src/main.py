"""Main ASGI application entrypoint and lifespan orchestration."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.router import router
from src.api.telemetry import get_host_metrics
from src.config import get_settings
from src.core.llm import GeminiLLM
from src.core.stt import WhisperSTT
from src.core.tts import KokoroTTS
from src.core.vad import SileroVAD

logging.basicConfig(level=logging.INFO)
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
    )
    app.state.vad = vad

    # 2. Faster-Whisper STT
    stt = WhisperSTT(
        model_name=settings.whisper_model_name,
        compute_type=settings.whisper_compute_type,
        cpu_threads=settings.whisper_cpu_threads,
        download_root=settings.whisper_model_dir,
        sample_rate=settings.sample_rate,
    )
    stt.warmup()
    app.state.stt = stt

    # 3. Gemini LLM
    llm = GeminiLLM(api_key=settings.gemini_api_key)
    app.state.llm = llm

    # 4. Kokoro TTS
    tts = KokoroTTS(
        model_path=settings.kokoro_model_path,
        voices_path=settings.kokoro_voices_path,
    )
    tts.warmup()
    app.state.tts = tts

    logger.info("Pipeline models successfully loaded into application state.")
    yield
    logger.info("Tearing down application state.")


def create_app() -> FastAPI:
    """Create and configure FastAPI application instance."""
    app = FastAPI(
        title="EchoSync AI",
        description="Low-Latency Edge/Cloud Hybrid Voice Agent",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(router)

    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", include_in_schema=False)
    @app.get("/app", include_in_schema=False)
    async def index() -> FileResponse:
        """Serve web client workbench."""
        return FileResponse(static_dir / "index.html")

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
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
