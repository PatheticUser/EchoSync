"""Unit tests for FastAPI health and readiness probe endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Initialize test client with lifespan pre-warming context."""
    with TestClient(app) as test_client:
        yield test_client


def test_healthz_liveness_probe(client: TestClient) -> None:
    """Verify GET /healthz returns HTTP 200 and healthy status."""
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_readiness_probe(client: TestClient) -> None:
    """Verify GET /ready returns HTTP 200 and confirms resident neural models."""
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["models"]["vad"] is True
    assert data["models"]["stt"] is True
    assert data["models"]["tts"] is True
    assert data["memory_rss_mb"] > 0


def test_static_workbench_routes(client: TestClient) -> None:
    """Verify GET / and GET /app return 200 OK with HTML content."""
    root_res = client.get("/")
    assert root_res.status_code == 200
    assert "text/html" in root_res.headers.get("content-type", "")
    assert "EchoSync" in root_res.text
    assert "Try Now" in root_res.text
    assert "audio-processor.js" not in root_res.text

    app_res = client.get("/app")
    assert app_res.status_code == 200
    assert "text/html" in app_res.headers.get("content-type", "")
    assert "audio-processor.js" in app_res.text
    assert "waveformCanvas" in app_res.text


def test_audio_processor_asset_served(client: TestClient) -> None:
    """Verify GET /static/audio-processor.js serves AudioWorklet code."""
    res = client.get("/static/audio-processor.js")
    assert res.status_code == 200
    assert "AudioProcessor" in res.text
    assert "registerProcessor" in res.text
