"""Unit and integration tests for WebSocket audio gateway and barge-in handling."""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from src.config import get_settings
from src.core.stt import TranscriptionResult
from src.core.vad import VADEvent, VADState
from src.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Initialize test client with app context."""
    with TestClient(app) as test_client:
        yield test_client


def test_websocket_connection_handshake(client: TestClient) -> None:
    """Verify WebSocket connection acceptance and initial LISTENING status."""
    with client.websocket_connect("/ws/audio") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "status"
        assert msg["data"]["state"] == "LISTENING"


def test_websocket_interrupt_handling(client: TestClient) -> None:
    """Verify client interrupt JSON control message resets state to LISTENING."""
    with client.websocket_connect("/ws/audio") as ws:
        # Initial greeting
        init_msg = ws.receive_json()
        assert init_msg["data"]["state"] == "LISTENING"

        # Send interrupt message
        ws.send_text(json.dumps({"type": "user_interrupt"}))
        status_msg = ws.receive_json()
        assert status_msg["type"] == "status"
        assert status_msg["data"]["state"] == "LISTENING"


def test_websocket_audio_ingestion_and_turn_flow(client: TestClient) -> None:
    """Verify simulated dialogue turn from VAD SPEECH_END through STT, LLM, and TTS delivery."""
    synthetic_audio = np.zeros(16000, dtype=np.float32)

    # Mock components to run turn deterministically and fast
    mock_vad_event = VADEvent(
        state=VADState.SPEECH_END,
        probability=0.01,
        audio_buffer=synthetic_audio,
        duration_ms=500.0,
    )
    mock_stt_result = TranscriptionResult(
        text="What is the weather?",
        duration_ms=45.0,
        audio_duration_s=1.0,
    )

    async def mock_stream_sentence_chunks(prompt: str, **kwargs):
        yield "The weather is sunny and warm."

    async def mock_synthesize_stream(text: str):
        # 2 chunks of 2048 bytes
        yield b"\x00" * 2048
        yield b"\x01" * 2048

    with (
        patch(
            "src.core.vad.SileroVAD.async_process_frame", new=AsyncMock(return_value=mock_vad_event)
        ),
        patch(
            "src.core.stt.WhisperSTT.async_transcribe", new=AsyncMock(return_value=mock_stt_result)
        ),
        patch(
            "src.core.llm.GeminiLLM.stream_sentence_chunks", side_effect=mock_stream_sentence_chunks
        ),
        patch("src.core.tts.KokoroTTS.synthesize_stream", side_effect=mock_synthesize_stream),
        client.websocket_connect("/ws/audio") as ws,
    ):
        init_msg = ws.receive_json()
        assert init_msg["data"]["state"] == "LISTENING"

        # Send 1024-byte audio chunk
        dummy_pcm_chunk = np.zeros(512, dtype=np.int16).tobytes()
        ws.send_bytes(dummy_pcm_chunk)

        # 1. PROCESSING_STT
        stt_status = ws.receive_json()
        assert stt_status["type"] == "status"
        assert stt_status["data"]["state"] == "PROCESSING_STT"

        # 2. Transcript
        transcript_msg = ws.receive_json()
        assert transcript_msg["type"] == "transcript"
        assert transcript_msg["data"]["text"] == "What is the weather?"

        # 3. STREAMING_LLM
        llm_status = ws.receive_json()
        assert llm_status["type"] == "status"
        assert llm_status["data"]["state"] == "STREAMING_LLM"

        # 4. SPEAKING
        speaking_status = ws.receive_json()
        assert speaking_status["type"] == "status"
        assert speaking_status["data"]["state"] == "SPEAKING"

        # 5. Audio Header
        audio_header = ws.receive_json()
        assert audio_header["type"] == "audio_header"
        assert audio_header["data"]["text_segment"] == "The weather is sunny and warm."
        assert audio_header["data"]["sample_rate"] == 24000

        # 6. Binary Audio Chunks
        chunk_1 = ws.receive_bytes()
        assert len(chunk_1) == 2048
        chunk_2 = ws.receive_bytes()
        assert len(chunk_2) == 2048

        # 7. LISTENING (turn finished)
        final_status = ws.receive_json()
        assert final_status["type"] == "status"
        assert final_status["data"]["state"] == "LISTENING"


# ---------------------------------------------------------------------------
# T2.1 / T2.2 / T2.3 admission gate tests. All rejections happen before accept
# and surface client-side as WebSocketDisconnect with the configured code.
# ---------------------------------------------------------------------------


def test_websocket_rejects_disallowed_origin(client: TestClient) -> None:
    """T2.1: a non-allowlisted Origin is refused with 4403 before accept."""
    with (
        pytest.raises(WebSocketDisconnect) as exc_info,
        client.websocket_connect("/ws/audio", headers={"Origin": "https://evil.example"}),
    ):
        pass
    assert exc_info.value.code == 4403


def test_websocket_accepts_allowlisted_origin(client: TestClient) -> None:
    """T2.1: an Origin on WS_ALLOWED_ORIGINS connects normally."""
    with client.websocket_connect("/ws/audio", headers={"Origin": "http://127.0.0.1:8000"}) as ws:
        msg = ws.receive_json()
        assert msg["data"]["state"] == "LISTENING"


def test_websocket_requires_origin_in_non_dev(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T2.1: outside development, a missing Origin (non-browser client) is refused."""
    monkeypatch.setenv("APP_ENV", "production")
    get_settings.cache_clear()
    try:
        with (
            pytest.raises(WebSocketDisconnect) as exc_info,
            client.websocket_connect("/ws/audio"),
        ):
            pass
        assert exc_info.value.code == 4403
    finally:
        get_settings.cache_clear()


def test_websocket_rejects_beyond_concurrency_cap(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T2.2: sessions beyond WS_MAX_CONCURRENT are refused with 4403."""
    monkeypatch.setattr("src.api.router._ws_concurrency_semaphore", asyncio.Semaphore(1))
    with client.websocket_connect("/ws/audio") as ws:
        assert ws.receive_json()["data"]["state"] == "LISTENING"
        # A second simultaneous session must be refused while the first holds the slot.
        with (
            pytest.raises(WebSocketDisconnect) as exc_info,
            client.websocket_connect("/ws/audio"),
        ):
            pass
        assert exc_info.value.code == 4403


def test_websocket_rejects_beyond_per_ip_cap(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T2.2: more simultaneous sessions than WS_MAX_PER_IP from one IP close 4403."""
    monkeypatch.setenv("WS_MAX_PER_IP", "1")
    get_settings.cache_clear()
    try:
        with client.websocket_connect("/ws/audio") as ws:
            assert ws.receive_json()["data"]["state"] == "LISTENING"
            with (
                pytest.raises(WebSocketDisconnect) as exc_info,
                client.websocket_connect("/ws/audio"),
            ):
                pass
            assert exc_info.value.code == 4403
    finally:
        get_settings.cache_clear()


def test_websocket_bearer_token_gate(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """T2.3: when WS_BEARER_TOKEN is set, missing/invalid creds close 4401 and valid open."""
    monkeypatch.setenv("WS_BEARER_TOKEN", "s3cret")
    get_settings.cache_clear()
    try:
        with (
            pytest.raises(WebSocketDisconnect) as missing_token,
            client.websocket_connect("/ws/audio"),
        ):
            pass
        assert missing_token.value.code == 4401

        with (
            pytest.raises(WebSocketDisconnect) as wrong_token,
            client.websocket_connect("/ws/audio", headers={"Authorization": "Bearer wrong"}),
        ):
            pass
        assert wrong_token.value.code == 4401

        with client.websocket_connect(
            "/ws/audio", headers={"Authorization": "Bearer s3cret"}
        ) as ws:
            msg = ws.receive_json()
            assert msg["data"]["state"] == "LISTENING"
    finally:
        get_settings.cache_clear()
