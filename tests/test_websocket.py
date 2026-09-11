"""Unit and integration tests for WebSocket audio gateway and barge-in handling."""

import json
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

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
