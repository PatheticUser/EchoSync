"""Pytest configuration and session fixtures for EchoSync test suite."""

import os

import pytest

from src.config import get_settings


@pytest.fixture(autouse=True, scope="session")
def setup_test_environment() -> None:
    """Set default test environment variables for unit test repeatability."""
    os.environ.setdefault("GEMINI_API_KEY", "test_gemini_api_key_placeholder")
    os.environ["TTS_ENGINE"] = "kokoro"
    os.environ["PORT"] = "8000"
    os.environ["WS_ALLOWED_ORIGINS"] = '["http://127.0.0.1:8000","http://localhost:8000"]'
    get_settings.cache_clear()
