"""Pytest configuration and session fixtures for EchoSync test suite."""

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def setup_test_environment() -> None:
    """Set default test environment variables if not present."""
    if "GEMINI_API_KEY" not in os.environ:
        os.environ["GEMINI_API_KEY"] = "test_gemini_api_key_placeholder"
