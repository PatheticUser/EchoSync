"""Unit tests and offline verification for upstream LLM streaming and chunking."""

import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai.errors import APIError

from src.core.llm import (
    GeminiLLM,
    SentenceChunker,
    clean_speech_text,
    is_retryable_llm_error,
)


@dataclass(slots=True)
class MockChunk:
    """Mock container mimicking Gemini GenerateContentResponse."""

    text: str | None


async def async_generator(items: list[Any]) -> AsyncIterator[Any]:
    """Helper to convert a list of items into an async generator."""
    for item in items:
        yield item


def test_clean_speech_text() -> None:
    """Verify markdown headers, bullets, asterisks, and code blocks are stripped."""
    raw_markdown = (
        "# Overview\n"
        "**Hello!** *Welcome* to `EchoSync`.\n"
        "- Point 1: Low latency voice pipeline.\n"
        "* Point 2: Edge compute.\n"
        "1. Step one is ready."
    )
    cleaned = clean_speech_text(raw_markdown)
    assert "#" not in cleaned
    assert "*" not in cleaned
    assert "`" not in cleaned
    assert "-" not in cleaned
    assert (
        "Hello! Welcome to EchoSync. Point 1: Low latency voice pipeline. Point 2: Edge compute. Step one is ready."
        == cleaned
    )


def test_sentence_chunker_deterministic() -> None:
    """Verify sentence accumulator splits clauses on boundaries with min_chars."""
    chunker = SentenceChunker(min_chars=20)
    tokens = [
        "Hello ",
        "there! ",
        "How ",
        "are you ",
        "doing today? ",
        "I am fine, thank you. ",
        "Goodbye!",
    ]

    emitted: list[str] = []
    for token in tokens:
        clauses = chunker.feed(token)
        emitted.extend(clauses)

    final = chunker.flush()
    if final:
        emitted.append(final)

    assert len(emitted) == 3
    # "Hello there!" (12 chars < 20) merges with "How are you doing today?"
    assert emitted[0] == "Hello there! How are you doing today?"
    assert emitted[1] == "I am fine, thank you."
    assert emitted[2] == "Goodbye!"


def test_sentence_chunker_empty_and_flush() -> None:
    """Verify chunker handles empty tokens and flushes remaining partial buffers."""
    chunker = SentenceChunker(min_chars=20)
    assert chunker.feed("") == []
    assert chunker.feed("   ") == []
    assert chunker.flush() is None

    # Feed text without boundary delimiter
    assert chunker.feed("This text has no punctuation yet") == []
    assert chunker.flush() == "This text has no punctuation yet"


@pytest.mark.asyncio
async def test_gemini_llm_stream_sentence_chunks_mocked() -> None:
    """Verify GeminiLLM stream_sentence_chunks yields cleanly parsed clauses from stream."""
    mock_chunks = [
        MockChunk(text="Paris is the capital of France. "),
        MockChunk(text="It is famous for the Eiffel Tower. "),
        MockChunk(text="Have a nice day!"),
    ]

    mock_client = MagicMock()
    mock_client.aio.models.generate_content_stream = AsyncMock(
        return_value=async_generator(mock_chunks)
    )

    llm = GeminiLLM(api_key="mock_key", client=mock_client)
    clauses: list[str] = []
    async for clause in llm.stream_sentence_chunks("What is the capital of France?"):
        clauses.append(clause)

    assert len(clauses) == 3
    assert clauses[0] == "Paris is the capital of France."
    assert clauses[1] == "It is famous for the Eiffel Tower."
    assert clauses[2] == "Have a nice day!"


@pytest.mark.asyncio
async def test_tenacity_retry_on_429_transient() -> None:
    """Verify tenacity retries on rate limit HTTP 429 and succeeds on subsequent try."""
    mock_chunks = [MockChunk(text="Recovered from 429 successfully.")]

    # First attempt raises APIError 429, second attempt succeeds
    error_429 = APIError(429, {"error": {"message": "Resource exhausted: Rate limit reached"}})
    mock_client = MagicMock()
    mock_client.aio.models.generate_content_stream = AsyncMock(
        side_effect=[error_429, async_generator(mock_chunks)]
    )

    llm = GeminiLLM(api_key="mock_key", client=mock_client)
    clauses: list[str] = []
    async for clause in llm.stream_sentence_chunks("Ping"):
        clauses.append(clause)

    assert len(clauses) == 1
    assert clauses[0] == "Recovered from 429 successfully."
    assert mock_client.aio.models.generate_content_stream.call_count == 2


@pytest.mark.asyncio
async def test_tenacity_retry_exhaustion() -> None:
    """Verify tenacity exhausts retries and raises after 3 attempts."""
    error_429 = APIError(429, {"error": {"message": "Rate limit exceeded"}})
    mock_client = MagicMock()
    mock_client.aio.models.generate_content_stream = AsyncMock(side_effect=error_429)

    llm = GeminiLLM(api_key="mock_key", client=mock_client)
    with pytest.raises(APIError):
        async for _ in llm.stream_tokens("Ping"):
            pass

    assert mock_client.aio.models.generate_content_stream.call_count == 3


def test_is_retryable_llm_error() -> None:
    """Verify retry filter on various error types."""
    assert is_retryable_llm_error(APIError(429, {"error": {"message": "Quota limit"}}))
    assert is_retryable_llm_error(APIError(503, {"error": {"message": "Service unavailable"}}))
    assert not is_retryable_llm_error(APIError(400, {"error": {"message": "Bad request"}}))
    assert not is_retryable_llm_error(ValueError("Invalid argument"))


@pytest.mark.asyncio
async def test_live_gemini_smoke_test() -> None:
    """Live smoke test against Google AI Studio API (skipped if key is dummy/missing)."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key or api_key.startswith(("your_", "test_")):
        pytest.skip("No real GEMINI_API_KEY present in environment")

    llm = GeminiLLM(api_key=api_key)
    tokens: list[str] = []
    async for token in llm.stream_tokens("Respond with 'EchoSync online.'"):
        tokens.append(token)

    full_text = "".join(tokens)
    assert len(full_text) > 0
