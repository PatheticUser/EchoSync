"""Unit tests and offline verification for upstream LLM streaming and chunking."""

import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from itertools import pairwise
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai.errors import APIError

from src.core.llm import (
    DEFAULT_VOICE_SYSTEM_PROMPT,
    GeminiLLM,
    SentenceChunker,
    build_contents,
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

    assert len(emitted) == 4
    # First chunk emits early once min 12 chars (low TTS latency), even though
    # min_chars=20 governs all subsequent boundaries.
    assert emitted[0] == "Hello there!"
    assert emitted[1] == "How are you doing today?"
    assert emitted[2] == "I am fine, thank you."
    assert emitted[3] == "Goodbye!"


def test_sentence_chunker_empty_and_flush() -> None:
    """Verify chunker handles empty tokens and flushes remaining partial buffers."""
    chunker = SentenceChunker(min_chars=20)
    assert chunker.feed("") == []
    assert chunker.feed("   ") == []
    assert chunker.flush() is None

    # Feed text without boundary delimiter
    assert chunker.feed("This text has no punctuation yet") == []
    assert chunker.flush() == "This text has no punctuation yet"


def test_sentence_chunker_splits_on_terminal_punctuation() -> None:
    """Verify standard mode splits at sentence-terminal punctuation (. ! ?)."""
    chunker = SentenceChunker(min_chars=20)

    emitted: list[str] = []
    for token in [
        "The system is fully operational. ",
        "It handles callbacks without latency. ",
        "Are there any pending alerts? ",
        "No issues!",
    ]:
        emitted.extend(chunker.feed(token))

    final = chunker.flush()
    if final:
        emitted.append(final)

    assert emitted == [
        "The system is fully operational.",
        "It handles callbacks without latency.",
        "Are there any pending alerts?",
        "No issues!",
    ]


def test_comma_does_not_split_when_clause_under_min_chars() -> None:
    """Verify a comma never ejects a short clause as its own TTS boundary."""
    chunker = SentenceChunker(min_chars=20)

    emitted: list[str] = []
    for token in [
        "We are live now. ",
        "A short phrase, continues onward. ",
    ]:
        emitted.extend(chunker.feed(token))

    # The comma clause ("A short phrase,") is under min_chars and must not
    # surface as its own TTS chunk; it is emitted with the full sentence.
    assert emitted == [
        "We are live now.",
        "A short phrase, continues onward.",
    ]


def test_comma_splits_long_first_clause_when_terminal_follows() -> None:
    """Verify the first-chunk fast path keeps comma as a boundary for a long
    clause while a sentence-terminal follows; standard mode folds the comma."""
    chunker = SentenceChunker(min_chars=20)

    emitted = chunker.feed("The quick brown fox jumps over the lazy dog, and keeps running. ")
    assert emitted == ["The quick brown fox jumps over the lazy dog,"]
    assert chunker.flush() == "and keeps running."

    # After the first chunk, standard mode emits the comma clause folded into
    # the full sentence delivered at the terminal punctuation.
    chunker = SentenceChunker(min_chars=20)
    assert chunker.feed("Setup complete. ") == ["Setup complete."]
    assert chunker.feed("The quick brown fox jumps over the lazy dog, and keeps running. ") == [
        "The quick brown fox jumps over the lazy dog, and keeps running."
    ]


def test_early_first_chunk_fast_path() -> None:
    """Verify the fast path emits a short first clause and respects disabling it."""
    # Fast path enabled (default): a short first clause still emits promptly.
    fast = SentenceChunker(min_chars=20)
    assert fast.feed("Hello there! ") == ["Hello there!"]

    # Fast path disabled: the same short clause is held until flush.
    slow = SentenceChunker(min_chars=20, early_first_chunk=False)
    assert slow.feed("Hello there! ") == []
    assert slow.flush() == "Hello there!"


def test_flush_returns_remainder_and_resets_state() -> None:
    """Verify flush returns unemitted text and resets the chunker for the next stream."""
    chunker = SentenceChunker(min_chars=20)

    assert chunker.feed("Turn complete. The next statement is still running") == ["Turn complete."]
    assert chunker.flush() == "The next statement is still running"

    # Fully reset: an empty flush is a no-op and the next clause emits promptly.
    assert chunker.flush() is None
    assert chunker.feed("Another turn. ") == ["Another turn."]


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


def test_build_contents_empty_history() -> None:
    """Verify a single user prompt when no conversation history exists."""
    contents = build_contents(None, "Hello there", max_turns=8)
    assert contents == [{"role": "user", "parts": [{"text": "Hello there"}]}]


def test_build_contents_preserves_alternation() -> None:
    """Verify user/model alternation is preserved and current prompt closes the list."""
    history = [
        ("user", "What is the capital of France?"),
        ("model", "Paris."),
        ("user", "Is it on the Seine?"),
        ("model", "Yes, it is."),
    ]
    contents = build_contents(history, "What is the Eiffel Tower?", max_turns=8)

    roles = [c["role"] for c in contents]
    assert roles == ["user", "model", "user", "model", "user"]
    assert contents[-1]["parts"][0]["text"] == "What is the Eiffel Tower?"
    assert contents[0]["parts"][0]["text"] == "What is the capital of France?"


def test_build_contents_trims_to_max_turns() -> None:
    """Verify history trims to the trailing `max_turns` turn pairs."""
    pairs = [pair for i in range(3) for pair in (("user", f"turn {i}"), ("model", f"reply {i}"))]
    contents = build_contents(pairs, "Now what?", max_turns=1)

    # Only the last pair survives trimming (2 messages) + closing user prompt
    assert len(contents) == 3
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"][0]["text"] == "turn 2"
    assert contents[1]["role"] == "model"
    assert contents[1]["parts"][0]["text"] == "reply 2"
    assert contents[2]["parts"][0]["text"] == "Now what?"


def test_build_contents_repairs_broken_role_sequence() -> None:
    """Verify leading model message is dropped and adjacent same-role texts merge."""
    history = [
        ("model", "Orphan echo."),
        ("model", "Still model."),
        ("user", "First real user line."),
    ]
    contents = build_contents(history, "Go", max_turns=8)

    assert contents[0]["role"] == "user"
    assert contents[0]["parts"][0]["text"] == "First real user line."
    assert contents[1]["role"] == "user"
    assert contents[1]["parts"][0]["text"] == "Go"
    assert len(contents) == 2


@pytest.mark.asyncio
async def test_stream_sentence_chunks_sends_history_contents() -> None:
    """Verify history is threaded into the Gemini contents payload."""
    mock_chunks = [MockChunk(text="Sure, the Eiffel Tower is in Paris.")]
    mock_client = MagicMock()
    mock_client.aio.models.generate_content_stream = AsyncMock(
        return_value=async_generator(mock_chunks)
    )

    llm = GeminiLLM(api_key="mock_key", client=mock_client)
    history = [("user", "Where is the Eiffel Tower?"), ("model", "It is in Paris.")]
    clauses: list[str] = []
    async for clause in llm.stream_sentence_chunks("Tell me more.", history=history, min_chars=5):
        clauses.append(clause)

    kwargs = mock_client.aio.models.generate_content_stream.call_args.kwargs
    sent_contents = kwargs["contents"]
    sent_roles = [c["role"] for c in sent_contents]
    assert sent_roles == ["user", "model", "user"]
    assert sent_contents[-1]["parts"][0]["text"] == "Tell me more."
    assert len(clauses) == 1


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


def test_default_voice_system_prompt_is_non_empty() -> None:
    """Verify the voice system prompt is defined and encodes the T1.5 constraints."""
    prompt = DEFAULT_VOICE_SYSTEM_PROMPT
    assert isinstance(prompt, str)
    assert prompt.strip()
    assert "1-3 complete" in prompt
    assert "Never omit words" in prompt
    assert "Never use fillers" in prompt
    assert "never trail off" in prompt
    assert "markdown" in prompt
    assert "like a sharp colleague" in prompt
    assert "say so plainly" in prompt


def test_build_contents_alternation_invariant() -> None:
    """Verify model-closed histories yield strictly alternating user/model roles
    that open and close on a user message (the Gemini contents contract)."""
    well_formed: list[list[tuple[str, str]] | None] = [
        None,
        [("user", "Where is the Eiffel Tower?"), ("model", "It is in Paris.")],
        [
            ("user", "What is the capital of France?"),
            ("model", "Paris."),
            ("user", "Is it on the Seine?"),
            ("model", "Yes, it is."),
        ],
    ]
    for history in well_formed:
        contents = build_contents(history, "Tell me more.", max_turns=8)
        roles = [c["role"] for c in contents]
        assert roles[0] == "user"
        assert roles[-1] == "user"
        assert all(prev != cur for prev, cur in pairwise(roles))


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
