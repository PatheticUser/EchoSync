"""Upstream LLM reasoning gateway and streaming sentence boundary chunker."""

import asyncio
import re
from collections.abc import AsyncIterator
from typing import Any

from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import SecretStr
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

DEFAULT_VOICE_SYSTEM_PROMPT = (
    "You are EchoSync, an ultra-low-latency conversational voice AI. "
    "Respond immediately, concisely, and naturally in 1-2 spoken sentences. "
    "Strictly never output Markdown syntax, asterisks, headers, code blocks, or bullet lists."
)


def clean_speech_text(text: str) -> str:
    """Strip markdown artifacts, list markers, headers, and excessive spacing."""
    # Remove full markdown header lines (e.g. # Overview)
    cleaned = re.sub(r"^\s*#+.*$", "", text, flags=re.MULTILINE)
    # Remove any remaining standalone hash symbols
    cleaned = re.sub(r"#+", "", cleaned)
    # Remove list markers (- item, * item, 1. item)
    cleaned = re.sub(r"^\s*(?:[-*+]|\d+\.)\s+", "", cleaned, flags=re.MULTILINE)
    # Remove markdown formatting characters (*, _, ~, `)
    cleaned = re.sub(r"[*_~`]", "", cleaned)
    # Collapse multiple whitespaces and newlines
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


class SentenceChunker:
    """Accumulator that parses streaming text tokens into vocalizable sentence clauses."""

    def __init__(self, min_chars: int = 20, early_first_chunk: bool = False) -> None:
        self.min_chars = min_chars
        self.early_first_chunk = early_first_chunk
        self._standard_pattern = re.compile(r"([.!?;:])(?:\s+|\n+)")
        self._first_pattern = (
            re.compile(r"([.!?;:,])(?:\s+|\n+)") if early_first_chunk else self._standard_pattern
        )
        self._buffer = ""
        self._is_first_chunk = early_first_chunk

    def feed(self, token: str) -> list[str]:
        """Add a token to the accumulator and return any completed sentence clauses."""
        self._buffer += token
        emitted: list[str] = []

        while True:
            pattern = self._first_pattern if self._is_first_chunk else self._standard_pattern
            min_len = 12 if self._is_first_chunk else self.min_chars
            match = pattern.search(self._buffer)
            if not match:
                break

            candidate = self._buffer[: match.end()].strip()
            if len(candidate) >= min_len:
                cleaned = clean_speech_text(candidate)
                if cleaned:
                    emitted.append(cleaned)
                    self._is_first_chunk = False
                self._buffer = self._buffer[match.end() :]
            else:
                next_match = pattern.search(self._buffer, pos=match.start() + 1)
                if next_match:
                    candidate_extended = self._buffer[: next_match.end()].strip()
                    if len(candidate_extended) >= min_len:
                        cleaned = clean_speech_text(candidate_extended)
                        if cleaned:
                            emitted.append(cleaned)
                            self._is_first_chunk = False
                        self._buffer = self._buffer[next_match.end() :]
                    else:
                        break
                else:
                    break

        return emitted

    def flush(self) -> str | None:
        """Flush any remaining buffered text upon stream completion."""
        self._is_first_chunk = True
        if not self._buffer.strip():
            self._buffer = ""
            return None
        cleaned = clean_speech_text(self._buffer)
        self._buffer = ""
        return cleaned if cleaned else None


def is_retryable_llm_error(exc: BaseException) -> bool:
    """Determine if an exception warrants an exponential backoff retry."""
    if isinstance(exc, APIError):
        status_code = getattr(exc, "code", None)
        if status_code in (429, 500, 502, 503, 504):
            return True
        msg = str(exc).lower()
        if "429" in msg or "resource_exhausted" in msg or "rate limit" in msg:
            return True
    return False


class GeminiLLM:
    """Gateway for Google GenAI streaming generation with resilient retry logic."""

    def __init__(
        self,
        api_key: str | SecretStr,
        model_name: str = "gemini-3.6-flash",
        system_prompt: str = DEFAULT_VOICE_SYSTEM_PROMPT,
        client: Any | None = None,
    ) -> None:
        key_str = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
        self.api_key = key_str
        self.model_name = model_name
        self.system_prompt = system_prompt
        self._client = client or genai.Client(api_key=self.api_key)

    async def _call_stream_with_retry(self, prompt: str) -> Any:
        """Execute generate_content_stream with Tenacity AsyncRetrying backoff."""
        retrying = AsyncRetrying(
            retry=retry_if_exception(is_retryable_llm_error),
            wait=wait_random_exponential(min=0.5, max=6.0, multiplier=2.0),
            stop=stop_after_attempt(3),
            reraise=True,
        )

        async for attempt in retrying:
            with attempt:
                config = types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    temperature=0.3,
                    max_output_tokens=150,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                )
                async with asyncio.timeout(5.0):
                    return await self._client.aio.models.generate_content_stream(
                        model=self.model_name,
                        contents=prompt,
                        config=config,
                    )
        raise RuntimeError("Failed to obtain streaming generator from Gemini API")

    async def stream_tokens(self, prompt: str) -> AsyncIterator[str]:
        """Stream individual text token chunks from Gemini."""
        stream = await self._call_stream_with_retry(prompt)
        async for chunk in stream:
            text = getattr(chunk, "text", None)
            if text:
                yield text

    async def stream_sentence_chunks(
        self,
        prompt: str,
        min_chars: int = 20,
    ) -> AsyncIterator[str]:
        """Stream grammatically complete sentence clauses ready for TTS synthesis."""
        chunker = SentenceChunker(min_chars=min_chars)
        async for token in self.stream_tokens(prompt):
            clauses = chunker.feed(token)
            for clause in clauses:
                yield clause

        final_clause = chunker.flush()
        if final_clause:
            yield final_clause
