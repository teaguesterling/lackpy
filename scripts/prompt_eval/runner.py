"""Ollama streaming runner: one shot generation with timeout and token counts."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class GenerationRecord:
    """One generation attempt's raw data.

    Attributes:
        model: Ollama model id (e.g. 'qwen2.5-coder:1.5b').
        raw: Raw content string returned by the model.
        tokens_eval: eval_count from the last streaming chunk
            (model-produced tokens).
        tokens_prompt: prompt_eval_count (input tokens consumed).
        duration_ms: Wall-clock milliseconds for the generation call.
        error: Error string if the call failed or timed out; None on
            success. An empty stream without an error is NOT an error.
    """

    model: str
    raw: str
    tokens_eval: int
    tokens_prompt: int
    duration_ms: float
    error: str | None = None


def generate_once(
    client: Any,
    model: str,
    system_prompt: str,
    user_message: str,
    temperature: float,
    timeout: int,
    keep_alive: str = "30m",
) -> GenerationRecord:
    """Run a single streaming chat against Ollama with timeout.

    Exceptions are captured and returned as an error field on the
    record, not raised. Timeouts are measured against wall-clock time
    and bail the stream early. Token counts are taken from the last
    chunk that reported them.

    Args:
        client: An ollama.Client-compatible object whose .chat(...)
            method accepts model, messages, options, stream, keep_alive
            kwargs and returns an iterable of chunks.
        model: Ollama model id.
        system_prompt: System message content.
        user_message: User message content.
        temperature: Sampling temperature (0.0–1.0).
        timeout: Seconds of wall-clock before bailing the stream.
        keep_alive: Ollama keep_alive directive (how long to keep the
            model loaded after the call). Defaults to "30m".

    Returns:
        A GenerationRecord capturing the raw output, token counts,
        duration, and any error string.
    """
    start = time.time()
    chunks: list[str] = []
    eval_count = 0
    prompt_eval_count = 0
    error: str | None = None
    try:
        stream = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            options={"temperature": temperature},
            stream=True,
            keep_alive=keep_alive,
        )
        for chunk in stream:
            elapsed = time.time() - start
            if elapsed > timeout:
                error = f"timeout: exceeded {timeout}s"
                break
            token = chunk.message.content or ""
            chunks.append(token)
            ec = getattr(chunk, "eval_count", None)
            if ec:
                eval_count = ec
            pec = getattr(chunk, "prompt_eval_count", None)
            if pec:
                prompt_eval_count = pec
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    duration_ms = (time.time() - start) * 1000
    return GenerationRecord(
        model=model,
        raw="".join(chunks),
        tokens_eval=eval_count,
        tokens_prompt=prompt_eval_count,
        duration_ms=duration_ms,
        error=error,
    )


def make_ollama_client(host: str = "http://localhost:11435"):
    """Construct an ollama.Client. Lazy-imports ollama so tests can mock.

    Returns an ollama.Client instance targeting `host`. Raises
    ImportError at call time if the ollama package is not installed —
    the harness only calls this when an actual sweep is about to run.
    """
    import ollama
    return ollama.Client(host=host)
