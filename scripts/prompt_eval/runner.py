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
    import concurrent.futures
    import threading

    start = time.time()
    chunks: list[str] = []
    eval_count = 0
    prompt_eval_count = 0
    error: str | None = None

    # Run the streaming chat in a thread so we can enforce a hard
    # timeout on the initial connection + model load, not just
    # between chunks. Without this, client.chat() blocks indefinitely
    # while ollama loads a large model.
    stream_iter = None
    cancel = threading.Event()

    def _start_stream():
        return client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            options={"temperature": temperature},
            stream=True,
            keep_alive=keep_alive,
        )

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_start_stream)
            try:
                stream_iter = future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                error = f"timeout: model load/prompt eval exceeded {timeout}s"
                cancel.set()

        if stream_iter is not None and not cancel.is_set():
            for chunk in stream_iter:
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


_CONNECTION_ERRORS = (
    "ConnectError",
    "ConnectionRefusedError",
    "RemoteProtocolError",
    "ConnectionError",
    "ServerDisconnectedError",
)


def is_connection_error(error: str | None) -> bool:
    """Check if an error string indicates ollama is down, not a model issue."""
    if not error:
        return False
    return any(e in error for e in _CONNECTION_ERRORS)


def check_ollama_health(client, timeout: float = 5.0) -> str | None:
    """Ping ollama to verify it's responsive.

    Returns None if healthy, or an error string if unreachable.
    """
    try:
        client.list()
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def wait_for_ollama(client, max_wait: int = 30, poll_interval: int = 5) -> str | None:
    """Wait for ollama to become responsive after a crash.

    Returns None once healthy, or the last error string if it doesn't
    recover within max_wait seconds.
    """
    import time as _time

    deadline = _time.time() + max_wait
    last_error = None
    while _time.time() < deadline:
        last_error = check_ollama_health(client)
        if last_error is None:
            return None
        _time.sleep(poll_interval)
    return last_error


def make_ollama_client(host: str = "http://localhost:11435"):
    """Construct an ollama.Client. Lazy-imports ollama so tests can mock.

    Returns an ollama.Client instance targeting `host`. Raises
    ImportError at call time if the ollama package is not installed —
    the harness only calls this when an actual sweep is about to run.
    """
    import ollama
    return ollama.Client(host=host)
