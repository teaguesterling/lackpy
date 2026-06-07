"""Sync↔async bridge for tool callables.

lackpy programs execute **synchronously** in :class:`RestrictedRunner` (the
"lacking" language has no ``await``). Some tool callables — notably MCP-backed
ones — are async. The bridge lets a sync tool callable, running on a worker
thread, invoke a coroutine on the service's event loop and block for its result.

Correctness depends on two things the service arranges:

1. The sync runner runs **off** the loop thread (``loop.run_in_executor``), so the
   loop is free to service the marshaled coroutine. Marshaling onto the same loop
   the runner blocks would deadlock.
2. Executions are **single-flight** (an ``asyncio.Lock`` in the service), so only
   one execution owns the bridge's ``loop`` at a time.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any, Awaitable


class AsyncBridge:
    """Marshals coroutines from a worker thread onto a target event loop.

    ``loop`` is set by the service for the duration of one execution and cleared
    afterwards (under the execution lock, so it is never ambiguous).
    """

    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None

    def call_sync(self, coro: Awaitable[Any], timeout: float | None = None) -> Any:
        """Run ``coro`` on the bridge's loop and block until it returns.

        Intended to be called from a worker thread (never the loop thread).

        Raises:
            RuntimeError: If no loop is active (called outside an execution).
            TimeoutError: If the coroutine does not complete within ``timeout``.
        """
        loop = self.loop
        if loop is None:
            raise RuntimeError(
                "AsyncBridge has no active loop; an async tool was called outside "
                "a bridged execution."
            )
        fut = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            return fut.result(timeout)
        except FuturesTimeout:
            fut.cancel()
            raise TimeoutError(f"async tool call timed out after {timeout}s")


def mark_async(fn: Any) -> Any:
    """Flag a callable as loop-bound so the service uses the threaded exec path."""
    fn._lackpy_async = True
    return fn


def is_async_callable(fn: Any) -> bool:
    """Whether ``fn`` was flagged via :func:`mark_async`."""
    return bool(getattr(fn, "_lackpy_async", False))
