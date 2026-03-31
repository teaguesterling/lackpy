"""Trace data structures and callable wrapping for execution recording."""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class TraceEntry:
    """A single tool call record in an execution trace.

    Attributes:
        step: Zero-based index of this call within the program execution.
        tool: Name of the tool that was called.
        args: Keyword argument mapping captured at call time.
        result: Summarized return value, or None if the call failed.
        duration_ms: Wall-clock time for the call in milliseconds.
        success: Whether the call completed without raising.
        error: Error message string if the call raised, otherwise None.
    """

    step: int
    tool: str
    args: dict[str, Any]
    result: Any
    duration_ms: float
    success: bool
    error: str | None


@dataclass
class Trace:
    """Execution trace accumulating tool call records and file side-effects.

    Attributes:
        entries: Ordered list of TraceEntry records, one per tool call.
        files_read: Paths of files read during execution (populated by tools).
        files_modified: Paths of files written or modified during execution.
    """

    entries: list[TraceEntry] = field(default_factory=list)
    files_read: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)


def make_traced(name: str, fn: Callable, trace: Trace) -> Callable:
    """Wrap a callable to record each invocation in a Trace.

    The wrapper captures positional and keyword arguments, measures wall-clock
    duration, appends a TraceEntry to ``trace``, and re-raises any exception
    after recording it.

    Args:
        name: Tool name to record in each TraceEntry.
        fn: The callable to wrap.
        trace: The Trace instance to append entries to.

    Returns:
        A wrapper callable with the same signature as ``fn``.
    """
    sig = inspect.signature(fn)
    param_names = list(sig.parameters.keys())

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        call_args: dict[str, Any] = {}
        for i, val in enumerate(args):
            key = param_names[i] if i < len(param_names) else f"arg{i}"
            call_args[key] = val
        call_args.update(kwargs)

        step = len(trace.entries)
        start = time.perf_counter()

        try:
            result = fn(*args, **kwargs)
            duration = (time.perf_counter() - start) * 1000
            trace.entries.append(TraceEntry(
                step=step, tool=name, args=call_args,
                result=_summarize(result), duration_ms=duration,
                success=True, error=None,
            ))
            return result
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            trace.entries.append(TraceEntry(
                step=step, tool=name, args=call_args,
                result=None, duration_ms=duration,
                success=False, error=str(e),
            ))
            raise

    return wrapper


def _summarize(value: Any, max_len: int = 500) -> Any:
    if isinstance(value, str) and len(value) > max_len:
        return value[:max_len] + f"... ({len(value)} chars)"
    return value
