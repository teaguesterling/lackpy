"""Trace data structures and callable wrapping for execution recording."""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .refusal import LackpyRefusal


@dataclass
class TraceEntry:
    """A single tool call record in an execution trace.

    Attributes:
        step: Zero-based index of this call within the program execution.
        tool: Name of the tool that was called (``"Path.read_text"`` for a
            ``Path`` stand-in member).
        args: Keyword argument mapping captured at call time; for a ``Path``
            member, a list holding the path as the program built it.
        result: Summarized return value, or None if the call failed.
        duration_ms: Wall-clock time for the call in milliseconds.
        success: Whether the call completed without raising.
        error: Error message string if the call raised, otherwise None.
    """

    step: int
    tool: str
    args: dict[str, Any] | list[Any]
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


def make_traced(name: str, fn: Callable, trace: Trace,
                kibitzer_session: Any = None) -> Callable:
    """Wrap a callable to record each invocation in a Trace.

    The wrapper captures positional and keyword arguments, measures wall-clock
    duration, appends a TraceEntry to ``trace``, and re-raises any exception
    or ``LackpyRefusal`` after recording it (design §8, gate 3).

    If a kibitzer_session is provided, each call is also reported to Kibitzer
    via after_call() for mode tracking and coaching.

    Args:
        name: Tool name to record in each TraceEntry.
        fn: The callable to wrap.
        trace: The Trace instance to append entries to.
        kibitzer_session: Optional KibitzerSession for per-call tracking.

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

        def report(success: bool) -> None:
            if kibitzer_session:
                kibitzer_session.after_call(name, call_args, success=success)

        return record_call(trace, name, call_args, lambda: fn(*args, **kwargs), report)

    return wrapper


def record_call(trace: Trace, tool: str, args: dict[str, Any] | list[Any],
                call: Callable[[], Any],
                on_done: Callable[[bool], None] | None = None,
                summarize: bool = True) -> Any:
    """Run ``call()``, append one TraceEntry for it, and return its result.

    A raised ``Exception`` or ``LackpyRefusal`` is recorded as a failed entry
    whose error is ``str(e)`` (for a refusal, that text carries the hint), then
    re-raised. ``summarize=False`` records no result (for a lazy iterator).
    """
    step = len(trace.entries)
    start = time.perf_counter()
    try:
        result = call()
    except (Exception, LackpyRefusal) as e:
        trace.entries.append(TraceEntry(
            step=step, tool=tool, args=args, result=None,
            duration_ms=(time.perf_counter() - start) * 1000,
            success=False, error=error_text(e),
        ))
        if on_done:
            on_done(False)
        raise
    trace.entries.append(TraceEntry(
        step=step, tool=tool, args=args,
        result=_summarize(result) if summarize else None,
        duration_ms=(time.perf_counter() - start) * 1000,
        success=True, error=None,
    ))
    if on_done:
        on_done(True)
    return result


def error_text(e: BaseException) -> str:
    """Return ``str(e)``, or a fixed text when ``str(e)`` itself raises.

    ``str(KeyError(n))`` raises ``ValueError`` for an int over 4,300 digits;
    the runner must still return a result (review I1), and a trace entry too.
    """
    try:
        return str(e)
    except (Exception, LackpyRefusal):
        return f"{type(e).__name__} (its message could not be converted to text)"


def _summarize(value: Any, max_len: int = 500) -> Any:
    if isinstance(value, str) and len(value) > max_len:
        return value[:max_len] + f"... ({len(value)} chars)"
    return value
