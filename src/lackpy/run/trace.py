"""Trace data structures, and the recording of one call (``record_call``)."""

from __future__ import annotations

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


def record_call(trace: Trace, tool: str,
                args: dict[str, Any] | list[Any] | Callable[[], list[Any]],
                call: Callable[[], Any],
                on_done: Callable[[bool], None] | None = None,
                summarize: bool = True) -> Any:
    """Run ``call()``, append one TraceEntry for it, and return its result.

    A raised ``Exception`` or ``LackpyRefusal`` is recorded as a failed entry
    whose error is ``str(e)`` (for a refusal, that text carries the hint), then
    re-raised. ``summarize=False`` records no result (for a lazy iterator).
    ``args`` may be a function that builds them; it runs inside the recorded
    region, and when it fails the entry's args are ``[]``.
    """
    step = len(trace.entries)
    start = time.perf_counter()
    try:
        if callable(args):
            built = args
            args = []
            args = built()
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
