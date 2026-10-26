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

    def to_data(self) -> dict[str, Any]:
        """The entry as a plain dict, for the wire."""
        return {"step": self.step, "tool": self.tool, "args": self.args,
                "result": self.result, "duration_ms": self.duration_ms,
                "success": self.success, "error": self.error}

    @classmethod
    def from_data(cls, d: dict[str, Any]) -> "TraceEntry":
        """``TypeError`` when ``tool`` is not a str: the parent reads entries from a
        worker, and every reader treats a TypeError as a bad message."""
        if type(d["tool"]) is not str:
            raise TypeError(f"a trace entry's tool must be str, not {type(d['tool']).__name__}")
        return cls(step=d["step"], tool=d["tool"], args=d["args"], result=d["result"],
                   duration_ms=d["duration_ms"], success=d["success"], error=d["error"])


@dataclass
class Trace:
    """Execution trace accumulating tool call records and file side-effects.

    Attributes:
        entries: Ordered list of TraceEntry records, one per tool call.
        files_read: Paths of files read during execution (populated by tools).
        files_modified: Paths of files written or modified during execution.
        listener: Called with each entry right after ``record_call`` appends it,
            so a worker can stream entries as they happen. Not part of equality.
    """

    entries: list[TraceEntry] = field(default_factory=list)
    files_read: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    listener: Callable[[TraceEntry], None] | None = field(default=None, repr=False, compare=False)


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
        _notify(trace)
        if on_done:
            on_done(False)
        raise
    trace.entries.append(TraceEntry(
        step=step, tool=tool, args=args,
        result=_summarize(result) if summarize else None,
        duration_ms=(time.perf_counter() - start) * 1000,
        success=True, error=None,
    ))
    _notify(trace)
    if on_done:
        on_done(True)
    return result


#: The suffix of a start entry's tool name (``"Path.write_text:start"``).
START_SUFFIX = ":start"


def announce_start(trace: Trace, tool: str, args: list[Any]) -> None:
    """Tell the listener that ``tool`` is about to start (Plan 3 Task 6, Review Focus 5).

    The start entry goes to the listener only, never into ``trace.entries``:
    an in-process run is never killed, so its trace and step numbers stay as
    they were. In a worker the entry streams to the parent before the write
    begins, so a cell killed in the middle of a write is known to have started
    it. The entry's ``step`` is the step its done entry will have.
    """
    if trace.listener is None:
        return
    entry = TraceEntry(step=len(trace.entries), tool=tool + START_SUFFIX, args=args,
                       result=None, duration_ms=0.0, success=True, error=None)
    try:
        trace.listener(entry)
    except Exception:
        pass


def _notify(trace: Trace) -> None:
    """Call the listener with the newest entry. Its failure never changes the run:
    the entry stays, ``on_done`` still runs, and the program's own exception propagates."""
    if trace.listener is None:
        return
    try:
        trace.listener(trace.entries[-1])
    except Exception:
        pass


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
    """A trace entry's result: a str over ``max_len`` chars is cut to ``max_len``.

    Bytes are recorded as ``repr`` text, cut to the first ``max_len`` bytes
    (review I2): raw bytes are not JSON, and a ``read_bytes`` result must not
    put the whole file in the trace.
    """
    if isinstance(value, str) and len(value) > max_len:
        return value[:max_len] + f"... ({len(value)} chars)"
    if isinstance(value, (bytes, bytearray)):
        text = repr(bytes(value[:max_len]))
        return text + f"... ({len(value)} bytes)" if len(value) > max_len else text
    return value
