"""Tests for trace data structures and callable tracing."""

import pytest

from lackpy.run.trace import Trace, TraceEntry, make_traced


def test_trace_entry_creation():
    entry = TraceEntry(
        step=0, tool="read", args={"path": "test.py"},
        result="file contents", duration_ms=1.5,
        success=True, error=None,
    )
    assert entry.step == 0
    assert entry.tool == "read"
    assert entry.args == {"path": "test.py"}
    assert entry.result == "file contents"
    assert entry.duration_ms == 1.5
    assert entry.success is True
    assert entry.error is None


def test_trace_creation():
    trace = Trace()
    assert trace.entries == []
    assert trace.files_read == []
    assert trace.files_modified == []


def test_make_traced_records_call():
    def read(path: str) -> str:
        return f"contents of {path}"

    trace = Trace()
    traced_read = make_traced("read", read, trace)

    result = traced_read("test.py")

    assert result == "contents of test.py"
    assert len(trace.entries) == 1
    entry = trace.entries[0]
    assert entry.step == 0
    assert entry.tool == "read"
    assert entry.args == {"path": "test.py"}
    assert entry.success is True
    assert entry.error is None
    assert entry.duration_ms >= 0


def test_make_traced_records_error():
    def bad_fn(path: str) -> str:
        raise FileNotFoundError("no such file")

    trace = Trace()
    traced_fn = make_traced("bad_fn", bad_fn, trace)

    with pytest.raises(FileNotFoundError, match="no such file"):
        traced_fn("missing.py")

    assert len(trace.entries) == 1
    entry = trace.entries[0]
    assert entry.step == 0
    assert entry.tool == "bad_fn"
    assert entry.args == {"path": "missing.py"}
    assert entry.success is False
    assert entry.error == "no such file"
    assert entry.result is None
