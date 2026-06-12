"""record_delegation — best-effort write of a delegate() result into blq.

blq.storage.BlqStorage.open is monkeypatched with a fake so these run without a
real .bird/ DuckDB; the no-bird and error paths assert the best-effort guarantees.
"""
from __future__ import annotations

import pytest

from lackpy.observ import record_delegation


def _result(success: bool = True) -> dict:
    return {
        "success": success,
        "program": "x = read_file('a')\nx",
        "grade": {"w": 1, "d": 1},
        "generation_tier": "woollama",
        "generation_time_ms": 12.0,
        "execution_time_ms": 3.0,
        "total_time_ms": 15.0,
        "trace": [{"step": 1, "tool": "read_file", "args": {}, "result": "...",
                   "duration_ms": 1, "success": True, "error": None}],
        "files_read": ["a"],
        "files_modified": [],
        "output": "...",
        "stdout": "",
        "error": None if success else "boom",
        "correction_strategy": None,
        "correction_attempts": 0,
    }


class _FakeStorage:
    def __init__(self):
        self.calls: list = []

    def write_run(self, run_meta, events=None, output=None):
        self.calls.append((run_meta, events, output))
        return f"run-{len(self.calls)}"

    def close(self):
        pass


@pytest.fixture
def fake_blq(tmp_path, monkeypatch):
    blq_storage = pytest.importorskip("blq.storage")  # optional dep; skip if absent
    (tmp_path / ".bird").mkdir()
    fake = _FakeStorage()
    monkeypatch.setattr(blq_storage.BlqStorage, "open", staticmethod(lambda d: fake))
    return fake


def test_noop_without_bird(tmp_path):
    # No .bird/ in the workspace → nothing recorded, no error.
    assert record_delegation(tmp_path, "do x", _result()) is None


def test_writes_run_for_success(tmp_path, fake_blq):
    rid = record_delegation(tmp_path, "count rows", _result())
    assert rid == "run-1"
    (run_meta, events, output), = fake_blq.calls
    assert run_meta["source_name"] == "delegate"
    assert run_meta["source_type"] == "import"
    assert run_meta["tag"] == "delegate"
    assert run_meta["exit_code"] == 0
    assert run_meta["command"] == 'lackpy delegate "count rows"'
    lackpy_meta = run_meta["environment"]["lackpy"]
    assert lackpy_meta["generation_tier"] == "woollama"
    assert lackpy_meta["tools"] == ["read_file"]
    assert events is None                       # no events on success
    assert output == b"x = read_file('a')\nx"   # the generated program


def test_failure_records_error_event(tmp_path, fake_blq):
    record_delegation(tmp_path, "x", _result(success=False))
    run_meta, events, _ = fake_blq.calls[0]
    assert run_meta["exit_code"] == 1
    assert len(events) == 1
    assert events[0]["severity"] == "error"
    assert events[0]["message"] == "boom"
    assert events[0]["tool_name"] == "lackpy"


def test_intent_is_truncated(tmp_path, fake_blq):
    record_delegation(tmp_path, "z" * 500, _result())
    cmd = fake_blq.calls[0][0]["command"]
    assert cmd == 'lackpy delegate "' + "z" * 200 + '"'


def test_swallows_storage_errors(tmp_path, monkeypatch):
    # A concurrent writer / lock error must never propagate out of delegate.
    blq_storage = pytest.importorskip("blq.storage")  # optional dep; skip if absent
    (tmp_path / ".bird").mkdir()

    def boom(_):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(blq_storage.BlqStorage, "open", staticmethod(boom))
    assert record_delegation(tmp_path, "x", _result()) is None
