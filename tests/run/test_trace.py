"""Tests for trace data structures and callable tracing."""

import pytest

from lackpy.run.trace import Trace, TraceEntry, record_call


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


def test_record_call_records_a_call():
    # Retired kit model: make_traced wrapped kit tools; record_call is what the Path stand-in uses.
    trace = Trace()
    result = record_call(trace, "Path.read_text", ["test.py"], lambda: "contents of test.py")
    assert result == "contents of test.py"
    (entry,) = trace.entries
    assert (entry.step, entry.tool, entry.args, entry.success, entry.error) == (
        0, "Path.read_text", ["test.py"], True, None)
    assert entry.duration_ms >= 0


def test_record_call_records_an_error_then_reraises():
    def bad():
        raise FileNotFoundError("no such file")

    trace = Trace()
    with pytest.raises(FileNotFoundError, match="no such file"):
        record_call(trace, "Path.read_text", ["missing.py"], bad)
    (entry,) = trace.entries
    assert (entry.success, entry.error, entry.result) == (False, "no such file", None)


def test_record_call_error_text_falls_back_when_str_raises():
    # Parked item 3 (Task 2 re-review, trace.py:98): str(KeyError(n)) raises for an
    # int over 4,300 digits. The entry gets a fixed text, and the original error propagates.
    big = 10 ** 5000

    def bad():
        raise KeyError(big)

    trace = Trace()
    with pytest.raises(KeyError):
        record_call(trace, "Path.read_text", ["x"], bad)
    (entry,) = trace.entries
    assert entry.success is False
    assert entry.error == "KeyError (its message could not be converted to text)"


def test_make_traced_is_gone():
    # Kit tools were the only callers; the kit is gone (stepy100 Task 7).
    from lackpy.run import trace as trace_module
    assert not hasattr(trace_module, "make_traced")


# --- stepy100 Plan 2, Task 5: Path I/O and refusals in the trace; data-only variables ---

from lackpy.run.paths import PathPolicy
from lackpy.run.refusal import LackpyRefusal
from lackpy.run.runner import RestrictedRunner


@pytest.fixture
def ws(tmp_path, monkeypatch):
    w = tmp_path / "ws"
    w.mkdir()
    (w / "a.txt").write_text("hello")
    monkeypatch.chdir(tmp_path)
    return w


def _run(program, ws):
    return RestrictedRunner().run(program, path_policy=PathPolicy(ws, []))


def test_path_read_and_write_are_traced(ws):
    program = ("from pathlib import Path\n"
               "t = Path('a.txt').read_text()\n"
               "Path('b.txt').write_text(t + '!')\n")
    result = _run(program, ws)
    assert result.success, result.error
    entries = result.trace.entries
    assert [(e.step, e.tool, e.args, e.success) for e in entries] == [
        (0, "Path.read_text", ["a.txt"], True),
        (1, "Path.write_text", ["b.txt"], True),
    ]
    assert entries[0].result == "hello"
    assert entries[0].error is None


def test_path_trace_args_are_the_path_as_built(ws):
    # The workspace-joined path must not reach the trace either.
    result = _run("from pathlib import Path\nPath('.').joinpath('a.txt').exists()", ws)
    assert result.success, result.error
    assert result.trace.entries[0].tool == "Path.exists"
    assert result.trace.entries[0].args == ["a.txt"]


def test_refused_path_call_appears_in_the_trace(ws):
    program = "from pathlib import Path\nPath('../outside.txt').read_text()"
    result = _run(program, ws)
    assert not result.success
    (entry,) = result.trace.entries
    assert entry.tool == "Path.read_text"
    assert entry.args == ["../outside.txt"]
    assert entry.success is False
    assert entry.result is None
    assert "not allowed in lackpy" in entry.error


def test_refused_path_call_is_traced_even_when_the_program_catches_exception(ws):
    program = ("from pathlib import Path\n"
               "try:\n"
               "    Path('../outside.txt').read_text()\n"
               "except Exception:\n"
               "    pass\n")
    result = _run(program, ws)
    assert not result.success
    assert [e.success for e in result.trace.entries] == [False]


def test_record_call_records_a_refusal_then_reraises():
    def tool():
        raise LackpyRefusal("reading secrets", "use Path")

    trace = Trace()
    with pytest.raises(LackpyRefusal):
        record_call(trace, "Path.read_text", ["x"], tool)
    (entry,) = trace.entries
    assert entry.success is False
    assert entry.result is None
    assert "use Path" in entry.error


def test_variables_hold_data_only(ws):
    program = ("import json\n"
               "from pathlib import Path\n"
               "def f():\n"
               "    return 1\n"
               "g = len\n"
               "lam = lambda: 2\n"
               "j = json\n"
               "P = Path\n"
               "p = Path('a.txt')\n"
               "t = type(1)\n"
               "T = type\n"
               "x = 1\n"
               "s = 'text'\n")
    result = _run(program, ws)
    assert result.success, result.error
    assert result.variables == {"x": 1, "s": "text"}


# --- Task 5 fix round: M1 (recursive data allowlist) and M4 (pattern args traced) ---

@pytest.mark.parametrize("program", [
    "v = (i for i in range(3))",           # generator
    "v = {'a': 1}.items()",                # dict view
    "try:\n    int('x')\nexcept ValueError as e:\n    v = e",  # exception instance
    "import json\nv = [json]",            # proxy nested in a list
    "from pathlib import Path\nv = [Path('a.txt')]",
    "v = {'f': len}",
    "v = (1, [type(1)])",
    "v = range(3)",
    "v = 1j",                              # complex is not in the allowlist
    "v = frozenset({1})",
])
def test_variables_drop_non_data_at_any_depth(ws, program):
    result = _run(program + "\nkeep = 1", ws)
    assert result.success, result.error
    assert result.variables == {"keep": 1}


def test_variables_keep_nested_data(ws):
    program = "v = {'a': [1, (2.5, b'x', None, True)], 's': {1, 'b'}, (1, 2): 'k'}"
    result = _run(program, ws)
    assert result.success, result.error
    assert result.variables == {"v": {"a": [1, (2.5, b"x", None, True)], "s": {1, "b"}, (1, 2): "k"}}


def test_variables_drop_a_self_referencing_list(ws):
    result = _run("v = []\nv.append(v)\nkeep = 1", ws)
    assert result.success, result.error
    assert result.variables == {"keep": 1}


def test_glob_pattern_refusal_is_traced(ws):
    # Review M4 (measured): str() of a type view refused before the trace entry existed.
    result = _run("from pathlib import Path\nlist(Path('.').glob(type(1)))", ws)
    assert not result.success
    assert [(e.tool, e.success) for e in result.trace.entries] == [("Path.glob", False)]
    assert "not allowed in lackpy" in result.trace.entries[0].error


def test_glob_pattern_that_cannot_be_text_is_traced(ws):
    # Review M4 (measured): str(10**5000) raised ValueError with no trace entry.
    result = _run("from pathlib import Path\nlist(Path('.').glob(10**5000))", ws)
    assert not result.success
    assert [(e.tool, e.success) for e in result.trace.entries] == [("Path.glob", False)]


def test_variables_walk_of_shared_structure_is_linear(ws):
    # Re-review: x = [x, x] doubles the tree per level; the walk runs on the host
    # after exec, outside the program timeout, so it must visit each object once.
    import time
    start = time.perf_counter()
    result = _run("x = 0\nfor i in range(40):\n    x = [x, x]\n", ws)
    elapsed = time.perf_counter() - start
    assert result.success, result.error
    assert "x" in result.variables
    assert elapsed < 1.0, elapsed
