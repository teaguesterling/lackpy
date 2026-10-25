"""ProcessRunner: a time limit enforced by killing a process (Plan 3, Task 4; spec D1-D5).

Every test that waits on a process has its own bound: each ``run`` has a
``timeout``, each thread join and subprocess wait has one, and the autouse
``_hang_guard`` dumps the stacks and exits if a test still hangs (pytest-timeout
is not installed), so a regression fails rather than hangs.
"""

from __future__ import annotations

import faulthandler
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from lackpy.run.paths import PathPolicy
from lackpy.run.process import STDOUT_CAP, ProcessRunner

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="the zygote needs Linux (pidfd, prctl)")

REGEX = 'import re\nre.search(r"(a+)+$", "a" * 26 + "b")'


@pytest.fixture(autouse=True)
def _hang_guard():
    faulthandler.dump_traceback_later(60, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture
def ws(tmp_path):
    w = tmp_path / "ws"
    w.mkdir()
    return w


@pytest.fixture
def runner():
    r = ProcessRunner()
    yield r
    r.close()


def _run(runner, program, ws, timeout=2.0, **kw):
    return runner.run(program, params=None, path_policy=PathPolicy(ws, []),
                      workspace=ws, timeout=timeout, **kw)


def _in_thread(fn):
    """Start ``fn`` in a thread; return a getter that joins it with a bound."""
    box: dict = {}
    t = threading.Thread(target=lambda: box.setdefault("value", fn()), daemon=True)
    t.start()

    def get(bound=10.0):
        t.join(bound)
        assert not t.is_alive(), "the run did not return within its bound"
        return box["value"]
    return get


def _wait_for(pred, bound=5.0):
    end = time.monotonic() + bound
    while time.monotonic() < end:
        value = pred()
        if value:
            return value
        time.sleep(0.01)
    raise AssertionError("condition not met within the bound")


def test_simple_run(runner, ws):
    result = _run(runner, "x = 1 + 1\nx", ws)
    assert result.success is True, result.error
    assert result.output == 2
    assert result.variables == {"x": 2}


def test_stdout_and_variables_cross(runner, ws):
    result = _run(runner, "print('hi')\ny = [1, (2, 3), b'z']\n", ws)
    assert result.success is True, result.error
    assert result.stdout == "hi\n"
    assert result.variables == {"y": [1, (2, 3), b"z"]}


def test_runs_in_the_workspace(runner, ws):
    (ws / "a.txt").write_text("hello")
    result = _run(runner, "from pathlib import Path\nPath('a.txt').read_text()", ws)
    assert result.success is True, result.error
    assert result.output == "hello"
    assert result.trace.files_read == ["a.txt"]


def test_regex_stops_at_limit(runner, ws):
    """Gate 1: a regex running in C is stopped at the limit (D1)."""
    start = time.monotonic()
    result = _run(runner, REGEX, ws)
    elapsed = time.monotonic() - start
    assert result.success is False
    assert result.error_phase == "timeout"
    assert result.error == "timeout: the run took longer than 2 s and was stopped"
    assert elapsed < 2 + 1, elapsed
    # The worker was killed, not left to run the regex out (D1).
    _wait_for(lambda: _proc_starttime(runner.last_pid) is None, bound=1.0)


def test_fifo_read_stops_at_limit(runner, ws):
    """Gate 2: a read blocked in the OS is stopped at the limit."""
    os.mkfifo(ws / "p")
    start = time.monotonic()
    result = _run(runner, "from pathlib import Path\nPath('p').read_text()", ws)
    assert result.error_phase == "timeout"
    assert time.monotonic() - start < 2 + 1


def test_memory_cap(ws):
    """Gate 3: the deque stops at the memory cap, and the error says so.

    timeout=10, not 2: filling 256 MiB took 1.1-2.4 s on longbottom (measured),
    so at 2 s the time limit would win the race and this would test the timeout.
    """
    r = ProcessRunner(memory_mb=256)
    try:
        result = _run(r, "import collections\nd = collections.deque(iter(int, 1))", ws, timeout=10)
    finally:
        r.close()
    assert result.success is False
    assert result.error == "memory: the run used more than 256 MiB and was stopped"
    assert result.error_phase is None


def test_trace_survives_kill(runner, ws):
    """Gate 5: the parent knows what the program wrote before it was stopped (D4)."""
    result = _run(runner, "from pathlib import Path\nPath('out.json').write_text('{}')\n"
                          "for i in iter(int, 1): pass", ws)
    assert result.error_phase == "timeout"
    assert "out.json" in result.trace.files_modified
    assert [e.tool for e in result.trace.entries] == ["Path.write_text"]


def test_crash_message(runner, ws):
    """Review Focus 2: a worker killed by a signal gives the crash text."""
    get = _in_thread(lambda: _run(runner, "for i in iter(int, 1): pass", ws, timeout=8))
    pid = _wait_for(lambda: runner.last_pid)
    time.sleep(0.3)  # let the worker reach the loop
    os.kill(pid, signal.SIGSEGV)
    result = get()
    assert result.success is False
    assert result.error_phase is None
    assert result.error == "the run process ended without a result (signal 11)"


def test_stdout_cap(runner, ws):
    """Review Focus 3: stdout past 10 MB is cut, and the run still finishes."""
    result = _run(runner, "for i in range(12):\n    print('x' * 1000000)\n'end'", ws, timeout=10)
    marker = "\n[lackpy: output cut at 10 MB]"
    assert result.success is True, result.error
    assert result.output == "end"
    assert result.stdout.endswith(marker)
    assert len(result.stdout) == STDOUT_CAP + len(marker)
    assert STDOUT_CAP == 10_000_000


def _proc_starttime(pid: int) -> str | None:
    """Field 22 of /proc/<pid>/stat, or None when the pid is gone or a zombie."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    fields = stat[stat.rindex(")") + 2:].split()
    if fields[0] == "Z":
        return None
    return fields[19]


def _descendants(root: int) -> set[int]:
    parents: dict[int, int] = {}
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            try:
                stat = Path(f"/proc/{entry}/stat").read_text()
            except (FileNotFoundError, ProcessLookupError):
                continue
            parents[int(entry)] = int(stat[stat.rindex(")") + 2:].split()[1])
    found: set[int] = set()
    frontier = {root}
    while frontier:
        children = {p for p, pp in parents.items() if pp in frontier and p not in found}
        found |= children
        frontier = children
    return found


def test_no_orphans(ws):
    """Review Focus 1: SIGKILL the service; within 2 s its zygote and worker are gone.

    The pids are taken from /proc before the kill: after it they are reparented,
    so a parent-chain walk would no longer find them. A zombie counts as gone.
    """
    script = textwrap.dedent(f"""
        import sys, threading, time
        from pathlib import Path
        from lackpy.run.paths import PathPolicy
        from lackpy.run.process import ProcessRunner
        ws = Path({str(ws)!r})
        r = ProcessRunner()
        threading.Thread(target=lambda: r.run("for i in iter(int, 1): pass", params=None,
            path_policy=PathPolicy(ws, []), workspace=ws, timeout=60), daemon=True).start()
        end = time.monotonic() + 10
        while r.last_pid is None and time.monotonic() < end:
            time.sleep(0.01)
        print(r._proc.pid, r.last_pid, flush=True)
        time.sleep(60)
    """)
    service = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    try:
        line = service.stdout.readline()
        zygote_pid, worker_pid = (int(x) for x in line.split())
        time.sleep(0.3)
        tree = _descendants(service.pid)
        assert {zygote_pid, worker_pid} <= tree, (tree, line)
        before = {p: _proc_starttime(p) for p in tree}
        os.kill(service.pid, signal.SIGKILL)
        service.wait(5)
        end = time.monotonic() + 2
        while time.monotonic() < end:
            alive = {p for p, st in before.items() if st is not None and _proc_starttime(p) == st}
            if not alive:
                break
            time.sleep(0.02)
        assert not alive, f"left behind after 2 s: {sorted(alive)}"
    finally:
        if service.poll() is None:
            service.kill()
            service.wait(5)


def test_zygote_restart(runner, ws):
    assert _run(runner, "1", ws).output == 1
    zygote = runner._proc
    zygote.kill()
    zygote.wait(5)
    result = _run(runner, "2", ws)
    assert result.success is True, result.error
    assert result.output == 2
    assert runner._proc.pid != zygote.pid


def test_zygote_env_scrubbed(monkeypatch, ws):
    """Ruling B: the environ op works only with --test-ops; the secret never reaches a worker."""
    monkeypatch.setenv("LACKPY_TEST_SECRET", "hunter2")
    r = ProcessRunner()
    r._zygote_flags = ("--test-ops",)
    try:
        env = r._worker_environ(timeout=5)
        assert "PATH" in env  # positive evidence that the op answered
        assert "LACKPY_TEST_SECRET" not in env
        assert set(env) <= {"PATH", "HOME", "LANG", "TZ"}
        assert b"hunter2" not in Path(f"/proc/{r._proc.pid}/environ").read_bytes()
    finally:
        r.close()


def test_environ_op_refused_without_test_ops(runner):
    assert runner._worker_environ(timeout=5) is None


def test_worker_validates_again(runner, ws):
    """Ledger ruling: the worker validates, so run() is safe for any caller."""
    result = _run(runner, "def f():\n    yield 1\ng = f()\ng.gi_frame", ws)
    assert result.success is False
    assert result.error.startswith("Validation failed:")


def test_rules_are_resolved_in_the_worker(runner, ws):
    assert _run(runner, "for i in [1]: pass", ws, rules=["no_loops"]).error.startswith(
        "Validation failed:")
    bad = _run(runner, "1", ws, rules=["no_such_rule"])
    assert bad.success is False
    assert "unknown rule 'no_such_rule'" in bad.error


def test_non_data_trace_field_is_sent_as_text(runner, ws):
    """Ledger ruling: a trace entry that is not wire data crosses as repr text; the run succeeds."""
    (ws / "a.txt").write_text("hi")
    result = _run(runner, "from pathlib import Path\ns = Path('a.txt').stat()\ns.st_size", ws)
    assert result.success is True, result.error
    assert result.output == 2
    [entry] = result.trace.entries
    assert entry.tool == "Path.stat"
    assert isinstance(entry.result, str)


def test_worker_holds_only_its_own_socket(runner, ws):
    """A worker holds no control socket and no other run's socket (measured from /proc)."""
    loop = "for i in iter(int, 1): pass"
    ga = _in_thread(lambda: _run(runner, loop, ws, timeout=3))
    first = _wait_for(lambda: runner.last_pid)
    gb = _in_thread(lambda: _run(runner, loop, ws, timeout=3))
    second = _wait_for(lambda: runner.last_pid if runner.last_pid != first else None)
    time.sleep(0.3)
    sockets = [fd for fd in os.listdir(f"/proc/{second}/fd") if int(fd) > 2
               and os.readlink(f"/proc/{second}/fd/{fd}").startswith("socket:")]
    ga(), gb()
    assert len(sockets) == 1, sockets


def test_small_memory_cap_keeps_ordinary_errors(ws):
    """A worker starts at about 47 MiB VmPeak; at a 64 MiB cap an ordinary error is not "memory"."""
    r = ProcessRunner(memory_mb=64)
    try:
        result = _run(r, "1 / 0", ws)
    finally:
        r.close()
    assert result.success is False
    assert "division" in result.error


def test_concurrent_runs_do_not_cross(runner, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for w, text in ((a, "A"), (b, "B")):
        w.mkdir()
        (w / "f.txt").write_text(text)
    prog = "from pathlib import Path\nPath('f.txt').read_text()"
    ga = _in_thread(lambda: _run(runner, prog, a))
    gb = _in_thread(lambda: _run(runner, prog, b))
    assert (ga().output, gb().output) == ("A", "B")
    assert os.getcwd() not in (str(a), str(b))
