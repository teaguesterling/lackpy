"""A literate document in a session worker, with a checkpoint fork for each cell (Plan 3 Task 6; D6, D7).

Every test that waits on a process has its own bound: each run has a cell
limit, each subprocess wait has a timeout, and the autouse ``_hang_guard``
dumps the stacks and exits if a test still hangs (pytest-timeout is not
installed), so a regression fails rather than hangs.
"""

from __future__ import annotations

import faulthandler
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="the session worker needs Linux (fork, pidfd)")


@pytest.fixture(autouse=True)
def _hang_guard():
    faulthandler.dump_traceback_later(90, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


def test_fork_inside_running_coroutine(tmp_path):
    """Step 0 spike: a fork at the midpoint of a coroutine under ``asyncio.run``.

    The child finishes the coroutine and writes its marker; the parent waits for
    the child, then finishes its own copy. Both copies must see their own state.
    """
    script = textwrap.dedent(f"""
        import asyncio, os
        out = {str(tmp_path)!r}

        async def helper(x):
            await asyncio.sleep(0)
            return x * 2

        async def main():
            a = await helper(21)
            pid = os.fork()
            if pid == 0:
                b = await helper(a)  # the event loop still works in the child
                with open(os.path.join(out, "child"), "w") as f:
                    f.write(f"{{a}} {{b}}")
                return "child"
            _, status = os.waitpid(pid, 0)
            c = await helper(a + 1)
            with open(os.path.join(out, "parent"), "w") as f:
                f.write(f"{{a}} {{c}} {{os.waitstatus_to_exitcode(status)}}")
            return "parent"

        role = asyncio.run(main())
        os._exit(0)
    """)
    proc = subprocess.run([sys.executable, "-I", "-c", script], timeout=30,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "child").read_text() == "42 84"
    assert (tmp_path / "parent").read_text() == "42 86 0"


# --- the session worker ------------------------------------------------------

import asyncio  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

from lackpy.interpreters.literate.parser import parse  # noqa: E402
from lackpy.profiles import Profile  # noqa: E402
from lackpy.run.paths import load_policy  # noqa: E402
from lackpy.run.process import ProcessRunner  # noqa: E402
from lackpy.service import LackpyService  # noqa: E402

_LIT = Profile(execution="literate")
# lackpy has no `while`; a catastrophic regex runs in C for far longer than any limit here.
LOOP = 'import re\nre.search(r"(a+)+$", "a" * 40 + "b")'


def _doc(*cells: str, prose: str = "") -> str:
    body = "\n\n".join(f"```lackpy\n{c}\n```" for c in cells)
    return body + (f"\n\n{prose}\n" if prose else "\n")


def _cell_number(doc: str, source: str) -> int:
    """The document number (from 1, prose counted) of the cell whose content is ``source``."""
    cells = parse(doc).cells
    return next(i for i, c in enumerate(cells) if c.content.strip() == source.strip()) + 1


@pytest.fixture
def env(tmp_path):
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    ws = tmp_path / "ws"
    ws.mkdir()

    def loader(workspace):
        return load_policy(workspace, home=home, managed=None)

    return {"ws": ws, "home": home, "loader": loader}


def _run(env, doc, timeout=1.0, **kw):
    service = LackpyService(workspace=env["ws"], path_policy_loader=env["loader"])
    return asyncio.run(service.run_program(doc, profile=_LIT, timeout=timeout, **kw))


@pytest.fixture
def runner():
    r = ProcessRunner()
    r._zygote_flags = ("--test-ops",)
    yield r
    r.close()


def _gone(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] == "Z"
    except FileNotFoundError:
        return True


def _wait_gone(pids, bound=5.0) -> list[int]:
    end = time.monotonic() + bound
    alive = list(pids)
    while alive and time.monotonic() < end:
        alive = [p for p in alive if not _gone(p)]
        time.sleep(0.02)
    return alive


def test_cell_timeout_does_not_end_document(env):
    doc = _doc("a = 1", LOOP, "b = a + 1", prose="B={b}")
    start = time.monotonic()
    r = _run(env, doc, timeout=1.0)
    elapsed = time.monotonic() - start
    assert r.error_phase == "timeout", r
    assert r.success is False
    assert "B=2" in r.stdout, r.stdout
    n = _cell_number(doc, LOOP)
    assert f"timeout: cell {n} took longer than 1 s and was stopped" in r.stdout
    assert r.variables.get("b") == 2
    assert elapsed < 6, elapsed


def test_timeout_names_written_file(env):
    """Gate 5, literate form: the note names the file the stopped cell wrote."""
    cell = "from pathlib import Path\nPath('out.json').write_text('{}')\n" + LOOP
    doc = _doc("a = 1", cell, "b = a + 1", prose="B={b}")
    r = _run(env, doc, timeout=1.0)
    assert r.error_phase == "timeout", r
    n = _cell_number(doc, cell)
    assert (f"timeout: cell {n} took longer than 1 s and was stopped; "
            f"it wrote out.json before it was stopped") in r.stdout, r.stdout
    assert "out.json" in r.trace.files_modified
    assert (env["ws"] / "out.json").read_text() == "{}"
    assert "B=2" in r.stdout


def test_write_started_not_confirmed(env, runner):
    """Review Focus 5 (Ruling C): a write that has a start entry but no done entry.

    The test-only hook blocks inside the stand-in's write, after its start entry,
    so the cell is killed in the middle of the write.
    """
    cell = "from pathlib import Path\nPath('big.txt').write_text('x' * 1000)"
    doc = _doc("a = 1", cell, "b = a + 1", prose="B={b}")
    policy = env["loader"](env["ws"])
    chan = runner.open_session(path_policy=policy, workspace=env["ws"], cell_timeout=1.0,
                               rule_names=None, context_config={"test_block_writes": True})
    try:
        r = chan.submit(doc, params=None)
    finally:
        chan.close()
    assert r.error_phase == "timeout", r
    n = _cell_number(doc, cell)
    assert (f"timeout: cell {n} took longer than 1 s and was stopped; it wrote "
            f"big.txt (write started, not confirmed) before it was stopped") in r.stdout, r.stdout
    assert "B=2" in r.stdout


def test_refusal_still_latches(env):
    """D4 unchanged: a refusal in cell 2 ends the document, in the session worker too."""
    refused = "from pathlib import Path\nPath('.claude/settings.json').write_text('{}')"
    doc = _doc("a = 1", refused, "c = 3", prose="C={c}")
    r = _run(env, doc, timeout=5.0)
    assert r.error_phase == "refused", r
    assert r.success is False
    assert "refused:" in r.stdout
    assert "c" not in r.variables
    assert not (env["ws"] / ".claude" / "settings.json").exists()


def test_callable_rule_refused_by_process_path(env, runner):
    policy = env["loader"](env["ws"])
    with pytest.raises(ValueError, match="the process runner takes rule names, not callables"):
        runner.open_session(path_policy=policy, workspace=env["ws"], cell_timeout=1.0,
                            rule_names=[lambda tree: []], context_config={})


def test_rule_names_apply_in_the_worker(env):
    """Ruling D: caller rules cross as names, and the worker validates every cell with them."""
    doc = _doc("a = 1", "for i in range(3):\n    a = a + i", prose="A={a}")
    r = _run(env, doc, timeout=5.0, rules=["no_loops"])
    assert r.success is False
    assert "loop" in (r.error or "").lower() + r.stdout.lower(), r


def test_session_close_leaves_no_process(env, runner):
    """Every checkpoint and cell process is gone after close."""
    doc = _doc("a = 1", "b = 2", "c = 3", prose="C={c}")
    policy = env["loader"](env["ws"])
    chan = runner.open_session(path_policy=policy, workspace=env["ws"], cell_timeout=5.0,
                               rule_names=None, context_config={})
    r = chan.submit(doc, params=None)
    pids = [pid for _, pid in chan.checkpoints] + [chan.pid]
    assert r.success, r
    assert len(pids) >= 4
    assert not [p for p in pids if _gone(p)]
    chan.close()
    assert _wait_gone(pids) == []


def test_zygote_death_leaves_no_process(env, runner):
    """A zygote that dies while a cell runs takes the whole session down (PDEATHSIG at each fork)."""
    doc = _doc("a = 1", "b = 2", LOOP)
    policy = env["loader"](env["ws"])
    chan = runner.open_session(path_policy=policy, workspace=env["ws"], cell_timeout=30.0,
                               rule_names=None, context_config={})
    box: dict = {}
    t = threading.Thread(target=lambda: box.setdefault("r", chan.submit(doc, params=None)),
                         daemon=True)
    t.start()
    end = time.monotonic() + 10
    while len(chan.checkpoints) < 3 and time.monotonic() < end:
        time.sleep(0.02)
    pids = [pid for _, pid in chan.checkpoints] + [chan.pid]
    assert len(pids) >= 4, chan.checkpoints
    os.kill(runner._proc.pid, 9)
    assert _wait_gone(pids) == []
    t.join(10)
    assert not t.is_alive()
    assert box["r"].success is False
    chan.close()


# --- the kernel's checkpoint hook, in-process --------------------------------

from lackpy.interpreters.base import ExecutionContext  # noqa: E402
from lackpy.interpreters.literate import LiterateInterpreter  # noqa: E402
from lackpy.run.session_worker import Resume  # noqa: E402
from lackpy.run.trace import TraceEntry  # noqa: E402


def _execute(doc, checkpoint):
    ctx = ExecutionContext(base_dir=".", params={}, config={"checkpoint": checkpoint})
    return asyncio.run(LiterateInterpreter().execute(doc, ctx))


def test_hook_timeout_reifies_the_cell_as_holes():
    """A "timeout" Resume: the cell does not run, its names are holes, a later reader is blocked."""
    doc = _doc("a = 1", "x = a * 2", "c = x + 1", "d = a + 1", prose="D={d}")
    target = _cell_number(doc, "x = a * 2") - 1
    calls = []
    wrote = TraceEntry(step=0, tool="Path.write_text", args=["w.txt"], result=1,
                       duration_ms=0.1, success=True, error=None)

    def checkpoint(index):
        calls.append(index)
        if index == target:
            return Resume("timeout", seconds=2.5, entries=(wrote,))
        return None

    r = _execute(doc, checkpoint)
    assert r.success is False
    assert r.metadata["error_phase"] == "timeout"
    assert (f"timeout: cell {target + 1} took longer than 2.5 s and was stopped; "
            f"it wrote w.txt before it was stopped") in r.output
    assert "D=2" in r.output
    assert "x" not in r.metadata["variables"] or repr(r.metadata["variables"]["x"]).startswith("⟨")
    assert any(e.entry_type == "timed_out" for e in r.metadata["ledger"].entries())
    assert any(e.entry_type == "hole_opened" and e.name == "c" for e in r.metadata["ledger"].entries())
    # The stopped cell's entries join the document's trace.
    assert [e.tool for e in r.metadata["trace"].entries] == ["Path.write_text"]
    assert calls == list(range(len(parse(doc).cells)))


def test_hook_resume_continues_over_the_new_document():
    """A "resume" Resume: the loop goes on at the same index over the new document's cells."""
    old = _doc("a = 1", "b = a + 100")
    new = _doc("a = 1", "b = a + 1", prose="B={b}")
    target = _cell_number(old, "b = a + 100") - 1
    sent = []

    def checkpoint(index):
        if index == target and not sent:
            sent.append(index)
            return Resume("resume", document=new)
        return None

    r = _execute(old, checkpoint)
    assert r.success, r.error
    assert r.metadata["variables"]["b"] == 2
    assert "B=2" in r.output
    assert r.metadata["error_phase"] is None


def test_partial_message_from_a_stopped_cell_is_dropped(env, runner):
    """A cell killed in the middle of a message leaves half a line on its own socket only.

    The parent drains that socket, drops the partial line, and the document goes
    on: the next cell's messages arrive whole on a new socket.
    """
    cell = "from pathlib import Path\nPath('p.txt').write_text('x')"
    doc = _doc("a = 1", cell, "b = a + 1", prose="B={b}")
    policy = env["loader"](env["ws"])
    chan = runner.open_session(path_policy=policy, workspace=env["ws"], cell_timeout=1.0,
                               rule_names=None, context_config={"test_partial_frame": True})
    try:
        r = chan.submit(doc, params=None)
    finally:
        chan.close()
    assert r.error_phase == "timeout", r
    assert "p.txt (write started, not confirmed)" in r.stdout
    assert "B=2" in r.stdout
    assert r.variables.get("b") == 2


def test_test_hooks_need_test_ops(env):
    """Without --test-ops the worker ignores the test-only hooks (ledger Rulings B, C)."""
    r = ProcessRunner()
    try:
        cell = "from pathlib import Path\nPath('t.txt').write_text('x')"
        chan = r.open_session(path_policy=env["loader"](env["ws"]), workspace=env["ws"],
                              cell_timeout=5.0, rule_names=None,
                              context_config={"test_block_writes": True})
        try:
            res = chan.submit(_doc(cell), params=None)
        finally:
            chan.close()
    finally:
        r.close()
    assert res.success, res
    assert (env["ws"] / "t.txt").read_text() == "x"


def test_hook_forks_each_dirty_rerun_too():
    """The re-exec pass calls the hook before each re-run, so a re-run is under the limit too."""
    doc = _doc("y = x + 1", "x = 1", prose="Y={y}")
    first = _cell_number(doc, "y = x + 1") - 1
    calls = []

    def checkpoint(index):
        calls.append(index)
        if calls.count(first) == 2:  # the re-run of the healed cell
            return Resume("timeout", seconds=1.0)
        return None

    r = _execute(doc, checkpoint)
    assert calls.count(first) == 2, calls
    assert r.metadata["error_phase"] == "timeout"
    assert f"timeout: cell {first + 1} took longer than 1 s and was stopped" in r.output
    dirty = [e for e in r.metadata["ledger"].entries() if e.entry_type == "dirty"]
    assert dirty and dirty[-1].detail["outcome"] == "reified"


def test_death_between_cell_end_and_cell_start_is_a_crash(env, runner):
    """A process that dies in its hook, before the next cell_start, is reported at once.

    The checkpoints above still hold the shared channel, so the parent must
    watch the process's own socket too, not wait for the cell limit.
    """
    doc = _doc("a = 1", "b = 2", "c = 3")
    policy = env["loader"](env["ws"])
    chan = runner.open_session(path_policy=policy, workspace=env["ws"], cell_timeout=30.0,
                               rule_names=None, context_config={"test_exit_in_hook": 1})
    start = time.monotonic()
    try:
        r = chan.submit(doc, params=None)
    finally:
        chan.close()
    assert time.monotonic() - start < 5
    assert r.success is False
    assert r.error_phase is None
    assert "the session process ended without a result" in (r.error or ""), r


def test_channel_events_send_and_kill(env, runner):
    """The SessionChannel primitives Task 7 builds on: send, events, kill, checkpoints."""
    doc = _doc("a = 1", LOOP)
    policy = env["loader"](env["ws"])
    chan = runner.open_session(path_policy=policy, workspace=env["ws"], cell_timeout=30.0,
                               rule_names=None, context_config={})
    try:
        chan.send({"op": "submit", "document": doc, "params": None, **chan._submit_base})
        seen = []
        for msg in chan.events(time.monotonic() + 10):
            seen.append(msg["event"])
            if msg["event"] == "cell_start" and msg["index"] == len(parse(doc).cells) - 1:
                break
        assert seen.count("cell_start") == len(parse(doc).cells)
        index, pid = chan.checkpoints[-1]
        chan.kill(pid)
        rest = list(chan.events(time.monotonic() + 5))
        assert rest[-1]["event"] == "ended" and rest[-1]["pid"] == pid
    finally:
        chan.close()


@pytest.mark.parametrize("entry", ["run_program", "delegate", "mcp.run_program",
                                   "lackpyctl render --execute"])
def test_every_entry_point_stops_a_cell_and_goes_on(env, monkeypatch, capsys, entry):
    """D9: each literate route runs in the session worker, with the limit on each cell."""
    from lackpy import service as service_module

    (env["ws"] / ".lackpy").mkdir()
    (env["ws"] / ".lackpy" / "config.toml").write_text('[profiles.lit]\nexecution = "literate"\n')
    monkeypatch.setattr(service_module, "load_policy", env["loader"])
    doc = _doc("a = 1", LOOP, "b = a + 1", prose="B={b}")
    note = f"timeout: cell {_cell_number(doc, LOOP)} took longer than 1 s and was stopped"
    service = LackpyService(workspace=env["ws"], path_policy_loader=env["loader"])
    start = time.monotonic()
    if entry == "run_program":
        r = asyncio.run(service.run_program(doc, profile="lit", timeout=1.0))
        text, phase = r.stdout, r.error_phase
    elif entry == "delegate":
        r = asyncio.run(service.delegate("x", profile="lit", _program_override=doc, timeout=1.0))
        text, phase = r.get("stdout", ""), r["error_phase"]
    elif entry == "mcp.run_program":
        pytest.importorskip("mcp", reason="mcp required for the MCP entry point")
        from lackpy.mcp import server
        server._service = service
        try:
            r = asyncio.run(server.run_program(doc, profile="lit", timeout=1.0))
        finally:
            server._service = None
        text, phase = r["stdout"], r["error_phase"]
    else:
        from lackpy.ctl import main
        path = env["ws"] / "doc.md"
        path.write_text(doc)
        code = main(["render", "--execute", str(path), "--base-dir", str(env["ws"]),
                     "--timeout", "1"])
        assert code == 1
        text, phase = capsys.readouterr().out, "timeout"
    assert time.monotonic() - start < 8
    assert phase == "timeout"
    assert note in text, text
    assert "B=2" in text
