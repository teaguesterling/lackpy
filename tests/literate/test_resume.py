"""A retry resumes from the checkpoint at the first changed cell (Plan 3 Task 7; spec D10).

``LiterateRun`` is the parent's handle on one session worker. Its ``submit``
applies the prefix rule: the cells before the first changed cell already ran,
so they do not run again. ``SessionRegistry`` holds the runs that MCP calls
name by ``session_id``.

Every run has a cell limit, and the autouse ``_hang_guard`` exits a test that
still hangs, so a regression fails rather than hangs.
"""

from __future__ import annotations

import asyncio
import faulthandler
import importlib.util
import os
import sys
import time
from pathlib import Path

import pytest

from lackpy.run.paths import load_policy
from lackpy.run.process import ProcessRunner

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="the session worker needs Linux (fork, pidfd)")

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "literate_agent.py"

# lackpy has no `while`; a catastrophic regex runs in C for far longer than any limit here.
LOOP = 'import re\nre.search(r"(a+)+$", "a" * 40 + "b")'
# Cell 1: one more line in log.txt, and a binding a later cell reads.
LOG = "from pathlib import Path\np = Path('log.txt')\np.write_text(p.read_text() + 'line\\n')\na = 20"
FIX = "b = a + 1"


@pytest.fixture(autouse=True)
def _hang_guard():
    faulthandler.dump_traceback_later(90, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


def _doc(*cells: str, prose: str = "B={b}") -> str:
    """Code cells, then one prose line. A cell that starts with ``"> "`` is prose."""
    parts = [c[2:] if c.startswith("> ") else f"```lackpy\n{c}\n```" for c in cells]
    return "\n\n".join(parts) + (f"\n\n{prose}\n" if prose else "\n")


@pytest.fixture
def env(tmp_path):
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "log.txt").write_text("")

    def loader(workspace):
        return load_policy(workspace, home=home, managed=None)

    return {"ws": ws, "home": home, "loader": loader}


@pytest.fixture
def runner():
    r = ProcessRunner()
    yield r
    r.close()


def _lines(env) -> int:
    return (env["ws"] / "log.txt").read_text().count("line")


def _literate_run(env, runner, **kw):
    from lackpy.run.sessions import LiterateRun

    kw.setdefault("cell_timeout", 1.0)
    return LiterateRun(runner, workspace=env["ws"], profile_name="lit",
                       path_policy=env["loader"](env["ws"]), **kw)


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


# --- the prefix rule, through LiterateRun --------------------------------------

def test_resume_after_timeout_skips_unchanged_prefix(env, runner):
    run = _literate_run(env, runner)
    try:
        first = run.submit(_doc(LOG, LOOP), None)
        assert first.error_phase == "timeout", first
        assert _lines(env) == 1
        second = run.submit(_doc(LOG, FIX), None)
    finally:
        run.close()
    assert second.success, second
    assert second.variables.get("b") == 21, second
    assert "B=21" in second.stdout, second.stdout
    assert _lines(env) == 1  # cell 1 did not run again


def test_changed_prefix_runs_fresh(env, runner):
    run = _literate_run(env, runner)
    try:
        first = run.submit(_doc(LOG, LOOP), None)
        assert first.error_phase == "timeout", first
        second = run.submit(_doc(LOG.replace("a = 20", "a = 30"), FIX), None)
    finally:
        run.close()
    assert second.success, second
    assert second.variables.get("b") == 31, second
    assert _lines(env) == 2  # cell 1 changed, so the document ran again from the start


def test_a_finished_document_is_not_resumed(env, runner):
    """The process that finished the document has left the kernel loop (review note).

    An appended cell has no checkpoint that can still run, so the document runs fresh:
    the result is right, and cell 1 runs again.
    """
    run = _literate_run(env, runner)
    try:
        # The document ends on its code cell: no prose segment after it has a checkpoint.
        first = run.submit(_doc(LOG, prose=""), None)
        assert first.success, first
        second = run.submit(_doc(LOG, FIX), None)
    finally:
        run.close()
    assert second.success, second
    assert second.variables.get("b") == 21
    assert _lines(env) == 2


def test_a_changed_prose_cell_resumes_from_its_segment(env, runner):
    """A prose cell takes no checkpoint of its own; a run of prose cells shares the
    checkpoint before the first of them. So an edit to the prose before the fixed
    cell still resumes after cell 1."""
    run = _literate_run(env, runner)
    try:
        first = run.submit(_doc(LOG, "> Now the slow part.", LOOP), None)
        assert first.error_phase == "timeout", first
        second = run.submit(_doc(LOG, "> Now the fixed part.", FIX), None)
    finally:
        run.close()
    assert second.success, second
    assert "Now the fixed part." in second.stdout
    assert "Now the slow part." not in second.stdout
    assert second.variables.get("b") == 21
    assert _lines(env) == 1


def test_resume_applies_the_params(env, runner):
    """The resumed cells get the submission's params, as the first run's cells did."""
    run = _literate_run(env, runner)
    try:
        first = run.submit(_doc(LOG, "params = {'n': 1}\n" + LOOP), {"n": 5})
        assert first.error_phase == "timeout", first
        second = run.submit(_doc(LOG, "params = {'n': 1}\nb = a + params['n']"), {"n": 5})
    finally:
        run.close()
    assert second.success, second
    assert second.variables.get("b") == 25, second
    assert _lines(env) == 1


def test_resume_twice_from_the_same_checkpoint(env, runner):
    """A checkpoint carries out each instruction in a child, so it is still there
    for the next retry."""
    run = _literate_run(env, runner)
    try:
        assert run.submit(_doc(LOG, LOOP), None).error_phase == "timeout"
        assert run.submit(_doc(LOG, LOOP + "\nz = 1"), None).error_phase == "timeout"
        third = run.submit(_doc(LOG, FIX), None)
    finally:
        run.close()
    assert third.success, third
    assert third.variables.get("b") == 21
    assert _lines(env) == 1


def test_checkpoints_over_the_cap_exit(env, runner):
    """Beyond max_checkpoints the oldest checkpoint process exits; a resume before it runs fresh."""
    cells = [LOG, "c1 = 1", "c2 = 2", "c3 = 3", "c4 = 4"]
    run = _literate_run(env, runner, max_checkpoints=2)
    try:
        first = run.submit(_doc(*cells, LOOP), None)
        assert first.error_phase == "timeout", first
        live = [pid for _, pid in run._channel.checkpoints]
        assert len(run._channel._stack) <= 3, run._channel._stack
        # The newest checkpoint is kept: a fix of the last cell resumes.
        second = run.submit(_doc(*cells, FIX), None)
        assert second.success, second
        assert _lines(env) == 1
        # Cell 2's checkpoint was dropped: a change there runs fresh.
        third = run.submit(_doc(LOG, "c1 = 10", *cells[2:], FIX), None)
        assert third.success, third
        assert _lines(env) == 2
    finally:
        run.close()
    assert _wait_gone(live) == []


def _epoll_targets(pid: int) -> list[str]:
    """The fds registered in each epoll instance that process ``pid`` holds."""
    out = []
    for fd in os.listdir(f"/proc/{pid}/fd"):
        try:
            if os.readlink(f"/proc/{pid}/fd/{fd}") != "anon_inode:[eventpoll]":
                continue
            with open(f"/proc/{pid}/fdinfo/{fd}") as f:
                out += [line.split()[1] for line in f if line.startswith("tfd:")]
        except OSError:
            continue
    return out


def test_a_finished_child_leaves_its_ancestors_epoll_alone(env, runner):
    """A forked child's asyncio.run teardown must not change the epoll instance of a
    checkpoint that can still resume (measured: without a private epoll in each child,
    the finished process's loop.close() removed the root's self-pipe registration)."""
    run = _literate_run(env, runner)
    try:
        r = run.submit(_doc("a = 1", "b = 2"), None)
        assert r.success, r
        root = run._channel.pid
        assert _epoll_targets(root), "the root's loop holds its self-pipe in epoll"
    finally:
        run.close()


# --- the registry ---------------------------------------------------------------

def _open(registry, env, runner, profile_name="lit"):
    return registry.open(runner=runner, workspace=env["ws"], profile_name=profile_name,
                         path_policy=env["loader"](env["ws"]), cell_timeout=1.0)


def test_unknown_session_id_errors(env):
    from lackpy.run.sessions import SessionError, SessionRegistry

    registry = SessionRegistry()
    with pytest.raises(SessionError) as e:
        registry.get("nope", workspace=env["ws"], profile_name="lit")
    assert str(e.value) == "unknown or expired session_id 'nope'"


def test_session_profile_mismatch_errors(env, runner):
    from lackpy.run.sessions import SessionError, SessionRegistry

    registry = SessionRegistry()
    run = _open(registry, env, runner)
    try:
        with pytest.raises(SessionError) as e:
            registry.get(run.session_id, workspace=env["ws"], profile_name="other")
        assert str(e.value) == f"session {run.session_id!r} belongs to profile 'lit' in {env['ws']}"
        with pytest.raises(SessionError):
            registry.get(run.session_id, workspace=env["ws"].parent, profile_name="lit")
        assert registry.get(run.session_id, workspace=env["ws"], profile_name="lit") is run
    finally:
        registry.close(run.session_id)


def test_idle_session_reaped(env, runner):
    from lackpy.run.sessions import SessionError, SessionRegistry

    now = [1000.0]
    registry = SessionRegistry(idle_seconds=5, clock=lambda: now[0])
    run = _open(registry, env, runner)
    assert run.submit(_doc("a = 1", "b = 2"), None).success
    pids = [pid for _, pid in run._channel.checkpoints] + [run._channel.pid]
    now[0] += 4
    assert registry.get(run.session_id, workspace=env["ws"], profile_name="lit") is run
    now[0] += 6
    with pytest.raises(SessionError, match="unknown or expired session_id"):
        registry.get(run.session_id, workspace=env["ws"], profile_name="lit")
    assert _wait_gone(pids) == []


def test_a_full_registry_closes_the_session_idle_longest(env, runner):
    from lackpy.run.sessions import SessionError, SessionRegistry

    now = [0.0]
    registry = SessionRegistry(max_sessions=2, clock=lambda: now[0])
    first = _open(registry, env, runner)
    now[0] += 1
    second = _open(registry, env, runner)
    now[0] += 1
    registry.get(first.session_id, workspace=env["ws"], profile_name="lit")  # first is used again
    now[0] += 1
    third = _open(registry, env, runner)
    try:
        with pytest.raises(SessionError):
            registry.get(second.session_id, workspace=env["ws"], profile_name="lit")
        assert registry.get(first.session_id, workspace=env["ws"], profile_name="lit") is first
    finally:
        registry.close(first.session_id)
        registry.close(third.session_id)


# --- literate_agent -------------------------------------------------------------

@pytest.fixture(scope="module")
def agent():
    spec = importlib.util.spec_from_file_location("literate_agent_script_resume", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_literate_agent_resubmission_resumes(agent, env):
    """Round 1 fails; the first resubmission times out in cell 2; the second fixes
    cell 2 only. Cell 1 writes its line once."""
    responses = [
        "<compute>\nx = 1 // 0\n</compute>\n",
        _doc(LOG, LOOP),
        _doc(LOG, FIX),
        "never asked",
    ]
    calls = []

    async def generate(prompt, **kwargs):
        calls.append(prompt)
        return responses.pop(0)

    out = asyncio.run(agent.run_literate_agent(
        "x", base_dir=str(env["ws"]), generate=generate, policy_loader=env["loader"],
        max_iterations=4, timeout=1.0))
    assert len(calls) == 3
    assert "took longer than 1 s" in calls[2]  # the model is shown the timeout
    assert "B=21" in out, out
    assert _lines(env) == 1
