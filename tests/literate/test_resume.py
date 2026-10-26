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
    """Beyond max_checkpoints, checkpoints exit (spec D10, fix round 1 I3): the one
    before the stopped cell and the latest are kept, the oldest of the rest go. A
    resume before a dropped one runs fresh."""
    cells = [LOG, "c1 = 1", "c2 = 2", "c3 = 3", "c4 = 4"]
    run = _literate_run(env, runner, max_checkpoints=2)
    try:
        first = run.submit(_doc(*cells, LOOP), None)
        assert first.error_phase == "timeout", first
        live = [pid for _, pid in run._channel.checkpoints]
        assert len(run._channel._stack) <= 3, run._channel._stack
        # The checkpoint before the stopped (last) cell is kept: its fix resumes.
        second = run.submit(_doc(*cells, FIX), None)
        assert second.success, second
        assert _lines(env) == 1
        # The checkpoint before cell 2 (index 1) was among the oldest: dropped, so a
        # change there runs fresh.
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
    run.release()  # get() leases the run (fix round 1); a leased run is never reaped
    now[0] += 6
    with pytest.raises(SessionError, match="unknown or expired session_id"):
        registry.get(run.session_id, workspace=env["ws"], profile_name="lit")
    assert _wait_gone(pids) == []


def test_a_full_registry_closes_the_session_idle_longest(env, runner):
    from lackpy.run.sessions import SessionError, SessionRegistry

    now = [0.0]
    registry = SessionRegistry(max_sessions=2, clock=lambda: now[0])
    # open() and get() lease a run (fix round 1); each lease ends where the run is last used.
    first = _open(registry, env, runner)
    first.release()
    now[0] += 1
    second = _open(registry, env, runner)
    second.release()
    now[0] += 1
    registry.get(first.session_id, workspace=env["ws"], profile_name="lit")  # first is used again
    first.release()
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


# --- Task 7 fix round 1 -----------------------------------------------------------

import threading  # noqa: E402

from lackpy.interpreters.literate.parser import parse  # noqa: E402

# Prose that runs for far longer than any limit here (no `**` in lackpy).
SLOW_PROSE = "> X={sum(range(10000000000))}"


def _cell_number(doc: str, text: str) -> int:
    """The document number (from 1, prose counted) of the first cell that holds ``text``."""
    return next(i for i, c in enumerate(parse(doc).cells) if text in c.content) + 1


def _bounded_submit(run, runner, doc, params=None, bound=15.0):
    """``run.submit`` in a thread; None if it did not return within ``bound`` seconds.

    On a hang the zygote is stopped, so the session's processes exit and the
    thread returns (a waiting checkpoint exits when the zygote is gone).
    """
    box = {}
    t = threading.Thread(target=lambda: box.setdefault("r", run.submit(doc, params)), daemon=True)
    t.start()
    t.join(bound)
    if t.is_alive():
        runner.close()
        t.join(10)
        return None
    return box["r"]


@pytest.mark.parametrize("where", ["segment_start", "middle", "trailing"])
def test_a_slow_prose_cell_times_out_and_the_document_goes_on(env, runner, where):
    """C1: a prose interpolation over the limit is reified as timed out, once.

    ``segment_start``: the document's first cell. ``middle``: between two code
    cells. ``trailing``: the last cell, with no code cell after it.
    """
    docs = {
        "segment_start": _doc(SLOW_PROSE, "b = 1"),
        "middle": _doc("a = 1", SLOW_PROSE, "b = a + 1"),
        "trailing": _doc("a = 1", "b = a + 1", SLOW_PROSE, prose=""),
    }
    doc = docs[where]
    run = _literate_run(env, runner)
    try:
        r = _bounded_submit(run, runner, doc)
        assert r is not None, "submit did not return: the slow prose cell ran again and again"
    finally:
        run.close()
    assert r.error_phase == "timeout", r
    n = _cell_number(doc, "sum(range(")
    assert r.stdout.count(f"timeout: cell {n} took longer than 1 s and was stopped") == 1, r.stdout
    assert r.variables.get("b") in (1, 2), r
    if where != "trailing":
        assert "B=" in r.stdout


def test_the_resume_route_validates_as_the_fresh_route(env, runner):
    """I2: a document the fresh route refuses as a whole is refused the same way on resume."""
    bad = _doc(LOG, prose="") + "\n```lackpy @nosuch\nb = 1\n```\n"  # a parse error
    chan = runner.open_session(path_policy=env["loader"](env["ws"]), workspace=env["ws"],
                               cell_timeout=1.0, rule_names=None, context_config={})
    try:
        fresh = chan.submit(bad, params=None)
    finally:
        chan.close()
    assert fresh.success is False and fresh.error, fresh
    chan = runner.open_session(path_policy=env["loader"](env["ws"]), workspace=env["ws"],
                               cell_timeout=1.0, rule_names=None, context_config={})
    try:
        assert chan.submit(_doc(LOG, LOOP), params=None).error_phase == "timeout"
        assert chan.can_resume(1)
        resumed = chan.resume(1, bad)
    finally:
        chan.close()
    assert (resumed.success, resumed.error) == (False, fresh.error)


def test_an_identical_resubmission_resumes_at_the_stopped_cell(env, runner):
    """Pending ruling A: k = min(first changed cell, first cell that did not complete)."""
    run = _literate_run(env, runner)
    try:
        # The document ends on the stopped cell: no prose segment after it.
        assert run.submit(_doc(LOG, LOOP, prose=""), None).error_phase == "timeout"
        again = run.submit(_doc(LOG, LOOP, prose=""), None)
    finally:
        run.close()
    assert again.error_phase == "timeout", again
    assert _lines(env) == 1  # cell 1 did not run again


# About 3.7 s of CPU on longbottom: over a 1 s limit, under a 15 s one.
SLOWISH = 'import re\nre.search(r"(a+)+$", "a" * 25 + "b")\nb = a + 1'


def test_a_longer_timeout_keeps_the_session(env):
    """Pending ruling B, Teague's retry path: the same document with a longer timeout
    resumes at the stopped cell, which now finishes; cell 1 wrote its line once."""
    from lackpy.profiles import Profile
    from lackpy.service import LackpyService

    service = LackpyService(workspace=env["ws"], path_policy_loader=env["loader"])
    lit = Profile(execution="literate")
    doc = _doc(LOG, SLOWISH, prose="")
    try:
        first = asyncio.run(service.run_program(doc, profile=lit, timeout=1))
        assert first.error_phase == "timeout", first
        second = asyncio.run(service.run_program(doc, profile=lit, timeout=15,
                                                 session_id=first.session_id))
    finally:
        service.close()
    assert second.success, second
    assert second.session_id == first.session_id
    assert second.variables.get("b") == 21
    assert _lines(env) == 1


def test_eviction_keeps_the_checkpoint_before_the_failed_cell(env, runner):
    """I3 (spec D10): with max_checkpoints=2, the checkpoint before the last cell that
    did not succeed is kept, as is the latest, while later cells add checkpoints."""
    cells = [LOG, LOOP] + [f"c{i} = {i}" for i in range(2, 7)]
    run = _literate_run(env, runner, max_checkpoints=2)
    try:
        first = run.submit(_doc(*cells), None)
        assert first.error_phase == "timeout", first
        assert len(run._channel.checkpoints) <= 3
        fixed = run.submit(_doc(LOG, FIX, *cells[2:]), None)
    finally:
        run.close()
    assert fixed.success, fixed
    assert fixed.variables.get("b") == 21
    assert _lines(env) == 1


@pytest.mark.parametrize("cap", [0, 1])
def test_max_checkpoints_below_two_is_refused(env, runner, cap):
    from lackpy.run.sessions import SessionRegistry

    with pytest.raises(ValueError, match="max_checkpoints"):
        _literate_run(env, runner, max_checkpoints=cap)
    with pytest.raises(ValueError, match="max_checkpoints"):
        SessionRegistry(max_checkpoints=cap)


def test_a_leased_run_is_never_evicted(env, runner):
    """Minor 7: open() and get() lease the run; eviction skips a leased run."""
    from lackpy.run.sessions import SessionRegistry

    registry = SessionRegistry(max_sessions=1)
    first = _open(registry, env, runner)  # leased, not yet submitted
    second = _open(registry, env, runner)
    try:
        assert first.session_id in registry  # not evicted while leased
        r = first.submit(_doc(LOG, FIX), None)
        assert r.success, r
        assert _lines(env) == 1
    finally:
        registry.close(first.session_id)
        registry.close(second.session_id)


def test_a_closed_run_refuses_submit(env, runner):
    """Minor 7: a closed LiterateRun never runs fresh silently."""
    run = _literate_run(env, runner)
    run.close()
    r = run.submit(_doc(LOG, FIX), None)
    assert r.success is False
    assert r.error == f"session {run.session_id!r} is closed"
    assert _lines(env) == 0


def test_literate_agent_resubmission_follows_continue(agent, env):
    """I4: a resubmitted document that ends in @continue keeps following @continue."""
    responses = [
        "<compute>\nx = 1 // 0\n</compute>\n",
        "<compute>\na = 2\n</compute>\n\n<compute continue>",
        "<compute>\nb = a + 1\n</compute>\n\nB={b}.\n",
        "never asked",
    ]
    calls = []

    async def generate(prompt, **kwargs):
        calls.append(prompt)
        return responses.pop(0)

    out = asyncio.run(agent.run_literate_agent(
        "x", base_dir=str(env["ws"]), generate=generate, policy_loader=env["loader"],
        max_iterations=4, timeout=5.0))
    assert len(calls) == 3
    assert "B=3." in out, out


def test_literate_agent_uses_the_configured_memory_cap(agent, env, monkeypatch):
    """Minor 11: the agent's runner takes sandbox.memory_mb from the workspace config."""
    (env["ws"] / ".lackpy").mkdir()
    (env["ws"] / ".lackpy" / "config.toml").write_text("[sandbox]\nmemory_mb = 384\n")
    seen = []
    real = agent.shared_runner
    monkeypatch.setattr(agent, "shared_runner", lambda mb: seen.append(mb) or real(mb))

    async def generate(prompt, **kwargs):
        return "Plain prose.\n"

    asyncio.run(agent.run_literate_agent("x", base_dir=str(env["ws"]), generate=generate,
                                         policy_loader=env["loader"]))
    assert seen == [384]


# --- Task 7 fix round 2: a measured memory budget for the checkpoints ------------

BIG = "big = list(range(1000000, 3800000))"  # about 100 MB of int objects


def _pss_mib(pid: int) -> float:
    with open(f"/proc/{pid}/smaps_rollup") as f:
        return next(int(line.split()[1]) for line in f if line.startswith("Pss:")) / 1024


def test_the_checkpoint_budget_bounds_a_large_value(env, runner):
    """Every later cell iterates a large value, so each checkpoint holds its own copy
    (I5: about 87 MiB each). With a 256 MiB budget, the checkpoints other than the two
    D10 keeps stay under it, and a resume at the stopped cell still works."""
    sums = [f"s{i} = sum(big)" for i in range(1, 13)]
    first_doc = _doc(LOG + "\n" + BIG, *sums, LOOP, "big = 0", prose="")
    # Final review I3: cell 1 builds the large value in 1.2-2.0 s under suite load,
    # so 2 s flaked; 10 s gives the cells that are not LOOP room (LOOP costs 10 s).
    run = _literate_run(env, runner, cell_timeout=10.0, max_checkpoint_mb=256)
    try:
        first = run.submit(first_doc, None)
        assert first.error_phase == "timeout", first
        stack = run._channel._stack
        deepest = stack[-1]
        kept = {deepest.pid, deepest.parent, run._channel._protect}
        others = sum(_pss_mib(p.pid) for p in stack if p.pid not in kept)
        assert others <= 256, (others, len(stack))
        fixed = run.submit(_doc(LOG + "\n" + BIG, *sums, FIX, "big = 0", prose=""), None)
    finally:
        run.close()
    assert fixed.success, fixed
    assert fixed.variables.get("b") == 21
    assert _lines(env) == 1


def test_budget_eviction_keeps_the_d10_checkpoints(env, runner):
    """A budget below one checkpoint drops every checkpoint but the D10 two, and does not loop."""
    cells = [LOG, LOOP] + [f"c{i} = {i}" for i in range(2, 7)]
    run = _literate_run(env, runner, max_checkpoint_mb=1)
    try:
        first = run.submit(_doc(*cells), None)
        assert first.error_phase == "timeout", first
        assert len(run._channel._stack) <= 3, run._channel._stack
        fixed = run.submit(_doc(LOG, FIX, *cells[2:]), None)
    finally:
        run.close()
    assert fixed.success, fixed
    assert _lines(env) == 1


def test_an_unreadable_smaps_counts_as_zero():
    from lackpy.run import process

    assert process._pss_kib(2 ** 31 - 1) == 0  # no such process: counted as 0, never raised


@pytest.mark.parametrize("budget", [0, -5, "big"])
def test_a_bad_checkpoint_budget_is_refused(env, runner, budget):
    from lackpy.run.sessions import SessionRegistry

    with pytest.raises(ValueError, match="max_checkpoint_mb"):
        _literate_run(env, runner, max_checkpoint_mb=budget)
    with pytest.raises(ValueError, match="max_checkpoint_mb"):
        SessionRegistry(max_checkpoint_mb=budget)


# --- final review I2: close does not wait for a running submit -------------------

def test_close_does_not_wait_for_a_running_submit(env, runner):
    """close() stops the session's processes at once; the in-flight submit then
    reports a failure, and the run refuses later submits."""
    import threading

    run = _literate_run(env, runner, cell_timeout=20.0)
    out: list = []
    worker = threading.Thread(target=lambda: out.append(run.submit(_doc(LOG, LOOP), None)))
    worker.start()
    try:
        end = time.monotonic() + 10
        while time.monotonic() < end and not (run._channel and run._channel.checkpoints):
            time.sleep(0.02)
        assert run._channel and run._channel.checkpoints, "the loop cell never started"
        time.sleep(0.3)  # the loop cell is running
        pids = [pid for _, pid in run._channel.checkpoints] + [run._channel.pid]
        start = time.monotonic()
        run.close()
        took = time.monotonic() - start
    finally:
        worker.join(30)
    assert took < 2.0, took  # well under the 20 s cell limit
    assert not worker.is_alive()
    assert out and out[0].success is False, out
    assert "ended without a result" in out[0].error, out[0].error
    assert run._channel is None  # the submit's thread closed the channel's fds
    assert _wait_gone(pids) == []
    assert run.submit(_doc(LOG, FIX), None).error == f"session {run.session_id!r} is closed"
