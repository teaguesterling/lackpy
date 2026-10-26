"""Service routing through the process runner, the timeout parameter and max_timeout
(Plan 3, Task 5; spec D8, D9; gates 1 and 7 through the service).

Every run here has its own bound: each program has a timeout of a few seconds,
and ``_hang_guard`` dumps the stacks and exits if a test still hangs.
"""

from __future__ import annotations

import asyncio
import faulthandler
import math
import os
import sys
import time

import pytest

from lackpy.config import LackpyConfig
from lackpy.profiles import Profile
from lackpy.run.paths import PathPolicy, load_policy
from lackpy.run.process import ProcessRunner
from lackpy.service import LackpyService

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="the zygote needs Linux (pidfd, prctl)")

REGEX = 'import re\nre.search(r"(a+)+$", "a" * 26 + "b")'
FOREVER = "for i in iter(int, 1):\n    pass"


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


def _service(ws, tmp_path, **config):
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True, exist_ok=True)
    cfg = LackpyConfig(config_dir=ws / ".lackpy", **config)
    return LackpyService(workspace=ws, config=cfg,
                         path_policy_loader=lambda w: load_policy(w, home=home, managed=None))


@pytest.fixture
def make(ws, tmp_path):
    made: list[LackpyService] = []

    def build(**config):
        svc = _service(ws, tmp_path, **config)
        made.append(svc)
        return svc
    yield build
    for svc in made:
        svc.close()


# --- the limit through the service (gate 1) ---------------------------------

async def test_run_program_timeout(make):
    """Gate 1 through the service: the regex stops at the requested limit."""
    svc = make()
    start = time.monotonic()
    result = await svc.run_program(REGEX, timeout=1)
    elapsed = time.monotonic() - start
    assert result.success is False
    assert result.error_phase == "timeout"
    assert result.error == "timeout: the run took longer than 1 s and was stopped"
    # The zygote's first start (about 0.3 s) is not charged to the program.
    assert elapsed < 1 + 1, elapsed


async def test_default_timeout_is_sandbox_timeout(make):
    """With no timeout, config.sandbox_timeout is the limit (D8)."""
    svc = make(sandbox_timeout=1)
    start = time.monotonic()
    result = await svc.run_program(FOREVER)
    elapsed = time.monotonic() - start
    assert result.error_phase == "timeout", result.error
    assert result.error == "timeout: the run took longer than 1 s and was stopped"
    assert elapsed < 1 + 1, elapsed


async def test_delegate_timeout(make):
    """delegate takes the same parameter and returns the timeout result as it is (D7)."""
    svc = make()
    result = await svc.delegate("x", _program_override=REGEX, timeout=1)
    assert result["success"] is False
    assert result["error_phase"] == "timeout"


async def test_delegate_result_carries_error_phase(make):
    svc = make()
    result = await svc.delegate("x", _program_override="1")
    assert result["success"] is True, result["error"]
    assert "error_phase" in result and result["error_phase"] is None


async def test_run_program_runs_in_a_worker(make, ws):
    """D9: the program runs in a forked worker, in the workspace; the service's cwd stays."""
    svc = make()
    cwd = os.getcwd()
    result = await svc.run_program("from pathlib import Path\nPath('w.txt').write_text('x')\n1")
    assert result.success, result.error
    assert (ws / "w.txt").read_text() == "x"
    assert os.getcwd() == cwd
    runner = svc._process_runner
    assert isinstance(runner, ProcessRunner)
    assert runner.last_pid is not None and runner.last_pid != os.getpid()


async def test_concurrent_runs(make, ws):
    """Review Focus 4: two runs under gather keep their own stdout and files."""
    svc = make()
    cwd = os.getcwd()
    prog = "from pathlib import Path\nPath('{0}.txt').write_text('{0}')\nprint('{0}')"
    a, b = await asyncio.gather(svc.run_program(prog.format("a")),
                                svc.run_program(prog.format("b")))
    assert a.success and b.success, (a.error, b.error)
    assert a.stdout.strip() == "a"
    assert b.stdout.strip() == "b"
    assert (ws / "a.txt").read_text() == "a"
    assert (ws / "b.txt").read_text() == "b"
    assert os.getcwd() == cwd


async def test_concurrent_runs_overlap(make):
    """Without _exec_lock, two slow runs take about the time of one, not two."""
    svc = make()
    await svc.run_program("1")  # start the zygote first
    start = time.monotonic()
    a, b = await asyncio.gather(svc.run_program(FOREVER, timeout=1),
                                svc.run_program(FOREVER, timeout=1))
    elapsed = time.monotonic() - start
    assert a.error_phase == b.error_phase == "timeout"
    assert elapsed < 1.8, elapsed


# --- the cap (gate 7) -------------------------------------------------------

async def test_timeout_over_cap_refuses(make):
    """Gate 7: over the profile's max_timeout refuses before validation and before the run."""
    svc = make(profiles={"short": {"max_timeout": 5}})
    # An invalid program: a refusal before validation gives the cap text, not the validator's.
    result = await svc.run_program("import os", profile="short", timeout=10)
    assert result.success is False
    assert result.error_phase == "refused"
    assert result.error == "timeout 10 s is over the profile's max_timeout (5 s, profile short)"
    assert svc._process_runner is None  # no zygote was started


async def test_timeout_over_default_cap_refuses(make):
    """With no max_timeout, the cap is config.sandbox_timeout."""
    svc = make(sandbox_timeout=3)
    result = await svc.run_program("1", timeout=3.5)
    assert result.error_phase == "refused"
    assert result.error == "timeout 3.5 s is over the profile's max_timeout (3 s, profile (none))"


async def test_unnamed_profile_cap_text(make):
    svc = make()
    result = await svc.run_program("1", profile=Profile(max_timeout=2), timeout=4)
    assert result.error == "timeout 4 s is over the profile's max_timeout (2 s, profile (unnamed))"


async def test_default_timeout_is_capped_by_the_profile(make):
    """Fix round 1 ruling: with no timeout, the limit is min(sandbox_timeout, cap)."""
    svc = make(profiles={"short": {"max_timeout": 1}})
    start = time.monotonic()
    result = await svc.run_program(FOREVER, profile="short")
    elapsed = time.monotonic() - start
    assert result.error_phase == "timeout", result.error
    assert result.error == "timeout: the run took longer than 1 s and was stopped"
    assert elapsed < 1 + 1, elapsed


async def test_default_timeout_is_capped_through_delegate(make):
    svc = make(profiles={"short": {"max_timeout": 1}})
    result = await svc.delegate("x", profile="short", _program_override=FOREVER)
    assert result["error_phase"] == "timeout", result["error"]
    assert result["error"] == "timeout: the run took longer than 1 s and was stopped"


async def test_default_timeout_is_capped_through_profile_default(make):
    svc = make(profile_default="short", profiles={"short": {"max_timeout": 1}})
    result = await svc.run_program(FOREVER)
    assert result.error_phase == "timeout", result.error
    assert result.error == "timeout: the run took longer than 1 s and was stopped"


async def test_execute_program_default_is_capped(make):
    """Minor 3: the internal path with timeout=None takes the same capped default."""
    svc = make(profiles={"short": {"max_timeout": 1}})
    rp = svc._resolve_profile("short")
    result = await svc._execute_program(FOREVER, rp, {})
    assert result.error_phase == "timeout", result.error
    assert result.error == "timeout: the run took longer than 1 s and was stopped"


async def test_explicit_timeout_over_a_lower_cap_still_refuses(make):
    svc = make(profile_default="short", profiles={"short": {"max_timeout": 1}})
    result = await svc.run_program("1", timeout=2)
    assert result.error_phase == "refused"
    assert result.error == "timeout 2 s is over the profile's max_timeout (1 s, profile short)"


async def test_max_timeout_raises_the_cap(make):
    svc = make(sandbox_timeout=3, profiles={"long": {"max_timeout": 300}})
    result = await svc.run_program("1", profile="long", timeout=200)
    assert result.success is True, result.error
    assert result.output == 1


async def test_timeout_at_the_cap_runs(make):
    svc = make(profiles={"short": {"max_timeout": 5}})
    result = await svc.run_program("1", profile="short", timeout=5)
    assert result.success is True, result.error


async def test_delegate_over_cap_refuses_before_generation(make, monkeypatch):
    svc = make(profiles={"short": {"max_timeout": 5}})

    async def no_generate(*a, **kw):
        raise AssertionError("generate ran for a refused request")
    monkeypatch.setattr(svc, "generate", no_generate)
    result = await svc.delegate("anything", profile="short", timeout=6)
    assert result["success"] is False
    assert result["error_phase"] == "refused"
    assert result["error"] == "timeout 6 s is over the profile's max_timeout (5 s, profile short)"


@pytest.mark.parametrize("bad", [math.nan, math.inf, -1, 0])
async def test_timeout_that_is_not_a_positive_number_refuses(make, bad):
    """A model sends these over MCP: a failed result, never a raise."""
    svc = make()
    result = await svc.run_program("1", timeout=bad)
    assert result.success is False
    assert result.error_phase == "refused"
    assert result.error == f"timeout must be a positive number of seconds, not {bad!r}"
    assert svc._process_runner is None


@pytest.mark.parametrize("bad", ["10", True, -5, 0])
async def test_max_timeout_that_is_not_a_positive_number_fails_closed(make, bad):
    svc = make(profiles={"p": {"max_timeout": bad}})
    result = await svc.run_program("1", profile="p")
    assert result.success is False
    assert f"max_timeout must be a positive number of seconds, not {bad!r}" in result.error


# --- rules, lifecycle and the platform --------------------------------------

async def test_rule_names_reach_the_worker(make, monkeypatch):
    """The caller's rule names cross to the worker, which validates again."""
    svc = make()
    sent = {}
    original = ProcessRunner.run

    def spy(self, program, **kw):
        sent["rules"] = kw.get("rules")
        return original(self, program, **kw)
    monkeypatch.setattr(ProcessRunner, "run", spy)
    result = await svc.run_program("x = 1\nx", rules=["no_loops", "max_depth(3)"])
    assert result.success, result.error
    assert sent["rules"] == ["no_loops", "max_depth(3)"]


async def test_callable_rule_validates_in_the_parent_only(make, monkeypatch):
    """A callable cannot cross: the parent applies it, the worker gets the names only."""
    svc = make()
    sent = {}
    original = ProcessRunner.run

    def spy(self, program, **kw):
        sent["rules"] = kw.get("rules")
        return original(self, program, **kw)
    monkeypatch.setattr(ProcessRunner, "run", spy)

    def allow(tree):
        return []

    def reject(tree):
        return ["rejected by the caller's rule"]
    ok = await svc.run_program("1", rules=[allow, "no_loops"])
    assert ok.success, ok.error
    assert sent["rules"] == ["no_loops"]
    bad = await svc.run_program("1", rules=[reject])
    assert bad.success is False
    assert "rejected by the caller's rule" in bad.error


def test_process_runner_refuses_callable_rules(ws):
    r = ProcessRunner()
    try:
        with pytest.raises(ValueError, match="the process runner takes rule names, not callables"):
            r.run("1", params=None, path_policy=PathPolicy(ws, []), workspace=ws,
                  timeout=2, rules=[lambda tree: []])
    finally:
        r.close()


async def test_services_share_one_runner(make):
    """Fix round 1 ruling: one ProcessRunner per process for each memory cap."""
    a, b, c = make(), make(), make(sandbox_memory_mb=300)
    for svc in (a, b, c):
        assert (await svc.run_program("1")).success
    assert a._process_runner is b._process_runner
    assert c._process_runner is not a._process_runner
    assert c._process_runner.memory_mb == 300


async def test_close_keeps_the_shared_runner(make):
    """A service's close() does not stop the zygote that other services share."""
    a, b = make(), make()
    assert (await a.run_program("1")).success
    zygote = a._process_runner._proc
    a.close()
    a.close()  # idempotent
    assert zygote.poll() is None
    assert (await b.run_program("1")).success
    assert b._process_runner._proc is zygote


async def test_close_shared_runners_stops_the_zygotes(make):
    from lackpy.run.process import close_shared_runners
    svc = make()
    assert (await svc.run_program("1")).success
    zygote = svc._process_runner._proc
    close_shared_runners()
    assert zygote.poll() is not None
    assert (await svc.run_program("1")).success  # a later run starts a new one


def test_shared_runners_are_closed_at_exit():
    """The shared runners are closed by an atexit handler."""
    import subprocess
    script = (
        "import atexit\n"
        "seen = []\n"
        "real = atexit.register\n"
        "atexit.register = lambda f, *a, **k: (seen.append(f), real(f, *a, **k))[1]\n"
        "import lackpy.run.process as p\n"
        "print(p.close_shared_runners in seen)\n")
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         timeout=30)
    assert out.stdout.strip() == "True", out.stderr


async def test_no_fork_is_a_failed_result(make, monkeypatch):
    """Without os.fork the run fails loudly, as a result, never in-process."""
    from lackpy.run.process import close_shared_runners
    close_shared_runners()  # a zygote that is already running would fork for us
    svc = make()
    monkeypatch.delattr(os, "fork")
    result = await svc.run_program("1")
    assert result.success is False
    assert result.error == "the process runner needs os.fork (Linux)"


async def test_memory_cap_is_the_config_value(make):
    svc = make(sandbox_memory_mb=300)
    assert (await svc.run_program("1")).success
    assert svc._process_runner.memory_mb == 300


async def test_literate_runs_in_a_session_worker(make, ws):
    """Task 6 moved literate execution into a session worker of the shared runner (D6)."""
    svc = make()
    doc = "Some prose.\n\n```lackpy\nx = 1 + 1\nx\n```\n"
    result = await svc.run_program(doc, profile=Profile(execution="literate"))
    assert result.success, result.error
    assert result.output == 2
    assert svc._process_runner is not None
    assert svc._process_runner.last_pid not in (None, os.getpid())


async def test_delegate_bounds_the_literate_checkpoints(make, monkeypatch):
    """Final review I1: delegate's literate session takes the registry's checkpoint
    bounds, as run_program's does; with none, its checkpoints were unbounded."""
    seen: list[dict] = []
    real = ProcessRunner.open_session

    def spy(self, **kw):
        seen.append(kw)
        return real(self, **kw)

    monkeypatch.setattr(ProcessRunner, "open_session", spy)
    svc = make()
    doc = "Some prose.\n\n```lackpy\nx = 1 + 1\nx\n```\n"
    result = await svc.delegate("x", profile=Profile(execution="literate"), _program_override=doc)
    assert result["success"], result
    assert len(seen) == 1
    assert seen[0].get("max_checkpoints") == svc._sessions.max_checkpoints == 32
    assert seen[0].get("max_checkpoint_mb") == svc._sessions.max_checkpoint_mb == 1024
