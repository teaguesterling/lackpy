"""A cell with ``rerun=no`` runs at most once in a session (spec E4).

Once such a cell has run in a ``LiterateRun``, whether it succeeded or failed,
a submit whose start (a resume, or a fresh run) is at or before it is refused
before it runs anything. ``rerun=yes`` is the default.

Every run has a cell limit, and the autouse ``_hang_guard`` exits a test that
still hangs, so a regression fails rather than hangs.
"""

from __future__ import annotations

import faulthandler
import sys

import pytest

from lackpy.run.paths import load_policy
from lackpy.run.process import ProcessRunner

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="the session worker needs Linux (fork, pidfd)")

# One more line in log.txt, then print "write".
WRITE = ("from pathlib import Path\np = Path('log.txt')\np.write_text(p.read_text() + 'line\\n')\n"
         "print('write')")


@pytest.fixture(autouse=True)
def _hang_guard():
    faulthandler.dump_traceback_later(90, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


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


def _doc(*cells: str) -> str:
    """Code cells. A cell that starts with ``"no: "`` has ``rerun=no``; ``"> "`` is prose."""
    parts = []
    for c in cells:
        if c.startswith("> "):
            parts.append(c[2:])
        elif c.startswith("no: "):
            parts.append(f"```lackpy rerun=no\n{c[4:]}\n```")
        else:
            parts.append(f"```lackpy\n{c}\n```")
    return "\n\n".join(parts) + "\n"


def _lines(env) -> int:
    return (env["ws"] / "log.txt").read_text().count("line")


def _literate_run(env, runner, **kw):
    from lackpy.run.sessions import LiterateRun

    kw.setdefault("cell_timeout", 5.0)
    return LiterateRun(runner, workspace=env["ws"], profile_name="lit",
                       path_policy=env["loader"](env["ws"]), **kw)


def test_rerun_no_refuses_a_resume_after_a_later_failure(env, runner):
    """The case of spec E4: the write cell ran, a later cell failed, and the next
    submit changes a cell before the write cell. The submit is refused."""
    run = _literate_run(env, runner)
    try:
        first = run.submit(_doc("a = 1", "no: " + WRITE, "1/0"), None)
        assert first.error.startswith("cell 2 (1/0): "), first
        second = run.submit(_doc("a = 2", "no: " + WRITE, "1/0"), None)
    finally:
        run.close()
    assert not second.success, second
    assert second.error_phase == "refused", second
    assert second.error == ("cell 1 (from pathlib import Path) has rerun=no and has already "
                            "run in this session; start a new session to run it again"), second
    assert first.stdout.count("write") + second.stdout.count("write") == 1
    assert _lines(env) == 1


def test_rerun_no_allows_a_resume_after_it(env, runner):
    """A change after the ``rerun=no`` cell resumes from the checkpoint before cell 2."""
    run = _literate_run(env, runner)
    try:
        assert not run.submit(_doc("a = 1", "no: " + WRITE, "1/0"), None).success
        second = run.submit(_doc("a = 1", "no: " + WRITE, "c = a + 1"), None)
    finally:
        run.close()
    assert second.success, second
    assert second.variables.get("c") == 2, second
    assert _lines(env) == 1  # cells 0 and 1 did not run again


def test_rerun_no_refuses_after_its_own_failure(env, runner):
    """A ``rerun=no`` cell that failed has run: its fix may not run it again."""
    run = _literate_run(env, runner)
    try:
        first = run.submit(_doc("a = 1", "no: " + WRITE + "\n1/0"), None)
        assert first.error.startswith("cell 1 (from pathlib import Path): "), first
        second = run.submit(_doc("a = 1", "no: " + WRITE), None)
    finally:
        run.close()
    assert second.error_phase == "refused", second
    assert "rerun=no" in second.error, second
    assert _lines(env) == 1


def test_rerun_no_refuses_a_fresh_run_in_the_same_session(env, runner):
    """Other params make the next submit run fresh. A fresh run starts at cell 0,
    so it is refused too (Review Focus 5)."""
    run = _literate_run(env, runner)
    try:
        doc = _doc("params = {'n': 1}", "no: " + WRITE)
        assert run.submit(doc, {"n": 1}).success
        second = run.submit(doc, {"n": 2})
    finally:
        run.close()
    assert second.error_phase == "refused", second
    assert "rerun=no" in second.error, second
    assert _lines(env) == 1


def test_rerun_no_after_prose_is_recorded(env, runner):
    """The prose before a ``rerun=no`` cell shares its checkpoint (one ``cell_start``
    at the prose's index). The cell is still recorded as run."""
    run = _literate_run(env, runner)
    try:
        assert not run.submit(_doc("a = 1", "> Now the write.", "no: " + WRITE, "1/0"), None).success
        second = run.submit(_doc("a = 2", "> Now the write.", "no: " + WRITE, "1/0"), None)
    finally:
        run.close()
    assert second.error_phase == "refused", second
    assert second.error.startswith("cell 2 (from pathlib import Path) has rerun=no"), second
    assert _lines(env) == 1


def test_a_refused_submit_keeps_the_session(env, runner):
    """A refusal runs nothing and changes nothing: the next submit compares with the
    document the session last ran, and resumes after the ``rerun=no`` cell."""
    run = _literate_run(env, runner)
    try:
        assert not run.submit(_doc("a = 1", "no: " + WRITE, "1/0"), None).success
        assert run.submit(_doc("a = 2", "no: " + WRITE, "1/0"), None).error_phase == "refused"
        third = run.submit(_doc("a = 1", "no: " + WRITE, "c = a + 1"), None)
    finally:
        run.close()
    assert third.success, third
    assert third.variables.get("c") == 2, third
    assert _lines(env) == 1


def test_rerun_yes_is_the_default(env, runner):
    """With no option, a resume from before the cell runs it again."""
    run = _literate_run(env, runner)
    try:
        assert not run.submit(_doc("a = 1", WRITE, "1/0"), None).success
        second = run.submit(_doc("a = 2", WRITE, "1/0"), None)
    finally:
        run.close()
    assert second.error_phase != "refused", second
    assert second.error.startswith("cell 2 (1/0): "), second
    assert _lines(env) == 2


def test_a_bad_rerun_value_is_a_validation_error(env, runner):
    run = _literate_run(env, runner)
    try:
        r = run.submit(_doc(WRITE).replace("```lackpy\n", "```lackpy rerun=maybe\n"), None)
    finally:
        run.close()
    assert not r.success, r
    assert "rerun='maybe' is not valid: use rerun=yes or rerun=no" in r.error, r
    assert _lines(env) == 0  # nothing ran


def test_the_parser_checks_the_rerun_value():
    from lackpy.interpreters.literate.parser import parse

    assert parse("```lackpy rerun=no\nx = 1\n```\n").errors == []
    assert parse("```lackpy rerun=yes\nx = 1\n```\n").errors == []
    errors = parse("```lackpy rerun=maybe\nx = 1\n```\n").errors
    assert len(errors) == 1 and "rerun='maybe' is not valid" in errors[0], errors
