"""Spec E11: ``printed`` is what the cells (or the program) printed, and nothing else.

A document's prose and rendered output are never in it. What a failed cell
printed before it failed is. ``stdout`` is unchanged: it stays the rendered
output of a document.
"""

from __future__ import annotations

import faulthandler
import sys

import pytest

from lackpy.interpreters.base import ExecutionContext
from lackpy.interpreters.literate import LiterateInterpreter
from lackpy.run.paths import load_policy
from lackpy.run.process import ProcessRunner
from lackpy.run.runner import RestrictedRunner
from lackpy.run.sessions import LiterateRun

needs_linux = pytest.mark.skipif(sys.platform != "linux",
                                 reason="the session worker needs Linux (fork, pidfd)")


@pytest.fixture(autouse=True)
def _hang_guard():
    faulthandler.dump_traceback_later(90, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


PROSE_DOC = '# Title\n\n```lackpy\nprint("a")\n```\n\n```lackpy\nx = 2\n```\n\n{x}\n'
FAILING_DOC = ('```lackpy\nprint("one")\n```\n\n```lackpy\nprint("two"); 1/0\n```\n\n'
               '```lackpy\nprint("three")\n```\n')


@pytest.fixture
def home(tmp_path):
    h = tmp_path / "home"
    (h / ".claude").mkdir(parents=True)
    return h


async def _interpret(tmp_path, home, doc):
    ctx = ExecutionContext(base_dir=tmp_path, config={
        "path_policy": load_policy(tmp_path, home=home, managed=None)})
    return await LiterateInterpreter().execute(doc, ctx)


@pytest.mark.asyncio
async def test_printed_holds_only_cell_prints(tmp_path, home):
    result = await _interpret(tmp_path, home, PROSE_DOC)
    assert result.success, result.error
    assert result.metadata["printed"] == "a\n"
    assert "Title" in result.output  # the rendered output still has the prose


@pytest.mark.asyncio
async def test_printed_keeps_the_failed_cells_output(tmp_path, home):
    result = await _interpret(tmp_path, home, FAILING_DOC)
    assert not result.success
    assert result.metadata["printed"] == "one\ntwo\n"


@needs_linux
@pytest.mark.parametrize("doc, printed, success", [
    (PROSE_DOC, "a\n", True),
    (FAILING_DOC, "one\ntwo\n", False),
])
def test_printed_crosses_the_session_worker(tmp_path, home, doc, printed, success):
    ws = tmp_path / "ws"
    ws.mkdir()
    runner = ProcessRunner()
    run = LiterateRun(runner, workspace=ws, profile_name="lit", cell_timeout=5.0,
                      path_policy=load_policy(ws, home=home, managed=None))
    try:
        result = run.submit(doc, None)
    finally:
        run.close()
        runner.close()
    assert result.success is success, result
    assert result.printed == printed
    assert "Title" not in result.printed
    if success:
        assert "Title" in result.stdout  # stdout is unchanged: the rendered document


def test_program_printed_equals_stdout():
    result = RestrictedRunner().run('print("hi")')
    assert result.printed == result.stdout == "hi\n"


@needs_linux
def test_program_printed_equals_stdout_across_the_process_runner(tmp_path, home):
    ws = tmp_path / "ws"
    ws.mkdir()
    runner = ProcessRunner()
    try:
        result = runner.run('print("hi")', params=None, workspace=ws, timeout=10,
                            path_policy=load_policy(ws, home=home, managed=None))
    finally:
        runner.close()
    assert result.success, result
    assert result.printed == result.stdout == "hi\n"


@needs_linux
def test_printed_survives_the_too_large_failure_fallback(tmp_path, home):
    """The fallback leaves out the variables of a failed document, not what it printed."""
    ws = tmp_path / "ws"
    ws.mkdir()
    doc = ('```lackpy\nprint("kept")\n```\n\n'
           '```lackpy\nbig = list(range(1000000, 3800000))\n```\n\n'
           '```lackpy\n1/0\n```\n')
    runner = ProcessRunner()
    run = LiterateRun(runner, workspace=ws, profile_name="lit", cell_timeout=10.0,
                      path_policy=load_policy(ws, home=home, managed=None))
    try:
        result = run.submit(doc, None)
    finally:
        run.close()
        runner.close()
    assert not result.success
    assert "big" not in result.variables
    assert result.printed == "kept\n"


TIMEOUT_DOC = ('```lackpy\nprint("one")\n```\n\n```lackpy\nprint("two")\n```\n\n'
               '```lackpy\nfor i in iter(int, 1): pass\n```\n')


@needs_linux
def test_printed_keeps_the_cells_before_a_timeout(tmp_path, home):
    """Task 5 review ruling: a timed-out document keeps what the earlier cells printed, in order."""
    ws = tmp_path / "ws"
    ws.mkdir()
    runner = ProcessRunner()
    run = LiterateRun(runner, workspace=ws, profile_name="lit", cell_timeout=3.0,
                      path_policy=load_policy(ws, home=home, managed=None))
    try:
        result = run.submit(TIMEOUT_DOC, None)
    finally:
        run.close()
        runner.close()
    assert not result.success
    assert result.error_phase == "timeout", result
    assert result.printed == "one\ntwo\n"


@needs_linux
def test_printed_keeps_the_timed_out_cells_own_prints(tmp_path, home):
    """Final review 4 (spec E11): the stopped cell's prints follow the earlier cells'."""
    ws = tmp_path / "ws"
    ws.mkdir()
    doc = ('```lackpy\nprint("before")\n```\n\n'
           '```lackpy\nprint("partial"); print("more", end="")\nfor i in iter(int, 1): pass\n```\n\n'
           '```lackpy\nprint("after")\n```\n')
    runner = ProcessRunner()
    run = LiterateRun(runner, workspace=ws, profile_name="lit", cell_timeout=2.0,
                      path_policy=load_policy(ws, home=home, managed=None))
    try:
        result = run.submit(doc, None)
    finally:
        run.close()
        runner.close()
    assert result.error_phase == "timeout", result
    assert result.printed == "before\npartial\nmore"


@needs_linux
def test_printed_on_a_resume_keeps_the_earlier_cells_prints(tmp_path, home):
    """Final review 8: a resume's ``printed`` holds what the cells before the resume point
    printed in the earlier submit (the checkpoint kept it), then the resumed cells' prints."""
    ws = tmp_path / "ws"
    ws.mkdir()
    runner = ProcessRunner()
    run = LiterateRun(runner, workspace=ws, profile_name="lit", cell_timeout=5.0,
                      path_policy=load_policy(ws, home=home, managed=None))
    try:
        first = run.submit(FAILING_DOC, None)
        channel = run._channel
        fixed = run.submit(FAILING_DOC.replace("; 1/0", ""), None)
        resumed = run._channel is channel  # a fresh run opens a new channel
    finally:
        run.close()
        runner.close()
    assert not first.success
    assert fixed.success, fixed
    assert resumed
    assert fixed.printed == "one\ntwo\nthree\n"
