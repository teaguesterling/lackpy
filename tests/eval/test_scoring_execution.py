"""Tests for stage 2 execution scoring."""

from pathlib import Path

from scripts.prompt_eval.intents import GateResult, Intent
from scripts.prompt_eval.scoring import run_execution, score_cell


TOYBOX = Path(__file__).parent / "fixtures" / "toybox"


def _assert_true(x):
    return True


def _assert_contains(needle):
    def check(x):
        return needle in str(x)
    return check


def test_python_execution_passes_assertion():
    """A known-good python program should run, return files, and pass a contains-assertion."""
    intent = Intent(
        id="pyexec.01",
        interpreter="python",
        difficulty="core",
        text="find a file",
        return_shape="list[str]",
        structural_gate=lambda p: GateResult(passed=True),
        exec_assertion=_assert_contains("app.py"),
    )
    program = "files = find_files('*.py')\nfiles"
    exec_res = run_execution(intent, program, toybox_dir=TOYBOX)
    assert exec_res.success, f"execution failed: {exec_res.error}"
    assert intent.exec_assertion(exec_res.output)


def test_score_cell_returns_0_on_gate_fail():
    intent = Intent(
        id="pyexec.02",
        interpreter="python",
        difficulty="core",
        text="x",
        return_shape="str",
        structural_gate=lambda p: GateResult(passed=False, errors=["bad"]),
        exec_assertion=_assert_true,
    )
    score = score_cell(intent, raw_generation="anything", toybox_dir=TOYBOX)
    assert score.score == 0
    assert not score.executed
    assert not score.gate.passed


def test_score_cell_returns_1_when_assertion_fails():
    intent = Intent(
        id="pyexec.03",
        interpreter="python",
        difficulty="core",
        text="x",
        return_shape="str",
        structural_gate=lambda p: GateResult(passed=True),
        exec_assertion=lambda x: False,
    )
    score = score_cell(intent, raw_generation="x = 1\nx", toybox_dir=TOYBOX)
    assert score.executed
    assert score.score == 1
    assert not score.assertion_passed


def test_score_cell_returns_2_on_full_pass():
    intent = Intent(
        id="pyexec.04",
        interpreter="python",
        difficulty="core",
        text="x",
        return_shape="int",
        structural_gate=lambda p: GateResult(passed=True),
        exec_assertion=lambda x: x == 42,
    )
    score = score_cell(intent, raw_generation="42", toybox_dir=TOYBOX)
    assert score.executed
    assert score.assertion_passed
    assert score.score == 2


def test_score_cell_catches_assertion_exception():
    """If the assertion callable raises, score_cell should treat it as a score-1 failure, not propagate."""
    intent = Intent(
        id="pyexec.05",
        interpreter="python",
        difficulty="core",
        text="x",
        return_shape="int",
        structural_gate=lambda p: GateResult(passed=True),
        exec_assertion=lambda x: 1 / 0,  # boom
    )
    score = score_cell(intent, raw_generation="42", toybox_dir=TOYBOX)
    assert score.executed
    assert score.score == 1
    assert score.exec_error is not None
    assert "ZeroDivisionError" in score.exec_error or "zero" in score.exec_error.lower()
