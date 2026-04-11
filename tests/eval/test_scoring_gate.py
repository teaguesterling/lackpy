"""Tests for stage 1 structural gating."""

from scripts.prompt_eval.intents import GateResult, Intent
from scripts.prompt_eval.scoring import CellScore, run_gate


def _dummy_intent(gate_impl):
    return Intent(
        id="test.01",
        interpreter="python",
        difficulty="core",
        text="test",
        return_shape="str",
        structural_gate=gate_impl,
        exec_assertion=lambda x: True,
    )


def _pass_gate(p: str) -> GateResult:
    return GateResult(passed=True)


def _fail_gate(p: str) -> GateResult:
    return GateResult(passed=False, errors=["nope"])


def test_run_gate_strips_code_fences():
    raw = "```python\nfiles = find_files('*.py')\nfiles\n```"
    sanitized, gr = run_gate(_dummy_intent(_pass_gate), raw)
    assert "```" not in sanitized
    assert "find_files" in sanitized
    assert gr.passed


def test_run_gate_passes_sanitized_to_intent_gate():
    called_with = []
    def capture(p: str) -> GateResult:
        called_with.append(p)
        return GateResult(passed=True)
    raw = "```\nhello\n```"
    run_gate(_dummy_intent(capture), raw)
    assert len(called_with) == 1
    assert "```" not in called_with[0]


def test_run_gate_returns_gate_failure():
    sanitized, gr = run_gate(_dummy_intent(_fail_gate), "anything")
    assert not gr.passed
    assert gr.errors == ["nope"]


def test_run_gate_handles_empty_raw():
    sanitized, gr = run_gate(_dummy_intent(_fail_gate), "")
    assert sanitized == ""
    assert not gr.passed


def test_cell_score_default_values():
    gr = GateResult(passed=False)
    cs = CellScore(raw_generation="x", sanitized_program="x", gate=gr)
    assert cs.executed is False
    assert cs.exec_output is None
    assert cs.exec_error is None
    assert cs.assertion_passed is False
    assert cs.score == 0
    assert cs.duration_ms_execution == 0.0
    assert cs.interpreter_metadata == {}
