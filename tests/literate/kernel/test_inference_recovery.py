"""Tests for InferenceRecoveryHandler - model-based recovery."""

import pytest

from lackpy.interpreters.literate.kernel.recovery import (
    InferenceRecoveryHandler,
    RecoveryAction,
    RecoveryContext,
)
from lackpy.interpreters.literate.kernel.plugins import PluginAdvice
from lackpy.interpreters.literate.parser import Cell


def _make_context(**kwargs) -> RecoveryContext:
    defaults = {
        "failed_cell": Cell(cell_type="prose", content="Value: {x:.1f if x > 0 else y:.1f}"),
        "error": "Invalid format specifier",
        "error_phase": "runtime",
        "scope": {"x": "float: 3.14", "y": "float: 2.71"},
        "cell_index": 5,
        "prior_output": "Previous output\n",
        "attempt": 0,
        "plugin_advice": None,
    }
    defaults.update(kwargs)
    return RecoveryContext(**defaults)


class TestInferenceRecoveryHandler:
    def test_fix_response_parsed(self):
        async def fake_infer(prompt: str) -> str:
            return "```lackpy @hidden\nval = x if x > 0 else y\n```\n\nValue: {val:.1f}"

        handler = InferenceRecoveryHandler(infer_fn=fake_infer, max_attempts=2)
        ctx = _make_context()
        action = handler.on_cell_error(ctx)
        assert action.kind == "fix"
        assert len(action.cells) == 2
        assert action.cells[0].cell_type == "hidden"
        assert action.cells[1].cell_type == "prose"

    def test_scratch_response_becomes_inspect(self):
        async def fake_infer(prompt: str) -> str:
            return "```lackpy @scratch\ntype(x)\n```"

        handler = InferenceRecoveryHandler(infer_fn=fake_infer, max_attempts=2)
        ctx = _make_context()
        action = handler.on_cell_error(ctx)
        assert action.kind == "inspect"
        assert "type(x)" in action.expr

    def test_empty_response_aborts(self):
        async def fake_infer(prompt: str) -> str:
            return ""

        handler = InferenceRecoveryHandler(infer_fn=fake_infer, max_attempts=2)
        ctx = _make_context()
        action = handler.on_cell_error(ctx)
        assert action.kind == "abort"

    def test_plugin_advice_included_in_prompt(self):
        prompts_seen: list[str] = []

        async def fake_infer(prompt: str) -> str:
            prompts_seen.append(prompt)
            return "```lackpy @hidden\nfoo = 1\n```"

        advice = PluginAdvice(hints=["Try importing the module"], doc_context=["see docs/x.md"])
        handler = InferenceRecoveryHandler(infer_fn=fake_infer, max_attempts=2)
        ctx = _make_context(plugin_advice=advice)
        handler.on_cell_error(ctx)
        assert "Try importing the module" in prompts_seen[0]

    def test_inspect_result_fed_back(self):
        call_count = [0]

        async def fake_infer(prompt: str) -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                return "```lackpy @scratch\ntype(x)\n```"
            return "```lackpy @hidden\nval = float(x)\n```\n\nValue: {val:.1f}"

        handler = InferenceRecoveryHandler(infer_fn=fake_infer, max_attempts=3)
        ctx = _make_context()
        action = handler.on_cell_error(ctx)
        assert action.kind == "inspect"

        action = handler.on_inspect_result(ctx, "'float'")
        assert action.kind == "fix"
        assert len(action.cells) == 2
