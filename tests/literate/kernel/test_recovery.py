"""Tests for recovery handler protocol."""

import pytest

from lackpy.interpreters.literate.kernel.recovery import (
    NoRecoveryHandler,
    RecoveryAction,
    RecoveryContext,
)
from lackpy.interpreters.literate.kernel.plugins import PluginAdvice
from lackpy.interpreters.literate.parser import Cell


def _make_context(**kwargs) -> RecoveryContext:
    defaults = {
        "failed_cell": Cell(cell_type="code", content="x = foo"),
        "error": "undefined names: 'foo'",
        "error_phase": "static",
        "scope": {"y": "int: 42"},
        "cell_index": 3,
        "prior_output": "Some output\n",
        "attempt": 0,
        "plugin_advice": None,
    }
    defaults.update(kwargs)
    return RecoveryContext(**defaults)


class TestRecoveryAction:
    def test_fix_action(self):
        cells = [Cell(cell_type="hidden", content="foo = 1")]
        action = RecoveryAction(kind="fix", cells=cells)
        assert action.kind == "fix"
        assert len(action.cells) == 1

    def test_inspect_action(self):
        action = RecoveryAction(kind="inspect", expr="type(x)")
        assert action.kind == "inspect"
        assert action.expr == "type(x)"

    def test_abort_action(self):
        action = RecoveryAction(kind="abort")
        assert action.kind == "abort"

    def test_skip_action(self):
        action = RecoveryAction(kind="skip")
        assert action.kind == "skip"


class TestNoRecoveryHandler:
    def test_always_aborts(self):
        handler = NoRecoveryHandler()
        ctx = _make_context()
        action = handler.on_cell_error(ctx)
        assert action.kind == "abort"

    def test_inspect_result_aborts(self):
        handler = NoRecoveryHandler()
        ctx = _make_context()
        action = handler.on_inspect_result(ctx, "42")
        assert action.kind == "abort"

    def test_max_attempts_is_zero(self):
        handler = NoRecoveryHandler()
        assert handler.max_attempts == 0


class TestRecoveryContext:
    def test_context_fields(self):
        ctx = _make_context(attempt=2)
        assert ctx.attempt == 2
        assert ctx.error == "undefined names: 'foo'"
        assert ctx.cell_index == 3

    def test_context_with_plugin_advice(self):
        advice = PluginAdvice(hints=["try importing foo"])
        ctx = _make_context(plugin_advice=advice)
        assert ctx.plugin_advice.hints == ["try importing foo"]
