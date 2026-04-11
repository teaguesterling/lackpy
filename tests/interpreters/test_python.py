"""Tests for the PythonInterpreter plugin.

These tests mirror the existing run_program behavior in LackpyService —
the PythonInterpreter should be a drop-in for the same execution path,
validated against the same kit and producing the same semantic output.
"""

import pytest
from pathlib import Path

from lackpy.interpreters import (
    ExecutionContext,
    PythonInterpreter,
    run_interpreter,
)
from lackpy.kit.registry import ResolvedKit
from lackpy.kit.toolbox import ArgSpec, ToolSpec
from lackpy.lang.grader import Grade


def _make_kit():
    """Build a minimal ResolvedKit with one callable tool for tests."""
    def _count(items):
        return len(items)

    tools = {
        "count": ToolSpec(
            name="count",
            provider="builtin",
            description="Count items",
            args=[ArgSpec(name="items", type="list")],
            returns="int",
            grade_w=1,
            effects_ceiling=1,
        ),
    }
    return ResolvedKit(
        tools=tools,
        callables={"count": _count},
        grade=Grade(w=1, d=1),
        description="count(items) -> int",
    )


class TestPythonValidation:
    def test_valid_program_passes(self):
        interp = PythonInterpreter()
        ctx = ExecutionContext(kit=_make_kit())
        result = interp.validate("count([1, 2, 3])", ctx)
        assert result.valid
        assert result.errors == []

    def test_missing_kit_fails(self):
        interp = PythonInterpreter()
        ctx = ExecutionContext(kit=None)
        result = interp.validate("1 + 1", ctx)
        assert not result.valid
        assert any("kit" in e for e in result.errors)

    def test_import_rejected(self):
        interp = PythonInterpreter()
        ctx = ExecutionContext(kit=_make_kit())
        result = interp.validate("import os", ctx)
        assert not result.valid

    def test_unknown_tool_rejected(self):
        interp = PythonInterpreter()
        ctx = ExecutionContext(kit=_make_kit())
        result = interp.validate("delete_everything()", ctx)
        assert not result.valid


class TestPythonExecution:
    @pytest.mark.asyncio
    async def test_execute_simple_program(self):
        interp = PythonInterpreter()
        ctx = ExecutionContext(kit=_make_kit())
        result = await run_interpreter(interp, "count([1, 2, 3])", ctx)
        assert result.success
        assert result.output == 3
        assert result.output_format == "python"

    @pytest.mark.asyncio
    async def test_execute_with_params(self):
        interp = PythonInterpreter()
        ctx = ExecutionContext(
            kit=_make_kit(),
            params={"data": [1, 2, 3, 4, 5]},
        )
        result = await run_interpreter(interp, "count(data)", ctx)
        assert result.success
        assert result.output == 5

    @pytest.mark.asyncio
    async def test_failed_validation_short_circuits(self):
        interp = PythonInterpreter()
        ctx = ExecutionContext(kit=_make_kit())
        result = await run_interpreter(interp, "import sys", ctx)
        assert not result.success
        assert "Validation failed" in result.error

    @pytest.mark.asyncio
    async def test_result_has_duration(self):
        interp = PythonInterpreter()
        ctx = ExecutionContext(kit=_make_kit())
        result = await run_interpreter(interp, "count([])", ctx)
        assert result.success
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_result_metadata_includes_trace(self):
        interp = PythonInterpreter()
        ctx = ExecutionContext(kit=_make_kit())
        result = await run_interpreter(interp, "count([1, 2])", ctx)
        assert result.success
        assert "trace" in result.metadata
