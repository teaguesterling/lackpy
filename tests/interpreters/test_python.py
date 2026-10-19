"""Tests for the PythonInterpreter plugin.

These tests mirror the run_program behavior in LackpyService: the
PythonInterpreter shares RestrictedRunner, the runtime namespace and the
workspace PathPolicy, and binds no tool callables (stepy100 Plan 2, Task 7).
"""

import json

import pytest

from lackpy.interpreters import (
    ExecutionContext,
    PythonInterpreter,
    run_interpreter,
)
from lackpy.lang.grader import Grade
from lackpy.run.paths import load_policy
from lackpy.service import KIT_REMOVED
from lackpy.tools.registry import ResolvedTools
from lackpy.tools.toolbox import ArgSpec, ToolSpec


def _make_kit():
    """A ResolvedTools with one callable tool: the retired kit model."""
    def _count(items):
        return len(items)

    tools = {
        "count": ToolSpec(
            name="count", provider="builtin", description="Count items",
            args=[ArgSpec(name="items", type="list")], returns="int",
            grade_w=1, effects_ceiling=1,
        ),
    }
    return ResolvedTools(tools=tools, callables={"count": _count},
                         grade=Grade(w=1, d=1), description="count(items) -> int")


@pytest.fixture
def ws(tmp_path):
    w = tmp_path / "ws"
    w.mkdir()
    (w / "a.txt").write_text("hello")
    return w


@pytest.fixture
def interp(tmp_path):
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    return PythonInterpreter(path_policy_loader=lambda w: load_policy(w, home=home, managed=None))


class TestPythonValidation:
    def test_valid_program_passes(self, interp):
        result = interp.validate("len([1, 2, 3])", ExecutionContext())
        assert result.valid
        assert result.errors == []

    def test_no_tools_needed(self, interp):
        # Retired kit model: a context without a resolved kit was invalid.
        result = interp.validate("1 + 1", ExecutionContext(tools=None))
        assert result.valid, result.errors

    def test_context_naming_tools_is_an_error(self, interp):
        result = interp.validate("count([1])", ExecutionContext(tools=_make_kit()))
        assert not result.valid
        assert result.errors == [KIT_REMOVED]

    def test_import_rejected(self, interp):
        result = interp.validate("import os", ExecutionContext())
        assert not result.valid

    def test_unknown_name_rejected(self, interp):
        result = interp.validate("delete_everything()", ExecutionContext())
        assert not result.valid


class TestPythonExecution:
    @pytest.mark.asyncio
    async def test_execute_simple_program(self, interp, ws):
        result = await run_interpreter(interp, "len([1, 2, 3])", ExecutionContext(base_dir=ws))
        assert result.success, result.error
        assert result.output == 3
        assert result.output_format == "python"

    @pytest.mark.asyncio
    async def test_execute_reads_through_the_path_policy(self, interp, ws):
        program = "from pathlib import Path\nPath('a.txt').read_text()"
        result = await run_interpreter(interp, program, ExecutionContext(base_dir=ws))
        assert result.success, result.error
        assert result.output == "hello"
        assert [e.tool for e in result.metadata["trace"].entries] == ["Path.read_text"]

    @pytest.mark.asyncio
    async def test_path_policy_is_loaded_from_the_base_dir(self, interp, ws):
        (ws / ".claude").mkdir()
        (ws / ".claude" / "settings.json").write_text(
            json.dumps({"permissions": {"deny": ["Read(./a.txt)"]}}))
        program = "from pathlib import Path\nPath('a.txt').read_text()"
        result = await run_interpreter(interp, program, ExecutionContext(base_dir=ws))
        assert not result.success
        assert "not allowed in lackpy" in result.error

    @pytest.mark.asyncio
    async def test_malformed_settings_is_a_failed_result(self, interp, ws):
        settings = ws / ".claude" / "settings.json"
        settings.parent.mkdir()
        settings.write_text("{")
        result = await run_interpreter(interp, "x = 1", ExecutionContext(base_dir=ws))
        assert not result.success
        assert str(settings) in result.error

    @pytest.mark.asyncio
    async def test_any_loader_exception_is_a_failed_result(self, ws):
        # Review M4: the loader is host code; whatever it raises is a failed result.
        def loader(base_dir):
            raise TypeError("boom")
        result = await run_interpreter(PythonInterpreter(path_policy_loader=loader), "x = 1",
                                       ExecutionContext(base_dir=ws))
        assert not result.success
        assert "TypeError" in result.error

    @pytest.mark.asyncio
    async def test_a_refusal_from_the_loader_is_a_failed_result(self, ws):
        # Review I4: LackpyRefusal is a BaseException; `except Exception` let it out.
        from lackpy.run.refusal import LackpyRefusal

        def loader(base_dir):
            raise LackpyRefusal("access to loop")
        result = await run_interpreter(PythonInterpreter(path_policy_loader=loader), "x = 1",
                                       ExecutionContext(base_dir=ws))
        assert not result.success
        assert "access to loop" in result.error

    @pytest.mark.asyncio
    async def test_a_symlink_loop_in_a_deny_rule_is_a_failed_result(self, interp, ws):
        (ws / "loop").symlink_to(ws / "loop")
        (ws / ".claude").mkdir()
        (ws / ".claude" / "settings.json").write_text(
            json.dumps({"permissions": {"deny": ["Read(./loop/secret)"]}}))
        result = await run_interpreter(interp, "x = 1\nx", ExecutionContext(base_dir=ws))
        assert not result.success
        assert "Read(./loop/secret)" in result.error

    @pytest.mark.asyncio
    async def test_deeply_nested_settings_name_the_file(self, interp, ws):
        settings = ws / ".claude" / "settings.json"
        settings.parent.mkdir()
        settings.write_text('{"permissions": {"deny": ' + "[" * 100000 + "]" * 100000 + '}}')
        result = await run_interpreter(interp, "x = 1", ExecutionContext(base_dir=ws))
        assert not result.success
        assert str(settings) in result.error

    @pytest.mark.asyncio
    async def test_context_naming_tools_fails(self, interp, ws):
        result = await run_interpreter(interp, "count([])", ExecutionContext(tools=_make_kit(), base_dir=ws))
        assert not result.success
        assert KIT_REMOVED in result.error

    @pytest.mark.asyncio
    async def test_execute_with_params(self, interp, ws):
        ctx = ExecutionContext(base_dir=ws, params={"data": [1, 2, 3, 4, 5]})
        # Task 5: params replace the program's own params literal; no bare names.
        result = await run_interpreter(interp, 'params = {"data": []}\nlen(params["data"])', ctx)
        assert result.success, result.error
        assert result.output == 5

    @pytest.mark.asyncio
    async def test_failed_validation_short_circuits(self, interp, ws):
        result = await run_interpreter(interp, "import sys", ExecutionContext(base_dir=ws))
        assert not result.success
        assert "Validation failed" in result.error

    @pytest.mark.asyncio
    async def test_result_has_duration(self, interp, ws):
        result = await run_interpreter(interp, "len([])", ExecutionContext(base_dir=ws))
        assert result.success
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_result_metadata_includes_trace(self, interp, ws):
        result = await run_interpreter(interp, "len([1, 2])", ExecutionContext(base_dir=ws))
        assert result.success
        assert "trace" in result.metadata


class TestSystemPromptHint:
    """stepy100 Task 8 (Task 7 review M3): the hint frames lackpy programs. The
    kit-era hint said "Use read_file()", "ORCHESTRATE, DO NOT IMPLEMENT" and
    "Do NOT define functions, classes, or use import"; the pre-prompt replaces them."""

    def test_hint_drops_the_kit_framing(self):
        hint = PythonInterpreter().system_prompt_hint()
        for phrase in ("read_file", "ORCHESTRATE", "Do NOT define functions",
                       "use import", "pre-loaded", "tool"):
            assert phrase not in hint, phrase

    def test_hint_names_the_language(self):
        hint = PythonInterpreter().system_prompt_hint()
        assert "lackpy" in hint
        assert "Python 3" in hint
