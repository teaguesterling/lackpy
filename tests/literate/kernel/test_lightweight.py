"""Tests for LightweightKernel - the exec-into-dict kernel backend."""

import pytest

from lackpy.interpreters.literate.kernel.lightweight import LightweightKernel
from lackpy.interpreters.literate.parser import Cell


@pytest.fixture
def kernel():
    return LightweightKernel()


class TestExecuteCell:
    def test_code_cell_executes(self, kernel):
        cell = Cell(cell_type="code", content="x = 42")
        result = kernel.execute_cell(cell, cell_index=0)
        assert result.success
        assert result.namespace_delta == {"x": 42}

    def test_prose_cell_produces_output(self, kernel):
        cell = Cell(cell_type="prose", content="Hello world")
        result = kernel.execute_cell(cell, cell_index=0)
        assert result.success
        assert "Hello world" in result.output

    def test_hidden_cell_no_output(self, kernel):
        cell = Cell(cell_type="hidden", content="secret = 99")
        result = kernel.execute_cell(cell, cell_index=0)
        assert result.success
        assert result.output == ""
        assert result.namespace_delta == {"secret": 99}

    def test_state_persists_across_cells(self, kernel):
        kernel.execute_cell(Cell(cell_type="hidden", content="x = 10"), cell_index=0)
        result = kernel.execute_cell(
            Cell(cell_type="prose", content="Value: {x}"), cell_index=1
        )
        assert result.success
        assert "10" in result.output

    def test_syntax_error_caught_statically(self, kernel):
        cell = Cell(cell_type="code", content="x = ")
        result = kernel.execute_cell(cell, cell_index=0)
        assert not result.success
        assert result.error_phase == "static"

    def test_undefined_name_caught_statically(self, kernel):
        cell = Cell(cell_type="code", content="print(undefined_var)")
        result = kernel.execute_cell(cell, cell_index=0)
        assert not result.success
        assert result.error_phase == "static"

    def test_runtime_error_caught(self, kernel):
        cell = Cell(cell_type="code", content="x = 1 / 0")
        result = kernel.execute_cell(cell, cell_index=0)
        assert not result.success
        assert result.error_phase == "runtime"
        assert "ZeroDivisionError" in result.error

    def test_write_cell_calls_write_file(self, kernel):
        written = {}
        kernel._namespace["write_file"] = lambda p, c: written.update({p: c})
        cell = Cell(
            cell_type="write",
            content="print('hi')",
            annotation_args={"path": "out.py"},
        )
        result = kernel.execute_cell(cell, cell_index=0)
        assert result.success
        assert written["out.py"] == "print('hi')"

    def test_continue_cell_sets_flag(self, kernel):
        cell = Cell(cell_type="continue", content="")
        result = kernel.execute_cell(cell, cell_index=0)
        assert result.success
        assert result.namespace_delta.get("__continue_requested__") is True


class TestInspect:
    def test_inspect_variable(self, kernel):
        kernel.execute_cell(Cell(cell_type="hidden", content="x = [1, 2, 3]"), cell_index=0)
        assert kernel.inspect("x") == "[1, 2, 3]"

    def test_inspect_expression(self, kernel):
        kernel.execute_cell(Cell(cell_type="hidden", content="x = 42"), cell_index=0)
        assert kernel.inspect("x * 2") == "84"

    def test_inspect_error(self, kernel):
        result = kernel.inspect("undefined_var")
        assert "NameError" in result


class TestScope:
    def test_get_scope_empty(self, kernel):
        scope = kernel.get_scope()
        assert scope == {}

    def test_get_scope_after_execution(self, kernel):
        kernel.execute_cell(Cell(cell_type="hidden", content="x = 42\ny = 'hi'"), cell_index=0)
        scope = kernel.get_scope()
        assert "x" in scope
        assert "int" in scope["x"]
        assert "y" in scope
        assert "str" in scope["y"]


class TestRestart:
    def test_restart_clears_state(self, kernel):
        kernel.execute_cell(Cell(cell_type="hidden", content="x = 42"), cell_index=0)
        kernel.restart()
        scope = kernel.get_scope()
        assert "x" not in scope
