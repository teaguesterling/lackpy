"""Tests for LightweightKernel: every cell runs through a CellSession (literate stepy Task 2)."""

import pytest

from lackpy.interpreters.literate.kernel.forgiveness import ErrorValue, Hole
from lackpy.interpreters.literate.kernel.lightweight import LightweightKernel
from lackpy.interpreters.literate.parser import Cell
from lackpy.run.cells import CellSession

from ..conftest import make_kernel


@pytest.fixture
def kernel():
    return make_kernel()


class TestConstruction:
    def test_the_kernel_takes_a_session_and_nothing_else(self):
        with pytest.raises(TypeError):
            LightweightKernel(namespace={})  # the host namespace argument is gone
        with pytest.raises(TypeError):
            LightweightKernel()  # a session is required
        session = CellSession(path_policy=None)
        assert LightweightKernel(session).session is session

    def test_the_host_exec_path_is_gone(self):
        from lackpy.interpreters.literate.kernel import lightweight

        assert not hasattr(lightweight, "_do_exec")
        assert not hasattr(lightweight, "_cwd")


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
        assert result.output == "Hello world\n"

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
        assert result.output == "Value: 10\n"

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

    def test_a_cell_the_validator_rejects_fails_in_phase_validate(self, kernel):
        result = kernel.execute_cell(Cell(cell_type="code", content="import os"), 0)
        assert not result.success
        assert result.error_phase == "validate"
        assert "Import" in result.error

    def test_host_builtins_are_not_reachable(self, kernel):
        # The old kernel exec'd with the host builtins module; open() worked.
        result = kernel.execute_cell(
            Cell(cell_type="code", content="t = open('/etc/hostname').read()"), 0)
        assert not result.success
        assert result.error_phase == "validate"

    def test_a_refusal_gives_phase_refused(self, kernel):
        result = kernel.execute_cell(
            Cell(cell_type="code", content="from pathlib import Path"), 0)
        assert not result.success
        assert result.error_phase == "refused"
        assert "not allowed in lackpy" in result.error

    def test_continue_cell_sets_flag(self, kernel):
        cell = Cell(cell_type="continue", content="")
        result = kernel.execute_cell(cell, cell_index=0)
        assert result.success
        assert result.namespace_delta.get("__continue_requested__") is True

    def test_scratch_cell_summarises_its_new_names(self, kernel):
        result = kernel.execute_cell(
            Cell(cell_type="scratch", content="a = 10\nb = 'hi'"), 0)
        assert result.success
        assert result.output == "[scratch: a=int, b=str]\n"
        assert kernel.lookup("a") == 10

    def test_code_cell_stdout_is_captured(self, kernel):
        result = kernel.execute_cell(Cell(cell_type="code", content="print('hi', 2)"), 0)
        assert result.success and result.output == "hi 2\n"

    def test_a_cell_gets_the_session_cell_id(self, kernel):
        result = kernel.execute_cell(Cell(cell_type="code", content="x = 1"), 0)
        assert result.cell_id is not None


class TestProse:
    def test_prose_format_spec(self, kernel):
        kernel.execute_cell(Cell(cell_type="code", content="total = 3.14159"), 0)
        result = kernel.execute_cell(Cell(cell_type="prose", content="Price: {total:.2f}"), 1)
        assert result.output == "Price: 3.14\n"

    def test_prose_conversion(self, kernel):
        kernel.execute_cell(Cell(cell_type="code", content="s = 'a'"), 0)
        result = kernel.execute_cell(Cell(cell_type="prose", content="S: {s!r}"), 1)
        assert result.output == "S: 'a'\n"

    def test_prose_nested_fstring_and_quotes(self, kernel):
        kernel.execute_cell(Cell(cell_type="code", content="items = ['a', 'b']\nd = {'k': 42}"), 0)
        result = kernel.execute_cell(
            Cell(cell_type="prose",
                 content="Items: {chr(10).join([f'- {x}' for x in items])} / {d['k']}"), 1)
        assert result.output == "Items: - a\n- b / 42\n"

    def test_prose_escapes_and_json_braces(self, kernel):
        kernel.execute_cell(Cell(cell_type="code", content="x = 1"), 0)
        result = kernel.execute_cell(
            Cell(cell_type="prose", content='Config: {"key": "v"} {{x}} and {x}'), 1)
        assert result.output == 'Config: {"key": "v"} {x} and 1\n'

    def test_empty_prose_is_a_blank_line(self, kernel):
        result = kernel.execute_cell(Cell(cell_type="prose", content="  \n "), 0)
        assert result.success and result.output == "\n"

    def test_prose_does_not_join_the_program(self, kernel):
        kernel.execute_cell(Cell(cell_type="code", content="x = 1"), 0)
        kernel.execute_cell(Cell(cell_type="prose", content="{x}"), 1)
        assert kernel.session._program == "x = 1"

    def test_a_large_collection_renders_truncated(self, kernel):
        kernel.execute_cell(Cell(cell_type="code", content="xs = list(range(100))"), 0)
        result = kernel.execute_cell(Cell(cell_type="prose", content="{xs}"), 1)
        assert "n=100" in result.output

    def test_a_runtime_error_in_an_interpolation_fails_the_prose_cell(self, kernel):
        kernel.execute_cell(Cell(cell_type="code", content="d = {}"), 0)
        result = kernel.execute_cell(Cell(cell_type="prose", content="V: {d['k']}"), 1)
        assert not result.success and result.error_phase == "runtime"
        assert "KeyError" in result.error


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

    def test_inspect_cannot_reach_the_host(self, kernel):
        result = kernel.inspect("open('/etc/hostname').read()")
        assert "Forbidden" in result or "not allowed" in result


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


class TestReifiedValues:
    def test_bind_accepts_only_forgiving_values(self, kernel):
        kernel.bind("h", Hole("h"))
        assert isinstance(kernel.lookup("h"), Hole)
        assert "h" in kernel.known_names()
        with pytest.raises(TypeError):
            kernel.bind("f", print)  # no host object enters the kernel's view

    def test_a_reified_value_never_enters_the_session(self, kernel):
        kernel.bind("e", ErrorValue(name="e", error="boom", error_phase="runtime"))
        assert "e" not in kernel.session.bound_names()

    def test_a_cell_that_rebinds_a_reified_name_clears_it(self, kernel):
        kernel.bind("x", Hole("x"))
        kernel.execute_cell(Cell(cell_type="code", content="x = 5"), 0)
        assert kernel.lookup("x") == 5


class TestTheLatchAtEveryEntryPoint:
    """Review I1: after a refusal every cell type and inspect refuse, also the
    ones that never call the session (prose text, @continue)."""

    @pytest.fixture
    def refused(self, kernel):
        result = kernel.execute_cell(Cell(cell_type="code", content="from pathlib import Path"), 0)
        assert result.error_phase == "refused"
        return kernel

    @pytest.mark.parametrize("cell", [
        Cell(cell_type="prose", content="Plain text"),
        Cell(cell_type="prose", content="   "),
        Cell(cell_type="continue", content=""),
        Cell(cell_type="scratch", content=""),
        Cell(cell_type="code", content="   "),
        Cell(cell_type="code", content="x = 1"),
        Cell(cell_type="code", content="x = "),  # a static failure too
    ])
    def test_every_cell_refuses_after_a_refusal(self, refused, cell):
        result = refused.execute_cell(cell, 1)
        assert not result.success and result.error_phase == "refused"
        assert "earlier refusal" in result.error and "pathlib" in result.error

    def test_rerun_refuses_after_a_refusal(self, refused):
        result = refused.rerun_cell(Cell(cell_type="prose", content="T"), 1, None)
        assert result.error_phase == "refused"

    def test_inspect_reports_the_latch(self, refused):
        assert "earlier refusal" in refused.inspect("1 + 1")
        assert refused.refusal is not None

    def test_a_refusing_inspect_latches(self, tmp_policy):
        kernel = make_kernel(tmp_policy)
        kernel.execute_cell(Cell(cell_type="code", content="from pathlib import Path"), 0)
        assert kernel.refusal is None
        assert "/etc/hostname" in kernel.inspect("Path('/etc/hostname').read_text()")
        assert kernel.refusal is not None


class TestNamesNotValues:
    """Review I2: a non-data binding is tracked by name, with an opaque marker."""

    def test_a_def_appears_in_the_delta_as_an_opaque_marker(self, kernel):
        from lackpy.interpreters.literate.kernel.versions import Opaque

        result = kernel.execute_cell(Cell(cell_type="code", content="def f(n):\n    return n"), 0)
        assert result.success
        assert isinstance(result.namespace_delta["f"], Opaque)
        assert result.namespace_delta["f"].type_name == "function"
        assert kernel.lookup("f") is result.namespace_delta["f"]
        assert "f" not in kernel.get_namespace()  # variables stay data only

    def test_a_rebind_gives_a_new_marker(self, kernel):
        assert kernel.execute_cell(Cell(cell_type="code", content="xs = zip([1], [2])"), 0).success
        first = kernel.lookup("xs")
        assert first is not None
        kernel.execute_cell(Cell(cell_type="code", content="ys = 1"), 1)
        assert kernel.lookup("xs") is first
        kernel.execute_cell(Cell(cell_type="code", content="xs = zip([3], [4])"), 2)
        assert kernel.lookup("xs") is not first

    def test_scratch_names_the_type_of_a_non_data_value(self, kernel):
        result = kernel.execute_cell(
            Cell(cell_type="scratch", content="def f():\n    return 1\na = 1"), 0)
        assert result.output == "[scratch: a=int, f=function]\n"


class TestProseOutputAndRepr:
    def test_prose_keeps_what_an_expression_printed(self, kernel):
        # Review M2: as python3's print(f"...") would: the side output first.
        result = kernel.execute_cell(Cell(cell_type="prose", content="A {print('side')} B"), 0)
        assert result.output == "side\nA None B\n"

    def test_prose_repr_of_a_function_has_no_address(self, kernel):
        # Review M7.
        kernel.execute_cell(Cell(cell_type="code", content="import json"), 0)
        result = kernel.execute_cell(Cell(cell_type="prose", content="{json.dumps}"), 1)
        assert result.success and "0x" not in result.output
        assert result.output == "<function dumps>\n"
