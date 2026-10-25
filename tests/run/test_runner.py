"""Tests for the v1 restricted runner."""

import json
import pathlib
import sys

import pytest

from lackpy.run.base import ExecutionResult
from lackpy.run.paths import PathPolicy
from lackpy.run.runner import RestrictedRunner


class TestEffectiveOutput:
    def test_typed_output_wins_over_stdout(self):
        # A present typed value is never overridden or coerced by captured stdout.
        r = ExecutionResult(success=True, output=42, stdout="ignored\n")
        assert r.effective_output == 42

    def test_falls_back_to_stripped_stdout(self):
        r = ExecutionResult(success=True, output=None, stdout="50\n")
        assert r.effective_output == "50"

    def test_none_when_no_output_and_no_stdout(self):
        r = ExecutionResult(success=True, output=None, stdout="")
        assert r.effective_output is None

    def test_falsy_typed_output_is_preserved(self):
        # output=0 / "" / [] are real values, not "missing" — keep them, don't
        # fall through to stdout.
        r = ExecutionResult(success=True, output=0, stdout="99\n")
        assert r.effective_output == 0


@pytest.fixture
def runner():
    return RestrictedRunner()


@pytest.fixture
def ws(tmp_path):
    w = tmp_path / "ws"
    w.mkdir()
    (w / "test.py").write_text("print('x')\n")
    (w / "a.py").write_text("")
    return w


def _run(program, ws, **kw):
    return RestrictedRunner().run(program, path_policy=PathPolicy(ws, []), **kw)


READ = "from pathlib import Path\n"


class TestNoToolNamespace:
    """stepy100 Task 7: the runner binds no tool callables."""

    def test_run_takes_no_namespace(self, runner):
        # Retired kit model: run(program, namespace) bound kit tools as traced globals.
        with pytest.raises(TypeError):
            runner.run("x = 1", {"read_file": lambda p: p})
        with pytest.raises(TypeError):
            runner.run("x = 1", namespace={"read_file": lambda p: p})

    def test_kit_names_are_unbound(self, runner):
        result = runner.run("read_file('x')")
        assert not result.success
        assert "read_file" in result.error


class TestBasicExecution:
    def test_simple_assignment_and_output(self, ws):
        result = _run(READ + "x = Path('test.py').read_text()\nlen(x)", ws)
        assert result.success, result.error
        assert result.output == len("print('x')\n")

    def test_captures_trace(self, ws):
        result = _run(READ + "x = Path('test.py').read_text()", ws)
        assert result.success, result.error
        assert [e.tool for e in result.trace.entries] == ["Path.read_text"]

    def test_captures_variables(self, ws):
        result = _run(READ + "x = Path('test.py').read_text()\ny = len(x)", ws)
        assert result.success
        assert "x" in result.variables
        assert "y" in result.variables

    def test_last_expression_captured(self, ws):
        result = _run(READ + "files = sorted(Path('.').glob('*.py'))\nlen(files)", ws)
        assert result.success, result.error
        assert result.output == 2

    def test_no_last_expression(self, ws):
        result = _run(READ + "x = Path('test.py').read_text()", ws)
        assert result.success
        assert result.output is None


class TestPrintCapture:
    def test_print_captured_to_stdout(self, runner):
        result = runner.run("print('hello')")
        assert result.success
        assert result.stdout == "hello\n"

    def test_print_does_not_escape_as_output(self, ws):
        # A trailing print(...) is a call returning None, so the typed output
        # stays None — the value is recoverable from stdout instead.
        result = _run(READ + "x = list(Path('.').glob('*.py'))\nprint(len(x))", ws)
        assert result.success
        assert result.output is None
        assert result.stdout == "2\n"

    def test_print_multiple_args_and_calls(self, runner):
        result = runner.run("print('a', 'b')\nprint('c')")
        assert result.success
        assert result.stdout == "a b\nc\n"

    def test_sep_and_end_none_mean_the_defaults(self, runner):
        # Review M2: python3 prints "1 2\n" for sep=None and end=None.
        result = runner.run("print(1, 2, sep=None)\nprint(3, end=None)\nprint(4, 5, sep=None, end=None)")
        assert result.success, result.error
        assert result.stdout == "1 2\n3\n4 5\n"

    def test_no_print_means_empty_stdout(self, runner):
        result = runner.run("files = [1, 2]\nlen(files)")
        assert result.success
        assert result.stdout == ""

    def test_partial_stdout_on_runtime_error(self, ws):
        result = _run(READ + "print('before')\nPath('missing.py').read_text()", ws)
        assert not result.success
        assert result.stdout == "before\n"


class TestParams:
    # Task 5: params replace the program's own params literal; no bare names (tests/run/test_params.py).
    def test_params_reach_the_program_through_its_literal(self, runner):
        result = runner.run('params = {"content": ""}\nlen(params["content"])',
                            params={"content": "hello world"})
        assert result.success, result.error
        assert result.output == 11

    def test_params_bind_no_variable_of_their_own(self, runner):
        result = runner.run('params = {"content": ""}\nx = len(params["content"])',
                            params={"content": "hello"})
        assert "content" not in result.variables
        assert result.variables == {"params": {"content": "hello"}, "x": 5}


class TestErrorHandling:
    def test_runtime_error_captured(self, ws):
        result = _run(READ + "x = Path('missing.py').read_text()", ws)
        assert not result.success
        assert "missing.py" in result.error


class TestSecurity:
    def test_builtins_restricted(self, runner):
        result = runner.run("compile('1', '', 'eval')")
        assert not result.success


class TestBuiltinsAtRuntime:
    def test_sort_by_is_no_longer_injected(self, runner):
        result = runner.run("sort_by([{'a': 1}], 'a')")
        assert not result.success

    def test_exception_class_is_available(self, runner):
        # Naming the class in an expression is enough. The instance is not data,
        # so the output is its repr (parked item 2).
        result = runner.run("KeyError('k')")
        assert result.success, result.error
        assert result.output == "KeyError('k')"


class TestOutputNeverRaisesOnTheHost:
    """Parked item 2: output and variables hold only what a host serializer can take."""

    @pytest.mark.parametrize("program, expected", [
        ("type(3)", "<class 'int'>"),
        ("from collections import Counter\nCounter('aab')", "Counter({'a': 2, 'b': 1})"),
        ("def f():\n    return 1\nf", None),
    ])
    def test_non_data_output_is_its_repr(self, runner, program, expected):
        result = runner.run(program)
        assert result.success, result.error
        assert isinstance(result.output, str)
        if expected is not None:
            assert result.output == expected
        json.dumps(result.output)

    def test_path_output_is_its_repr(self, ws):
        result = _run(READ + "Path('a')", ws)
        assert result.success, result.error
        assert result.output == repr(pathlib.Path("a"))

    def test_module_proxy_output_is_text(self, runner):
        result = runner.run("import json\njson")
        assert result.success, result.error
        assert isinstance(result.output, str)

    @pytest.mark.parametrize("program", ["10 ** 5000", "[1, 10 ** 5000]", "{'k': (10 ** 5000,)}"])
    def test_output_whose_text_cannot_be_made_is_a_failed_result(self, runner, program):
        result = runner.run(program)
        assert not result.success
        assert result.output is None
        assert "cannot be converted to text" in result.error

    def test_data_output_is_unchanged(self, runner):
        result = runner.run("x = {'a': [1, 2.5, None, True, b'x', ('t',)]}\nx")
        assert result.output == {"a": [1, 2.5, None, True, b"x", ("t",)]}

    def test_huge_int_is_not_a_variable(self, runner):
        result = runner.run("x = 10 ** 5000\ny = [x]\nz = 1")
        assert result.success, result.error
        assert result.variables == {"z": 1}

    def test_int_at_the_text_limit_is_kept(self, runner):
        digits = sys.get_int_max_str_digits()
        result = runner.run(f"x = 10 ** {digits - 1}")
        assert result.variables == {"x": 10 ** (digits - 1)}


class TestCompileFailure:
    def test_compile_error_is_a_failed_result_not_an_exception(self, runner):
        # Final review I2(b): any drift between validate() and compile() must not raise out of run().
        result = runner.run("a, *b, *c = [1, 2, 3]")
        assert not result.success
        assert "Compile error" in result.error


class TestRuntimeNamespace:
    """stepy100 Task 2: the runner builds the namespace and never raises."""

    def test_runs_without_a_kit_namespace(self, runner):
        result = runner.run("x = [1, 2]\nlen(x)")
        assert result.success, result.error
        assert result.output == 2

    def test_variables_exclude_namespace_names(self, runner):
        result = runner.run("x = type(3).__name__")
        assert result.success, result.error
        assert result.variables == {"x": "int"}

    def test_file_is_the_program_file(self, runner):
        result = runner.run("__file__", program_file="/w/p.py")
        assert result.output == "/w/p.py"

    def test_refusal_is_a_failed_result(self, runner):
        result = runner.run("type(3).mro")
        assert not result.success
        assert result.error == "type(...).mro is not allowed in lackpy"

    def test_refusal_sets_error_phase(self, runner):
        refused = runner.run("type(3).mro")
        assert refused.error_phase == "refused"
        assert runner.run("1/0").error_phase is None
        assert runner.run("1").error_phase is None

    def test_refusal_escapes_except_exception(self, runner):
        result = runner.run("try:\n    type(3).mro\nexcept Exception:\n    pass")
        assert not result.success
        assert "not allowed in lackpy" in result.error

    def test_deep_unary_expression_does_not_raise(self, runner):
        # Design §8, gate 4: a 3,000-deep expression is a failed result.
        result = runner.run("-" * 3000 + "1")
        assert not result.success
        assert result.error

    def test_params_cannot_replace_the_builtins(self, runner):
        # A param or kit tool named like a namespace global must not replace it.
        import builtins
        # Since Task 5 a param binds no name at all; this program has no params
        # literal, so the run fails before it starts, naming the key.
        result = runner.run("open", params={"__builtins__": vars(builtins)})
        assert not result.success
        assert "'__builtins__'" in result.error
        # With a literal, a dict of functions has no literal form (ruling c).
        result = runner.run('params = {"__builtins__": {}}\nopen', params={"__builtins__": vars(builtins)})
        assert not result.success
        assert "'__builtins__'" in result.error

    def test_unprintable_error_does_not_raise(self, runner):
        # Review I1: str(KeyError(n)) raises ValueError for an int over 4,300 digits.
        program = (
            "n = 1\n"
            "for i in range(5000):\n"
            "    n = n * 10\n"
            "d = {}\n"
            "d[n]\n"
        )
        result = runner.run(program)
        assert not result.success
        assert "KeyError" in result.error


_NAMESPACE_KEYS = ["type", "print", "__file__", "__name__", "__builtins__", "len", "KeyError"]


class TestNameCollisions:
    """A param cannot collide with a namespace name: it binds no name (Task 5).

    Review M3's kit-tool collision check went with the kit (parked item 7):
    there is no tool namespace left to collide (TestNoToolNamespace).
    """

    @pytest.mark.parametrize("name", _NAMESPACE_KEYS)
    def test_param_named_like_a_namespace_name_binds_nothing(self, runner, name):
        program = f'params = {{"{name}": "default"}}\nx = params["{name}"]\ny = len(str(type(1)))'
        result = runner.run(program, params={name: "caller value"})
        assert result.success, result.error
        assert result.variables["x"] == "caller value"
        assert result.variables["y"] == len("<class 'int'>")
