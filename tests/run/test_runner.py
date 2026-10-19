"""Tests for the v1 restricted runner."""

import pytest

from lackpy.run.base import ExecutionResult
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
def mock_namespace():
    return {
        "read_file": lambda path: f"contents of {path}",
        "find_files": lambda pattern: ["a.py", "b.py"],
    }


class TestBasicExecution:
    def test_simple_assignment_and_output(self, runner, mock_namespace):
        result = runner.run("x = read_file('test.py')\nlen(x)", mock_namespace)
        assert result.success
        assert result.output == len("contents of test.py")

    def test_captures_trace(self, runner, mock_namespace):
        result = runner.run("x = read_file('f.py')", mock_namespace)
        assert result.success
        assert len(result.trace.entries) == 1
        assert result.trace.entries[0].tool == "read_file"

    def test_captures_variables(self, runner, mock_namespace):
        result = runner.run("x = read_file('f.py')\ny = len(x)", mock_namespace)
        assert result.success
        assert "x" in result.variables
        assert "y" in result.variables

    def test_last_expression_captured(self, runner, mock_namespace):
        result = runner.run("files = find_files('*.py')\nlen(files)", mock_namespace)
        assert result.success
        assert result.output == 2

    def test_no_last_expression(self, runner, mock_namespace):
        result = runner.run("x = read_file('f.py')", mock_namespace)
        assert result.success
        assert result.output is None


class TestPrintCapture:
    def test_print_captured_to_stdout(self, runner, mock_namespace):
        result = runner.run("print('hello')", mock_namespace)
        assert result.success
        assert result.stdout == "hello\n"

    def test_print_does_not_escape_as_output(self, runner, mock_namespace):
        # A trailing print(...) is a call returning None, so the typed output
        # stays None — the value is recoverable from stdout instead.
        result = runner.run("x = find_files('*.py')\nprint(len(x))", mock_namespace)
        assert result.success
        assert result.output is None
        assert result.stdout == "2\n"

    def test_print_multiple_args_and_calls(self, runner, mock_namespace):
        result = runner.run("print('a', 'b')\nprint('c')", mock_namespace)
        assert result.success
        assert result.stdout == "a b\nc\n"

    def test_no_print_means_empty_stdout(self, runner, mock_namespace):
        result = runner.run("files = find_files('*.py')\nlen(files)", mock_namespace)
        assert result.success
        assert result.stdout == ""

    def test_partial_stdout_on_runtime_error(self, runner):
        def bad_read(path):
            raise FileNotFoundError("no such file")
        ns = {"read_file": bad_read}
        result = runner.run("print('before')\nread_file('missing.py')", ns)
        assert not result.success
        assert result.stdout == "before\n"


class TestParams:
    # Task 5: params replace the program's own params literal; no bare names (tests/run/test_params.py).
    def test_params_reach_the_program_through_its_literal(self, runner, mock_namespace):
        result = runner.run('params = {"content": ""}\nlen(params["content"])', mock_namespace,
                            params={"content": "hello world"})
        assert result.success, result.error
        assert result.output == 11

    def test_params_bind_no_variable_of_their_own(self, runner, mock_namespace):
        result = runner.run('params = {"content": ""}\nx = len(params["content"])', mock_namespace,
                            params={"content": "hello"})
        assert "content" not in result.variables
        assert result.variables == {"params": {"content": "hello"}, "x": 5}


class TestErrorHandling:
    def test_runtime_error_captured(self, runner, mock_namespace):
        def bad_read(path):
            raise FileNotFoundError("no such file")
        ns = {"read_file": bad_read}
        result = runner.run("x = read_file('missing.py')", ns)
        assert not result.success
        assert "no such file" in result.error


class TestSecurity:
    def test_builtins_restricted(self, runner, mock_namespace):
        result = runner.run("compile('1', '', 'eval')", mock_namespace)
        assert not result.success


class TestBuiltinsAtRuntime:
    def test_sort_by_is_no_longer_injected(self, runner, mock_namespace):
        result = runner.run("sort_by([{'a': 1}], 'a')", mock_namespace)
        assert not result.success

    def test_exception_class_is_available(self, runner, mock_namespace):
        # try is still forbidden (Task 8); naming the class in an expression is enough.
        result = runner.run("KeyError('k')", mock_namespace)
        assert result.success, result.error
        assert isinstance(result.output, KeyError)


class TestCompileFailure:
    def test_compile_error_is_a_failed_result_not_an_exception(self, runner):
        # Final review I2(b): any drift between validate() and compile() must not raise out of run().
        result = runner.run("a, *b, *c = [1, 2, 3]", {})
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

    def test_refusal_escapes_except_exception(self, runner):
        result = runner.run("try:\n    type(3).mro\nexcept Exception:\n    pass")
        assert not result.success
        assert "not allowed in lackpy" in result.error

    def test_deep_unary_expression_does_not_raise(self, runner):
        # Design §8, gate 4: a 3,000-deep expression is a failed result.
        result = runner.run("-" * 3000 + "1")
        assert not result.success
        assert result.error

    def test_kit_namespace_tools_run_inside_the_new_builtins(self, runner, mock_namespace):
        # A kit tool and an approved builtin both work in one program.
        result = runner.run("files = find_files('*.py')\nlen(sorted(files))", mock_namespace)
        assert result.success, result.error
        assert result.output == 2
        assert [e.tool for e in result.trace.entries] == ["find_files"]

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
    """Review M3: a kit tool must not collide with a namespace name.

    Neither side may win silently: the caller's value and the namespace's
    value are both meaningful. The run fails at bind time, naming the name.
    A param cannot collide since Task 5: it binds no name (ruling a).
    """

    @pytest.mark.parametrize("name", _NAMESPACE_KEYS)
    def test_param_named_like_a_namespace_name_binds_nothing(self, runner, name):
        program = f'params = {{"{name}": "default"}}\nx = params["{name}"]\ny = len(str(type(1)))'
        result = runner.run(program, params={name: "caller value"})
        assert result.success, result.error
        assert result.variables["x"] == "caller value"
        assert result.variables["y"] == len("<class 'int'>")

    @pytest.mark.parametrize("name", _NAMESPACE_KEYS)
    def test_tool_collision_is_a_failed_result(self, runner, name):
        called = []
        result = runner.run("x = 1", {name: lambda *a: called.append(a)})
        assert not result.success
        assert repr(name) in result.error
        assert "collides" in result.error
        assert called == []

    def test_non_colliding_names_still_bind(self, runner):
        result = runner.run('params = {"n": 0}\nx = params["n"] + 1', {"tool": lambda: 0}, params={"n": 1})
        assert result.success, result.error
        assert result.variables == {"params": {"n": 1}, "x": 2}
