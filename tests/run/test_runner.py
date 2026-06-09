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
    def test_params_available_as_variables(self, runner, mock_namespace):
        result = runner.run("len(content)", mock_namespace, params={"content": "hello world"})
        assert result.success
        assert result.output == 11

    def test_params_not_in_output_variables(self, runner, mock_namespace):
        result = runner.run("x = len(content)", mock_namespace, params={"content": "hello"})
        assert "content" not in result.variables
        assert "x" in result.variables


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


class TestSortByBuiltin:
    def test_sort_by_dict_key(self, runner, mock_namespace):
        result = runner.run(
            "items = [{'name': 'b', 'val': 2}, {'name': 'a', 'val': 1}]\nsort_by(items, 'name')",
            mock_namespace,
        )
        assert result.success
        assert result.output == [{'name': 'a', 'val': 1}, {'name': 'b', 'val': 2}]

    def test_sort_by_reverse(self, runner, mock_namespace):
        result = runner.run(
            "items = [{'v': 1}, {'v': 3}, {'v': 2}]\nsort_by(items, 'v', reverse=True)",
            mock_namespace,
        )
        assert result.success
        assert result.output == [{'v': 3}, {'v': 2}, {'v': 1}]
