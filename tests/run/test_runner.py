"""Tests for the v1 restricted runner."""

import pytest

from lackpy.run.runner import RestrictedRunner


@pytest.fixture
def runner():
    return RestrictedRunner()


@pytest.fixture
def mock_namespace():
    return {
        "read": lambda path: f"contents of {path}",
        "glob": lambda pattern: ["a.py", "b.py"],
    }


class TestBasicExecution:
    def test_simple_assignment_and_output(self, runner, mock_namespace):
        result = runner.run("x = read('test.py')\nlen(x)", mock_namespace)
        assert result.success
        assert result.output == len("contents of test.py")

    def test_captures_trace(self, runner, mock_namespace):
        result = runner.run("x = read('f.py')", mock_namespace)
        assert result.success
        assert len(result.trace.entries) == 1
        assert result.trace.entries[0].tool == "read"

    def test_captures_variables(self, runner, mock_namespace):
        result = runner.run("x = read('f.py')\ny = len(x)", mock_namespace)
        assert result.success
        assert "x" in result.variables
        assert "y" in result.variables

    def test_last_expression_captured(self, runner, mock_namespace):
        result = runner.run("files = glob('*.py')\nlen(files)", mock_namespace)
        assert result.success
        assert result.output == 2

    def test_no_last_expression(self, runner, mock_namespace):
        result = runner.run("x = read('f.py')", mock_namespace)
        assert result.success
        assert result.output is None


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
        ns = {"read": bad_read}
        result = runner.run("x = read('missing.py')", ns)
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
