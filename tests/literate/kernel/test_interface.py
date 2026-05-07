"""Tests for kernel interface types."""

from lackpy.interpreters.literate.kernel.interface import CellResult


class TestCellResult:
    def test_successful_result(self):
        result = CellResult(
            success=True,
            output="hello\n",
            error=None,
            error_phase=None,
            namespace_delta={"x": 42},
            cell_index=0,
        )
        assert result.success
        assert result.output == "hello\n"
        assert result.namespace_delta == {"x": 42}

    def test_static_failure(self):
        result = CellResult(
            success=False,
            output=None,
            error="name 'foo' is not defined",
            error_phase="static",
            namespace_delta={},
            cell_index=2,
        )
        assert not result.success
        assert result.error_phase == "static"

    def test_runtime_failure(self):
        result = CellResult(
            success=False,
            output=None,
            error="ZeroDivisionError: division by zero",
            error_phase="runtime",
            namespace_delta={},
            cell_index=1,
        )
        assert result.error_phase == "runtime"
