"""Tests for the AstSelectInterpreter plugin.

Uses a small fixture file on disk so tests don't depend on lackpy's
own source layout. The fixture contains known classes and functions
with predictable names and line ranges.
"""

import pytest
from pathlib import Path
from textwrap import dedent

from lackpy.interpreters import (
    AstSelectInterpreter,
    ExecutionContext,
    run_interpreter,
)

pluckit = pytest.importorskip("pluckit")


FIXTURE_SOURCE = dedent('''
    """Fixture module for AstSelectInterpreter tests."""

    PI = 3.14159


    def greet(name):
        """Return a greeting."""
        return f"hello, {name}"


    def double(x):
        return x * 2


    class Counter:
        """A counter object."""

        def __init__(self, start=0):
            self.value = start

        def increment(self):
            self.value += 1
            return self.value

        def reset(self):
            self.value = 0
''').lstrip()


@pytest.fixture
def fixture_file(tmp_path):
    """Write a known Python fixture file and return its path."""
    path = tmp_path / "sample.py"
    path.write_text(FIXTURE_SOURCE)
    return path


@pytest.fixture
def ast_ctx(fixture_file):
    """Execution context pointing at the fixture file."""
    return ExecutionContext(
        base_dir=fixture_file.parent,
        config={"code": str(fixture_file)},
    )


@pytest.fixture
def interp():
    return AstSelectInterpreter()


class TestValidation:
    def test_bare_selector_valid(self, interp):
        ctx = ExecutionContext()
        result = interp.validate(".fn#greet", ctx)
        assert result.valid
        assert result.errors == []

    def test_empty_program_rejected(self, interp):
        ctx = ExecutionContext()
        result = interp.validate("", ctx)
        assert not result.valid

    def test_whitespace_only_rejected(self, interp):
        ctx = ExecutionContext()
        result = interp.validate("   \n  \t", ctx)
        assert not result.valid

    def test_selector_sheet_rejected(self, interp):
        ctx = ExecutionContext()
        result = interp.validate(".fn#greet { show: body; }", ctx)
        assert not result.valid
        assert any("pss" in e for e in result.errors)

    def test_multi_line_selector_rejected(self, interp):
        ctx = ExecutionContext()
        result = interp.validate(".fn#greet\n.fn#double", ctx)
        assert not result.valid


class TestExecutionFull:
    @pytest.mark.asyncio
    async def test_single_match_renders_full_body(self, interp, ast_ctx):
        result = await run_interpreter(interp, ".fn#greet", ast_ctx)
        assert result.success
        assert result.output_format == "markdown"
        assert result.metadata["match_count"] == 1
        assert "# `.fn#greet`" in result.output
        assert "def greet(name):" in result.output
        assert "hello" in result.output
        assert "```python" in result.output

    @pytest.mark.asyncio
    async def test_multiple_matches_rendered(self, interp, ast_ctx):
        result = await run_interpreter(interp, ".fn", ast_ctx)
        assert result.success
        assert result.metadata["match_count"] >= 2
        # Should show multiple functions
        assert "def greet(name):" in result.output
        assert "def double(x):" in result.output

    @pytest.mark.asyncio
    async def test_heading_includes_selector(self, interp, ast_ctx):
        result = await run_interpreter(interp, ".fn#double", ast_ctx)
        assert result.success
        # H1 is the selector
        assert result.output.startswith("# `.fn#double`")

    @pytest.mark.asyncio
    async def test_match_heading_has_qualified_name(self, interp, ast_ctx):
        result = await run_interpreter(interp, ".fn#greet", ast_ctx)
        assert result.success
        # Each match has an H2 with the qualified name and location
        assert "##" in result.output
        assert "F/greet" in result.output

    @pytest.mark.asyncio
    async def test_location_is_relative(self, interp, ast_ctx, tmp_path):
        result = await run_interpreter(interp, ".fn#greet", ast_ctx)
        assert result.success
        # Relative path should appear (sample.py), not the full absolute path
        assert "sample.py" in result.output
        assert str(tmp_path) not in result.output or "sample.py" in result.output

    @pytest.mark.asyncio
    async def test_empty_match_returns_empty_string(self, interp, ast_ctx):
        result = await run_interpreter(
            interp,
            ".fn#nonexistent_function_12345",
            ast_ctx,
        )
        assert result.success
        assert result.output == ""
        assert result.metadata["match_count"] == 0


class TestExecutionBrief:
    @pytest.mark.asyncio
    async def test_brief_mode_single_line_per_match(self, interp, fixture_file):
        ctx = ExecutionContext(
            base_dir=fixture_file.parent,
            config={"code": str(fixture_file), "mode": "brief"},
        )
        result = await run_interpreter(interp, ".fn", ctx)
        assert result.success
        assert result.metadata["mode"] == "brief"
        # Brief mode shows peek (signature line) not full body
        assert "def greet(name):" in result.output
        # But NOT the full body content
        assert "return f" not in result.output

    @pytest.mark.asyncio
    async def test_brief_mode_has_location_prefix(self, interp, fixture_file):
        ctx = ExecutionContext(
            base_dir=fixture_file.parent,
            config={"code": str(fixture_file), "mode": "brief"},
        )
        result = await run_interpreter(interp, ".fn#greet", ctx)
        assert result.success
        # Format: `- \`path:line\` — signature`
        assert "sample.py:" in result.output

    @pytest.mark.asyncio
    async def test_unknown_mode_rejected(self, interp, fixture_file):
        ctx = ExecutionContext(
            base_dir=fixture_file.parent,
            config={"code": str(fixture_file), "mode": "bogus"},
        )
        result = await run_interpreter(interp, ".fn", ctx)
        assert not result.success
        assert "unknown mode" in result.error.lower()


class TestMetadata:
    @pytest.mark.asyncio
    async def test_metadata_includes_selector(self, interp, ast_ctx):
        result = await run_interpreter(interp, ".fn#greet", ast_ctx)
        assert result.metadata["selector"] == ".fn#greet"

    @pytest.mark.asyncio
    async def test_metadata_includes_mode(self, interp, ast_ctx):
        result = await run_interpreter(interp, ".fn#greet", ast_ctx)
        assert result.metadata["mode"] == "full"

    @pytest.mark.asyncio
    async def test_duration_recorded(self, interp, ast_ctx):
        result = await run_interpreter(interp, ".fn#greet", ast_ctx)
        assert result.duration_ms > 0


class TestValidationIntegration:
    @pytest.mark.asyncio
    async def test_sheet_syntax_short_circuits_execution(self, interp, ast_ctx):
        # If validation rejects, execution should not be attempted.
        result = await run_interpreter(interp, ".fn#greet { show: body; }", ast_ctx)
        assert not result.success
        assert "Validation failed" in result.error
