"""Tests for the PssInterpreter plugin.

pss is a thin wrapper over pluckit's AstViewer — most of its behavior
is delegated to pluckit. These tests verify the wrapper's contract:
config resolution, validation, error handling, and metadata, without
duplicating pluckit's own view() tests.
"""

import pytest
from pathlib import Path
from textwrap import dedent

from lackpy.interpreters import (
    ExecutionContext,
    PssInterpreter,
    run_interpreter,
)

pluckit = pytest.importorskip("pluckit")


FIXTURE_SOURCE = dedent('''
    """Fixture module for PssInterpreter tests."""


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
''').lstrip()


@pytest.fixture
def fixture_file(tmp_path):
    """Write a known Python fixture file and return its path."""
    path = tmp_path / "sample.py"
    path.write_text(FIXTURE_SOURCE)
    return path


@pytest.fixture
def pss_ctx(fixture_file):
    """Execution context pointing at the fixture file."""
    return ExecutionContext(
        base_dir=fixture_file.parent,
        config={"code": str(fixture_file)},
    )


@pytest.fixture
def interp():
    return PssInterpreter()


class TestValidation:
    def test_empty_program_rejected(self, interp):
        ctx = ExecutionContext()
        result = interp.validate("", ctx)
        assert not result.valid

    def test_whitespace_only_rejected(self, interp):
        ctx = ExecutionContext()
        result = interp.validate("   \n\t", ctx)
        assert not result.valid

    def test_bare_selector_is_valid(self, interp):
        """pss accepts bare selectors (they behave like ast-select)."""
        ctx = ExecutionContext()
        result = interp.validate(".fn#greet", ctx)
        assert result.valid

    def test_single_rule_with_declarations_valid(self, interp):
        ctx = ExecutionContext()
        result = interp.validate(".fn#greet { show: body; }", ctx)
        assert result.valid

    def test_multi_rule_sheet_valid(self, interp):
        ctx = ExecutionContext()
        program = (
            ".fn#greet { show: body; }\n"
            ".class#Counter { show: outline; }"
        )
        result = interp.validate(program, ctx)
        assert result.valid

    def test_unbalanced_braces_rejected(self, interp):
        ctx = ExecutionContext()
        result = interp.validate(".fn#greet { show: body", ctx)
        assert not result.valid
        assert any("brace" in e.lower() for e in result.errors)

    def test_extra_closing_brace_rejected(self, interp):
        ctx = ExecutionContext()
        result = interp.validate(".fn#greet { show: body; } }", ctx)
        assert not result.valid


class TestExecution:
    @pytest.mark.asyncio
    async def test_multi_rule_sheet_renders(self, interp, pss_ctx):
        program = (
            ".fn#greet { show: body; }\n"
            ".fn#double { show: body; }"
        )
        result = await run_interpreter(interp, program, pss_ctx)
        assert result.success
        assert result.output_format == "markdown"
        # Both matches appear in the output
        assert "def greet(name):" in result.output
        assert "def double(x):" in result.output

    @pytest.mark.asyncio
    async def test_single_rule_renders(self, interp, pss_ctx):
        result = await run_interpreter(
            interp, ".fn#greet { show: body; }", pss_ctx
        )
        assert result.success
        assert "def greet(name):" in result.output

    @pytest.mark.asyncio
    async def test_bare_selector_renders(self, interp, pss_ctx):
        """A bare selector with no declaration block still works."""
        result = await run_interpreter(interp, ".fn#greet", pss_ctx)
        assert result.success
        assert "def greet(name):" in result.output

    @pytest.mark.asyncio
    async def test_output_is_markdown_format(self, interp, pss_ctx):
        result = await run_interpreter(
            interp, ".fn#greet { show: body; }", pss_ctx
        )
        assert result.success
        # pluckit's markdown output uses code fences
        assert "```" in result.output

    @pytest.mark.asyncio
    async def test_missing_code_config_fails(self, interp, fixture_file):
        """pss requires explicit 'code' in config."""
        ctx = ExecutionContext(base_dir=fixture_file.parent)
        result = await run_interpreter(
            interp, ".fn#greet { show: body; }", ctx
        )
        assert not result.success
        assert "code" in result.error.lower()

    @pytest.mark.asyncio
    async def test_validation_short_circuits(self, interp, pss_ctx):
        """Unbalanced braces should fail before calling pluckit."""
        result = await run_interpreter(interp, ".fn#greet { show: body", pss_ctx)
        assert not result.success
        assert "Validation failed" in result.error


class TestMetadata:
    @pytest.mark.asyncio
    async def test_rule_count_single(self, interp, pss_ctx):
        result = await run_interpreter(
            interp, ".fn#greet { show: body; }", pss_ctx
        )
        assert result.metadata["rule_count"] == 1

    @pytest.mark.asyncio
    async def test_rule_count_multiple(self, interp, pss_ctx):
        program = (
            ".fn#greet { show: body; }\n"
            ".fn#double { show: signature; }\n"
            ".class#Counter { show: outline; }"
        )
        result = await run_interpreter(interp, program, pss_ctx)
        assert result.metadata["rule_count"] == 3

    @pytest.mark.asyncio
    async def test_rule_count_bare_selector(self, interp, pss_ctx):
        """A bare selector counts as one rule."""
        result = await run_interpreter(interp, ".fn#greet", pss_ctx)
        assert result.metadata["rule_count"] == 1

    @pytest.mark.asyncio
    async def test_metadata_includes_code(self, interp, pss_ctx, fixture_file):
        result = await run_interpreter(
            interp, ".fn#greet { show: body; }", pss_ctx
        )
        assert result.metadata["code"] == str(fixture_file)

    @pytest.mark.asyncio
    async def test_metadata_includes_format(self, interp, pss_ctx):
        result = await run_interpreter(
            interp, ".fn#greet { show: body; }", pss_ctx
        )
        assert result.metadata["format"] == "markdown"

    @pytest.mark.asyncio
    async def test_duration_recorded(self, interp, pss_ctx):
        result = await run_interpreter(
            interp, ".fn#greet { show: body; }", pss_ctx
        )
        assert result.duration_ms > 0


class TestRegistry:
    def test_pss_is_registered(self):
        from lackpy.interpreters import INTERPRETERS, get_interpreter
        assert "pss" in INTERPRETERS
        assert get_interpreter("pss") is PssInterpreter
