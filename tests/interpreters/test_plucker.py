"""Tests for the PluckerInterpreter plugin.

plucker is a thin wrapper over PythonInterpreter with a pluckit-specific
kit. These tests verify the wrapper's behavior: kit installation,
validation delegation, real pluckit execution, and that the chain's
terminal operation determines the output type.
"""

import pytest
from pathlib import Path
from textwrap import dedent

from lackpy.interpreters import (
    ExecutionContext,
    PluckerInterpreter,
    run_interpreter,
)

pluckit = pytest.importorskip("pluckit")


FIXTURE_SOURCE = dedent('''
    """Fixture module for PluckerInterpreter tests."""


    def greet(name):
        """Return a greeting."""
        return f"hello, {name}"


    def double(x):
        return x * 2


    async def fetch_data(url):
        """Async function for testing :async selector."""
        return None


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
def plucker_ctx(fixture_file):
    """Execution context with the fixture file as the default code."""
    return ExecutionContext(
        base_dir=fixture_file.parent,
        config={"code": str(fixture_file)},
    )


@pytest.fixture
def interp():
    return PluckerInterpreter()


class TestValidation:
    def test_valid_chain_passes(self, interp):
        ctx = ExecutionContext()
        program = 'source("src/**/*.py").find(".fn#main").names()'
        result = interp.validate(program, ctx)
        assert result.valid
        assert result.errors == []

    def test_multi_step_chain_passes(self, interp):
        ctx = ExecutionContext()
        program = (
            'p = source("src/**/*.py")\n'
            'fns = p.find(".fn#main")\n'
            'fns.names()'
        )
        result = interp.validate(program, ctx)
        assert result.valid

    def test_unknown_bare_name_rejected(self, interp):
        ctx = ExecutionContext()
        result = interp.validate('unknown_fn("x")', ctx)
        assert not result.valid
        assert any("unknown_function" in e.lower() or "unknown" in e.lower()
                   for e in result.errors)

    def test_import_rejected(self, interp):
        ctx = ExecutionContext()
        result = interp.validate("import os", ctx)
        assert not result.valid

    def test_empty_program_is_valid_noop(self, interp):
        """An empty plucker program is a valid no-op, same as empty Python."""
        ctx = ExecutionContext()
        result = interp.validate("", ctx)
        assert result.valid


class TestExecution:
    @pytest.mark.asyncio
    async def test_names_returns_list_of_strings(self, interp, plucker_ctx, fixture_file):
        program = f'source("{fixture_file}").find(".fn").names()'
        result = await run_interpreter(interp, program, plucker_ctx)
        assert result.success
        assert isinstance(result.output, list)
        assert all(isinstance(n, str) for n in result.output)
        # Our fixture defines greet, double, fetch_data, __init__, increment
        assert "greet" in result.output or "double" in result.output

    @pytest.mark.asyncio
    async def test_count_returns_int(self, interp, fixture_file):
        program = f'source("{fixture_file}").find(".fn").count()'
        result = await run_interpreter(interp, program, ExecutionContext())
        assert result.success
        assert isinstance(result.output, int)
        assert result.output > 0

    @pytest.mark.asyncio
    async def test_view_returns_markdown_string(self, interp, fixture_file):
        program = (
            f'source("{fixture_file}").view'
            '(".fn#greet { show: signature; }")'
        )
        result = await run_interpreter(interp, program, ExecutionContext())
        assert result.success
        assert isinstance(result.output, str)
        # Pluckit's view output has code fences
        assert "```" in result.output

    @pytest.mark.asyncio
    async def test_default_code_from_context(self, interp, plucker_ctx):
        """source() with no args uses context.config['code']."""
        program = 'source().find(".fn#greet").names()'
        result = await run_interpreter(interp, program, plucker_ctx)
        assert result.success
        assert "greet" in result.output

    @pytest.mark.asyncio
    async def test_explicit_code_overrides_default(
        self, interp, plucker_ctx, fixture_file
    ):
        """source(code) with an arg overrides context.config['code']."""
        program = f'source("{fixture_file}").find(".fn#greet").names()'
        result = await run_interpreter(interp, program, plucker_ctx)
        assert result.success

    @pytest.mark.asyncio
    async def test_multi_step_chain(self, interp, fixture_file):
        program = (
            f'p = source("{fixture_file}")\n'
            'fns = p.find(".fn")\n'
            'fns.names()'
        )
        result = await run_interpreter(interp, program, ExecutionContext())
        assert result.success
        assert isinstance(result.output, list)

    @pytest.mark.asyncio
    async def test_missing_code_with_no_default(self, interp):
        """source() with no args and no context default should fail."""
        program = 'source().find(".fn").count()'
        result = await run_interpreter(interp, program, ExecutionContext())
        assert not result.success


class TestOutputMetadata:
    @pytest.mark.asyncio
    async def test_metadata_includes_interpreter_name(self, interp, fixture_file):
        program = f'source("{fixture_file}").find(".fn").count()'
        result = await run_interpreter(interp, program, ExecutionContext())
        assert result.metadata.get("interpreter") == "plucker"

    @pytest.mark.asyncio
    async def test_output_format_is_python(self, interp, fixture_file):
        """plucker output varies by terminal — format tag stays 'python'."""
        program = f'source("{fixture_file}").find(".fn").count()'
        result = await run_interpreter(interp, program, ExecutionContext())
        assert result.output_format == "python"

    @pytest.mark.asyncio
    async def test_duration_recorded(self, interp, fixture_file):
        program = f'source("{fixture_file}").find(".fn").count()'
        result = await run_interpreter(interp, program, ExecutionContext())
        assert result.duration_ms > 0


class TestRegistry:
    def test_plucker_is_registered(self):
        from lackpy.interpreters import INTERPRETERS, get_interpreter
        assert "plucker" in INTERPRETERS
        assert get_interpreter("plucker") is PluckerInterpreter
