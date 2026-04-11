"""Tests for the interpreter base protocol, registry, and context types."""

from pathlib import Path

import pytest

from lackpy.interpreters import (
    INTERPRETERS,
    ExecutionContext,
    InterpreterExecutionResult,
    InterpreterValidationResult,
    get_interpreter,
    list_interpreters,
    register_interpreter,
)


class TestExecutionContext:
    def test_default_fields(self):
        ctx = ExecutionContext()
        assert ctx.kit is None
        assert ctx.params == {}
        assert ctx.config == {}
        assert ctx.extra_rules is None
        assert ctx.base_dir == Path.cwd()

    def test_explicit_fields(self):
        ctx = ExecutionContext(
            kit="stub",
            params={"x": 1},
            base_dir=Path("/tmp"),
            config={"mode": "brief"},
            extra_rules=["rule"],
        )
        assert ctx.kit == "stub"
        assert ctx.params == {"x": 1}
        assert ctx.base_dir == Path("/tmp")
        assert ctx.config == {"mode": "brief"}
        assert ctx.extra_rules == ["rule"]


class TestValidationResult:
    def test_valid(self):
        r = InterpreterValidationResult(valid=True)
        assert r.valid
        assert r.errors == []
        assert r.warnings == []

    def test_invalid_with_errors(self):
        r = InterpreterValidationResult(valid=False, errors=["bad", "worse"])
        assert not r.valid
        assert len(r.errors) == 2


class TestExecutionResult:
    def test_successful_result(self):
        r = InterpreterExecutionResult(
            success=True,
            output="hello",
            output_format="markdown",
            duration_ms=12.3,
        )
        assert r.success
        assert r.output == "hello"
        assert r.output_format == "markdown"
        assert r.error is None

    def test_failed_result(self):
        r = InterpreterExecutionResult(
            success=False,
            error="boom",
            output_format="none",
        )
        assert not r.success
        assert r.error == "boom"


class TestRegistry:
    def test_python_is_registered(self):
        assert "python" in INTERPRETERS

    def test_ast_select_is_registered(self):
        assert "ast-select" in INTERPRETERS

    def test_get_interpreter_returns_class(self):
        cls = get_interpreter("python")
        assert cls.name == "python"

    def test_get_interpreter_unknown_raises(self):
        with pytest.raises(KeyError) as excinfo:
            get_interpreter("nonexistent")
        # The error message should list available interpreters
        msg = str(excinfo.value)
        assert "python" in msg
        assert "ast-select" in msg

    def test_list_interpreters_has_metadata(self):
        entries = list_interpreters()
        names = {e["name"] for e in entries}
        assert "python" in names
        assert "ast-select" in names
        for e in entries:
            assert "description" in e
            assert len(e["description"]) > 0

    def test_register_without_name_raises(self):
        class Bogus:
            pass

        with pytest.raises(ValueError, match="name"):
            register_interpreter(Bogus)

    def test_re_register_warns_but_wins(self):
        class Fake1:
            name = "temp-test-interp"
            description = "fake1"

        class Fake2:
            name = "temp-test-interp"
            description = "fake2"

        register_interpreter(Fake1)
        with pytest.warns(UserWarning, match="re-registered"):
            register_interpreter(Fake2)
        assert INTERPRETERS["temp-test-interp"] is Fake2
        # Clean up so we don't pollute other tests
        del INTERPRETERS["temp-test-interp"]
