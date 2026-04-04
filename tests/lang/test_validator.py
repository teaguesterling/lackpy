"""Tests for the AST validator."""

import pytest

from lackpy.lang.validator import validate, ValidationResult


class TestForbiddenNodes:
    def test_rejects_import(self):
        result = validate("import os")
        assert not result.valid
        assert any("Import" in e for e in result.errors)

    def test_rejects_import_from(self):
        result = validate("from os import path")
        assert not result.valid
        assert any("Import" in e for e in result.errors)

    def test_rejects_function_def(self):
        result = validate("def foo():\n    pass")
        assert not result.valid
        assert any("FunctionDef" in e for e in result.errors)

    def test_rejects_async_function_def(self):
        result = validate("async def foo():\n    pass")
        assert not result.valid
        assert any("AsyncFunctionDef" in e for e in result.errors)

    def test_rejects_class_def(self):
        result = validate("class Foo:\n    pass")
        assert not result.valid
        assert any("ClassDef" in e for e in result.errors)

    def test_rejects_standalone_lambda(self):
        """Lambda is only allowed as key= argument, not standalone."""
        assert not validate("x = lambda: 1", allowed_names=set()).valid

    def test_rejects_while(self):
        result = validate("while True:\n    pass")
        assert not result.valid
        assert any("While" in e for e in result.errors)

    def test_rejects_try(self):
        result = validate("try:\n    x = 1\nexcept:\n    pass")
        assert not result.valid
        assert any("Try" in e for e in result.errors)

    def test_rejects_raise(self):
        result = validate("raise ValueError('bad')")
        assert not result.valid
        assert any("Raise" in e for e in result.errors)

    def test_rejects_global(self):
        result = validate("global x")
        assert not result.valid
        assert any("Global" in e for e in result.errors)

    def test_rejects_assert(self):
        result = validate("assert True")
        assert not result.valid
        assert any("Assert" in e for e in result.errors)

    def test_rejects_delete(self):
        result = validate("del x")
        assert not result.valid
        assert any("Delete" in e for e in result.errors)

    def test_rejects_yield(self):
        # yield outside a function is a syntax error — that is fine
        result = validate("yield 1")
        assert not result.valid


class TestForbiddenNames:
    def test_rejects_dunder_import(self):
        result = validate("x = __import__('os')")
        assert not result.valid
        assert any("__import__" in e for e in result.errors)

    def test_rejects_open(self):
        result = validate("f = open('file.txt')")
        assert not result.valid
        assert any("open" in e for e in result.errors)

    def test_rejects_getattr(self):
        result = validate("v = getattr(x, 'attr')", allowed_names={"x"})
        assert not result.valid
        assert any("getattr" in e for e in result.errors)

    def test_rejects_input(self):
        result = validate("x = input('prompt')")
        assert not result.valid
        assert any("input" in e for e in result.errors)

    def test_rejects_breakpoint(self):
        result = validate("breakpoint()")
        assert not result.valid
        assert any("breakpoint" in e for e in result.errors)

    def test_rejects_type(self):
        result = validate("t = type(x)", allowed_names={"x"})
        assert not result.valid
        assert any("type" in e for e in result.errors)


class TestNamespaceCheck:
    def test_rejects_unknown_function(self):
        result = validate("x = mystery_function(1)")
        assert not result.valid
        assert any("mystery_function" in e for e in result.errors)

    def test_accepts_allowed_builtin(self):
        result = validate("n = len([1, 2, 3])")
        assert result.valid, result.errors

    def test_accepts_kit_function(self):
        result = validate("data = read_file('file.txt')", allowed_names={"read_file"})
        assert result.valid, result.errors

    def test_rejects_call_not_in_kit(self):
        result = validate("data = read_file('file.txt')")
        assert not result.valid
        assert any("read_file" in e for e in result.errors)


class TestStringCheck:
    def test_rejects_dunder_in_string(self):
        result = validate("x = '__class__'")
        assert not result.valid
        assert any("__" in e for e in result.errors)

    def test_accepts_normal_string(self):
        result = validate("x = 'hello world'")
        assert result.valid, result.errors


class TestForLoopCheck:
    def test_accepts_for_over_call(self):
        result = validate("for i in range(10):\n    x = i")
        assert result.valid, result.errors

    def test_accepts_for_over_variable(self):
        result = validate("items = [1, 2, 3]\nfor i in items:\n    x = i")
        assert result.valid, result.errors


class TestValidPrograms:
    def test_simple_read_and_len(self):
        code = "data = read_file('file.txt')\nn = len(data)"
        result = validate(code, allowed_names={"read_file"})
        assert result.valid, result.errors
        assert "read_file" in result.calls
        assert "len" in result.calls
        assert "data" in result.variables

    def test_list_comprehension(self):
        code = "nums = [1, 2, 3]\nsquares = [x * x for x in nums]"
        result = validate(code)
        assert result.valid, result.errors

    def test_f_string(self):
        code = "name = 'world'\ngreeting = f'hello {name}'"
        result = validate(code)
        assert result.valid, result.errors

    def test_multiple_assigns_and_calls(self):
        code = (
            "data = read_file('input.txt')\n"
            "lines = data.split('\\n')\n"
            "n = len(lines)\n"
            "result = sorted(lines)\n"
        )
        result = validate(code, allowed_names={"read_file"})
        assert result.valid, result.errors

    def test_returns_syntax_error(self):
        result = validate("def (broken syntax")
        assert not result.valid
        assert any("Parse error" in e for e in result.errors)

    def test_sort_by_allowed(self):
        result = validate("sort_by([{'a': 1}], 'a')")
        assert result.valid
        assert "sort_by" in result.calls


class TestLambdaRestriction:
    def test_allows_lambda_as_sort_key(self):
        result = validate(
            "sorted([{'a': 1}], key=lambda x: x['a'])",
            allowed_names=set(),
        )
        assert result.valid

    def test_allows_lambda_in_sort_by(self):
        result = validate(
            "sort_by([{'a': 1}], key=lambda x: x['a'])",
            allowed_names=set(),
        )
        assert result.valid

    def test_allows_lambda_with_min_max(self):
        result = validate(
            "items = [{'v': 1}, {'v': 2}]\nmin(items, key=lambda x: x['v'])",
            allowed_names=set(),
        )
        assert result.valid

    def test_rejects_lambda_as_value(self):
        result = validate("x = lambda: 1", allowed_names=set())
        assert not result.valid
        assert any("key= argument" in e for e in result.errors)

    def test_rejects_lambda_as_positional_arg(self):
        result = validate("sorted([1, 2], lambda x: x)", allowed_names=set())
        assert not result.valid

    def test_rejects_lambda_in_non_key_kwarg(self):
        result = validate(
            "sorted([1], reverse=lambda x: x)",
            allowed_names=set(),
        )
        assert not result.valid
