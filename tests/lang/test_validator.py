"""Tests for the AST validator."""

import pytest

from lackpy.lang.validator import validate, ValidationResult


class TestForbiddenNodes:
    def test_rejects_import(self):
        result = validate("import os")
        assert not result.valid
        assert any("Import" in e for e in result.errors)
        assert "safety.import" in [v.rule_id for v in result.violations]

    def test_rejects_import_from(self):
        result = validate("from os import path")
        assert not result.valid
        assert any("Import" in e for e in result.errors)
        assert "safety.import" in [v.rule_id for v in result.violations]

    def test_accepts_top_level_function_def(self):
        assert validate("def foo():\n    pass").valid

    def test_rejects_async_function_def(self):
        result = validate("async def foo():\n    pass")
        assert not result.valid
        assert any("AsyncFunctionDef" in e for e in result.errors)

    def test_rejects_class_def(self):
        result = validate("class Foo:\n    pass")
        assert not result.valid
        assert any("ClassDef" in e for e in result.errors)

    def test_rejects_standalone_lambda(self):
        """A lambda assigned to a name stays rejected (style.lambda-assign)."""
        result = validate("x = lambda: 1", allowed_names=set())
        assert not result.valid
        assert "style.lambda-assign" in [v.rule_id for v in result.violations]

    def test_rejects_while(self):
        result = validate("while True:\n    pass")
        assert not result.valid
        assert any("While" in e for e in result.errors)

    def test_rejects_try(self):
        # Task 8: Try is an allowed node; a bare except stays rejected by safety.except-base.
        result = validate("try:\n    x = 1\nexcept:\n    pass")
        assert not result.valid
        assert "safety.except-base" in {v.rule_id for v in result.violations}

    def test_rejects_raise(self):
        result = validate("raise ValueError('bad')")
        assert not result.valid
        assert any("Raise" in e for e in result.errors)

    def test_rejects_global(self):
        result = validate("global x")
        assert not result.valid
        assert any("Global" in e for e in result.errors)

    def test_rejects_assert(self):
        """An assert needs a message (style.assert-message)."""
        result = validate("assert True")
        assert not result.valid
        assert "style.assert-message" in [v.rule_id for v in result.violations]

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
    def test_accepts_normal_string(self):
        result = validate("x = 'hello world'")
        assert result.valid, result.errors

    def test_accepts_dunder_in_string(self):
        """Dunder strings are data, not a sink. See TestDunderStringsAreNotASink.

        The blanket rejection this replaces made ordinary Python authoring
        impossible: no ``__init__.py`` could be written or read, no
        ``if __name__ == "__main__":`` emitted, no ``def __init__`` generated.
        """
        assert validate("x = '__class__'").valid
        for program in (
            "write_file('src/pkg/__init__.py', '')",
            "read_file('src/pkg/__init__.py')",
            'write_file("m.py", \'if __name__ == "__main__":\\n    pass\\n\')',
            'write_file("c.py", "class C:\\n    def __init__(self):\\n        pass\\n")',
        ):
            r = validate(program, allowed_names={"read_file", "write_file"})
            assert r.valid, f"{program!r} -> {r.errors}"


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

    def test_sort_by_rejected(self):
        result = validate("sort_by([{'a': 1}], 'a')")
        assert not result.valid
        assert any(v.rule_id == "safety.unknown-call" for v in result.violations)


class TestLambdaRestriction:
    def test_allows_lambda_as_sort_key(self):
        result = validate(
            "sorted([{'a': 1}], key=lambda x: x['a'])",
            allowed_names=set(),
        )
        assert result.valid

    def test_sort_by_with_lambda_rejected(self):
        result = validate(
            "sort_by([{'a': 1}], key=lambda x: x['a'])",
            allowed_names=set(),
        )
        assert not result.valid
        assert any(v.rule_id == "safety.unknown-call" for v in result.violations)

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

    def test_lambda_positions_are_key_defaultdict_and_re_sub_only(self):
        def ids(src):
            return [v.rule_id for v in validate(src, allowed_names=set()).violations]
        # positional lambda to sorted is a runtime TypeError (key is keyword-only)
        assert "style.lambda-position" in ids("xs = [1]\nsorted(xs, lambda r: r)")
        assert "style.lambda-position" in ids("list(lambda: 1)")
        assert validate("from collections import defaultdict\nd = defaultdict(lambda: 0)").valid
        assert validate("import collections\nd = collections.defaultdict(lambda: 0)").valid
        assert validate('import re\ns = re.sub("a", lambda m: m.group(0), "abc")').valid
        assert validate('import re\np = re.compile("a")\ns = p.sub(lambda m: m.group(0), "abc")').valid
        assert "style.lambda-position" in ids('import re\ns = re.sub(lambda m: 1, "a", "abc")')

    def test_rejects_lambda_in_non_key_kwarg(self):
        result = validate(
            "sorted([1], reverse=lambda x: x)",
            allowed_names=set(),
        )
        assert not result.valid


# Final review I2(a): break and continue compile only inside a loop of their own scope.
# They validated at top level and directly in a def, and then compile() raised SyntaxError.
@pytest.mark.parametrize("program", [
    "break",
    "continue",
    "if True:\n    break",
    "def f():\n    break\n    return 1\nx = f()",
    "def f():\n    continue",
    # a def body inside a for is its own scope
    "for i in [1]:\n    def f():  # ste: style.def-top-level, test\n        break",
    # a for's else clause is not its body
    "for i in [1]:\n    print(i)\nelse:\n    break",
])
def test_break_continue_outside_a_for_body_rejected(program):
    result = validate(program)
    assert any(v.rule_id == "safety.node" and ("Break" in v.message or "Continue" in v.message)
               for v in result.violations), result.violations


@pytest.mark.parametrize("program", [
    "for i in [1]:\n    break",
    "for i in [1]:\n    if i:\n        continue\n    print(i)",
    "for i in [1]:\n    with open_thing() as t:\n        break",
    "d = {}\nfor k in ['a']:\n    try:\n        x = d[k]\n    except KeyError:\n        continue\n    print(x)",
    "for i in [1]:\n    for j in [2]:\n        print(j)\n    else:\n        break",
    "def f(xs):\n    for x in xs:\n        if x:\n            break\n    return 1",
])
def test_break_continue_inside_a_for_body_allowed(program):
    result = validate(program, allowed_names={"open_thing"})
    assert not any("Break" in v.message or "Continue" in v.message for v in result.violations), \
        result.violations


# Final review I2(b): a program that passes every rule but fails to compile is invalid.
@pytest.mark.parametrize("program", [
    "a, *b, *c = [1, 2, 3]",
    "def f(a, a):\n    return a",
    "print(sep='', sep='')",
])
def test_program_that_does_not_compile_is_invalid(program):
    result = validate(program)
    assert not result.valid
    assert any(v.rule_id == "safety.parse" and "Compile error" in v.message
               for v in result.violations), result.violations


def test_calls_is_documented_as_advisory():
    # Final review I3: calls is best-effort, not a soundness boundary.
    assert "advisory" in ValidationResult.__doc__
