"""Tests for custom validation rules."""

import ast

from lackpy.lang.rules import no_loops, max_depth, max_calls, no_nested_calls


class TestNoLoops:
    def test_rejects_for_loop(self):
        tree = ast.parse("for x in range(10):\n    print(x)")
        errors = no_loops(tree)
        assert len(errors) > 0

    def test_accepts_no_loops(self):
        tree = ast.parse("x = 1\ny = 2")
        errors = no_loops(tree)
        assert errors == []


class TestMaxDepth:
    def test_rejects_deep_nesting(self):
        code = "if True:\n    if True:\n        if True:\n            x = 1"
        tree = ast.parse(code)
        errors = max_depth(2)(tree)
        assert len(errors) > 0

    def test_accepts_shallow_nesting(self):
        tree = ast.parse("if True:\n    x = 1")
        errors = max_depth(2)(tree)
        assert errors == []


class TestMaxCalls:
    def test_rejects_too_many_calls(self):
        tree = ast.parse("a = read_file('x')\nb = read_file('y')\nc = read_file('z')")
        errors = max_calls(2)(tree)
        assert len(errors) > 0

    def test_accepts_within_limit(self):
        tree = ast.parse("a = read_file('x')\nb = read_file('y')")
        errors = max_calls(2)(tree)
        assert errors == []


class TestNoNestedCalls:
    def test_rejects_nested_call(self):
        tree = ast.parse("len(read_file('x'))")
        errors = no_nested_calls(tree)
        assert len(errors) > 0

    def test_accepts_flat_calls(self):
        tree = ast.parse("x = read_file('f')\nlen(x)")
        errors = no_nested_calls(tree)
        assert errors == []
