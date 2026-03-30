"""Custom validation rules for lackpy programs.

Each rule is a callable: (ast.Module) -> list[str].
Empty list means the rule passes.
Rules can only tighten the core checks, never loosen them.
"""

from __future__ import annotations

import ast
from typing import Callable

Rule = Callable[[ast.Module], list[str]]


def no_loops(tree: ast.Module) -> list[str]:
    errors = []
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            errors.append(f"For-loop forbidden (line {node.lineno})")
    return errors


def max_depth(limit: int) -> Rule:
    def _check(tree: ast.Module) -> list[str]:
        errors: list[str] = []
        def _walk(node: ast.AST, depth: int) -> None:
            if depth > limit:
                lineno = getattr(node, "lineno", "?")
                errors.append(f"Nesting depth {depth} exceeds limit {limit} (line {lineno})")
                return
            for child in ast.iter_child_nodes(node):
                child_depth = depth + 1 if _is_block(child) else depth
                _walk(child, child_depth)
        _walk(tree, 0)
        return errors
    return _check


def _is_block(node: ast.AST) -> bool:
    return isinstance(node, (ast.If, ast.For, ast.With))


def max_calls(limit: int) -> Rule:
    def _check(tree: ast.Module) -> list[str]:
        count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Call))
        if count > limit:
            return [f"Too many calls: {count} exceeds limit {limit}"]
        return []
    return _check


def no_nested_calls(tree: ast.Module) -> list[str]:
    errors = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for arg in node.args:
                if isinstance(arg, ast.Call):
                    lineno = getattr(node, "lineno", "?")
                    errors.append(f"Nested call at line {lineno}: assign inner call to a variable first")
            for kw in node.keywords:
                if isinstance(kw.value, ast.Call):
                    lineno = getattr(node, "lineno", "?")
                    errors.append(f"Nested call in keyword arg at line {lineno}: assign inner call to a variable first")
    return errors
