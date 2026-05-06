"""Static analysis for cell pre-flight checks.

Performs two checks before executing a cell:
1. compile() - catches syntax errors, malformed f-strings
2. AST name resolution - catches undefined references
"""

from __future__ import annotations

import ast
import builtins

_BUILTINS = set(dir(builtins))


class StaticAnalysisError(Exception):
    pass


def check_cell(code: str, known_names: set[str]) -> None:
    try:
        tree = compile(code, "<cell>", "exec", ast.PyCF_ONLY_AST)
    except SyntaxError as e:
        raise StaticAnalysisError(f"syntax error: {e.msg} (line {e.lineno})") from e

    defined = _collect_definitions(tree)
    referenced = _collect_references(tree)

    available = known_names | _BUILTINS | defined
    undefined = referenced - available

    if undefined:
        names = ", ".join(f"'{n}'" for n in sorted(undefined))
        raise StaticAnalysisError(f"undefined names: {names}")


def _collect_definitions(tree: ast.Module) -> set[str]:
    defined: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                defined.update(_names_from_target(target))
        elif isinstance(node, ast.AugAssign):
            pass
        elif isinstance(node, ast.AnnAssign) and node.target:
            defined.update(_names_from_target(node.target))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(node.name)
            defined.update(_names_from_args(node.args))
        elif isinstance(node, ast.Lambda):
            defined.update(_names_from_args(node.args))
        elif isinstance(node, ast.ClassDef):
            defined.add(node.name)
        elif isinstance(node, ast.For):
            defined.update(_names_from_target(node.target))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                defined.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                defined.add(alias.asname or alias.name)
        elif isinstance(node, ast.With):
            for item in node.items:
                if item.optional_vars:
                    defined.update(_names_from_target(item.optional_vars))
        elif isinstance(node, ast.NamedExpr):
            defined.add(node.target.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            for gen in node.generators:
                defined.update(_names_from_target(gen.target))
    return defined


def _collect_references(tree: ast.Module) -> set[str]:
    refs: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            refs.add(node.id)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            refs.add(node.target.id)
    return refs


def _names_from_args(args: ast.arguments) -> set[str]:
    names: set[str] = set()
    for arg in args.posonlyargs + args.args + args.kwonlyargs:
        names.add(arg.arg)
    if args.vararg:
        names.add(args.vararg.arg)
    if args.kwarg:
        names.add(args.kwarg.arg)
    return names


def _names_from_target(node: ast.expr) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    elif isinstance(node, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for elt in node.elts:
            names.update(_names_from_target(elt))
        return names
    elif isinstance(node, ast.Starred):
        return _names_from_target(node.value)
    return set()
