"""Static analysis for cell pre-flight checks.

Called by LightweightKernel.execute_cell() BEFORE every cell evaluation.
Two checks:
  1. compile() — catches syntax errors, malformed f-strings
  2. AST name resolution — walks the AST to find Name(Load) nodes and
     verifies each against known_names (kernel namespace) + builtins.
     Catches forward references like using {x} before defining x.

Raises StaticAnalysisError on failure; the kernel returns this as
CellResult(success=False, error_phase="static").
"""

from __future__ import annotations

import ast
import builtins

_BUILTINS = set(dir(builtins))

#: Public view of the builtin-name set ``check_cell`` resolves against.
#: Callers doing their own pre-flight name resolution (the batch path's
#: forgiveness pre-pass) must use THIS set so they can never disagree with
#: the kernel's own static check.
BUILTIN_NAMES: frozenset[str] = frozenset(_BUILTINS)


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


def collect_names(tree: ast.Module) -> tuple[set[str], set[str]]:
    """Public ``(definitions, references)`` extraction for one cell's AST.

    Exactly the sets :func:`check_cell` computes internally — exposed so the
    batch path's forgiveness pre-pass (L1.1) resolves names with the same
    rules the kernel enforces.  (These per-cell def/ref sets are also the raw
    material for the future dirty-subgraph dependency graph, L1.4.)
    """
    return _collect_definitions(tree), _collect_references(tree)


def collect_bindings(tree: ast.Module) -> set[str]:
    """Names a cell binds in the KERNEL NAMESPACE when it executes.

    Narrower than :func:`_collect_definitions`, which includes scope-local
    names (function/lambda parameters, comprehension targets, class-body
    assignments) because for *reference checking* anything defined anywhere
    in the cell is resolvable.  For *binding* purposes — e.g. deciding which
    names a blocked cell would have bound, so they can be reified as chained
    holes — only module-level bindings count: binding a hole to a function
    parameter would pollute the namespace with a name that never exists.

    Walks top-level statements, descending into compound statements
    (``for``/``while``/``if``/``with``/``try``) but never into new scopes
    (``def``/``class``/lambdas/comprehensions).  Top-level walrus targets in
    bare expressions are not collected (rare; they simply won't be reified).
    """
    bound: set[str] = set()

    def visit(stmts: list[ast.stmt]) -> None:
        for node in stmts:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    bound.update(_names_from_target(target))
            elif isinstance(node, ast.AnnAssign):
                # `x: int` alone declares without binding; only `x: int = v`
                # binds.
                if node.value is not None and node.target:
                    bound.update(_names_from_target(node.target))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    bound.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    bound.add(alias.asname or alias.name)
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                bound.update(_names_from_target(node.target))
                visit(node.body)
                visit(node.orelse)
            elif isinstance(node, (ast.While, ast.If)):
                visit(node.body)
                visit(node.orelse)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    if item.optional_vars:
                        bound.update(_names_from_target(item.optional_vars))
                visit(node.body)
            elif isinstance(node, ast.Try):
                visit(node.body)
                visit(node.orelse)
                visit(node.finalbody)
                for handler in node.handlers:
                    visit(handler.body)

    visit(tree.body)
    return bound


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
