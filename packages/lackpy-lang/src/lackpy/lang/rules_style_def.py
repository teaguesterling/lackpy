"""Style rules for function definitions: top-level def only, short, no recursion (Task 7)."""

from __future__ import annotations

import ast

from .grammar import ALLOWED_BUILTINS
from .registry import Context, Violation, register

MAX_DEF_STATEMENTS = 12


def _top_level_defs(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    """The first top-level def of each name."""
    defs: dict[str, ast.FunctionDef] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef):
            defs.setdefault(stmt.name, stmt)
    return defs


def defined_before(tree: ast.Module, name: str, node: ast.AST) -> bool:
    """Whether the top-level def of ``name`` ends before ``node`` starts, by source position."""
    fn = _top_level_defs(tree).get(name)
    return fn is not None and (fn.end_lineno, fn.end_col_offset) <= (node.lineno, node.col_offset)


def _all_defs(ctx: Context):
    return [n for n in ast.walk(ctx.tree) if isinstance(n, ast.FunctionDef)]


def _own_returns(fn: ast.FunctionDef):
    """The ``return`` statements of ``fn`` itself, not those of a def nested in it."""
    stack = list(fn.body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Return):
            yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            stack.extend(ast.iter_child_nodes(node))


def _is_constant(node: ast.AST | None) -> bool:
    if isinstance(node, ast.Constant):
        return node.value is None or isinstance(node.value, (bool, int, float, complex, str))
    if isinstance(node, ast.Tuple):
        return all(_is_constant(e) for e in node.elts)
    if (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)):
        v = node.operand.value
        return isinstance(v, (int, float, complex)) and not isinstance(v, bool)
    return False


@register(
    "style.def-top-level", "style",
    summary="Write def only at the top level of the program, with no decorator.",
    rationale="A def inside a block or another def hides its scope. A decorator runs code that "
              "the reader cannot see.",
    bad="def outer():\n    def inner():\n        return 1\n    return inner()",
    good="def inner():\n    return 1\ndef outer():\n    return inner()",
)
def check_def_top_level(ctx: Context) -> list[Violation]:
    out = []
    for fn in _all_defs(ctx):
        if not isinstance(ctx.parents.get(id(fn)), ast.Module):
            out.append(Violation(
                "style.def-top-level", fn.lineno,
                f"Function '{fn.name}' at line {fn.lineno} must be defined at the top level"))
        elif fn.decorator_list:
            out.append(Violation(
                "style.def-top-level", fn.lineno,
                f"Function '{fn.name}' at line {fn.lineno} must not have a decorator"))
    return out


@register(
    "style.def-before-use", "style",
    summary="Call a program function only after its def ends.",
    rationale="Order shows the dependencies. It also rules out recursion, "
              "because a function is not complete inside its own body.",
    bad="def f(n):\n    return f(n)",
    good="def f(n):\n    return n\nx = f(1)",
)
def check_def_before_use(ctx: Context) -> list[Violation]:
    out = []
    for node in ast.walk(ctx.tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in ctx.defined_functions
                and not defined_before(ctx.tree, node.func.id, node)):
            out.append(Violation(
                "style.def-before-use", node.lineno,
                f"Function '{node.func.id}' is called at line {node.lineno} "
                f"before its def at line {ctx.defined_functions[node.func.id]} ends "
                f"(recursion is not allowed)"))
    return out


@register(
    "style.def-signature", "style",
    summary="Take plain parameters. Use no *args or **kwargs, and only constant defaults.",
    rationale="A default is evaluated once, when the def runs. A constant default has no hidden "
              "state. *args and **kwargs hide the parameters.",
    bad="def f(a=[]):\n    return a",
    good="def f(a=None):\n    return a",
)
def check_def_signature(ctx: Context) -> list[Violation]:
    out = []
    for fn in _all_defs(ctx):
        args = fn.args
        if args.vararg or args.kwarg:
            star = args.vararg or args.kwarg
            out.append(Violation(
                "style.def-signature", fn.lineno,
                f"Function '{fn.name}' at line {fn.lineno} must not take "
                f"{'*' if args.vararg else '**'}{star.arg}"))
        defaults = list(args.defaults) + [d for d in args.kw_defaults if d is not None]
        if not all(_is_constant(d) for d in defaults):
            out.append(Violation(
                "style.def-signature", fn.lineno,
                f"Function '{fn.name}' at line {fn.lineno} has a default that is not a constant "
                f"(use None, a bool, a number, a string, or a tuple of them)"))
    return out


@register(
    "style.def-return-mode", "style",
    summary="Return a value from every return statement, or from none.",
    rationale="A function that sometimes returns a value and sometimes None makes the caller "
              "check both.",
    bad="def f(x):\n    if x:\n        return 1\n    return",
    good="def f(x):\n    if x:\n        return 1\n    return 0",
)
def check_def_return_mode(ctx: Context) -> list[Violation]:
    out = []
    for fn in _all_defs(ctx):
        returns = list(_own_returns(fn))
        if any(r.value is None for r in returns) and any(r.value is not None for r in returns):
            out.append(Violation(
                "style.def-return-mode", fn.lineno,
                f"Function '{fn.name}' at line {fn.lineno} mixes return with a value "
                f"and return without a value"))
    return out


@register(
    "style.def-returns-function", "style",
    summary="Do not return a function.",
    rationale="A returned function carries hidden state and is called later, "
              "where the validator cannot follow it.",
    bad="def f():\n    return lambda: 1",
    good="def f():\n    return 1",
)
def check_def_returns_function(ctx: Context) -> list[Violation]:
    out = []
    for fn in _all_defs(ctx):
        for r in _own_returns(fn):
            if isinstance(r.value, ast.Lambda) or (
                    isinstance(r.value, ast.Name) and r.value.id in ctx.defined_functions):
                out.append(Violation(
                    "style.def-returns-function", r.lineno,
                    f"Function '{fn.name}' returns a function at line {r.lineno}"))
    return out


@register(
    "style.def-length", "style",
    summary=f"Keep a function to {MAX_DEF_STATEMENTS} statements, nested ones included.",
    rationale="A short function fits in one read. A long one needs a split into named steps.",
    bad="def f():\n" + "".join(f"    a{i} = {i}\n" for i in range(MAX_DEF_STATEMENTS + 1)),
    good="def f():\n" + "".join(f"    a{i} = {i}\n" for i in range(MAX_DEF_STATEMENTS)),
)
def check_def_length(ctx: Context) -> list[Violation]:
    out = []
    for fn in _all_defs(ctx):
        count = sum(isinstance(n, ast.stmt) for n in ast.walk(fn)) - 1
        if count > MAX_DEF_STATEMENTS:
            out.append(Violation(
                "style.def-length", fn.lineno,
                f"Function '{fn.name}' at line {fn.lineno} has {count} statements "
                f"(at most {MAX_DEF_STATEMENTS})"))
    return out


@register(
    "style.def-shadow", "style",
    summary="Do not name a def after a builtin or an import binding.",
    rationale="A def that reuses a builtin name changes what that name means for the rest "
              "of the program.",
    bad="def sorted(x):\n    return x",
    good="def sort_rows(x):\n    return x",
)
def check_def_shadow(ctx: Context) -> list[Violation]:
    return [
        Violation("style.def-shadow", fn.lineno,
                  f"Function name '{fn.name}' at line {fn.lineno} reuses a builtin or imported name")
        for fn in _all_defs(ctx)
        if fn.name in ALLOWED_BUILTINS or fn.name in ctx.member_bindings
        or fn.name in ctx.module_bindings
    ]
