"""Style rules for try: a small body, no else or finally, approved handler shapes (Task 8).

The design (section 4.3, "Rules for try") gives four handler shapes:

  - skip:    ``except E: continue``, when the try is in a ``for`` body
  - return:  ``except E: return <value>``, when the try is in a ``def``
  - ignore:  ``except E: pass``, when the try body assigns no name
  - default: ``except E: x = <value>``, assigning every name the body assigns

Print calls and assignments may come before the shape, and a handler may
also end with no shape and fall through (design 4.3: handlers that report and
fall through conform when they satisfy the same rule). One idea covers all of
them: after a try, every name that the body assigns is assigned in each
handler, or the handler leaves the flow (definite assignment).
``safety.except-base`` owns every check on the handler's classes.
"""

from __future__ import annotations

import ast
from collections import Counter

from .registry import Context, Violation, register
from .rules_safety import program_bindings

MAX_TRY_BODY = 3


def _tries(ctx: Context):
    return (n for n in ast.walk(ctx.tree) if isinstance(n, ast.Try))


@register(
    "style.try-body", "style",
    summary=f"Put at most {MAX_TRY_BODY} statements in a try body.",
    rationale="A small body ties the handler to the statement that can fail.",
    bad="d = {}\ntry:\n    a = d['a']\n    b = d['b']\n    c = d['c']\n    e = d['e']\n"
        "except KeyError:\n    a = b = c = e = 0",
    good="d = {}\ntry:\n    a = d['a']\nexcept KeyError:\n    a = 0",
)
def check_try_body(ctx: Context) -> list[Violation]:
    return [Violation("style.try-body", node.lineno,
                      f"Try at line {node.lineno} has {len(node.body)} statements in its body "
                      f"(at most {MAX_TRY_BODY})")
            for node in _tries(ctx) if len(node.body) > MAX_TRY_BODY]


@register(
    "style.try-else-finally", "style",
    summary="Do not use else or finally on a try. Use with for cleanup.",
    rationale="else and finally add paths that the reader must follow. "
              "A with statement does the cleanup.",
    bad="d = {}\ntry:\n    x = d['k']\nexcept KeyError:\n    x = 0\nfinally:\n    print('done')",
    good="d = {}\ntry:\n    x = d['k']\nexcept KeyError:\n    x = 0\nprint('done')",
)
def check_try_else_finally(ctx: Context) -> list[Violation]:
    out = []
    for node in _tries(ctx):
        for word, body in (("else", node.orelse), ("finally", node.finalbody)):
            if body:
                out.append(Violation(
                    "style.try-else-finally", node.lineno,
                    f"Try at line {node.lineno} must not have {word}:"))
    return out


_SCOPES = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.Lambda,
           ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _assigned_names(stmts: list[ast.stmt]) -> set[str]:
    """The names that ``stmts`` bind, as seen after them.

    Comprehension and lambda variables are left out (their own scope). An
    augmented assignment is left out: it needs an earlier binding of the name.
    """
    names: set[str] = set()
    stack: list[ast.AST] = list(stmts)
    while stack:
        node = stack.pop()
        if isinstance(node, _SCOPES):
            continue
        if isinstance(node, ast.AugAssign):
            stack.append(node.value)
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        stack.extend(ast.iter_child_nodes(node))
    return names


def _in_for_body(node: ast.AST, ctx: Context) -> bool:
    """Whether ``node`` is inside the body of a ``for`` loop of its own function or program."""
    child, parent = node, ctx.parents.get(id(node))
    while parent is not None:
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return False
        if isinstance(parent, ast.For) and child in parent.body:
            return True
        child, parent = parent, ctx.parents.get(id(parent))
    return False


def _in_def(node: ast.AST, ctx: Context) -> bool:
    parent = ctx.parents.get(id(node))
    while parent is not None:
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return True
        parent = ctx.parents.get(id(parent))
    return False


def _is_print(stmt: ast.stmt) -> bool:
    return (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Name) and stmt.value.func.id == "print")


def _is_assign(stmt: ast.stmt) -> bool:
    return isinstance(stmt, ast.Assign) or (isinstance(stmt, ast.AnnAssign) and stmt.value is not None)


def _handler_problem(try_node: ast.Try, handler: ast.ExceptHandler,
                     body_names: set[str], ctx: Context) -> str | None:
    """Why ``handler`` has no approved shape, or None when it has one.

    The handler is print calls and assignments, then optionally one of
    ``continue``, ``return <value>`` or ``pass``. A handler that does not
    leave the flow (no ``continue`` or ``return``) falls through, so it must
    assign every name that the try body assigns (definite assignment).
    """
    stmts = handler.body
    last = stmts[-1]
    terminal = last if isinstance(last, (ast.Continue, ast.Return, ast.Pass)) else None
    lead = stmts[:-1] if terminal is not None else stmts
    if not all(_is_print(s) or _is_assign(s) for s in lead):
        return ("only print calls and assignments may come before the end "
                "(continue, return <value>, pass, or nothing)")
    if isinstance(terminal, ast.Continue):
        return None if _in_for_body(try_node, ctx) else "continue is only allowed in a for body"
    if isinstance(terminal, ast.Return):
        if terminal.value is None:
            return "return must give a value"
        return None if _in_def(try_node, ctx) else "return is only allowed in a def"
    # Python deletes the as-name when the handler ends, so it assigns nothing after the try.
    missing = body_names - (_assigned_names(lead) - {handler.name})
    if missing:
        return (f"it falls through and leaves {', '.join(sorted(missing))} unassigned "
                f"(assign it, or use continue or return)")
    return None


@register(
    "style.try-handler", "style",
    summary="An except may hold only print calls and assignments. It may end with continue "
            "(in a for body), `return <value>` (in a def) or pass. An except that does not "
            "continue or return must assign every name that the try body assigns.",
    rationale="After a try, every name that the body assigns must have a value, or the "
              "handler must leave the flow. Each shape gives one sentence in the procedure, "
              "for example 'If d[k] raises KeyError, use 0 for x.'",
    bad="d = {}\ntry:\n    x = d['k']\nexcept KeyError:\n    print('missing')\nprint(x)",
    good="d = {}\ntry:\n    x = d['k']\nexcept KeyError:\n    print('missing')\n    x = 0\nprint(x)",
)
def check_try_handler(ctx: Context) -> list[Violation]:
    out = []
    # Python deletes the as-name when the handler ends, so an outer binding of
    # that name is gone after the try. Other except-as sites of the same name
    # bind and delete their own value, so they do not count.
    handler_names = Counter(n.name for n in ast.walk(ctx.tree)
                            if isinstance(n, ast.ExceptHandler) and n.name)
    other_bindings = Counter(program_bindings(ctx.tree)) - handler_names
    for handler in (n for n in ast.walk(ctx.tree) if isinstance(n, ast.ExceptHandler)):
        if handler.name and other_bindings[handler.name]:
            out.append(Violation(
                "style.try-handler", handler.lineno,
                f"Except at line {handler.lineno} binds '{handler.name}', which the program "
                f"also binds; the handler deletes it at its end (use another name, such as err)"))
    for node in _tries(ctx):
        body_names = _assigned_names(node.body)
        for handler in node.handlers:
            problem = _handler_problem(node, handler, body_names, ctx)
            if problem:
                out.append(Violation(
                    "style.try-handler", handler.lineno,
                    f"Except at line {handler.lineno} has no approved shape: {problem}"))
    return out
