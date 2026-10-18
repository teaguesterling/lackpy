"""Style rules for try: named errors, a small body, approved handler shapes (Task 8).

The design (section 4.3, "Rules for try") gives four handler shapes:

  - skip:    ``except E: continue``, when the try is in a ``for`` body
  - return:  ``except E: return <value>``, when the try is in a ``def``
  - ignore:  ``except E: pass``, when the try body assigns no name
  - default: ``except E: x = <value>``, assigning every name the body assigns

Print calls may come before the shape. One idea covers all four: after a try,
every name that the body assigns is assigned in each handler, or the handler
leaves the flow (definite assignment). ``safety.except-base`` owns the bare
``except`` and the non-Exception classes.
"""

from __future__ import annotations

import ast
import importlib

from .builtins_policy import ALLOWED
from .dictionary import MODULES
from .registry import Context, Violation, register
from .rules_safety import BASE_ONLY_EXCEPTIONS, handler_classes

MAX_TRY_BODY = 3


def _exception_members() -> frozenset[tuple[str, str]]:
    """The (module, member) pairs in ``MODULES`` whose real attribute is an exception class."""
    found = set()
    for module, members in MODULES.items():
        mod = importlib.import_module(module)
        for member in members:
            value = getattr(mod, member, None)
            if isinstance(value, type) and issubclass(value, Exception):
                found.add((module, member))
    return frozenset(found)


EXCEPTION_MEMBERS = _exception_members()  # for example ("json", "JSONDecodeError"), ("re", "error")


def _tries(ctx: Context):
    return (n for n in ast.walk(ctx.tree) if isinstance(n, ast.Try))


@register(
    "style.try-named", "style",
    summary="Name each error class in an except: a builtin error such as KeyError, "
            "or a dictionary error such as json.JSONDecodeError. Do not use Exception.",
    rationale="The named class says which failure the handler expects. "
              "Exception also catches the defects that the program must show.",
    bad="d = {}\ntry:\n    x = d['k']\nexcept Exception:\n    x = 0",
    good="d = {}\ntry:\n    x = d['k']\nexcept KeyError:\n    x = 0",
)
def check_try_named(ctx: Context) -> list[Violation]:
    out = []
    for node in ast.walk(ctx.tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        for cls in handler_classes(node):
            if isinstance(cls, ast.Name) and cls.id in BASE_ONLY_EXCEPTIONS:
                continue  # safety.except-base owns these
            if not _named_class(cls, ctx):
                out.append(Violation(
                    "style.try-named", node.lineno,
                    f"Except at line {node.lineno} must name error classes, not "
                    f"'{ast.unparse(cls)}' (use a builtin error such as KeyError, "
                    f"or json.JSONDecodeError or re.error)"))
    return out


def _named_class(cls: ast.expr, ctx: Context) -> bool:
    if isinstance(cls, ast.Name):
        if cls.id in ctx.member_bindings:
            return ctx.member_bindings[cls.id] in EXCEPTION_MEMBERS
        return cls.id != "Exception" and ALLOWED.get(cls.id) == "exception"
    if isinstance(cls, ast.Attribute) and isinstance(cls.value, ast.Name):
        module = ctx.module_bindings.get(cls.value.id)
        return (module, cls.attr) in EXCEPTION_MEMBERS
    return False


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


def _handler_problem(try_node: ast.Try, handler: ast.ExceptHandler,
                     body_names: set[str], ctx: Context) -> str | None:
    """Why ``handler`` has no approved shape, or None when it has one."""
    stmts = handler.body
    i = 0
    while i < len(stmts) and _is_print(stmts[i]):
        i += 1
    tail = stmts[i:]
    if not tail:
        return "it ends in print (end in continue, return <value>, pass or assignments)"
    if len(tail) == 1 and isinstance(tail[0], ast.Continue):
        return None if _in_for_body(try_node, ctx) else "continue is only allowed in a for body"
    if len(tail) == 1 and isinstance(tail[0], ast.Return):
        if tail[0].value is None:
            return "return must give a value"
        return None if _in_def(try_node, ctx) else "return is only allowed in a def"
    if len(tail) == 1 and isinstance(tail[0], ast.Pass):
        if body_names:
            return (f"pass leaves {', '.join(sorted(body_names))} unassigned "
                    f"(assign it, or use continue or return)")
        return None
    if all(isinstance(s, ast.Assign) or (isinstance(s, ast.AnnAssign) and s.value is not None)
           for s in tail):
        missing = body_names - _assigned_names(tail)
        if missing:
            return f"it does not assign {', '.join(sorted(missing))}"
        return None
    return ("only print calls may come before the last statement "
            "(continue, return <value>, pass or assignments)")


@register(
    "style.try-handler", "style",
    summary="End each except in continue (in a for body), return <value> (in a def), "
            "pass (when the try body assigns no name), or assignments to every name "
            "that the try body assigns. Only print calls may come before it.",
    rationale="After a try, every name that the body assigns must have a value, or the "
              "handler must leave the flow. Each shape gives one sentence in the procedure, "
              "for example 'If d[k] raises KeyError, use 0 for x.'",
    bad="d = {}\ntry:\n    x = d['k']\nexcept KeyError:\n    pass\nprint(x)",
    good="d = {}\ntry:\n    x = d['k']\nexcept KeyError:\n    x = 0\nprint(x)",
)
def check_try_handler(ctx: Context) -> list[Violation]:
    out = []
    for node in _tries(ctx):
        body_names = _assigned_names(node.body)
        for handler in node.handlers:
            problem = _handler_problem(node, handler, body_names, ctx)
            if problem:
                out.append(Violation(
                    "style.try-handler", handler.lineno,
                    f"Except at line {handler.lineno} has no approved shape: {problem}"))
    return out
