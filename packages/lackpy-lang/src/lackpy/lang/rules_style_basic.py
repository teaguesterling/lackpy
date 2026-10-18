"""Style rules for the simple constructs: assert, for, lambda (Task 6)."""

from __future__ import annotations

import ast

from .registry import Context, Violation, register


def _lambda_arg_ok(call: ast.Call, index: int, ctx: Context) -> bool:
    """A positional lambda is allowed as defaultdict's factory or the repl of re.sub/subn.

    Names resolve through the import bindings, so ``import re as regex`` works.
    """
    func = call.func
    if isinstance(func, ast.Name):
        target = ctx.member_bindings.get(func.id)
        if target == ("collections", "defaultdict"):
            return index == 0
        if target in (("re", "sub"), ("re", "subn")):
            return index == 1
        return False
    if isinstance(func, ast.Attribute):
        module = (ctx.module_bindings.get(func.value.id)
                  if isinstance(func.value, ast.Name) else None)
        if func.attr == "defaultdict":
            return module == "collections" and index == 0
        if func.attr in ("sub", "subn"):
            # re.sub(pattern, repl, s) has repl second; pattern.sub(repl, s) has it first.
            return index == (1 if module == "re" else 0)
    return False


@register(
    "style.assert-message", "style",
    summary="Give every assert a message that is a string or an f-string.",
    rationale="The message says what the assert protects. A name or a call hides it.",
    bad="x = 1\nassert x",
    good='x = 1\nassert x == 1, "x is one"',
)
def check_assert_message(ctx: Context) -> list[Violation]:
    out = []
    for node in ast.walk(ctx.tree):
        if isinstance(node, ast.Assert):
            msg = node.msg
            ok = isinstance(msg, ast.JoinedStr) or (
                isinstance(msg, ast.Constant) and isinstance(msg.value, str))
            if not ok:
                out.append(Violation(
                    "style.assert-message", node.lineno,
                    f"Assert at line {node.lineno} must have a string or f-string message",
                ))
    return out


def _has_call(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Call) for n in ast.walk(node))


@register(
    "style.for-iter", "style",
    summary="A for loop iterates over a call, a variable, or an expression with no call.",
    rationale="A call or a name shows what the loop reads. "
              "A call buried in a larger expression hides it.",
    bad="x = [3, 1]\nfor k in sorted(x)[:3]:\n    print(k)",
    good="x = [3, 1]\nfor k in x[:3]:\n    print(k)",
)
def check_for_iter(ctx: Context) -> list[Violation]:
    out = []
    for node in ast.walk(ctx.tree):
        if (isinstance(node, ast.For)
                and not isinstance(node.iter, (ast.Call, ast.Name))
                and _has_call(node.iter)):
            out.append(Violation(
                "style.for-iter", node.lineno,
                f"For-loop at line {node.lineno} must iterate over "
                f"a function call or variable",
            ))
    return out


@register(
    "style.lambda-position", "style",
    summary="Use lambda only as a key= value, a defaultdict factory, or the repl of re.sub.",
    rationale="A lambda in a key= argument is short and has one job. "
              "A lambda in other places hides logic.",
    bad="xs = [1]\nm = sorted(xs, lambda r: r)",
    good="words = sorted(['b', 'a'], key=lambda w: w)",
)
def check_lambda_position(ctx: Context) -> list[Violation]:
    out = []
    for node in ast.walk(ctx.tree):
        if isinstance(node, ast.Lambda):
            parent = ctx.parents.get(id(node))
            if isinstance(parent, ast.Assign):
                continue  # style.lambda-assign owns this case
            if isinstance(parent, ast.keyword) and parent.arg == "key":
                continue
            if (isinstance(parent, ast.Call) and node in parent.args
                    and _lambda_arg_ok(parent, parent.args.index(node), ctx)):
                continue
            out.append(Violation(
                "style.lambda-position", node.lineno,
                f"Lambda at line {node.lineno} is only allowed as a key= argument "
                f"(e.g., key=lambda x: x['field']), as the factory of "
                f"defaultdict(...), or as the repl argument of re.sub or re.subn",
            ))
    return out


@register(
    "style.lambda-assign", "style",
    summary="Do not assign a lambda to a name.",
    rationale="A named lambda is a function in disguise. Use a comprehension or a key= argument.",
    bad="f = lambda x: x",
    good="words = sorted(['b', 'a'], key=lambda w: w)",
)
def check_lambda_assign(ctx: Context) -> list[Violation]:
    return [
        Violation("style.lambda-assign", node.lineno,
                  f"Lambda assigned to a name at line {node.lineno}: "
                  f"use it as a key= argument instead")
        for node in ast.walk(ctx.tree)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Lambda)
    ]
