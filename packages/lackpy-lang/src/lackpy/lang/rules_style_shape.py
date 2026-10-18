"""Style rules for the shape of an expression (Task 9).

Each rule keeps one expression easy to read in one pass: short chains, shallow
call nesting, I/O at the start of a chain, and comprehensions with one loop.
"""

from __future__ import annotations

import ast

from .dictionary import PATH_IO_METHODS
from .registry import Context, Violation, register

MAX_CALL_NESTING = 2
MAX_METHOD_CHAIN = 3
MAX_ATTR_NAMES = 3  # a.b.c is allowed, a.b.c.d is not
ELIF_BRANCHES = 3

_COMPS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
# A call inside these counts from its own position, not from the enclosing call.
_BOUNDARY = _COMPS + (ast.Lambda,)


def _calls(ctx: Context):
    return (n for n in ast.walk(ctx.tree) if isinstance(n, ast.Call))


def _nearest_calls(node: ast.AST):
    """Yield the nearest Call nodes below ``node``, without crossing a comprehension or lambda."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.Call):
            yield child
        elif not isinstance(child, _BOUNDARY):
            yield from _nearest_calls(child)


def _arg_calls(call: ast.Call):
    """The calls that sit in the arguments of ``call`` (positional or keyword)."""
    for arg in [*call.args, *(k.value for k in call.keywords)]:
        if isinstance(arg, ast.Call):
            yield arg
        elif not isinstance(arg, _BOUNDARY):
            yield from _nearest_calls(arg)


def _is_io(node: ast.AST) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr in PATH_IO_METHODS)


def _is_method_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)


def _receiver_chain(call: ast.Call):
    """Yield the nodes of the receiver of ``call``, from the nearest outward."""
    node = call.func.value if isinstance(call.func, ast.Attribute) else None
    while node is not None:
        yield node
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            node = node.func.value
        elif isinstance(node, (ast.Attribute, ast.Subscript)):
            node = node.value
        else:
            node = None


def _depth(call: ast.Call) -> int:
    return 1 + max((_depth(c) for c in _arg_calls(call)), default=0)


@register(
    "style.genexp-consumed", "style",
    summary="Pass a generator expression straight to a call. Do not store it.",
    rationale="A stored generator runs later, once, and hides its cost. "
              "A call that consumes it makes the order of work clear.",
    bad="xs = [1, 2]\ng = (x for x in xs)\nprint(g)",
    good="xs = [1, 2]\nn = sum(x for x in xs)\nprint(n)",
)
def check_genexp_consumed(ctx: Context) -> list[Violation]:
    out = []
    for node in ast.walk(ctx.tree):
        if not isinstance(node, ast.GeneratorExp):
            continue
        parent = ctx.parents.get(id(node))
        if isinstance(parent, ast.keyword):
            parent = ctx.parents.get(id(parent))
        if not (isinstance(parent, ast.Call) and node is not parent.func):
            out.append(Violation(
                "style.genexp-consumed", node.lineno,
                f"Generator expression at line {node.lineno} must be an argument of a call",
                hint="Use a list comprehension, or pass it to sum, any, all, next or sorted"))
    return out


@register(
    "style.attr-chain", "style",
    summary=f"Use at most {MAX_ATTR_NAMES} names in a dotted chain (a.b.c).",
    rationale="A long chain reaches through objects the reader must hold in mind. "
              "A named step says what each part is.",
    bad="x = {}\ny = x.a.b.c",
    good="x = {}\nb = x.a.b\ny = b.c",
)
def check_attr_chain(ctx: Context) -> list[Violation]:
    out = []
    for node in ast.walk(ctx.tree):
        if not isinstance(node, ast.Attribute):
            continue
        parent = ctx.parents.get(id(node))
        if isinstance(parent, ast.Attribute) and parent.value is node:
            continue  # only the outermost link reports
        links, base = 0, node
        while isinstance(base, ast.Attribute):
            links += 1
            base = base.value
        if isinstance(base, ast.Name) and links + 1 > MAX_ATTR_NAMES:
            out.append(Violation(
                "style.attr-chain", node.lineno,
                f"Attribute chain at line {node.lineno} has {links + 1} names "
                f"(at most {MAX_ATTR_NAMES})",
                hint="Assign the middle of the chain to a name"))
    return out


@register(
    "style.method-chain", "style",
    summary=f"Chain at most {MAX_METHOD_CHAIN} method calls.",
    rationale="Each call in a chain changes the value. A long chain hides the type at each step.",
    bad="s = ' a b '\ny = s.strip().lower().split().pop()",
    good="s = ' a b '\nwords = s.strip().lower().split()\ny = words.pop()",
)
def check_method_chain(ctx: Context) -> list[Violation]:
    out = []
    for call in _calls(ctx):
        if not _is_method_call(call):
            continue
        parent = ctx.parents.get(id(call))
        grand = ctx.parents.get(id(parent)) if parent is not None else None
        if (isinstance(parent, ast.Attribute) and parent.value is call
                and isinstance(grand, ast.Call) and grand.func is parent):
            continue  # only the outermost link reports
        links, node = 0, call
        while _is_method_call(node):
            links += 1
            node = node.func.value
        if links > MAX_METHOD_CHAIN:
            out.append(Violation(
                "style.method-chain", call.lineno,
                f"Method chain at line {call.lineno} has {links} calls (at most {MAX_METHOD_CHAIN})",
                hint="Assign part of the chain to a name"))
    return out


@register(
    "style.call-nesting", "style",
    summary=f"Nest calls at most {MAX_CALL_NESTING} deep in arguments.",
    rationale="print(len(x)) reads at once. A third level makes the reader unwrap from the inside.",
    bad="x = [3, 1]\nprint(len(sorted(x)))",
    good="x = [3, 1]\nn = len(sorted(x))\nprint(n)",
)
def check_call_nesting(ctx: Context) -> list[Violation]:
    out, covered = [], set()
    for call in _calls(ctx):  # outer calls come before the calls inside them
        if id(call) in covered:
            continue
        if _depth(call) > MAX_CALL_NESTING:
            out.append(Violation(
                "style.call-nesting", call.lineno,
                f"Call at line {call.lineno} nests {_depth(call)} deep (at most {MAX_CALL_NESTING})",
                hint="Assign the inner call to a name"))
            stack = list(_arg_calls(call))
            while stack:
                inner = stack.pop()
                covered.add(id(inner))
                stack.extend(_arg_calls(inner))
    return out


@register(
    "style.io-position", "style",
    summary="Keep an I/O call out of arguments. In a chain, make it the first method call.",
    rationale="I/O is the step that can fail. Put it on a line of its own, "
              "or at the start of a chain, so the reader finds it.",
    bad="import json\nfrom pathlib import Path\np = 'a.json'\nd = json.loads(Path(p).read_text())",
    good="import json\nfrom pathlib import Path\np = 'a.json'\ntext = Path(p).read_text()\nd = json.loads(text)",
)
def check_io_position(ctx: Context) -> list[Violation]:
    out = []
    for call in _calls(ctx):
        for arg in _arg_calls(call):
            for node in [arg, *_receiver_chain(arg)]:
                if _is_io(node):
                    out.append(Violation(
                        "style.io-position", node.lineno,
                        f"I/O call .{node.func.attr}() at line {node.lineno} must not be an argument",
                        hint="Assign the I/O result to a name first"))
        if _is_io(call) and any(_is_method_call(n) for n in _receiver_chain(call)):
            out.append(Violation(
                "style.io-position", call.lineno,
                f"I/O call .{call.func.attr}() at line {call.lineno} must be the first "
                "method call in its chain",
                hint="Start the chain with the I/O call"))
    return out


@register(
    "style.comprehension-nested", "style",
    summary="Do not put a comprehension inside a comprehension.",
    rationale="Two loops in one expression make the reader track two scopes. Name the inner result.",
    bad="rows = [[1], [2]]\ny = [[a for a in r] for r in rows]",
    good="rows = [[1], [2]]\ny = [list(r) for r in rows]",
)
def check_comprehension_nested(ctx: Context) -> list[Violation]:
    out = []
    for node in ast.walk(ctx.tree):
        if not isinstance(node, _COMPS):
            continue
        parent = ctx.parents.get(id(node))
        while parent is not None and not isinstance(parent, (*_COMPS, ast.stmt)):
            parent = ctx.parents.get(id(parent))
        if isinstance(parent, _COMPS):
            out.append(Violation(
                "style.comprehension-nested", node.lineno,
                f"Comprehension at line {node.lineno} is inside another comprehension",
                hint="Use a for loop, or name the inner result"))
    return out


@register(
    "style.comprehension-multi-for", "style",
    summary="Use one for clause in a comprehension.",
    rationale="Two for clauses read in an order that surprises people. Use a for loop.",
    bad="rows = [[1], [2]]\ny = [a for r in rows for a in r]",
    good="rows = [[1], [2]]\ny = []\nfor r in rows:\n    y.extend(r)",
)
def check_comprehension_multi_for(ctx: Context) -> list[Violation]:
    return [Violation("style.comprehension-multi-for", node.lineno,
                      f"Comprehension at line {node.lineno} has {len(node.generators)} for clauses",
                      hint="Use a for loop")
            for node in ast.walk(ctx.tree)
            if isinstance(node, _COMPS) and len(node.generators) > 1]


@register(
    "style.elif-chain", "style",
    summary=f"Do not chain {ELIF_BRANCHES} or more if branches. Use a dict lookup.",
    rationale="A long if/elif chain hides that every branch picks one value. "
              "A dict states the mapping.",
    bad="x = 1\nif x == 1:\n    y = 'a'\nelif x == 2:\n    y = 'b'\nelse:\n    y = 'c'",
    good="x = 1\ny = {1: 'a', 2: 'b'}.get(x, 'c')",
)
def check_elif_chain(ctx: Context) -> list[Violation]:
    out = []
    for node in ast.walk(ctx.tree):
        if not isinstance(node, ast.If):
            continue
        parent = ctx.parents.get(id(node))
        if isinstance(parent, ast.If) and parent.orelse == [node]:
            continue  # only the top if reports
        branches, cur = 1, node
        while len(cur.orelse) == 1 and isinstance(cur.orelse[0], ast.If):
            cur = cur.orelse[0]
            branches += 1
        if cur.orelse:
            branches += 1
        if branches >= ELIF_BRANCHES:
            out.append(Violation(
                "style.elif-chain", node.lineno,
                f"If at line {node.lineno} has {branches} branches (fewer than {ELIF_BRANCHES} allowed)",
                hint="Use a dict lookup"))
    return out
