"""The validator's checks, ported into the rule registry with no change in behaviour.

Each check keeps the exact message string of the step it came from in
``validate()``. ``src/lackpy/infer/distill.py`` and ``tests/test_distill.py``
match on those strings. Registration order is run order.

``style.for-iter`` and ``style.lambda-position`` live here for now, with their
old behaviour. A later task changes them.
"""

from __future__ import annotations

import ast
import builtins

from .dictionary import MODULES, hint_for
from .grammar import ALLOWED_BUILTINS, ALLOWED_NODES, DENIED_ATTRIBUTES, FORBIDDEN_NAMES
from .registry import Context, Violation, register


@register(
    "safety.parse", "safety",
    summary="The program must parse as Python.",
    rationale="The other rules read the syntax tree. A program that does not parse has no tree.",
    bad="x = (",
    good="x = 1",
)
def check_parse(ctx: Context) -> list[Violation]:
    # validate() emits safety.parse itself when ast.parse fails, because no
    # Context exists then. A Context always holds a parsed tree, so this is empty.
    return []


@register(
    "safety.node", "safety",
    summary="Use only the syntax that the language allows.",
    rationale="Each allowed node type is a reviewed part of the sandbox. "
              "An unknown node type can open a path out of it.",
    bad="class Point:\n    pass",
    good="x = [1, 2, 3]",
)
def check_node(ctx: Context) -> list[Violation]:
    out = []
    for node in ast.walk(ctx.tree):
        node_type = type(node)
        if node_type not in ALLOWED_NODES:
            line = getattr(node, "lineno", None)
            out.append(Violation(
                "safety.node", line,
                f"Forbidden AST node: {node_type.__name__}"
                + (f" at line {node.lineno}" if hasattr(node, "lineno") else ""),
            ))
    return out


@register(
    "safety.forbidden-name", "safety",
    summary="Do not use a forbidden name, such as open or getattr.",
    rationale="These names reach the file system, the interpreter, or object internals "
              "outside the sandbox.",
    bad="f = open('notes.txt')",
    good="n = len([1, 2])",
)
def check_forbidden_name(ctx: Context) -> list[Violation]:
    out = []
    for node in ast.walk(ctx.tree):
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            out.append(Violation("safety.forbidden-name", node.lineno,
                                 f"Forbidden name: '{node.id}' at line {node.lineno}",
                                 hint=hint_for(node.id)))
    return out


# Attribute check — the sandbox boundary (GHSA-hpcj-3c97-43jm).
#
# ast.Attribute is allowed (data objects legitimately expose methods like
# ``.split``/``.items``/``.get``), but attribute *names* are restricted so an
# attribute chain cannot walk from a benign value to a type or module object:
#   - any name starting with "_" is denied (safety.attr-private) — this covers the
#     entire dunder traversal escape family (``__class__``/``__bases__``/
#     ``__subclasses__``/``__globals__``/``__mro__``/``__init__``/
#     ``__getattribute__``/…) as well as single-underscore private gadgets
#     (e.g. ``()._module``).
#   - a small set of non-underscore frame/generator internals (``f_globals``,
#     ``gi_frame``, …) is denied explicitly (safety.attr-denied, DENIED_ATTRIBUTES).
# This is an allow-by-shape / deny-by-name rule: it does not enumerate every
# safe attribute (that set is open-ended for arbitrary data), so it is a
# denylist on the dangerous shape. The known-complete escape surface in this
# subset is underscore-prefixed names; the explicit frame set covers the only
# non-underscore internals that expose a namespace.
#
# A removed rule, for the record: a blanket rejection of string literals that
# contain "__". It existed to stop getattr(obj, "__class__") -> __bases__ ->
# __subclasses__(). That sink is closed independently and more completely:
#
#   - getattr/setattr/delattr/hasattr/vars/globals/locals/dir/type/__import__
#     are all in FORBIDDEN_NAMES; eval/exec/compile are not in ALLOWED_BUILTINS.
#   - direct attribute access (obj.__class__) is rejected by the underscore
#     prefix rule, independent of any string.
#   - str.format/format_map DO turn a string into an attribute lookup, and
#     are therefore denied by name in DENIED_ATTRIBUTES. An earlier version of
#     this comment asserted no such call existed; it was wrong, and
#     "{0.__init__.__globals__[SECRET]}".format(obj) read a module global
#     through it. Any future builtin that takes an attribute name as DATA
#     belongs on that deny list too -- that, not the string literal, is the
#     sink. sort_by was the other one; it has been removed.
#
# It was also incomplete on its own terms — "_" + "_class_" + "_" evaluates to
# the same string and passed — so it never carried the weight it appeared to.
#
# Its cost was real: any string containing "__" was rejected, so a program
# could neither write nor read "__init__.py", emit `if __name__ ==
# "__main__":`, nor generate a class with a `def __init__`. Scaffolding a
# Python package was impossible.
#
# What this newly permits is a dunder string as DATA: a subscript key
# (obj["__class__"] is __getitem__, not attribute access), a filename, or a
# literal in generated source. What it must never permit is a dunder string
# reaching a sink that resolves it as an attribute name -- see the deny list
# above. Asserted in TestDunderStringsAreNotASink.

def _attribute_violation(rule_id: str, node: ast.Attribute) -> Violation:
    return Violation(
        rule_id, node.lineno,
        f"Forbidden attribute access: '.{node.attr}' at line {node.lineno} "
        f"(private/dunder/internal attribute access is not permitted)",
        hint=hint_for(node.attr),
    )


@register(
    "safety.attr-private", "safety",
    summary="Do not use an attribute whose name starts with an underscore.",
    rationale="Underscore attributes lead from a value to its type, its module and its globals. "
              "That chain is the main way out of a Python sandbox.",
    bad="name = 'a'.__class__",
    good="name = 'a'.upper()",
)
def check_attr_private(ctx: Context) -> list[Violation]:
    return [_attribute_violation("safety.attr-private", node)
            for node in ast.walk(ctx.tree)
            if isinstance(node, ast.Attribute) and node.attr.startswith("_")]


@register(
    "safety.attr-denied", "safety",
    summary="Do not use a denied attribute, such as format or f_globals.",
    rationale="These attributes have no underscore, but they expose frames and globals, "
              "or they turn a string into an attribute lookup.",
    bad="text = '{}'.format(1)",
    good="text = f'{1}'",
)
def check_attr_denied(ctx: Context) -> list[Violation]:
    return [_attribute_violation("safety.attr-denied", node)
            for node in ast.walk(ctx.tree)
            if isinstance(node, ast.Attribute)
            and not node.attr.startswith("_") and node.attr in DENIED_ATTRIBUTES]



# Imports — the dictionary (dictionary.MODULES) is the whole import surface.
#
# validate() builds ctx.module_bindings and ctx.member_bindings from the
# approved top-level imports. An import anywhere else is rejected, so each
# binding names a module or a member for the whole program. A binding by
# value (``s = statistics``) is not tracked: the runtime proxy (Plan 2) closes
# that gap. The message keeps the old safety.node text for an import
# statement, byte for byte, because consumers match on it.

_BUILTIN_NAMES = frozenset(dir(builtins))


def bound_before(tree: ast.Module, name: str, node: ast.AST) -> bool:
    """Whether a top-level from-import that binds ``name`` ends before ``node`` starts.

    A call to a member binding is admitted only after its import, by source
    position. Before it, the name is not bound yet at run time, so the call
    reaches something else.
    """
    for stmt in tree.body:
        if isinstance(stmt, ast.ImportFrom) and any((a.asname or a.name) == name for a in stmt.names):
            return (stmt.end_lineno, stmt.end_col_offset) <= (node.lineno, node.col_offset)
    return False


def _import_problem(node: ast.Import | ast.ImportFrom, ctx: Context) -> tuple[bool, list[str]]:
    """Return (rejected, subjects). The subjects are the words to look up a hint for."""
    subjects: list[str] = []
    rejected = not isinstance(ctx.parents.get(id(node)), ast.Module)
    bound: list[str] = []
    if isinstance(node, ast.Import):
        for a in node.names:
            if a.name not in MODULES:  # also rejects a dotted name, ``import m.sub``
                rejected = True
                subjects += [a.name, a.name.split(".")[0]]
            bound.append(a.asname or a.name)
    else:
        module = node.module or ""
        if node.level or module not in MODULES:
            rejected = True
            subjects += [module, module.split(".")[0]]
        for a in node.names:
            if a.name not in MODULES.get(module, ()):  # also rejects ``*``
                rejected = True
                subjects.insert(0, f"{module}.{a.name}")
            bound.append(a.asname or a.name)
    # A binding must not start with an underscore (``import json as __builtins__``
    # would replace the program's builtins), and must not reuse a builtin, a
    # forbidden name or a caller-allowed name: ``from json import loads as eval``
    # would make safety.unknown-call admit ``eval(...)`` and hide a tool call.
    if any(b.startswith("_") or b in _BUILTIN_NAMES or b in FORBIDDEN_NAMES
           or b in ctx.allowed_names for b in bound):
        rejected = True
    return rejected, subjects


@register(
    "safety.import", "safety",
    summary="Import only a dictionary module, at the top level, with import m or from m import x.",
    rationale="An import gives the program a module. Each dictionary module and member is reviewed. "
              "Other modules reach the file system, processes or the interpreter.",
    bad="import os\nnames = os.listdir('.')",
    good="import json\ndata = json.loads('[1]')",
)
def check_import(ctx: Context) -> list[Violation]:
    out = []
    for node in ast.walk(ctx.tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            rejected, subjects = _import_problem(node, ctx)
            if rejected:
                out.append(Violation(
                    "safety.import", node.lineno,
                    f"Forbidden AST node: {type(node).__name__} at line {node.lineno}",
                    hint=hint_for(*subjects),
                ))
    return out


@register(
    "safety.module-attr", "safety",
    summary="Use an imported module only as module.member, with an approved member. "
            "Do not assign to it.",
    rationale="A module holds more than its approved members, for example the modules that it imports. "
              "A module used as a value hides which member the program reads. "
              "An assignment to a module member changes the module for the host.",
    bad="import json\ndata = json.load('[1]')",
    good="import json\ndata = json.loads('[1]')",
)
def check_module_attr(ctx: Context) -> list[Violation]:
    bindings = ctx.module_bindings.keys() | ctx.member_bindings.keys()
    out = []
    for node in ast.walk(ctx.tree):
        # A module binding is only ever the base of an attribute (``json.loads``).
        # As a value (``s = statistics``, ``[json]``) it would carry the module
        # past this check.
        if isinstance(node, ast.Name) and node.id in ctx.module_bindings:
            parent = ctx.parents.get(id(node))
            if not (isinstance(parent, ast.Attribute) and parent.value is node):
                out.append(Violation(
                    "safety.module-attr", node.lineno,
                    f"Forbidden use of module: '{node.id}' at line {node.lineno} "
                    f"(use only {node.id}.<member>)",
                ))
            continue
        if not isinstance(node, ast.Attribute):
            continue
        # A store anywhere along a chain from a binding (``collections.Counter.x = f``)
        # changes a host module or class.
        root = node.value
        while isinstance(root, ast.Attribute):
            root = root.value
        if (not isinstance(node.ctx, ast.Load) and isinstance(root, ast.Name)
                and root.id in bindings):
            out.append(Violation(
                "safety.module-attr", node.lineno,
                f"Forbidden assignment to module member: '{ast.unparse(node)}' at line {node.lineno}",
            ))
            continue
        if not (isinstance(node.value, ast.Name) and node.value.id in ctx.module_bindings):
            continue
        module = ctx.module_bindings[node.value.id]
        subject = f"{module}.{node.attr}"
        if node.attr not in MODULES[module]:
            out.append(Violation(
                "safety.module-attr", node.lineno,
                f"Unknown module member: '{subject}' at line {node.lineno} (not in the dictionary)",
                hint=hint_for(subject),
            ))
    return out


# Namespace check — every ast.Call is inspected (not only ``func is ast.Name``):
# a call whose target is an attribute (``data.split(...)``) is permitted only
# because the attribute name already passed the attribute rules; a call whose
# target is a Subscript or another Call/expression is default-denied
# (safety.call-target) — those exotic call forms are not part of the intended
# subset and were an unchecked bypass before.

@register(
    "safety.unknown-call", "safety",
    summary="Call only an allowed builtin, a name that the caller allows, or an imported member.",
    rationale="The validator must know each function that a program calls. "
              "An unknown name can be a missing tool or a guess.",
    bad="x = frobnicate(1)",
    good="from collections import Counter\nc = Counter([1, 1])\nn = len(c)",
)
def check_unknown_call(ctx: Context) -> list[Violation]:
    allowed = ALLOWED_BUILTINS | ctx.allowed_names
    out = []
    for node in ast.walk(ctx.tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name in ctx.member_bindings and bound_before(ctx.tree, name, node):
                continue
            if name not in allowed and name not in FORBIDDEN_NAMES:
                out.append(Violation(
                    "safety.unknown-call", node.lineno,
                    f"Unknown function: '{name}' at line {node.lineno} "
                    f"(not in kit or builtins)",
                    hint=hint_for(name),
                ))
    return out


@register(
    "safety.call-target", "safety",
    summary="Call a function by its name, or call a method on a value.",
    rationale="A call of a subscript or of a call result hides which function runs. "
              "The validator cannot check it.",
    bad="funcs = [len]\nn = funcs[0]([1])",
    good="n = len([1])",
)
def check_call_target(ctx: Context) -> list[Violation]:
    out = []
    for node in ast.walk(ctx.tree):
        if isinstance(node, ast.Call) and not isinstance(node.func, (ast.Name, ast.Attribute)):
            out.append(Violation(
                "safety.call-target", node.lineno,
                f"Unsupported call target ({type(node.func).__name__}) at line "
                f"{node.lineno}: only named functions and method calls are allowed",
            ))
    return out


@register(
    "style.for-iter", "style",
    summary="A for loop iterates over a function call or a variable.",
    rationale="A named iterable shows what the loop reads.",
    bad="for c in 'abc':\n    print(c)",
    good="items = [1, 2]\nfor i in items:\n    print(i)",
)
def check_for_iter(ctx: Context) -> list[Violation]:
    out = []
    for node in ast.walk(ctx.tree):
        if isinstance(node, ast.For) and not isinstance(node.iter, (ast.Call, ast.Name)):
            out.append(Violation(
                "style.for-iter", node.lineno,
                f"For-loop at line {node.lineno} must iterate over "
                f"a function call or variable",
            ))
    return out


@register(
    "style.lambda-position", "style",
    summary="Use lambda only as the value of a key= argument.",
    rationale="A lambda in a key= argument is short and has one job. "
              "A lambda in other places hides logic.",
    bad="f = lambda x: x",
    good="words = sorted(['b', 'a'], key=lambda w: w)",
)
def check_lambda_position(ctx: Context) -> list[Violation]:
    out = []
    for node in ast.walk(ctx.tree):
        if isinstance(node, ast.Lambda):
            parent = ctx.parents.get(id(node))
            if not (isinstance(parent, ast.keyword) and parent.arg == "key"):
                out.append(Violation(
                    "style.lambda-position", node.lineno,
                    f"Lambda at line {node.lineno} is only allowed as key= argument "
                    f"(e.g., key=lambda x: x['field'])",
                ))
    return out
