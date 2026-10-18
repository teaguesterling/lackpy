"""The validator's checks, ported into the rule registry with no change in behaviour.

Each check keeps the exact message string of the step it came from in
``validate()``. ``src/lackpy/infer/distill.py`` and ``tests/test_distill.py``
match on those strings. Registration order is run order.
"""

from __future__ import annotations

import ast
import builtins
import importlib

from . import builtins_policy
from .dictionary import MODULES, hint_for
from .grammar import ALLOWED_BUILTINS, ALLOWED_NODES, DENIED_ATTRIBUTES, FORBIDDEN_NAMES
from .registry import Context, Violation, register


@register(
    "safety.parse", "safety",
    summary="The program must parse and compile as Python.",
    rationale="The other rules read the syntax tree. A program that does not parse has no tree. "
              "A program that does not compile cannot run.",
    bad="x = (",
    good="x = 1",
)
def check_parse(ctx: Context) -> list[Violation]:
    # validate() emits safety.parse itself when ast.parse fails, because no
    # Context exists then. A Context always holds a parsed tree, so this is empty.
    # validate() also compiles a program that passes every rule (a compile error
    # there, such as two starred targets, is safety.parse too).
    return []


def in_for_body(node: ast.AST, ctx: Context) -> bool:
    """Whether ``node`` is inside the body of a ``for`` loop of its own function or program.

    A def or lambda between them starts a new scope, and a ``for``'s else clause is
    not its body, so neither counts.
    """
    child, parent = node, ctx.parents.get(id(node))
    while parent is not None:
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return False
        if isinstance(parent, ast.For) and child in parent.body:
            return True
        child, parent = parent, ctx.parents.get(id(parent))
    return False


def _inside_def(node: ast.AST, ctx: Context) -> bool:
    parent = ctx.parents.get(id(node))
    while parent is not None:
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return True
        parent = ctx.parents.get(id(parent))
    return False


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
        if node_type is ast.Return and not _inside_def(node, ctx):
            # A return outside a function is a SyntaxError when the program compiles.
            out.append(Violation(
                "safety.node", node.lineno,
                f"Forbidden AST node: Return at line {node.lineno} (return is only allowed inside a def)"))
        elif node_type in (ast.Break, ast.Continue) and not in_for_body(node, ctx):
            # Outside a loop of its own scope, break and continue are a SyntaxError at compile.
            word = node_type.__name__.lower()
            out.append(Violation(
                "safety.node", node.lineno,
                f"Forbidden AST node: {node_type.__name__} at line {node.lineno} "
                f"({word} is only allowed inside a for body of the same def or program)"))
        elif node_type not in ALLOWED_NODES:
            line = getattr(node, "lineno", None)
            out.append(Violation(
                "safety.node", line,
                f"Forbidden AST node: {node_type.__name__}"
                + (f" at line {node.lineno}" if hasattr(node, "lineno") else ""),
            ))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.decorator_list:
            # A decorator runs when the def runs, and its result replaces the function.
            out.append(Violation(
                "safety.node", node.lineno,
                f"Forbidden decorator on function '{node.name}' at line {node.lineno} "
                f"(a decorator runs code when the def runs; call the function instead)"))
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


# Dunder names and `type` — the names that lead to the interpreter's internals.
#
# A name that starts with "__" reaches the module namespace (``__builtins__``,
# ``__spec__``, ``__loader__``) or the import machinery (``__import__``). Only
# two are data: ``__name__`` and ``__file__``, and only as a load. A program
# that stores or binds a dunder name can change what the runtime reads.
#
# ``type`` returns a class, and a class leads to ``object`` and every subclass.
# The one use a program needs is the class name, so ``type(<expr>).__name__``
# is the only form, and the result is a str.

_DUNDER_LOADS = frozenset({"__name__", "__file__"})


def _bound_names(tree: ast.Module):
    """Yield (name, line) for each binding that is not an ``ast.Name``.

    These are parameters, def and class names, except-as names and import
    aliases. Stores to an ``ast.Name`` are found through its ``ctx``.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            yield node.arg, node.lineno
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node.name, node.lineno
        elif isinstance(node, ast.ExceptHandler) and node.name:
            yield node.name, node.lineno
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                if a.name != "*":
                    yield a.asname or a.name.split(".")[0], node.lineno


@register(
    "safety.dunder-name", "safety",
    summary="Do not use a name that starts with two underscores. "
            "You can read `__name__` and `__file__`.",
    rationale="Dunder names lead to the builtins, the import machinery and the module namespace. "
              "`__name__` and `__file__` are only strings.",
    bad="x = __builtins__",
    good="if __name__ == '__main__':\n    print(1)",
)
def check_dunder_name(ctx: Context) -> list[Violation]:
    found: list[tuple[str, int]] = []
    for node in ast.walk(ctx.tree):
        if (isinstance(node, ast.Name) and node.id.startswith("__")
                and not (node.id in _DUNDER_LOADS and isinstance(node.ctx, ast.Load))):
            found.append((node.id, node.lineno))
    found += [(name, line) for name, line in _bound_names(ctx.tree) if name.startswith("__")]
    return [Violation("safety.dunder-name", line,
                      f"Forbidden dunder name: '{name}' at line {line} "
                      f"(only __name__ and __file__, and only to read)")
            for name, line in sorted(found, key=lambda f: f[1])]


def _is_type_call(node: ast.AST) -> bool:
    """Whether ``node`` is ``type(<expr>)``: one positional argument, nothing else."""
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "type" and len(node.args) == 1
            and not isinstance(node.args[0], ast.Starred) and not node.keywords)


def _is_type_name(node: ast.AST) -> bool:
    """Whether ``node`` is the load ``type(<expr>).__name__``."""
    return (isinstance(node, ast.Attribute) and node.attr == "__name__"
            and isinstance(node.ctx, ast.Load) and _is_type_call(node.value))


@register(
    "safety.type-form", "safety",
    summary="Use type only as `type(<expr>).__name__`.",
    rationale="type returns a class. A class leads to object and to every subclass. "
              "The name of the class is a string, and it is safe.",
    bad="t = type(3)",
    good="n = type(3).__name__",
)
def check_type_form(ctx: Context) -> list[Violation]:
    lines: list[int] = []
    for node in ast.walk(ctx.tree):
        if isinstance(node, ast.Name) and node.id == "type":
            call = ctx.parents.get(id(node))
            if not (isinstance(call, ast.Call) and call.func is node and _is_type_call(call)
                    and _is_type_name(ctx.parents.get(id(call)))):
                lines.append(node.lineno)
    lines += [line for name, line in _bound_names(ctx.tree) if name == "type"]
    return [Violation("safety.type-form", line,
                      f"Forbidden use of 'type' at line {line} (use only type(<expr>).__name__)")
            for line in sorted(lines)]


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
#   - getattr/setattr/delattr/hasattr/vars/globals/locals/dir/__import__
#     are all in FORBIDDEN_NAMES; eval/exec/compile are not in ALLOWED_BUILTINS;
#     safety.type-form admits type only as type(<expr>).__name__.
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


def _allowed_private(node: ast.Attribute, ctx: Context) -> bool:
    """The two underscore attributes that a program may read.

    ``<module>.__version__`` on a module binding, and ``type(<expr>).__name__``.
    Both are strings. Neither leads to a type, a module or a namespace.
    """
    if not isinstance(node.ctx, ast.Load):
        return False
    if node.attr == "__version__":
        return isinstance(node.value, ast.Name) and node.value.id in ctx.module_bindings
    return _is_type_name(node)


@register(
    "safety.attr-private", "safety",
    summary="Do not use an attribute whose name starts with an underscore. "
            "You can read `module.__version__` and `type(<expr>).__name__`.",
    rationale="Underscore attributes lead from a value to its type, its module and its globals. "
              "That chain is the main way out of a Python sandbox.",
    bad="name = 'a'.__class__",
    good="name = 'a'.upper()",
)
def check_attr_private(ctx: Context) -> list[Violation]:
    return [_attribute_violation("safety.attr-private", node)
            for node in ast.walk(ctx.tree)
            if isinstance(node, ast.Attribute) and node.attr.startswith("_")
            and not _allowed_private(node, ctx)]


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
    out.extend(_rebindings(ctx.tree))
    return out


def _rebindings(tree: ast.Module) -> list[Violation]:
    """A name bound by an import is never bound again, anywhere in the program.

    A rebinding would hide what a call reaches: after ``Counter = read_file``,
    ``Counter('x')`` is a tool call that safety.unknown-call admits and ``calls``
    leaves out. A second import of the same name is a rebinding too.
    """
    first: dict[str, int] = {}
    found: list[tuple[str, int]] = []
    imports = sorted((n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))),
                     key=lambda n: (n.lineno, n.col_offset))
    for stmt in imports:
        for a in stmt.names:
            if a.name == "*":
                continue
            name = a.asname or a.name.split(".")[0]
            if name in first:
                found.append((name, stmt.lineno))
            else:
                first[name] = stmt.lineno
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            name = node.id  # assignment, for, comprehension, with-as, starred targets
        elif isinstance(node, ast.arg):
            name = node.arg  # def and lambda parameters
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, ast.ExceptHandler) and node.name:
            name = node.name
        else:
            continue
        if name in first:
            found.append((name, node.lineno))
    return [Violation("safety.import", line,
                      f"Rebinding of imported name: '{name}' at line {line} "
                      f"(imported at line {first[name]})")
            for name, line in sorted(found, key=lambda f: f[1])]


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
        # __version__ is a str; safety.attr-private admits it on a module binding.
        if node.attr not in MODULES[module] and node.attr != "__version__":
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

def _rebound(tree: ast.Module, name: str) -> bool:
    """Whether ``name`` is bound anywhere other than by a ``def``.

    After ``g = read_file``, a call ``g(...)`` reaches a tool, but ``calls``
    leaves out the names of program functions. A rebound def name is no longer
    a known function.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == name and not isinstance(node.ctx, ast.Load):
            return True
        if isinstance(node, ast.arg) and node.arg == name:
            return True
        if isinstance(node, ast.ClassDef) and node.name == name:
            return True
        if isinstance(node, ast.ExceptHandler) and node.name == name:
            return True
    return False


@register(
    "safety.unknown-call", "safety",
    summary="Call only an allowed builtin, a name that the caller allows, a program def "
            "or an imported member.",
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
            if name == "type":  # safety.type-form owns every use of `type`
                continue
            if name in ctx.defined_functions and not _rebound(ctx.tree, name):
                continue  # style.def-before-use owns the position of the call
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


# except — every handler-class check lives here, in the safety tier.
#
# BaseException and its direct subclasses outside Exception (SystemExit,
# KeyboardInterrupt, GeneratorExit, BaseExceptionGroup) carry interpreter
# control, and the sandbox's own refusals and limits (Plan 2). A handler that
# catches them would swallow that control. A bare ``except`` catches them too.
#
# RecursionError, MemoryError and RuntimeError are limit signals. A handler
# that catches RecursionError can retry the recursion, and the program then
# runs without end (measured: a key= recursion that catches RecursionError
# ran past 60 s). RuntimeError is RecursionError's base, so it is listed too.
#
# Every other route to a limit signal is closed the same way, so no waiver can
# reopen it: ``Exception`` (it catches RecursionError), and any class that does
# not resolve statically to an approved class. A name bound anywhere in the
# program (``E = RecursionError``, ``errs = (KeyError, RecursionError)``,
# ``def f(KeyError)``) is not the builtin class, so it is rejected. The class
# expression is evaluated when an exception arrives, so it must be a name, a
# dotted name, or a tuple of them; a call there would run code (a tool).

BASE_ONLY_EXCEPTIONS = frozenset(
    name for name in dir(builtins)
    if isinstance(getattr(builtins, name), type)
    and issubclass(getattr(builtins, name), BaseException)
    and not issubclass(getattr(builtins, name), Exception)
)
LIMIT_SIGNALS = frozenset({"RecursionError", "RuntimeError", "MemoryError"})
UNCATCHABLE = BASE_ONLY_EXCEPTIONS | LIMIT_SIGNALS | {"Exception"}

_APPROVED_HINT = "name an approved error class, for example KeyError, json.JSONDecodeError or re.error"


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


def handler_classes(handler: ast.ExceptHandler) -> list[ast.expr]:
    """The class expressions that ``handler`` names: one, or each item of a tuple."""
    if handler.type is None:
        return []
    if isinstance(handler.type, ast.Tuple):
        return list(handler.type.elts)
    return [handler.type]


def is_dotted_name(node: ast.expr) -> bool:
    """Whether ``node`` is a Name, or an Attribute chain rooted on a Name."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return isinstance(node, ast.Name)


def program_bindings(tree: ast.Module) -> list[str]:
    """Every binding in the program, one entry per binding site.

    Assignment, augmented, for, comprehension and with targets, parameters,
    def and class names, except-as names and import aliases.
    """
    names = [n.id for n in ast.walk(tree)
             if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load)]
    return names + [name for name, _ in _bound_names(tree)]


def _approved_class(cls: ast.expr, ctx: Context) -> bool:
    """Whether ``cls`` resolves statically to an approved exception class."""
    if isinstance(cls, ast.Name):
        if cls.id in ctx.member_bindings:
            return ctx.member_bindings[cls.id] in EXCEPTION_MEMBERS
        return cls.id not in UNCATCHABLE and builtins_policy.ALLOWED.get(cls.id) == "exception"
    if isinstance(cls, ast.Attribute) and isinstance(cls.value, ast.Name):
        module = ctx.module_bindings.get(cls.value.id)
        return (module, cls.attr) in EXCEPTION_MEMBERS
    return False


@register(
    "safety.except-base", "safety",
    summary="Catch only a named error class, such as KeyError or json.JSONDecodeError, or a "
            "tuple of them. Do not catch Exception, BaseException, RecursionError, "
            "RuntimeError or MemoryError.",
    rationale="A bare except, Exception and BaseException also catch the signals that stop "
              "the program, such as a time limit or a recursion limit. A handler of "
              "RecursionError can retry without end. A name that the program binds, or a call, "
              "can hide any class.",
    bad="d = {}\ntry:\n    x = d['k']\nexcept Exception:\n    x = 0",
    good="d = {}\ntry:\n    x = d['k']\nexcept KeyError:\n    x = 0",
)
def check_except_base(ctx: Context) -> list[Violation]:
    out = []
    bound = set(program_bindings(ctx.tree)) - ctx.member_bindings.keys()
    for node in ast.walk(ctx.tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if node.type is None:
            out.append(Violation(
                "safety.except-base", node.lineno,
                f"Bare except at line {node.lineno} (name the error class, for example KeyError)"))
            continue
        for cls in handler_classes(node):
            if not is_dotted_name(cls):
                message = (f"Forbidden except expression '{ast.unparse(cls)}' at line {node.lineno} "
                           f"(use a class name, a dotted name, or a tuple of them, "
                           f"for example KeyError)")
            elif isinstance(cls, ast.Name) and cls.id in UNCATCHABLE:
                message = (f"Forbidden except of '{cls.id}' at line {node.lineno} "
                           f"(catch a subclass of Exception that is not a limit signal, "
                           f"for example KeyError)")
            elif isinstance(cls, ast.Name) and cls.id in bound:
                message = (f"Forbidden except of '{cls.id}' at line {node.lineno} "
                           f"('{cls.id}' is bound in the program; {_APPROVED_HINT})")
            elif not _approved_class(cls, ctx):
                message = (f"Forbidden except of '{ast.unparse(cls)}' at line {node.lineno} "
                           f"(not an error class; {_APPROVED_HINT})")
            else:
                continue
            out.append(Violation("safety.except-base", node.lineno, message))
    return out


# finally runs on every path out of the try, an exception in flight included. A
# ``return``, ``continue`` or ``break`` in it discards that exception: a
# RecursionError (so key= recursion retries without end), or a sandbox refusal
# (a BaseException that the runner must see). No waiver can lift this, so finally
# is a safety rule. A with statement does cleanup.

@register(
    "safety.try-finally", "safety",
    summary="Do not use finally on a try. Use a with statement for cleanup.",
    rationale="A return, continue or break in finally discards the exception in flight, "
              "such as a recursion limit or a sandbox refusal.",
    bad="d = {}\ntry:\n    x = d['k']\nexcept KeyError:\n    x = 0\nfinally:\n    print('done')",
    good="d = {}\ntry:\n    x = d['k']\nexcept KeyError:\n    x = 0\nprint('done')",
)
def check_try_finally(ctx: Context) -> list[Violation]:
    return [Violation("safety.try-finally", node.lineno,
                      f"Try at line {node.lineno} must not have finally: "
                      f"(it can discard an exception in flight; use with for cleanup)")
            for node in ast.walk(ctx.tree) if isinstance(node, ast.Try) and node.finalbody]
