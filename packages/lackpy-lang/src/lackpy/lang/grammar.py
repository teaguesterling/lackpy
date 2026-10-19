"""AST node sets defining the lackpy restricted Python subset."""

import ast

from .builtins_policy import allowed_builtin_names

ALLOWED_NODES: set[type] = {
    # Structural
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign,
    ast.For, ast.If, ast.With, ast.withitem,
    ast.Assert, ast.Continue, ast.Break, ast.Pass,
    # Top-level def, with the style.def-* rules (Yield, Global and Nonlocal stay forbidden)
    ast.FunctionDef, ast.Return,
    # try with named errors, with the style.try-* rules and safety.except-base (TryStar stays forbidden)
    ast.Try, ast.ExceptHandler,
    # Expressions
    ast.Call, ast.Name, ast.Attribute, ast.Subscript,
    ast.List, ast.Dict, ast.Tuple, ast.Set,
    ast.ListComp, ast.DictComp, ast.SetComp,
    ast.Compare, ast.BoolOp, ast.UnaryOp, ast.BinOp,
    ast.JoinedStr, ast.FormattedValue,
    ast.Constant, ast.Starred, ast.Slice,
    ast.GeneratorExp,
    # Comprehension internals
    ast.comprehension, ast.IfExp,
    # Lambda (restricted to key= argument — enforced by validator)
    ast.Lambda, ast.arguments, ast.arg,
    # Keyword arguments
    ast.keyword,
    # Imports (only dictionary modules, in the approved forms — safety.import)
    ast.Import, ast.ImportFrom, ast.alias,
    # Context nodes
    ast.Load, ast.Store, ast.Del,
    # Operators
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.FloorDiv,
    ast.BitAnd, ast.BitOr, ast.BitXor,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Is, ast.IsNot, ast.In, ast.NotIn,
    ast.And, ast.Or, ast.Not, ast.USub, ast.UAdd,
}

FORBIDDEN_NODES: set[type] = {
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.While,
    ast.Raise,
    ast.Global, ast.Nonlocal,
    ast.Yield, ast.YieldFrom,
    ast.Await, ast.AsyncFor, ast.AsyncWith,
    ast.Delete,
    ast.Match,
}

# except* parses from Python 3.11; this package supports 3.10.
if hasattr(ast, "TryStar"):
    FORBIDDEN_NODES.add(ast.TryStar)

FORBIDDEN_NAMES: frozenset[str] = frozenset({
    "__import__", "open",
    "globals", "locals", "vars", "dir",
    "getattr", "setattr", "delattr", "hasattr",
    "__builtins__", "__build_class__",
    "breakpoint", "exit", "quit",
    # `type` is not here: safety.type-form owns every use of that name.
    "super", "classmethod", "staticmethod", "property",
    "memoryview", "bytearray", "bytes",
    "map", "filter", "reduce",
    "input",
    # stdlib modules that models reach for instead of using kit tools (issue #4).
    # pathlib is an approved module now (dictionary.MODULES).
    "os", "sys", "subprocess", "shutil",
})

# Attribute names that are always denied, even without an underscore prefix.
# The primary attribute guard is a *prefix* rule (reject any attribute whose name
# starts with "_"), which covers the whole dunder-traversal escape family
# (__class__/__bases__/__subclasses__/__globals__/__mro__/__init__/…) and the
# single-underscore private-attribute gadgets (e.g. ``()._module``).
#
# These names have NO leading underscore, so the prefix rule would miss them. They
# are frame/generator/coroutine/traceback internals that expose a real namespace
# (``f_globals``/``f_builtins``). GeneratorExp is in the subset, so a generator
# object CAN be constructed, and this list is what keeps
# ``(x for x in y).gi_frame.f_globals`` unreachable. Do not prune it.
#
# Try is allowed now (Task 8), so ``except E as e`` gives a program an exception
# object. The guards on that object are: ``__traceback__`` (and every other
# underscore attribute) by the prefix rule, ``tb_frame`` and ``tb_next`` here for
# a traceback reached by any route (``e.with_traceback(None)`` returns the
# exception, not a traceback), ``mro`` here for the exception class, and ``obj``
# here because ``AttributeError.obj`` holds the object that lacked the
# attribute, which can be any object, a module included. ``name``, ``path`` and
# ``args`` stay allowed: ``ImportError.name``/``.path`` are strings, and ``name``
# is a ``Path`` member. Raise and sys remain excluded.
DENIED_ATTRIBUTES: frozenset[str] = frozenset({
    "f_globals", "f_builtins", "f_locals", "f_code", "f_back",
    "gi_frame", "gi_code", "cr_frame", "cr_code",
    "ag_frame", "ag_code", "tb_frame", "tb_next",
    "func_globals", "func_code",
    # str.format/format_map traverse attributes driven by the *format string*,
    # which is runtime data the AST never sees:
    #     "{0.__init__.__globals__[SECRET]}".format(obj)
    # Every name in that chain is invisible to Step 3.5 -- the only attribute in
    # the source is ``format`` itself, which has no leading underscore. This is
    # the one call form that turns a string into an attribute lookup, so it is
    # denied by name. f-strings are the replacement and are safe for the exact
    # reason format is not: their interpolations are real AST expressions and go
    # through the attribute rule like anything else.
    "format", "format_map",
    # Class.mro() walks to `object` and from there to every subclass.
    "mro",
    # AttributeError.obj: the object that lacked the attribute (any object).
    "obj",
})

ALLOWED_BUILTINS: frozenset[str] = allowed_builtin_names()
