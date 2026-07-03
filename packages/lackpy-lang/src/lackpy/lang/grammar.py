"""AST node sets defining the lackpy restricted Python subset."""

import ast

ALLOWED_NODES: set[type] = {
    # Structural
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign,
    ast.For, ast.If, ast.With, ast.withitem,
    # Expressions
    ast.Call, ast.Name, ast.Attribute, ast.Subscript,
    ast.List, ast.Dict, ast.Tuple, ast.Set,
    ast.ListComp, ast.DictComp, ast.SetComp,
    ast.Compare, ast.BoolOp, ast.UnaryOp, ast.BinOp,
    ast.JoinedStr, ast.FormattedValue,
    ast.Constant, ast.Starred, ast.Slice,
    # Comprehension internals
    ast.comprehension, ast.IfExp,
    # Lambda (restricted to key= argument — enforced by validator)
    ast.Lambda, ast.arguments, ast.arg,
    # Keyword arguments
    ast.keyword,
    # Context nodes
    ast.Load, ast.Store, ast.Del,
    # Operators
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.FloorDiv,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Is, ast.IsNot, ast.In, ast.NotIn,
    ast.And, ast.Or, ast.Not, ast.USub, ast.UAdd,
}

FORBIDDEN_NODES: set[type] = {
    ast.Import, ast.ImportFrom,
    ast.FunctionDef, ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.While,
    ast.Try, ast.ExceptHandler,
    ast.Raise,
    ast.Global, ast.Nonlocal,
    ast.Yield, ast.YieldFrom,
    ast.Await, ast.AsyncFor, ast.AsyncWith,
    ast.Assert,
    ast.Delete,
    ast.Match,
}

FORBIDDEN_NAMES: frozenset[str] = frozenset({
    "__import__", "open",
    "globals", "locals", "vars", "dir",
    "getattr", "setattr", "delattr", "hasattr",
    "__builtins__", "__build_class__",
    "breakpoint", "exit", "quit",
    "type", "super", "classmethod", "staticmethod", "property",
    "memoryview", "bytearray", "bytes",
    "map", "filter", "reduce",
    "input",
    # stdlib modules that models reach for instead of using kit tools (issue #4)
    "os", "sys", "pathlib", "subprocess", "shutil",
})

# Attribute names that are always denied, even without an underscore prefix.
# The primary attribute guard is a *prefix* rule (reject any attribute whose name
# starts with "_"), which covers the whole dunder-traversal escape family
# (__class__/__bases__/__subclasses__/__globals__/__mro__/__init__/…) and the
# single-underscore private-attribute gadgets (e.g. ``()._module``).
#
# These names have NO leading underscore, so the prefix rule would miss them. They
# are frame/generator/coroutine/traceback internals that expose a real namespace
# (``f_globals``/``f_builtins``). They are believed UNREACHABLE in the current
# subset (no GeneratorExp / Try / Raise / sys => no frame, generator, or traceback
# object can be constructed), but they are denied explicitly as defense-in-depth
# so a future grammar addition cannot silently re-open the route.
DENIED_ATTRIBUTES: frozenset[str] = frozenset({
    "f_globals", "f_builtins", "f_locals", "f_code", "f_back",
    "gi_frame", "gi_code", "cr_frame", "cr_code",
    "ag_frame", "ag_code", "tb_frame", "tb_next",
    "func_globals", "func_code",
})

ALLOWED_BUILTINS: frozenset[str] = frozenset({
    "len", "sorted", "reversed", "enumerate", "zip", "range",
    "min", "max", "sum", "any", "all", "abs", "round",
    "str", "int", "float", "bool", "list", "dict", "set", "tuple",
    "isinstance", "print",
    "sort_by",
})
