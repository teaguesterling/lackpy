"""Every Python builtin is classified: allowed, or denied with a reason.

``DENIED`` is a literal inventory on purpose. It is never computed from
``dir(builtins)``. A computed list would pass ``test_every_builtin_is_classified``
by construction, and a new Python release could then add a builtin silently.
With a literal list, a new name fails that test until a person classifies it.

The inventory was taken from CPython 3.12. Names that exist only in other
versions may also appear here.

``ALLOWED`` has two halves. The functions and types are a literal list. The
exception classes are a rule: every builtin that is a subclass of ``Exception``.
That rule excludes ``BaseException``, ``SystemExit``, ``KeyboardInterrupt``,
``GeneratorExit`` and ``BaseExceptionGroup`` (all denied below).
"""

from __future__ import annotations

import builtins as _builtins

_ALLOWED_FUNCTIONS: tuple[str, ...] = (
    "print", "len", "sorted", "str", "sum", "set", "dict", "range", "max", "list",
    "enumerate", "isinstance", "int", "any", "repr", "min", "next", "bool", "zip",
    "round", "all", "chr", "tuple", "float", "abs", "ord", "iter", "reversed",
    "frozenset", "hex",
)

ALLOWED: dict[str, str] = {name: "function" for name in _ALLOWED_FUNCTIONS}
ALLOWED.update({
    name: "exception"
    for name in dir(_builtins)
    if isinstance(getattr(_builtins, name), type)
    and issubclass(getattr(_builtins, name), Exception)
})
ALLOWED["ExceptionGroup"] = "exception"

# name -> reason category. Written out name by name; see the module docstring.
DENIED: dict[str, str] = {
    # reflection: reads or writes attributes, identity or namespaces by name
    "getattr": "reflection", "setattr": "reflection", "delattr": "reflection",
    "hasattr": "reflection", "globals": "reflection", "locals": "reflection",
    "vars": "reflection", "dir": "reflection", "id": "reflection",
    "hash": "reflection", "callable": "reflection", "issubclass": "reflection",
    "format": "reflection", "ascii": "reflection",
    # code-execution: compiles or runs code, or loads modules
    "eval": "code-execution", "exec": "code-execution",
    "compile": "code-execution", "__import__": "code-execution",
    "__build_class__": "code-execution", "breakpoint": "code-execution",
    # io: files, the console, and the site-added help text
    "open": "io", "input": "io", "help": "io", "copyright": "io",
    "credits": "io", "license": "io",
    # interpreter-control: ends or steers the interpreter. True, False, None and
    # Ellipsis parse as constants, never as names, so no program needs them here.
    "exit": "interpreter-control", "quit": "interpreter-control",
    "BaseException": "interpreter-control", "SystemExit": "interpreter-control",
    "KeyboardInterrupt": "interpreter-control",
    "GeneratorExit": "interpreter-control",
    "BaseExceptionGroup": "interpreter-control",
    "aiter": "interpreter-control", "anext": "interpreter-control",
    "NotImplemented": "interpreter-control", "Ellipsis": "interpreter-control",
    "True": "interpreter-control", "False": "interpreter-control",
    "None": "interpreter-control",
    # type-machinery: builds or inspects types. Plan 2 adds a safe `type` stand-in.
    "type": "type-machinery", "object": "type-machinery",
    "super": "type-machinery", "property": "type-machinery",
    "classmethod": "type-machinery", "staticmethod": "type-machinery",
    "slice": "type-machinery", "complex": "type-machinery",
    # use-comprehension: a comprehension replaces each of these
    "map": "use-comprehension", "filter": "use-comprehension",
    # binary: bytes, and number formats that are not decimal or hex
    "bytes": "binary", "bytearray": "binary", "memoryview": "binary",
    "bin": "binary", "oct": "binary", "divmod": "binary", "pow": "binary",
    # dunder: module-level dunder names
    "__debug__": "dunder", "__doc__": "dunder", "__loader__": "dunder",
    "__name__": "dunder", "__package__": "dunder", "__spec__": "dunder",
}


def allowed_builtin_names() -> frozenset[str]:
    """The names a program may use without a tool. ``grammar.ALLOWED_BUILTINS``."""
    return frozenset(ALLOWED)
