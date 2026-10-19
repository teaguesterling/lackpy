"""The restricted ``__import__`` and the module proxies (stepy100 Plan 2, Task 3).

This module is part of the sandbox boundary. A program never holds a real
module. ``__import__`` serves a ``ModuleProxy`` for a module in
``dictionary.MODULES`` and refuses every other import. The proxy exposes the
dictionary members of its module only. The static check cannot follow an alias
(``s = statistics; s.sys``), so the proxy is what closes that route.

The program keeps its ``import`` statements, so it is still plain Python. This
replaces the AST import rewrite of design §4.3 with the same effect.

``pathlib`` is not served until Task 4 binds the ``Path`` stand-in (ruling C):
a real ``Path`` would give unconfined host file access.
"""

from __future__ import annotations

import collections as _collections
import importlib
import types
from typing import Any, Callable

from ..lang import dictionary
from .refusal import LackpyRefusal, refuse

_PATH_PENDING_HINT = "Path arrives with the PathPolicy"

# Mutable host classes that a program could change: each run gets a subclass.
_PER_RUN_CLASSES: dict[str, frozenset[str]] = {
    "collections": frozenset({"Counter", "defaultdict", "deque", "OrderedDict"}),
}


def _per_run_subclass(base: type) -> type:
    """Return a fresh subclass of ``base`` that prints exactly as ``base`` does.

    It keeps the base's ``__name__``, ``__qualname__`` and ``__module__``, so the
    repr of an instance and of the class reads as under plain Python. It adds an
    instance ``__dict__`` only where the base has one.
    """
    body: dict[str, Any] = {"__qualname__": base.__qualname__, "__module__": base.__module__}
    if not base.__dictoffset__:
        body["__slots__"] = ()
    return type(base.__name__, (base,), body)


class ModuleProxy:
    """A module as a program sees it: the dictionary members, and nothing else.

    A member is the real object, except for the per-run classes. A name the
    real module has, but the dictionary does not list, refuses. A name the real
    module lacks raises ``AttributeError``, as plain Python does. ``__version__``
    gives the real value. Setting or deleting an attribute always refuses.
    """

    __slots__ = ("_lackpy_name", "_lackpy_module", "_lackpy_members")

    def __init__(self, name: str) -> None:
        module = importlib.import_module(name)
        per_run = _PER_RUN_CLASSES.get(name, frozenset())
        members: dict[str, Any] = {}
        for member in dictionary.MODULES[name]:
            if not hasattr(module, member):
                continue  # e.g. datetime.UTC before Python 3.11
            value = getattr(module, member)
            if isinstance(value, types.ModuleType):
                # No dictionary member is a module today; never hand one out.
                raise LackpyRefusal(f"{name}.{member} (a module)")
            if member in per_run:
                value = _per_run_subclass(value)
            members[member] = value
        object.__setattr__(self, "_lackpy_name", name)
        object.__setattr__(self, "_lackpy_module", module)
        object.__setattr__(self, "_lackpy_members", members)

    def __getattr__(self, attr: str) -> Any:
        # Reached only when normal lookup fails: every non-underscore name.
        members = object.__getattribute__(self, "_lackpy_members")
        if attr in members:
            return members[attr]
        name = object.__getattribute__(self, "_lackpy_name")
        module = object.__getattribute__(self, "_lackpy_module")
        if attr == "__version__" and hasattr(module, attr):
            return module.__version__
        if not hasattr(module, attr):
            raise AttributeError(f"module {name!r} has no attribute {attr!r}")
        refuse(f"{name}.{attr}", f"{name}.{attr}")

    def __setattr__(self, attr: str, value: Any) -> None:
        refuse(f"{object.__getattribute__(self, '_lackpy_name')}.{attr} = ...")

    def __delattr__(self, attr: str) -> None:
        refuse(f"del {object.__getattribute__(self, '_lackpy_name')}.{attr}")

    def __repr__(self) -> str:
        return repr(object.__getattribute__(self, "_lackpy_module"))


def make_restricted_import() -> Callable[..., ModuleProxy]:
    """Return a fresh ``__import__`` for one program run.

    Each run gets its own proxies, so no class state crosses runs (Review
    Focus 5). Within a run, a module imported twice is the same proxy.
    """
    proxies: dict[str, ModuleProxy] = {}

    def restricted_import(name: str, globals: Any = None, locals: Any = None,
                          fromlist: Any = (), level: int = 0) -> ModuleProxy:
        if level:
            refuse("a relative import")
        if name == "pathlib":
            raise LackpyRefusal(f"import {name}", _PATH_PENDING_HINT)
        if name not in dictionary.MODULES:
            refuse(f"import {name}", name)
        if name not in proxies:
            proxies[name] = ModuleProxy(name)
        return proxies[name]

    return restricted_import
