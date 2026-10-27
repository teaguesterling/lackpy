"""The restricted ``__import__`` and the module proxies (stepy100 Plan 2, Task 3).

This module is part of the sandbox boundary. A program never holds a real
module. ``__import__`` serves a ``ModuleProxy`` for a module in
``dictionary.MODULES`` and refuses every other import. The proxy exposes the
dictionary members of its module only. The static check cannot follow an alias
(``s = statistics; s.sys``), so the proxy is what closes that route.

The program keeps its ``import`` statements, so it is still plain Python. This
replaces the AST import rewrite of design §4.3 with the same effect.

``pathlib`` is served only when a ``PathPolicy`` is given: its proxy's ``Path``
is the stand-in from ``paths.make_path_class``. With no policy, ``pathlib``
refuses (ruling C, fail closed): a real ``Path`` would give unconfined host
file access.
"""

from __future__ import annotations

import importlib
import types
import weakref
from typing import TYPE_CHECKING, Any, Callable, NamedTuple

from ..lang import dictionary
from .paths import PathPolicy, make_path_class
from .refusal import LackpyRefusal, refuse

if TYPE_CHECKING:
    from .trace import Trace

_PATH_PENDING_HINT = "Path arrives with the PathPolicy"


class _ProxyState(NamedTuple):
    name: str
    module: types.ModuleType
    members: dict[str, Any]


# Each proxy's state lives here, not on the proxy, so no attribute of a proxy
# leads to the real module (review M1).
_STATE: "weakref.WeakKeyDictionary[ModuleProxy, _ProxyState]" = weakref.WeakKeyDictionary()


class ModuleProxy:
    """A module as a program sees it: the dictionary members, and nothing else.

    Each member is the real object. A name the real module has, but the
    dictionary does not list, refuses. A name the real module lacks raises
    ``AttributeError``, as plain Python does. ``__version__`` gives the real
    value. Setting or deleting an attribute always refuses. The proxy holds no
    state of its own: ``_STATE`` maps it to its module.
    """

    __slots__ = ("__weakref__",)

    def __init__(self, name: str, replacements: dict[str, Any] | None = None,
                 member_names: Any = None) -> None:
        module = importlib.import_module(name)
        members: dict[str, Any] = {}
        for member in (dictionary.MODULES[name] if member_names is None else member_names):
            if replacements and member in replacements:
                members[member] = replacements[member]
                continue
            if not hasattr(module, member):
                continue  # e.g. datetime.UTC before Python 3.11
            value = getattr(module, member)
            if isinstance(value, types.ModuleType):
                # No dictionary member is a module today; never hand one out.
                raise LackpyRefusal(f"{name}.{member} (a module)")
            members[member] = value
        _STATE[self] = _ProxyState(name, module, members)

    def __getattr__(self, attr: str) -> Any:
        # Reached only when normal lookup fails: every non-dunder name.
        name, module, members = _STATE[self]
        if attr in members:
            return members[attr]
        if attr == "__version__" and hasattr(module, attr):
            return module.__version__
        if not hasattr(module, attr):
            raise AttributeError(f"module {name!r} has no attribute {attr!r}")
        refuse(f"{name}.{attr}", f"{name}.{attr}")

    def __setattr__(self, attr: str, value: Any) -> None:
        refuse(f"{_STATE[self].name}.{attr} = ...")

    def __delattr__(self, attr: str) -> None:
        refuse(f"del {_STATE[self].name}.{attr}")

    def __repr__(self) -> str:
        return repr(_STATE[self].module)


def make_restricted_import(path_policy: PathPolicy | None = None,
                           trace: "Trace | None" = None) -> Callable[..., ModuleProxy]:
    """Return a fresh ``__import__`` for one program run.

    With ``path_policy``, ``pathlib`` is a proxy whose ``Path`` is the stand-in
    bound to that policy, one class per run. Without it, ``pathlib`` refuses.
    With ``trace``, the stand-in records its I/O calls there.

    Within a run, a module imported twice is the same proxy, as in plain
    Python. The members are the real, shared objects: the static
    ``safety.attr-store`` rule and the absent ``setattr`` keep a program from
    changing them (Review Focus 5, as re-ruled).
    """
    proxies: dict[str, ModuleProxy] = {}

    def restricted_import(name: str, globals: Any = None, locals: Any = None,
                          fromlist: Any = (), level: int = 0) -> ModuleProxy:
        if level:
            refuse("a relative import")
        if name == "pathlib" and path_policy is None:
            raise LackpyRefusal(f"import {name}", _PATH_PENDING_HINT)
        if name not in dictionary.MODULES:
            refuse(f"import {name}", name)
        if "." in name and not fromlist:
            # ``import a.b.c as X`` runs IMPORT_NAME a.b.c, then IMPORT_FROM b and c.
            # Serve a chain of shells, each exposing only the next part, so that X
            # is the proxy for a.b.c and no shell leads anywhere else.
            return _dotted_chain(name, restricted_import(name, fromlist=("*",)))
        if name not in proxies:
            if name == "pathlib":
                proxies[name] = ModuleProxy(name, {"Path": make_path_class(path_policy, trace)})
            else:
                proxies[name] = ModuleProxy(name)
        return proxies[name]

    return restricted_import


def _dotted_chain(name: str, leaf: ModuleProxy) -> ModuleProxy:
    """The shell proxy for the top of ``a.b.c``: ``a`` exposes ``b``, ``b`` exposes ``c``, ``c`` is ``leaf``."""
    parts = name.split(".")
    node = leaf
    for i in range(len(parts) - 1, 0, -1):
        node = ModuleProxy(".".join(parts[:i]), {parts[i]: node}, (parts[i],))
    return node
