"""The exec globals for a lackpy program (stepy100 Plan 2, Task 2).

This module is part of the sandbox boundary. A program's ``__builtins__`` holds
the names in ``builtins_policy.ALLOWED`` and a restricted ``__import__``
(``modules.make_restricted_import``, one per run). Everything else is absent:
the real ``__import__``, ``__build_class__``, ``open``,
``eval``, ``exec``, ``compile`` and the reflection builtins. The globals add
``__name__``, ``__file__``, ``print`` and a ``type`` stand-in.
"""

from __future__ import annotations

import builtins as _builtins
from typing import TYPE_CHECKING, Any, Callable

from ..lang import builtins_policy
from .modules import make_restricted_import
from .paths import PathPolicy
from .refusal import refuse

if TYPE_CHECKING:
    from .trace import Trace


class _TypeView:
    """The result of ``type(x)``. It exposes ``__name__`` only.

    The view holds the name and the repr text, never the real type, so no
    attribute chain can reach ``object`` or ``__subclasses__``. Where the view
    cannot give the CPython answer (equality, hashing, calling it), it refuses.
    """

    __slots__ = ("_name", "_repr")

    def __init__(self, name: str, text: str) -> None:
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_repr", text)

    def __getattribute__(self, attr: str) -> Any:
        if attr == "__name__":
            return object.__getattribute__(self, "_name")
        refuse(f"type(...).{attr}")

    def __setattr__(self, attr: str, value: Any) -> None:
        refuse(f"type(...).{attr} = ...")

    def __delattr__(self, attr: str) -> None:
        refuse(f"del type(...).{attr}")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        refuse("calling type(...)")

    def __eq__(self, other: object) -> bool:
        refuse("comparing type(...)")

    def __ne__(self, other: object) -> bool:
        refuse("comparing type(...)")

    def __hash__(self) -> int:
        refuse("hashing type(...)")

    def __repr__(self) -> str:
        return object.__getattribute__(self, "_repr")

    __str__ = __repr__


class _TypeStandIn:
    """The ``type`` name in a program: ``type(x)`` gives a ``_TypeView``.

    It is an instance, not a function, so it has no ``__globals__`` or
    ``__closure__`` to read. Every attribute refuses.
    """

    __slots__ = ()

    def __call__(self, *args: Any, **kwargs: Any) -> _TypeView:
        if kwargs or len(args) != 1:
            refuse("type() with other than one argument")
        (value,) = args
        if isinstance(value, (_TypeView, _TypeStandIn)):
            refuse("type(type(...))")
        real = type(value)
        return _TypeView(real.__name__, repr(real))

    def __getattribute__(self, attr: str) -> Any:
        refuse(f"type.{attr}")

    def __setattr__(self, attr: str, value: Any) -> None:
        refuse(f"type.{attr} = ...")

    def __delattr__(self, attr: str) -> None:
        refuse(f"del type.{attr}")

    def __repr__(self) -> str:
        return "<class 'type'>"

    __str__ = __repr__


_TYPE = _TypeStandIn()


def build_namespace(*, program_file: str | None, path_policy: PathPolicy | None = None,
                    print_fn: Callable[..., None],
                    trace: "Trace | None" = None) -> dict[str, Any]:
    """Return fresh exec globals for one program run.

    Args:
        program_file: The program's path, for ``__file__``. None gives ``"<stdin>"``.
        path_policy: The policy for file access. With it, ``pathlib`` serves the
            ``Path`` stand-in; with None, ``pathlib`` refuses (ruling C).
        print_fn: The function bound as ``print``, in the globals and the builtins.
        trace: Where the ``Path`` stand-in records its I/O calls. None records nothing.
    """
    program_builtins: dict[str, Any] = {
        name: getattr(_builtins, name) for name in builtins_policy.ALLOWED
    }
    program_builtins["print"] = print_fn
    program_builtins["__import__"] = make_restricted_import(path_policy, trace)
    return {
        "__builtins__": program_builtins,
        "__name__": "__main__",
        "__file__": program_file if program_file is not None else "<stdin>",
        "type": _TYPE,
        "print": print_fn,
    }
