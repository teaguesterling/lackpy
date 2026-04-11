"""Interpreter plugin registry.

Interpreters register themselves at import time so the service layer
and CLI can resolve names to interpreter classes. The registry is a
flat dict — no priority ordering, no multi-backend resolution. Names
collide last-write-wins with a warning, same policy as the compiler
registry in umwelt.
"""

from __future__ import annotations

import warnings
from typing import Any

from .base import Interpreter


INTERPRETERS: dict[str, type] = {}


def register_interpreter(interpreter_cls: type) -> None:
    """Register an interpreter plugin by its ``name`` attribute.

    Args:
        interpreter_cls: A class implementing the :class:`Interpreter`
            protocol. Its ``name`` attribute becomes the registry key.

    Warns:
        UserWarning: If an interpreter with the same name is already
            registered. The new registration wins; the warning exists
            to catch accidental collisions in tests or plugin discovery.
    """
    name = getattr(interpreter_cls, "name", None)
    if not name:
        raise ValueError(
            f"{interpreter_cls.__name__} has no 'name' attribute; "
            "interpreters must declare a unique string name."
        )
    if name in INTERPRETERS and INTERPRETERS[name] is not interpreter_cls:
        warnings.warn(
            f"Interpreter '{name}' is being re-registered; "
            f"replacing {INTERPRETERS[name].__name__} with {interpreter_cls.__name__}.",
            UserWarning,
            stacklevel=2,
        )
    INTERPRETERS[name] = interpreter_cls


def get_interpreter(name: str) -> type:
    """Look up an interpreter class by its registered name.

    Args:
        name: The interpreter name as it appears in the registry.

    Returns:
        The interpreter class.

    Raises:
        KeyError: If no interpreter is registered under that name.
            The error message lists the available names to help the
            caller correct a typo.
    """
    if name not in INTERPRETERS:
        available = ", ".join(sorted(INTERPRETERS)) or "(none registered)"
        raise KeyError(
            f"Unknown interpreter: {name!r}. Available: {available}"
        )
    return INTERPRETERS[name]


def list_interpreters() -> list[dict[str, Any]]:
    """Return metadata for every registered interpreter.

    Used by CLI help and introspection tools to list what's available.
    """
    return [
        {
            "name": cls.name,
            "description": getattr(cls, "description", ""),
        }
        for cls in INTERPRETERS.values()
    ]
