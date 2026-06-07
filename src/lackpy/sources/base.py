"""The ToolSource protocol.

A ``ToolSource`` is a superset of the resolution-only ``ToolProvider``
(``kit/providers/base.py``): it adds ``discover()``, so the source — not bare
lackpy code — owns the set of tool names and their full specs.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

from ..kit.toolbox import ToolSpec


@runtime_checkable
class ToolSource(Protocol):
    @property
    def name(self) -> str: ...

    def available(self) -> bool:
        """Whether this source can currently contribute tools."""
        ...

    def discover(self) -> list[ToolSpec]:
        """Return the full specs for every tool this source provides."""
        ...

    def resolve(self, spec: ToolSpec) -> Callable[..., Any]:
        """Return the callable implementation for one of this source's specs."""
        ...
