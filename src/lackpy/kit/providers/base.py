"""Base protocol for tool providers."""

from __future__ import annotations

from typing import Any, Callable, Protocol

from ..toolbox import ToolSpec


class ToolProvider(Protocol):
    @property
    def name(self) -> str: ...
    def available(self) -> bool: ...
    def resolve(self, tool_spec: ToolSpec) -> Callable[..., Any]: ...
