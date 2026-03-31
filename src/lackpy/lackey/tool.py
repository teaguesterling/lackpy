"""Tool descriptor for binding tools to providers in Lackey classes."""

from __future__ import annotations

from typing import Any


class Tool:
    """Descriptor that binds a tool to a provider.

    When accessed on a Lackey instance (self.read), returns a callable
    that delegates to the resolved provider implementation.

    When accessed on the class (CountLines.read), returns the Tool
    descriptor itself for introspection.
    """

    def __init__(self, provider: Any = None) -> None:
        self._provider = provider
        self._name: str | None = None

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        return obj._resolved_tools[self._name]

    def __repr__(self) -> str:
        provider_str = ""
        if self._provider is not None:
            provider_str = f", provider={self._provider!r}"
        return f"Tool(name={self._name!r}{provider_str})"
