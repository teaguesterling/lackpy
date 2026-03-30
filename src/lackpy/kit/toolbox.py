"""Plugin-based tool provider store."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ArgSpec:
    name: str
    type: str = "Any"
    description: str = ""


@dataclass
class ToolSpec:
    name: str
    provider: str
    provider_config: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    args: list[ArgSpec] = field(default_factory=list)
    returns: str = "Any"
    grade_w: int = 3
    effects_ceiling: int = 3


class Toolbox:
    def __init__(self) -> None:
        self.tools: dict[str, ToolSpec] = {}
        self._providers: dict[str, Any] = {}

    def register_provider(self, provider: Any) -> None:
        self._providers[provider.name] = provider

    def register_tool(self, spec: ToolSpec) -> None:
        self.tools[spec.name] = spec

    def resolve(self, name: str) -> Callable[..., Any]:
        if name not in self.tools:
            raise KeyError(f"Unknown tool: {name}")
        spec = self.tools[name]
        provider = self._providers.get(spec.provider)
        if provider is None:
            raise KeyError(f"No provider '{spec.provider}' registered for tool '{name}'")
        return provider.resolve(spec)

    def list_tools(self) -> list[ToolSpec]:
        return list(self.tools.values())

    def format_description(self, tool_names: list[str]) -> str:
        lines = []
        for name in tool_names:
            spec = self.tools.get(name)
            if spec is None:
                continue
            args_str = ", ".join(a.name for a in spec.args)
            lines.append(f"  {spec.name}({args_str}) -> {spec.returns}: {spec.description}")
        return "\n".join(lines)
