"""Plugin-based tool provider store."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ArgSpec:
    """Specification for a single tool argument.

    Attributes:
        name: Argument name as it appears in the function signature.
        type: Type annotation string (e.g. ``"str"``, ``"int"``).
        description: Human-readable description of the argument.
    """

    name: str
    type: str = "Any"
    description: str = ""


@dataclass
class ToolSpec:
    """Specification for a registered tool.

    Attributes:
        name: Unique tool name used in lackpy programs.
        provider: Name of the provider plugin that implements this tool.
        provider_config: Provider-specific configuration dict.
        description: Human-readable description shown in the inference prompt.
        args: Ordered list of argument specifications.
        returns: Return type annotation string.
        grade_w: World coupling level (0–3).
        effects_ceiling: Effects ceiling level (0–3).
    """

    name: str
    provider: str
    provider_config: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    args: list[ArgSpec] = field(default_factory=list)
    returns: str = "Any"
    grade_w: int = 3
    effects_ceiling: int = 3


class Toolbox:
    """Registry of tool providers and their resolved tool specs.

    Tools are registered via providers (plugin objects with a ``name`` attribute
    and a ``resolve`` method). Individual tools can also be registered directly
    via ``register_tool``.
    """

    def __init__(self) -> None:
        self.tools: dict[str, ToolSpec] = {}
        self._providers: dict[str, Any] = {}

    def register_provider(self, provider: Any) -> None:
        """Register a tool provider plugin and load its tools into the registry.

        Args:
            provider: A provider object with a ``name`` attribute. The provider's
                tools are made available for resolution via ``resolve``.
        """
        self._providers[provider.name] = provider

    def register_tool(self, spec: ToolSpec) -> None:
        """Register a single tool spec directly.

        Args:
            spec: The ToolSpec to add. Overwrites any existing spec with the same name.
        """
        self.tools[spec.name] = spec

    def resolve(self, name: str) -> Callable[..., Any]:
        """Return the callable implementation for a named tool.

        Args:
            name: The tool name to resolve.

        Returns:
            A callable that implements the tool.

        Raises:
            KeyError: If the tool name is not registered or its provider is not loaded.
        """
        if name not in self.tools:
            raise KeyError(f"Unknown tool: {name}")
        spec = self.tools[name]
        provider = self._providers.get(spec.provider)
        if provider is None:
            raise KeyError(f"No provider '{spec.provider}' registered for tool '{name}'")
        return provider.resolve(spec)

    def list_tools(self) -> list[ToolSpec]:
        """Return all registered tool specs.

        Returns:
            A list of all ToolSpec objects in registration order.
        """
        return list(self.tools.values())

    def format_description(self, tool_names: list[str]) -> str:
        """Build a formatted namespace description string for a subset of tools.

        Each tool is rendered as ``name(args) -> return_type: description``.
        Tools not found in the registry are silently skipped.

        Args:
            tool_names: Names of the tools to include in the description.

        Returns:
            A newline-joined string of tool signatures, suitable for inference prompts.
        """
        lines = []
        for name in tool_names:
            spec = self.tools.get(name)
            if spec is None:
                continue
            args_str = ", ".join(a.name for a in spec.args)
            lines.append(f"  {spec.name}({args_str}) -> {spec.returns}: {spec.description}")
        return "\n".join(lines)
