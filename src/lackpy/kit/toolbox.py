"""Plugin-based tool provider store."""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Names that will confuse small language models if used as tool names.
# Includes Python stdlib top-level modules and builtin functions.
_MASKING_NAMES: frozenset[str] = frozenset(
    getattr(sys, "stdlib_module_names", set()) | set(dir(__builtins__))
)


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
        examples: Tagged examples for retrieval-augmented prompting. Each
            example is a dict with keys: ``intent``, ``code``, ``tags``.
            At inference time, relevant examples are selected from the
            pool of all kit tool examples and injected into the prompt.
        docs: Path to a markdown documentation file, relative to the
            provider's package root. Resolved lazily at query time via
            the provider's ``resolve_docs`` method.
    """

    name: str
    provider: str
    provider_config: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    args: list[ArgSpec] = field(default_factory=list)
    returns: str = "Any"
    grade_w: int = 3
    effects_ceiling: int = 3
    examples: list[dict] = field(default_factory=list)
    docs: str | None = None


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

        Warns:
            UserWarning: If the tool name masks a Python stdlib module or builtin,
                which causes small language models to generate incorrect code.
        """
        if spec.name in _MASKING_NAMES:
            warnings.warn(
                f"Tool name '{spec.name}' masks a Python stdlib module or builtin. "
                f"Small language models may generate `{spec.name}.{spec.name}()` or "
                f"`import {spec.name}` instead of calling the tool directly. "
                f"Consider a more specific name (e.g. '{spec.name}_file').",
                UserWarning,
                stacklevel=2,
            )
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

    def resolve_docs(self, name: str, docs_root: Path) -> Path | None:
        """Return the absolute path to a tool's documentation file, or None.

        Args:
            name: Tool name to look up.
            docs_root: Root directory to resolve relative ``docs`` paths against.
        """
        if name not in self.tools:
            return None
        spec = self.tools[name]
        if not spec.docs:
            return None
        resolved = docs_root / spec.docs
        return resolved if resolved.exists() else None

    def docs_index(self) -> dict[str, str | None]:
        """Return a mapping of tool name to docs relative path for all tools."""
        return {name: spec.docs for name, spec in self.tools.items()}

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


ARGSPEC_TYPE_MAP: dict[str, type] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "dict": dict,
    "list": list,
    "Any": Any,
}


def resolve_python_type(type_str: str) -> type:
    """Map an ArgSpec type string to a Python type.

    Returns ``Any`` for unrecognized type strings (e.g. ``"list[str]"``).
    """
    return ARGSPEC_TYPE_MAP.get(type_str, Any)
