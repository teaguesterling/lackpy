"""Config-defined tool source.

Tools are *fully defined* in configuration — each entry carries its own spec
(name, args, returns, grade, docs) AND how to resolve its callable (provider +
module/function). Nothing is hard-coded in lackpy source.
"""

from __future__ import annotations

from typing import Any, Callable

from ..tools.providers.python import PythonProvider
from ..tools.toolbox import ArgSpec, ToolSpec


class ConfigToolSource:
    """A :class:`~lackpy.sources.base.ToolSource` backed by config dicts.

    Args:
        tool_defs: A list of fully-defined tool dicts (see ``default_tools.toml``
            for the shape). Each needs at least ``name``; ``provider`` defaults to
            ``"python"`` and resolves via ``module``/``function``.
        name: The source's name (used as provenance / for diagnostics).
    """

    def __init__(self, tool_defs: list[dict[str, Any]] | None, name: str = "config") -> None:
        self._name = name
        self._tool_defs = tool_defs or []
        self._python = PythonProvider()

    @property
    def name(self) -> str:
        return self._name

    def available(self) -> bool:
        return bool(self._tool_defs)

    def discover(self) -> list[ToolSpec]:
        return [self._spec_from_dict(d) for d in self._tool_defs]

    def resolve(self, spec: ToolSpec) -> Callable[..., Any]:
        if spec.provider == "python":
            return self._python.resolve(spec)
        raise ValueError(
            f"ConfigToolSource cannot resolve provider {spec.provider!r} for tool "
            f"{spec.name!r} (supported: 'python')"
        )

    @staticmethod
    def _spec_from_dict(d: dict[str, Any]) -> ToolSpec:
        name = d.get("name")
        if not name:
            raise ValueError(f"Tool definition missing 'name': {d!r}")
        # Fold module/function into provider_config so PythonProvider resolves it.
        provider_config = dict(d.get("provider_config", {}))
        if "module" in d:
            provider_config.setdefault("module", d["module"])
        if "function" in d:
            provider_config.setdefault("function", d["function"])
        args = [
            ArgSpec(
                name=a["name"],
                type=a.get("type", "Any"),
                description=a.get("description", ""),
            )
            for a in d.get("args", [])
        ]
        return ToolSpec(
            name=name,
            provider=d.get("provider", "python"),
            provider_config=provider_config,
            description=d.get("description", ""),
            args=args,
            returns=d.get("returns", "Any"),
            grade_w=d.get("grade_w", 3),
            effects_ceiling=d.get("effects_ceiling", 3),
            examples=d.get("examples", []),
            docs=d.get("docs"),
        )
