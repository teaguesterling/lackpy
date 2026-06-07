"""Virtual / harness-provided tool source (RFC 0002 §7).

A virtual tool is *fully declared* in config (name, args, returns, grade) but its
implementation is supplied by the host/harness at call time via a resolver, and
may or may not be available. Enforcement is two-layered (see LackpyService):

- generation-time visibility gate: unavailable virtual tools are filtered out of
  the kit before inference, so the model never composes against an absent tool;
- call-time raise: if a tool was visible at generation but the harness withdrew it
  before the call, the proxy raises (→ failed ExecutionResult via make_traced).
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

from ..kit.toolbox import ArgSpec, ToolSpec

# A harness resolver maps a tool name to its current implementation, or None if
# the harness does not currently offer it.
HarnessResolver = Callable[[str], Callable[..., Any] | None]


class VirtualToolSource:
    """Declares harness-provided tools; resolution defers to a harness resolver."""

    def __init__(self, tool_defs: list[dict[str, Any]] | None,
                 resolver: HarnessResolver | None, name: str = "virtual") -> None:
        self._name = name
        self._tool_defs = tool_defs or []
        self._resolver = resolver

    @property
    def name(self) -> str:
        return self._name

    def available(self) -> bool:
        # Source-level: contribute the declared specs whenever any are configured.
        # Per-tool availability is enforced by the gate + the call-time proxy.
        return bool(self._tool_defs)

    def discover(self) -> list[ToolSpec]:
        return [self._spec_from_dict(d) for d in self._tool_defs]

    def is_available(self, name: str) -> bool:
        """Whether the harness currently offers ``name`` (used by the gate)."""
        return self._resolver is not None and self._resolver(name) is not None

    def resolve(self, spec: ToolSpec) -> Callable[..., Any]:
        resolver = self._resolver
        name = spec.name
        param_names = [a.name for a in spec.args]

        def proxy(*args: Any, **kwargs: Any) -> Any:
            fn = resolver(name) if resolver is not None else None
            if fn is None:
                raise RuntimeError(f"virtual tool {name!r} is unavailable")
            call_args = dict(zip(param_names, args))
            call_args.update(kwargs)
            return fn(**call_args)

        proxy.__signature__ = inspect.Signature(
            [inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD) for n in param_names]
        )
        proxy.__name__ = name
        return proxy

    @staticmethod
    def _spec_from_dict(d: dict[str, Any]) -> ToolSpec:
        name = d.get("name")
        if not name:
            raise ValueError(f"Virtual tool definition missing 'name': {d!r}")
        args = [
            ArgSpec(name=a["name"], type=a.get("type", "Any"), description=a.get("description", ""))
            for a in d.get("args", [])
        ]
        return ToolSpec(
            name=name,
            provider="virtual",
            description=d.get("description", ""),
            args=args,
            returns=d.get("returns", "Any"),
            grade_w=d.get("grade_w", 3),
            effects_ceiling=d.get("effects_ceiling", 3),
            docs=d.get("docs"),
        )
