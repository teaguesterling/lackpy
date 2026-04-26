"""UmweltPolicySource: world-model policy from umwelt's PolicyEngine."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

from ..types import PolicyContext, PolicyResult, ToolConstraints


def _parse_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


class UmweltPolicySource:
    """Restricts tools based on umwelt's resolved capability-taxon policy.

    Highest priority. Can restrict the kit's tool set but cannot
    grant tools the kit doesn't have. Kit resolution (S1) is the
    ground truth for what's available.
    """

    name = "umwelt"
    priority = 100

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def resolve(self, current: PolicyResult, context: PolicyContext) -> PolicyResult:
        tool_entries = self._engine.resolve_all(type="tool")

        allowed: set[str] = set()
        denied: set[str] = set()
        constraints: dict[str, ToolConstraints] = {}

        for entry in tool_entries:
            name = entry["id"]
            if entry.get("visible") == "false":
                denied.add(name)
            else:
                allowed.add(name)

            if entry.get("max_level") or entry.get("allow_patterns") or entry.get("deny_patterns"):
                constraints[name] = ToolConstraints(
                    max_level=_parse_int(entry.get("max_level")),
                    allow_patterns=tuple(entry.get("allow_patterns", ())),
                    deny_patterns=tuple(entry.get("deny_patterns", ())),
                )

        effective_allowed = current.allowed_tools & frozenset(allowed)
        effective_denied = current.denied_tools | frozenset(denied)

        return current.replace(
            allowed_tools=effective_allowed,
            denied_tools=effective_denied,
            tool_constraints=MappingProxyType(constraints) if constraints else current.tool_constraints,
            resolved=False,
        )
