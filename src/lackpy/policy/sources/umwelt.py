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


def _split_patterns(value: Any) -> tuple[str, ...]:
    """umwelt stores list-valued properties as comma-separated strings; lackpy's
    ToolConstraints wants a tuple. (A bare str would otherwise iterate per-char.)"""
    if not value:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(v).strip() for v in value if str(v).strip())
    return tuple(s.strip() for s in str(value).split(",") if s.strip())


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
        # Thread the active operating mode so mode-scoped rules resolve correctly.
        # Without a mode, an explicit empty context yields the unscoped baseline rather
        # than letting every mode's rules compete (which over-restricts every mode).
        mode = context.get("mode")
        resolve_ctx = {"mode": mode} if mode else {}
        tool_entries = self._engine.resolve_all(type="tool", context=resolve_ctx)

        allowed: set[str] = set()
        denied: set[str] = set()
        constraints: dict[str, ToolConstraints] = {}

        for entry in tool_entries:
            # umwelt's resolve_all(type="tool") yields
            #   {entity_id, type_name, classes, attributes, properties:{...}}
            # with property keys hyphenated and values as strings.
            name = entry["entity_id"]
            props = entry.get("properties", {})
            if props.get("allow") == "false":
                denied.add(name)
            else:
                allowed.add(name)

            max_level = props.get("max-level")
            allow_patterns = _split_patterns(props.get("allow-patterns"))
            deny_patterns = _split_patterns(props.get("deny-patterns"))
            if max_level or allow_patterns or deny_patterns:
                constraints[name] = ToolConstraints(
                    max_level=_parse_int(max_level),
                    allow_patterns=allow_patterns,
                    deny_patterns=deny_patterns,
                )

        effective_allowed = current.allowed_tools & frozenset(allowed)
        effective_denied = current.denied_tools | frozenset(denied)

        return current.replace(
            allowed_tools=effective_allowed,
            denied_tools=effective_denied,
            tool_constraints=MappingProxyType(constraints) if constraints else current.tool_constraints,
            resolved=False,
        )
