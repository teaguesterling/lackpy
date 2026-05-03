"""Tool provisioning: classify tools as embedded, bridged, or unavailable."""

from __future__ import annotations

from enum import Enum
from typing import Any

from .constraints import BridgedToolPolicy


class ToolProvisionKind(Enum):
    EMBEDDED = "embedded"
    BRIDGED = "bridged"
    UNAVAILABLE = "unavailable"


def classify_tool(
    spec: Any,
    provider: Any,
    bridge_policy: BridgedToolPolicy | None,
) -> ToolProvisionKind:
    get_source = getattr(provider, "get_source", None)
    if get_source is not None:
        source = get_source(spec)
        if source is not None:
            return ToolProvisionKind.EMBEDDED

    if bridge_policy is None or not bridge_policy.allowed:
        return ToolProvisionKind.UNAVAILABLE

    if bridge_policy.allowed_tools and spec.name in bridge_policy.allowed_tools:
        return ToolProvisionKind.BRIDGED

    provider_name = getattr(provider, "name", getattr(spec, "provider", None))
    if bridge_policy.allowed_providers and provider_name in bridge_policy.allowed_providers:
        return ToolProvisionKind.BRIDGED

    return ToolProvisionKind.UNAVAILABLE


def partition_kit(
    tools: dict[str, tuple[Any, Any]],
    bridge_policy: BridgedToolPolicy | None,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    embedded: dict[str, Any] = {}
    bridged: dict[str, Any] = {}
    unavailable: list[str] = []

    for name, (spec, provider) in tools.items():
        kind = classify_tool(spec, provider, bridge_policy)
        if kind == ToolProvisionKind.EMBEDDED:
            embedded[name] = (spec, provider)
        elif kind == ToolProvisionKind.BRIDGED:
            bridged[name] = (spec, provider)
        else:
            unavailable.append(name)

    return embedded, bridged, unavailable
