"""Tests for tool provisioning decisions (embedded vs bridged vs unavailable)."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from lackpy.sandbox.constraints import BridgedToolPolicy


class TestToolProvisioningDecisions:
    def test_tool_with_source_is_embedded(self):
        from lackpy.sandbox.provisioning import classify_tool, ToolProvisionKind
        provider = MagicMock()
        provider.get_source = MagicMock(return_value="def read_file(path): ...")
        spec = MagicMock()
        spec.name = "read_file"
        result = classify_tool(spec, provider, bridge_policy=None)
        assert result == ToolProvisionKind.EMBEDDED

    def test_tool_without_source_no_bridge_is_unavailable(self):
        from lackpy.sandbox.provisioning import classify_tool, ToolProvisionKind
        provider = MagicMock()
        provider.get_source = MagicMock(return_value=None)
        spec = MagicMock()
        spec.name = "mcp_query"
        spec.provider = "mcp"
        result = classify_tool(spec, provider, bridge_policy=None)
        assert result == ToolProvisionKind.UNAVAILABLE

    def test_tool_without_source_bridge_allowed_by_name(self):
        from lackpy.sandbox.provisioning import classify_tool, ToolProvisionKind
        provider = MagicMock()
        provider.get_source = MagicMock(return_value=None)
        provider.name = "mcp"
        spec = MagicMock()
        spec.name = "mcp_query"
        spec.provider = "mcp"
        policy = BridgedToolPolicy(allowed=True, allowed_tools=("mcp_query",))
        result = classify_tool(spec, provider, bridge_policy=policy)
        assert result == ToolProvisionKind.BRIDGED

    def test_tool_without_source_bridge_allowed_by_provider(self):
        from lackpy.sandbox.provisioning import classify_tool, ToolProvisionKind
        provider = MagicMock()
        provider.get_source = MagicMock(return_value=None)
        provider.name = "mcp"
        spec = MagicMock()
        spec.name = "mcp_query"
        spec.provider = "mcp"
        policy = BridgedToolPolicy(allowed=True, allowed_providers=("mcp",))
        result = classify_tool(spec, provider, bridge_policy=policy)
        assert result == ToolProvisionKind.BRIDGED

    def test_tool_without_source_bridge_denied(self):
        from lackpy.sandbox.provisioning import classify_tool, ToolProvisionKind
        provider = MagicMock()
        provider.get_source = MagicMock(return_value=None)
        provider.name = "mcp"
        spec = MagicMock()
        spec.name = "mcp_query"
        spec.provider = "mcp"
        policy = BridgedToolPolicy(allowed=False)
        result = classify_tool(spec, provider, bridge_policy=policy)
        assert result == ToolProvisionKind.UNAVAILABLE

    def test_provider_without_get_source_treated_as_no_source(self):
        from lackpy.sandbox.provisioning import classify_tool, ToolProvisionKind
        provider = MagicMock(spec=["name", "available", "resolve"])
        provider.name = "legacy"
        spec = MagicMock()
        spec.name = "legacy_tool"
        spec.provider = "legacy"
        result = classify_tool(spec, provider, bridge_policy=None)
        assert result == ToolProvisionKind.UNAVAILABLE


class TestPartitionKit:
    def test_partition_returns_embedded_and_bridged(self):
        from lackpy.sandbox.provisioning import partition_kit, ToolProvisionKind
        spec_a = MagicMock()
        spec_a.name = "read_file"
        spec_a.provider = "builtin"
        spec_b = MagicMock()
        spec_b.name = "mcp_query"
        spec_b.provider = "mcp"

        provider_a = MagicMock()
        provider_a.name = "builtin"
        provider_a.get_source = MagicMock(return_value="def read_file(path): ...")
        provider_b = MagicMock()
        provider_b.name = "mcp"
        provider_b.get_source = MagicMock(return_value=None)

        tools = {"read_file": (spec_a, provider_a), "mcp_query": (spec_b, provider_b)}
        policy = BridgedToolPolicy(allowed=True, allowed_providers=("mcp",))
        embedded, bridged, unavailable = partition_kit(tools, policy)
        assert "read_file" in embedded
        assert "mcp_query" in bridged
        assert len(unavailable) == 0


class TestBuiltinProviderGetSource:
    def test_get_source_returns_string(self):
        from lackpy.kit.providers.builtin import BuiltinProvider
        from lackpy.kit.toolbox import ToolSpec
        provider = BuiltinProvider()
        spec = ToolSpec(name="read_file", description="Read a file", provider="builtin", args=[], grade_w=0, effects_ceiling="read")
        source = provider.get_source(spec)
        assert source is not None
        assert "def _builtin_read" in source

    def test_get_source_unknown_tool_returns_none(self):
        from lackpy.kit.providers.builtin import BuiltinProvider
        from lackpy.kit.toolbox import ToolSpec
        provider = BuiltinProvider()
        spec = ToolSpec(name="nonexistent", description="", provider="builtin", args=[], grade_w=0, effects_ceiling="none")
        source = provider.get_source(spec)
        assert source is None
