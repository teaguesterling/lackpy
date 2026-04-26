"""Tests for UmweltPolicySource."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from lackpy.policy.types import PolicyResult, PolicyContext, ToolConstraints
from lackpy.policy.sources.umwelt import UmweltPolicySource
from lackpy.kit.registry import ResolvedKit
from lackpy.lang.grader import Grade


class FakePolicyEngine:
    """Stub PolicyEngine that returns pre-configured tool entries."""

    def __init__(self, tool_entries: list[dict]):
        self._entries = tool_entries

    def resolve_all(self, type: str = "tool"):
        return self._entries


@pytest.fixture
def kit():
    return ResolvedKit(
        tools={}, callables={}, grade=Grade(w=0, d=0), description="",
    )


class TestUmweltPolicySourceBasic:
    def test_name_and_priority(self):
        source = UmweltPolicySource(FakePolicyEngine([]))
        assert source.name == "umwelt"
        assert source.priority == 100

    def test_restricts_to_kit_intersection(self, kit):
        engine = FakePolicyEngine([
            {"id": "read_file", "visible": "true"},
            {"id": "edit_file", "visible": "true"},
            {"id": "bash", "visible": "true"},
        ])
        source = UmweltPolicySource(engine)
        current = PolicyResult(allowed_tools=frozenset({"read_file", "edit_file"}))
        context: PolicyContext = {"kit": kit}
        result = source.resolve(current, context)
        assert result.allowed_tools == frozenset({"read_file", "edit_file"})
        assert "bash" not in result.allowed_tools

    def test_denies_tools_marked_invisible(self, kit):
        engine = FakePolicyEngine([
            {"id": "read_file", "visible": "true"},
            {"id": "edit_file", "visible": "false"},
        ])
        source = UmweltPolicySource(engine)
        current = PolicyResult(
            allowed_tools=frozenset({"read_file", "edit_file"}),
        )
        context: PolicyContext = {"kit": kit}
        result = source.resolve(current, context)
        assert result.allowed_tools == frozenset({"read_file"})
        assert "edit_file" in result.denied_tools

    def test_cannot_grant_tools_kit_lacks(self, kit):
        engine = FakePolicyEngine([
            {"id": "read_file", "visible": "true"},
            {"id": "bash", "visible": "true"},
        ])
        source = UmweltPolicySource(engine)
        current = PolicyResult(allowed_tools=frozenset({"read_file"}))
        context: PolicyContext = {"kit": kit}
        result = source.resolve(current, context)
        assert result.allowed_tools == frozenset({"read_file"})
        assert "bash" not in result.allowed_tools

    def test_never_marks_resolved(self, kit):
        source = UmweltPolicySource(FakePolicyEngine([]))
        context: PolicyContext = {"kit": kit}
        result = source.resolve(PolicyResult(), context)
        assert result.resolved is False


class TestUmweltPolicySourceConstraints:
    def test_sets_tool_constraints(self, kit):
        engine = FakePolicyEngine([
            {
                "id": "read_file",
                "visible": "true",
                "max_level": "2",
                "allow_patterns": ["src/**/*.py"],
                "deny_patterns": ["*.secret"],
            },
        ])
        source = UmweltPolicySource(engine)
        current = PolicyResult(allowed_tools=frozenset({"read_file"}))
        context: PolicyContext = {"kit": kit}
        result = source.resolve(current, context)
        assert "read_file" in result.tool_constraints
        tc = result.tool_constraints["read_file"]
        assert tc.max_level == 2
        assert tc.allow_patterns == ("src/**/*.py",)
        assert tc.deny_patterns == ("*.secret",)

    def test_no_constraints_when_not_specified(self, kit):
        engine = FakePolicyEngine([
            {"id": "read_file", "visible": "true"},
        ])
        source = UmweltPolicySource(engine)
        current = PolicyResult(allowed_tools=frozenset({"read_file"}))
        context: PolicyContext = {"kit": kit}
        result = source.resolve(current, context)
        assert "read_file" not in result.tool_constraints

    def test_merges_denied_with_existing(self, kit):
        engine = FakePolicyEngine([
            {"id": "bash", "visible": "false"},
        ])
        source = UmweltPolicySource(engine)
        current = PolicyResult(
            allowed_tools=frozenset({"read_file", "bash"}),
            denied_tools=frozenset({"rm_rf"}),
        )
        context: PolicyContext = {"kit": kit}
        result = source.resolve(current, context)
        assert "rm_rf" in result.denied_tools
        assert "bash" in result.denied_tools


class TestUmweltPolicySourcePreservesOtherFields:
    def test_preserves_hints_and_docs(self, kit):
        engine = FakePolicyEngine([
            {"id": "read_file", "visible": "true"},
        ])
        source = UmweltPolicySource(engine)
        current = PolicyResult(
            allowed_tools=frozenset({"read_file"}),
            prompt_hints=("use read_file for files",),
            docs=("docs/tools/read_file.md",),
            namespace_desc="read_file(path) -> str",
        )
        context: PolicyContext = {"kit": kit}
        result = source.resolve(current, context)
        assert result.prompt_hints == ("use read_file for files",)
        assert result.docs == ("docs/tools/read_file.md",)
        assert result.namespace_desc == "read_file(path) -> str"
