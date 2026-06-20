"""Tests for UmweltPolicySource.

The stub PolicyEngine returns the SAME shape umwelt's real PolicyEngine.resolve_all
produces — {entity_id, type_name, classes, attributes, properties:{...}} with hyphenated
string-valued props. (An earlier flat-dict stub let a real KeyError drift hide; the
retritis bench/phase4 conformance test is the cross-repo net against the live engine.)
"""

from __future__ import annotations

import pytest

from lackpy.policy.types import PolicyResult, PolicyContext, ToolConstraints
from lackpy.policy.sources.umwelt import UmweltPolicySource
from lackpy.tools.registry import ResolvedTools
from lackpy.lang.grader import Grade


def _tool(entity_id: str, **properties) -> dict:
    """Build an entry in umwelt's real resolve_all(type='tool') shape."""
    return {
        "entity_id": entity_id,
        "type_name": "tool",
        "classes": [],
        "attributes": {"name": entity_id},
        "properties": {k.replace("_", "-"): v for k, v in properties.items()},
    }


class FakePolicyEngine:
    """Stub PolicyEngine returning entries in umwelt's real resolve_all shape."""

    def __init__(self, tool_entries: list[dict]):
        self._entries = tool_entries

    def resolve_all(self, type: str = "tool", context=None):
        return self._entries


@pytest.fixture
def tools():
    return ResolvedTools(
        tools={}, callables={}, grade=Grade(w=0, d=0), description="",
    )


class TestUmweltPolicySourceBasic:
    def test_name_and_priority(self):
        source = UmweltPolicySource(FakePolicyEngine([]))
        assert source.name == "umwelt"
        assert source.priority == 100

    def test_restricts_to_kit_intersection(self, tools):
        engine = FakePolicyEngine([
            _tool("read_file", allow="true"),
            _tool("edit_file", allow="true"),
            _tool("bash", allow="true"),
        ])
        source = UmweltPolicySource(engine)
        current = PolicyResult(allowed_tools=frozenset({"read_file", "edit_file"}))
        context: PolicyContext = {"tools": tools}
        result = source.resolve(current, context)
        assert result.allowed_tools == frozenset({"read_file", "edit_file"})
        assert "bash" not in result.allowed_tools

    def test_denies_tools_marked_not_allowed(self, tools):
        engine = FakePolicyEngine([
            _tool("read_file", allow="true"),
            _tool("edit_file", allow="false"),
        ])
        source = UmweltPolicySource(engine)
        current = PolicyResult(
            allowed_tools=frozenset({"read_file", "edit_file"}),
        )
        context: PolicyContext = {"tools": tools}
        result = source.resolve(current, context)
        assert result.allowed_tools == frozenset({"read_file"})
        assert "edit_file" in result.denied_tools

    def test_cannot_grant_tools_kit_lacks(self, tools):
        engine = FakePolicyEngine([
            _tool("read_file", allow="true"),
            _tool("bash", allow="true"),
        ])
        source = UmweltPolicySource(engine)
        current = PolicyResult(allowed_tools=frozenset({"read_file"}))
        context: PolicyContext = {"tools": tools}
        result = source.resolve(current, context)
        assert result.allowed_tools == frozenset({"read_file"})
        assert "bash" not in result.allowed_tools

    def test_never_marks_resolved(self, tools):
        source = UmweltPolicySource(FakePolicyEngine([]))
        context: PolicyContext = {"tools": tools}
        result = source.resolve(PolicyResult(), context)
        assert result.resolved is False


class TestUmweltPolicySourceConstraints:
    def test_sets_tool_constraints(self, tools):
        engine = FakePolicyEngine([
            _tool(
                "read_file",
                allow="true",
                max_level="2",
                allow_patterns="src/**/*.py",
                deny_patterns="*.secret",
            ),
        ])
        source = UmweltPolicySource(engine)
        current = PolicyResult(allowed_tools=frozenset({"read_file"}))
        context: PolicyContext = {"tools": tools}
        result = source.resolve(current, context)
        assert "read_file" in result.tool_constraints
        tc = result.tool_constraints["read_file"]
        assert tc.max_level == 2
        assert tc.allow_patterns == ("src/**/*.py",)
        assert tc.deny_patterns == ("*.secret",)

    def test_comma_separated_patterns_split(self, tools):
        """umwelt serializes list props as comma-separated strings — must split, not
        iterate per-character."""
        engine = FakePolicyEngine([
            _tool("edit_file", allow="true", allow_patterns="src/**,tests/**"),
        ])
        source = UmweltPolicySource(engine)
        current = PolicyResult(allowed_tools=frozenset({"edit_file"}))
        result = source.resolve(current, {"tools": tools})
        assert result.tool_constraints["edit_file"].allow_patterns == ("src/**", "tests/**")

    def test_no_constraints_when_not_specified(self, tools):
        engine = FakePolicyEngine([
            _tool("read_file", allow="true"),
        ])
        source = UmweltPolicySource(engine)
        current = PolicyResult(allowed_tools=frozenset({"read_file"}))
        context: PolicyContext = {"tools": tools}
        result = source.resolve(current, context)
        assert "read_file" not in result.tool_constraints

    def test_merges_denied_with_existing(self, tools):
        engine = FakePolicyEngine([
            _tool("bash", allow="false"),
        ])
        source = UmweltPolicySource(engine)
        current = PolicyResult(
            allowed_tools=frozenset({"read_file", "bash"}),
            denied_tools=frozenset({"rm_rf"}),
        )
        context: PolicyContext = {"tools": tools}
        result = source.resolve(current, context)
        assert "rm_rf" in result.denied_tools
        assert "bash" in result.denied_tools


class TestUmweltPolicySourcePreservesOtherFields:
    def test_preserves_hints_and_docs(self, tools):
        engine = FakePolicyEngine([
            _tool("read_file", allow="true"),
        ])
        source = UmweltPolicySource(engine)
        current = PolicyResult(
            allowed_tools=frozenset({"read_file"}),
            prompt_hints=("use read_file for files",),
            docs=("docs/tools/read_file.md",),
            namespace_desc="read_file(path) -> str",
        )
        context: PolicyContext = {"tools": tools}
        result = source.resolve(current, context)
        assert result.prompt_hints == ("use read_file for files",)
        assert result.docs == ("docs/tools/read_file.md",)
        assert result.namespace_desc == "read_file(path) -> str"
