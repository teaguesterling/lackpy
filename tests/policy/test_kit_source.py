"""Tests for ToolsPolicySource."""

from __future__ import annotations

import pytest

from lackpy.policy.types import PolicyResult, PolicyContext
from lackpy.policy.sources.tools import ToolsPolicySource
from lackpy.tools.registry import ResolvedTools, resolve_tools
from lackpy.tools.toolbox import Toolbox, ToolSpec, ArgSpec
from lackpy.tools.providers.builtin import BuiltinProvider
from lackpy.lang.grader import Grade


@pytest.fixture
def toolbox():
    tb = Toolbox()
    tb.register_provider(BuiltinProvider())
    for name, desc, grade in [
        ("read_file", "Read file contents", 1),
        ("find_files", "Find files matching pattern", 1),
        ("edit_file", "Edit a file", 3),
    ]:
        tb.register_tool(ToolSpec(
            name=name, provider="builtin", description=desc,
            args=[ArgSpec(name="path", type="str")],
            returns="str", grade_w=grade, effects_ceiling=grade,
        ))
    return tb


@pytest.fixture
def tools(toolbox):
    return resolve_tools(["read_file", "find_files"], toolbox)


class TestToolsPolicySource:
    def test_sets_allowed_tools_from_kit(self, toolbox, tools):
        source = ToolsPolicySource(toolbox)
        context: PolicyContext = {"tools": tools}
        result = source.resolve(PolicyResult(), context)
        assert result.allowed_tools == frozenset({"read_file", "find_files"})

    def test_sets_grade_from_kit(self, toolbox, tools):
        source = ToolsPolicySource(toolbox)
        context: PolicyContext = {"tools": tools}
        result = source.resolve(PolicyResult(), context)
        assert result.grade.w == 1
        assert result.grade.d == 1

    def test_sets_namespace_desc(self, toolbox, tools):
        source = ToolsPolicySource(toolbox)
        context: PolicyContext = {"tools": tools}
        result = source.resolve(PolicyResult(), context)
        assert result.namespace_desc is not None
        assert "read_file" in result.namespace_desc

    def test_never_marks_resolved(self, toolbox, tools):
        source = ToolsPolicySource(toolbox)
        context: PolicyContext = {"tools": tools}
        result = source.resolve(PolicyResult(), context)
        assert result.resolved is False

    def test_priority_is_zero(self, toolbox):
        source = ToolsPolicySource(toolbox)
        assert source.priority == 0

    def test_name_is_kit(self, toolbox):
        source = ToolsPolicySource(toolbox)
        assert source.name == "tools"

    def test_high_grade_kit(self, toolbox):
        tools = resolve_tools(["read_file", "edit_file"], toolbox)
        source = ToolsPolicySource(toolbox)
        context: PolicyContext = {"tools": tools}
        result = source.resolve(PolicyResult(), context)
        assert result.grade.w == 3
        assert result.allowed_tools == frozenset({"read_file", "edit_file"})
