"""Tests for the toolbox — tool provider store."""

import pytest

from lackpy.kit.toolbox import Toolbox, ToolSpec, ArgSpec
from lackpy.kit.providers.builtin import BuiltinProvider


@pytest.fixture
def toolbox():
    tb = Toolbox()
    tb.register_provider(BuiltinProvider())
    return tb


def test_register_tool(toolbox):
    spec = ToolSpec(name="read", provider="builtin", provider_config={},
                    description="Read file contents",
                    args=[ArgSpec(name="path", type="str", description="File path")],
                    returns="str", grade_w=1, effects_ceiling=1)
    toolbox.register_tool(spec)
    assert "read" in toolbox.tools


def test_resolve_tool(toolbox):
    spec = ToolSpec(name="read", provider="builtin", provider_config={},
                    description="Read file contents",
                    args=[ArgSpec(name="path", type="str", description="File path")],
                    returns="str")
    toolbox.register_tool(spec)
    fn = toolbox.resolve("read")
    assert callable(fn)


def test_resolve_unknown_tool_raises(toolbox):
    with pytest.raises(KeyError):
        toolbox.resolve("nonexistent")


def test_list_tools(toolbox):
    spec = ToolSpec(name="read", provider="builtin", provider_config={},
                    description="Read file contents", args=[], returns="str")
    toolbox.register_tool(spec)
    tools = toolbox.list_tools()
    assert len(tools) == 1
    assert tools[0].name == "read"


def test_format_namespace_description(toolbox):
    spec = ToolSpec(name="read", provider="builtin", provider_config={},
                    description="Read file contents",
                    args=[ArgSpec(name="path", type="str", description="File path")],
                    returns="str")
    toolbox.register_tool(spec)
    desc = toolbox.format_description(["read"])
    assert "read(path)" in desc
    assert "str" in desc
