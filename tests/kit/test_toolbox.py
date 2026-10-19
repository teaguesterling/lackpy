"""Tests for the toolbox — tool provider store."""

import pytest

from lackpy.tools.toolbox import Toolbox, ToolSpec, ArgSpec
try:
    from lackpy.tools.providers.builtin import BuiltinProvider
except ImportError:  # removed with the kit in stepy100 Plan 2, Task 7
    BuiltinProvider = None


@pytest.fixture
def toolbox():
    tb = Toolbox()
    tb.register_provider(BuiltinProvider())
    return tb


@pytest.mark.xfail(reason="kit removed in stepy100 Plan 2; tool registry migration pending", strict=True)
def test_register_tool(toolbox):
    spec = ToolSpec(name="read_file", provider="builtin", provider_config={},
                    description="Read file contents",
                    args=[ArgSpec(name="path", type="str", description="File path")],
                    returns="str", grade_w=1, effects_ceiling=1)
    toolbox.register_tool(spec)
    assert "read_file" in toolbox.tools


@pytest.mark.xfail(reason="kit removed in stepy100 Plan 2; tool registry migration pending", strict=True)
def test_resolve_tool(toolbox):
    spec = ToolSpec(name="read_file", provider="builtin", provider_config={},
                    description="Read file contents",
                    args=[ArgSpec(name="path", type="str", description="File path")],
                    returns="str")
    toolbox.register_tool(spec)
    fn = toolbox.resolve("read_file")
    assert callable(fn)


@pytest.mark.xfail(reason="kit removed in stepy100 Plan 2; tool registry migration pending", strict=True)
def test_resolve_unknown_tool_raises(toolbox):
    with pytest.raises(KeyError):
        toolbox.resolve("nonexistent")


@pytest.mark.xfail(reason="kit removed in stepy100 Plan 2; tool registry migration pending", strict=True)
def test_list_tools(toolbox):
    spec = ToolSpec(name="read_file", provider="builtin", provider_config={},
                    description="Read file contents", args=[], returns="str")
    toolbox.register_tool(spec)
    tools = toolbox.list_tools()
    assert len(tools) == 1
    assert tools[0].name == "read_file"


@pytest.mark.xfail(reason="kit removed in stepy100 Plan 2; tool registry migration pending", strict=True)
def test_format_namespace_description(toolbox):
    spec = ToolSpec(name="read_file", provider="builtin", provider_config={},
                    description="Read file contents",
                    args=[ArgSpec(name="path", type="str", description="File path")],
                    returns="str")
    toolbox.register_tool(spec)
    desc = toolbox.format_description(["read_file"])
    assert "read_file(path)" in desc
    assert "str" in desc
