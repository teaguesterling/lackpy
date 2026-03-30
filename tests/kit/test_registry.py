"""Tests for kit resolution."""

import pytest

from lackpy.kit.registry import resolve_kit, ResolvedKit
from lackpy.kit.toolbox import Toolbox, ToolSpec, ArgSpec
from lackpy.kit.providers.builtin import BuiltinProvider


@pytest.fixture
def toolbox():
    tb = Toolbox()
    tb.register_provider(BuiltinProvider())
    for name, desc, grade in [("read", "Read file", 1), ("glob", "Find files", 1), ("edit", "Edit file", 3)]:
        tb.register_tool(ToolSpec(
            name=name, provider="builtin", description=desc,
            args=[ArgSpec(name="path", type="str")],
            returns="str", grade_w=grade, effects_ceiling=grade,
        ))
    return tb


class TestResolveFromList:
    def test_resolves_explicit_list(self, toolbox):
        kit = resolve_kit(["read", "glob"], toolbox)
        assert "read" in kit.tools
        assert "glob" in kit.tools
        assert "edit" not in kit.tools
        assert kit.grade.w == 1

    def test_unknown_tool_in_list_raises(self, toolbox):
        with pytest.raises(KeyError):
            resolve_kit(["read", "nonexistent"], toolbox)


class TestResolveFromName:
    def test_resolves_predefined_kit(self, toolbox, tmp_path):
        kit_file = tmp_path / "debug.kit"
        kit_file.write_text("---\nname: debug\ndescription: Read-only\n---\nread\nglob\n")
        kit = resolve_kit("debug", toolbox, kits_dir=tmp_path)
        assert "read" in kit.tools
        assert "glob" in kit.tools
        assert "edit" not in kit.tools

    def test_unknown_kit_name_raises(self, toolbox, tmp_path):
        with pytest.raises(FileNotFoundError):
            resolve_kit("nonexistent", toolbox, kits_dir=tmp_path)


class TestResolveFromDict:
    def test_resolves_dict_mapping(self, toolbox):
        kit = resolve_kit({"reader": "read", "finder": "glob"}, toolbox)
        assert "reader" in kit.tools
        assert "finder" in kit.tools
        assert kit.grade.w == 1


class TestResolvedKitDescription:
    def test_has_namespace_description(self, toolbox):
        kit = resolve_kit(["read"], toolbox)
        assert "read" in kit.description
        assert len(kit.description) > 0


class TestQuartermaster:
    def test_none_raises_not_implemented(self, toolbox):
        with pytest.raises(NotImplementedError):
            resolve_kit(None, toolbox)
