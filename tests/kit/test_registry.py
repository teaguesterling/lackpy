"""Tests for kit resolution."""

import pytest

from lackpy.kit.registry import resolve_kit, ResolvedKit
from lackpy.kit.toolbox import Toolbox, ToolSpec, ArgSpec
from lackpy.kit.providers.builtin import BuiltinProvider


@pytest.fixture
def toolbox():
    tb = Toolbox()
    tb.register_provider(BuiltinProvider())
    for name, desc, grade in [("read_file", "Read file", 1), ("find_files", "Find files", 1), ("edit_file", "Edit file", 3)]:
        tb.register_tool(ToolSpec(
            name=name, provider="builtin", description=desc,
            args=[ArgSpec(name="path", type="str")],
            returns="str", grade_w=grade, effects_ceiling=grade,
        ))
    return tb


class TestResolveFromList:
    def test_resolves_explicit_list(self, toolbox):
        kit = resolve_kit(["read_file", "find_files"], toolbox)
        assert "read_file" in kit.tools
        assert "find_files" in kit.tools
        assert "edit_file" not in kit.tools
        assert kit.grade.w == 1

    def test_unknown_tool_in_list_raises(self, toolbox):
        with pytest.raises(KeyError):
            resolve_kit(["read_file", "nonexistent"], toolbox)


class TestResolveFromName:
    def test_resolves_predefined_kit(self, toolbox, tmp_path):
        kit_file = tmp_path / "debug.kit"
        kit_file.write_text("---\nname: debug\ndescription: Read-only\n---\nread_file\nfind_files\n")
        kit = resolve_kit("debug", toolbox, kits_dir=tmp_path)
        assert "read_file" in kit.tools
        assert "find_files" in kit.tools
        assert "edit_file" not in kit.tools

    def test_unknown_kit_name_raises(self, toolbox, tmp_path):
        with pytest.raises(FileNotFoundError):
            resolve_kit("nonexistent", toolbox, kits_dir=tmp_path)


class TestResolveFromDict:
    def test_resolves_dict_mapping(self, toolbox):
        kit = resolve_kit({"reader": "read_file", "finder": "find_files"}, toolbox)
        assert "reader" in kit.tools
        assert "finder" in kit.tools
        assert kit.grade.w == 1


class TestResolvedKitDescription:
    def test_has_namespace_description(self, toolbox):
        kit = resolve_kit(["read_file"], toolbox)
        assert "read_file" in kit.description
        assert len(kit.description) > 0


class TestExtraTools:
    def test_extra_tools_merged_into_list_kit(self, toolbox):
        kit = resolve_kit(["read_file"], toolbox, extra_tools=["edit_file"])
        assert "read_file" in kit.tools
        assert "edit_file" in kit.tools
        assert kit.grade.w == 3

    def test_extra_tools_merged_into_named_kit(self, toolbox, tmp_path):
        kit_file = tmp_path / "readonly.kit"
        kit_file.write_text("---\nname: readonly\n---\nread_file\n")
        kit = resolve_kit("readonly", toolbox, kits_dir=tmp_path, extra_tools=["edit_file"])
        assert "read_file" in kit.tools
        assert "edit_file" in kit.tools

    def test_extra_tools_duplicate_ignored(self, toolbox):
        kit = resolve_kit(["read_file", "find_files"], toolbox, extra_tools=["read_file"])
        assert len(kit.tools) == 2

    def test_extra_tools_standalone_with_none_kit(self, toolbox):
        kit = resolve_kit(None, toolbox, extra_tools=["read_file"])
        assert "read_file" in kit.tools
        assert len(kit.tools) == 1

    def test_extra_tools_unknown_raises(self, toolbox):
        with pytest.raises(KeyError):
            resolve_kit(["read_file"], toolbox, extra_tools=["nonexistent"])

    def test_none_kit_string_resolves_empty(self, toolbox):
        kit = resolve_kit("none", toolbox)
        assert len(kit.tools) == 0

    def test_none_kit_string_with_extra_tools(self, toolbox):
        kit = resolve_kit("none", toolbox, extra_tools=["read_file"])
        assert "read_file" in kit.tools
        assert len(kit.tools) == 1


class TestQuartermaster:
    def test_none_raises_not_implemented(self, toolbox):
        with pytest.raises(NotImplementedError):
            resolve_kit(None, toolbox)
