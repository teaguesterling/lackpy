"""Tests for kit resolution."""

import pytest

from lackpy.tools.registry import resolve_tools, ResolvedTools
from lackpy.tools.toolbox import Toolbox, ToolSpec, ArgSpec
from lackpy.tools.providers.builtin import BuiltinProvider


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
        kit = resolve_tools(["read_file", "find_files"], toolbox)
        assert "read_file" in kit.tools
        assert "find_files" in kit.tools
        assert "edit_file" not in kit.tools
        assert kit.grade.w == 1

    def test_unknown_tool_in_list_raises(self, toolbox):
        with pytest.raises(KeyError) as ei:
            resolve_tools(["read_file", "nonexistent"], toolbox)
        # Actionable failure: names the tool, explains where tools come from, and
        # lists what's available — not a bare "Unknown tool".
        msg = str(ei.value)
        assert "nonexistent" in msg
        assert "configured source" in msg
        assert "read_file" in msg          # available tools surfaced


class TestResolveFromName:
    def test_resolves_predefined_kit(self, toolbox, tmp_path):
        kit_file = tmp_path / "debug.kit"
        kit_file.write_text("---\nname: debug\ndescription: Read-only\n---\nread_file\nfind_files\n")
        kit = resolve_tools("debug", toolbox, kits_dir=tmp_path)
        assert "read_file" in kit.tools
        assert "find_files" in kit.tools
        assert "edit_file" not in kit.tools

    def test_unknown_kit_name_raises(self, toolbox, tmp_path):
        with pytest.raises(FileNotFoundError):
            resolve_tools("nonexistent", toolbox, kits_dir=tmp_path)


class TestResolveFromDict:
    def test_resolves_dict_mapping(self, toolbox):
        kit = resolve_tools({"reader": "read_file", "finder": "find_files"}, toolbox)
        assert "reader" in kit.tools
        assert "finder" in kit.tools
        assert kit.grade.w == 1


class TestResolvedToolsDescription:
    def test_has_namespace_description(self, toolbox):
        kit = resolve_tools(["read_file"], toolbox)
        assert "read_file" in kit.description
        assert len(kit.description) > 0


class TestExtraTools:
    def test_extra_tools_merged_into_list_kit(self, toolbox):
        kit = resolve_tools(["read_file"], toolbox, extra_tools=["edit_file"])
        assert "read_file" in kit.tools
        assert "edit_file" in kit.tools
        assert kit.grade.w == 3

    def test_extra_tools_merged_into_named_kit(self, toolbox, tmp_path):
        kit_file = tmp_path / "readonly.kit"
        kit_file.write_text("---\nname: readonly\n---\nread_file\n")
        kit = resolve_tools("readonly", toolbox, kits_dir=tmp_path, extra_tools=["edit_file"])
        assert "read_file" in kit.tools
        assert "edit_file" in kit.tools

    def test_extra_tools_duplicate_ignored(self, toolbox):
        kit = resolve_tools(["read_file", "find_files"], toolbox, extra_tools=["read_file"])
        assert len(kit.tools) == 2

    def test_extra_tools_standalone_with_none_kit(self, toolbox):
        kit = resolve_tools(None, toolbox, extra_tools=["read_file"])
        assert "read_file" in kit.tools
        assert len(kit.tools) == 1

    def test_extra_tools_unknown_raises(self, toolbox):
        with pytest.raises(KeyError):
            resolve_tools(["read_file"], toolbox, extra_tools=["nonexistent"])

    def test_none_kit_string_resolves_empty(self, toolbox):
        kit = resolve_tools("none", toolbox)
        assert len(kit.tools) == 0

    def test_none_kit_string_with_extra_tools(self, toolbox):
        kit = resolve_tools("none", toolbox, extra_tools=["read_file"])
        assert "read_file" in kit.tools
        assert len(kit.tools) == 1


class TestQuartermaster:
    def test_none_raises_not_implemented(self, toolbox):
        with pytest.raises(NotImplementedError):
            resolve_tools(None, toolbox)


class TestUnresolvableProfileErrors:
    """A bare `--profile <token>` is read as a profile NAME, not a tool.

    That is the most common way to misuse this API — `--profile log` looks for
    log.profile and fails with program='' and no usable reason. These assert the
    error says what actually went wrong and gives the invocation that works.
    """

    def test_names_the_tool_and_the_fix(self, toolbox, tmp_path):
        with pytest.raises(FileNotFoundError) as e:
            resolve_tools("read_file", toolbox, kits_dir=tmp_path)
        msg = str(e.value)
        assert "IS a registered tool, not a profile" in msg
        assert "--profile none --tools read_file" in msg

    def test_default_debug_profile_explains_itself(self, toolbox, tmp_path):
        """config.profile_default is 'debug' and no debug profile ships, so
        omitting --profile fails looking for a file that never existed.

        The message must not assert the profile was *defaulted*: this branch
        fires on the name, and `--profile debug` typed explicitly lands here too.
        It is not reachable from this function which of the two happened.
        """
        with pytest.raises(FileNotFoundError) as e:
            resolve_tools("debug", toolbox, kits_dir=tmp_path)
        msg = str(e.value)
        assert "No 'debug' profile ships" in msg
        assert "built-in default" in msg
        assert "--profile none --tools" in msg
        assert "Nothing specified a profile" not in msg, (
            "message claims the profile was defaulted, which this branch "
            "cannot know -- `--profile debug` reaches it too"
        )

    def test_unknown_name_lists_available_profiles(self, toolbox, tmp_path):
        (tmp_path / "fix.profile").write_text("read_file\n")
        with pytest.raises(FileNotFoundError) as e:
            resolve_tools("nosuchprofile", toolbox, kits_dir=tmp_path)
        assert "fix" in str(e.value)
