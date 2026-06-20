"""Phase 1 — Profile / resolve_profile (RFC 0002 increment 5, additive).

Covers the core promise: a profile composes the existing resolve_kit (a kit is the
degenerate, tools-only profile), carries per-task inference + the language/execution
axes, falls back to service defaults, and parses from [profiles.<name>] config.
"""
from __future__ import annotations

import pytest

from lackpy.config import load_config
from lackpy.kit.providers.builtin import BuiltinProvider
from lackpy.kit.registry import resolve_kit
from lackpy.kit.toolbox import ArgSpec, ToolSpec, Toolbox
from lackpy.profiles import Profile, ResolvedProfile, resolve_profile


@pytest.fixture
def toolbox():
    tb = Toolbox()
    tb.register_provider(BuiltinProvider())
    for name, desc, grade in [("read_file", "Read file", 1), ("find_files", "Find files", 1),
                              ("edit_file", "Edit file", 3)]:
        tb.register_tool(ToolSpec(
            name=name, provider="builtin", description=desc,
            args=[ArgSpec(name="path", type="str")],
            returns="str", grade_w=grade, effects_ceiling=grade,
        ))
    return tb


class TestDegenerate:
    def test_tools_only_profile_equals_a_kit(self, toolbox):
        # The core invariant: a tools-only profile resolves to exactly what resolve_kit does.
        rp = resolve_profile(["read_file", "find_files"], toolbox)
        rk = resolve_kit(["read_file", "find_files"], toolbox)
        assert isinstance(rp, ResolvedProfile)
        assert set(rp.tools.tools) == set(rk.tools)
        assert rp.grade == rk.grade                 # grade passthrough = the tools' grade
        assert rp.description == rk.description
        assert rp.language == "restricted-python"   # axis defaults — today's path
        assert rp.execution == "one-shot"
        assert rp.model is None and rp.mode is None  # nothing carried, nothing defaulted

    def test_none_string_is_empty(self, toolbox):
        rp = resolve_profile("none", toolbox)
        assert rp.tools.tools == {}

    def test_extra_tools_merged_and_grade_raised(self, toolbox):
        rp = resolve_profile(Profile(tools=["read_file"], extra_tools=["edit_file"]), toolbox)
        assert set(rp.tools.tools) == {"read_file", "edit_file"}
        assert rp.grade.w == 3                       # edit_file (3/3) raises the join


class TestInferenceCarry:
    def test_profile_object_carries_inference_and_axes(self, toolbox):
        rp = resolve_profile(
            Profile(tools=["read_file"], model="ollama/qwen2.5-coder:3b", mode="spm",
                    language="shlack", execution="literate"),
            toolbox,
        )
        assert rp.model == "ollama/qwen2.5-coder:3b"
        assert rp.mode == "spm"
        assert rp.language == "shlack"
        assert rp.execution == "literate"

    def test_defaults_fallback_and_profile_wins(self, toolbox):
        defaults = {"model": "default/model", "mode": "1-shot"}
        # unset → falls back to defaults
        rp = resolve_profile(Profile(tools=["read_file"]), toolbox, defaults=defaults)
        assert rp.model == "default/model" and rp.mode == "1-shot"
        # set → profile wins over the default
        rp2 = resolve_profile(Profile(tools=["read_file"], model="prof/model"), toolbox,
                              defaults=defaults)
        assert rp2.model == "prof/model" and rp2.mode == "1-shot"


class TestByName:
    def test_resolve_named_profile_from_registry(self, toolbox):
        profiles = {"fast": {"tools": ["read_file", "find_files"],
                             "model": "ollama/qwen2.5-coder:3b", "mode": "1-shot"}}
        rp = resolve_profile("fast", toolbox, profiles=profiles)
        assert set(rp.tools.tools) == {"read_file", "find_files"}
        assert rp.model == "ollama/qwen2.5-coder:3b" and rp.mode == "1-shot"

    def test_str_not_a_named_profile_is_a_toolset(self, toolbox):
        # A str that isn't a configured profile is a bare tool-set value → resolve_kit
        # treats it as a kit/profile file name (here: missing → FileNotFoundError).
        with pytest.raises(FileNotFoundError):
            resolve_profile("not-a-profile", toolbox, profiles={"fast": {}}, kits_dir=None)


class TestFromConfig:
    def test_valid_table(self):
        p = Profile.from_config({"tools": ["read_file"], "model": "m", "execution": "literate"})
        assert p.tools == ["read_file"] and p.model == "m" and p.execution == "literate"
        assert p.language == "restricted-python"     # default for unset axis

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="unknown profile key"):
            Profile.from_config({"toolz": ["read_file"]})


def test_config_parses_profiles_table(tmp_path):
    cfg_file = tmp_path / ".lackpy" / "config.toml"
    cfg_file.parent.mkdir()
    cfg_file.write_text('''
[profiles.fast]
tools = ["read_file", "find_files"]
model = "ollama/qwen2.5-coder:3b"
mode  = "1-shot"

[profiles.notebook]
tools     = "filesystem"
language  = "shlack"
execution = "literate"
''')
    cfg = load_config(tmp_path)
    assert "fast" in cfg.profiles and "notebook" in cfg.profiles
    assert cfg.profiles["fast"]["model"] == "ollama/qwen2.5-coder:3b"
    assert cfg.profiles["notebook"]["execution"] == "literate"


def test_profile_file_extension_loads(tmp_path, toolbox):
    """Regression (review #2): a .profile file is listed AND loadable, like .kit."""
    from lackpy.kit.registry import resolve_kit
    kits = tmp_path / "kits"
    kits.mkdir()
    (kits / "fs.profile").write_text("---\nname: fs\n---\nread_file\nfind_files\n")
    rt = resolve_kit("fs", toolbox, kits_dir=kits)
    assert set(rt.tools) == {"read_file", "find_files"}
