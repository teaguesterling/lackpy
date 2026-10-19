"""load_policy: the settings sources that feed a PathPolicy (stepy100 Plan 2, Task 4).

Every test passes a temporary ``home`` and ``managed``. No test reads the real
``/etc/claude-code`` or the real ``~/.claude``.
"""

import json
import sys
from pathlib import Path

import pytest

from lackpy.run.paths import load_policy, sources
from lackpy.run.refusal import LackpyRefusal


@pytest.fixture
def ws(tmp_path):
    w = tmp_path / "ws"
    (w / ".claude").mkdir(parents=True)
    (w / ".lackpy").mkdir()
    return w


@pytest.fixture
def home(tmp_path):
    h = tmp_path / "home"
    (h / ".claude").mkdir(parents=True)
    return h


@pytest.fixture
def managed(tmp_path):
    d = tmp_path / "etc" / "claude-code"
    d.mkdir(parents=True)
    return d / "managed-settings.json"


def write_settings(path, permissions=None, **top):
    data = dict(top)
    if permissions is not None:
        data["permissions"] = permissions
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def allowed(policy, path, op="read"):
    try:
        policy.check(Path(path), op)
    except LackpyRefusal:
        return False
    return True


def load(ws, home, managed):
    return load_policy(ws, home=home, managed=managed)


class TestManagedSettingsPath:
    @pytest.mark.parametrize("platform, expected", [
        ("linux", "/etc/claude-code/managed-settings.json"),
        ("darwin", "/Library/Application Support/ClaudeCode/managed-settings.json"),
        ("win32", r"C:\Program Files\ClaudeCode\managed-settings.json"),
    ])
    def test_platform_path(self, platform, expected):
        assert str(sources.managed_settings_path(platform)) == expected

    def test_default_is_for_this_platform(self):
        assert sources.MANAGED_SETTINGS == sources.managed_settings_path(sys.platform)

    def test_windows_never_uses_programdata(self):
        assert "ProgramData" not in str(sources.managed_settings_path("win32"))


class TestSources:
    def test_no_files_gives_the_default_policy(self, ws, home, managed):
        policy = load(ws, home, managed)
        assert allowed(policy, ws / "a")
        assert not allowed(policy, home / "a")

    def test_workspace_settings_deny(self, ws, home, managed):
        write_settings(ws / ".claude" / "settings.json", {"deny": ["Read(./secrets/**)"]})
        policy = load(ws, home, managed)
        assert not allowed(policy, ws / "secrets" / "a")
        assert not allowed(policy, ws / "secrets" / "a", "write")

    def test_local_settings_deny(self, ws, home, managed):
        write_settings(ws / ".claude" / "settings.local.json", {"deny": ["Edit(./out/**)"]})
        policy = load(ws, home, managed)
        assert allowed(policy, ws / "out" / "a")
        assert not allowed(policy, ws / "out" / "a", "write")

    def test_user_settings_anchor_is_dot_claude(self, ws, home, managed):
        write_settings(home / ".claude" / "settings.json", {"allow": ["Read(/agents/**)"]})
        policy = load(ws, home, managed)
        assert allowed(policy, home / ".claude" / "agents" / "x.md")

    def test_ask_is_a_deny(self, ws, home, managed):
        write_settings(ws / ".claude" / "settings.json", {"ask": ["Read(./a)"]})
        assert not allowed(load(ws, home, managed), ws / "a")

    def test_non_path_rules_are_ignored(self, ws, home, managed):
        write_settings(ws / ".claude" / "settings.json",
                       {"deny": ["Bash(rm:*)", "Write(./a)"], "allow": ["WebFetch"]})
        policy = load(ws, home, managed)
        assert allowed(policy, ws / "a", "write")

    def test_additional_directories(self, ws, home, managed, tmp_path):
        extra = tmp_path / "extra"
        write_settings(ws / ".claude" / "settings.json", {"additionalDirectories": [str(extra), "~/proj"]})
        policy = load(ws, home, managed)
        assert allowed(policy, extra / "a")
        assert allowed(policy, home / "proj" / "a", "write")

    def test_lackpy_config_paths_table(self, ws, home, managed, tmp_path):
        extra = tmp_path / "extra"
        (ws / ".lackpy" / "config.toml").write_text(
            '[paths]\n'
            'deny = ["Read(.env)"]\n'
            f'allow = ["Read(/{tmp_path}/shared/**)"]\n'
            f'additional_directories = ["{extra}"]\n'
        )
        policy = load(ws, home, managed)
        assert not allowed(policy, ws / ".env")
        assert allowed(policy, tmp_path / "shared" / "a")
        assert allowed(policy, extra / "a", "write")

    def test_lackpy_config_without_paths_table(self, ws, home, managed):
        (ws / ".lackpy" / "config.toml").write_text('[inference]\norder = ["templates"]\n')
        assert allowed(load(ws, home, managed), ws / "a")

    def test_managed_none_skips_managed(self, ws, home):
        assert allowed(load_policy(ws, home=home, managed=None), ws / "a")


class TestManaged:
    def test_managed_deny(self, ws, home, managed):
        write_settings(managed, {"deny": ["Read(.env)"]})
        assert not allowed(load(ws, home, managed), ws / ".env")

    def test_dropin_directory_is_merged_in_order(self, ws, home, managed):
        dropins = managed.parent / "managed-settings.d"
        write_settings(dropins / "20-b.json", {"deny": ["Read(./b)"]})
        write_settings(dropins / "10-a.json", {"deny": ["Read(./a)", "Read(./b)"]})
        write_settings(dropins / ".hidden.json", {"deny": ["Read(./hidden)"]})
        (dropins / "notes.txt").write_text("{not json")
        policy = load(ws, home, managed)
        assert not allowed(policy, ws / "a")
        assert not allowed(policy, ws / "b")
        assert allowed(policy, ws / "hidden")
        # Duplicates are removed: one line for Read(./b).
        assert sum("Read(./b)" in line for line in policy.describe()) == 1

    def test_dropins_without_base_file(self, ws, home, managed):
        write_settings(managed.parent / "managed-settings.d" / "a.json", {"deny": ["Read(./a)"]})
        assert not allowed(load(ws, home, managed), ws / "a")

    def test_managed_only_ignores_other_rules(self, ws, home, managed, tmp_path):
        extra = tmp_path / "extra"
        write_settings(managed, {"deny": ["Read(./m)"]}, allowManagedPermissionRulesOnly=True)
        write_settings(ws / ".claude" / "settings.json",
                       {"allow": [f"Read(/{tmp_path}/shared/**)"], "deny": ["Read(./p)"],
                        "additionalDirectories": [str(extra)]})
        (ws / ".lackpy" / "config.toml").write_text('[paths]\ndeny = ["Read(./q)"]\n')
        policy = load(ws, home, managed)
        assert not allowed(policy, ws / "m")
        assert allowed(policy, ws / "p")
        assert allowed(policy, ws / "q")
        assert not allowed(policy, tmp_path / "shared" / "a")
        # Design call (Claude): additionalDirectories still define default-readable dirs.
        assert allowed(policy, extra / "a")

    def test_managed_only_set_in_a_dropin(self, ws, home, managed):
        write_settings(managed.parent / "managed-settings.d" / "a.json", {},
                       allowManagedPermissionRulesOnly=True)
        write_settings(ws / ".claude" / "settings.json", {"deny": ["Read(./p)"]})
        assert allowed(load(ws, home, managed), ws / "p")


class TestMalformed:
    @pytest.mark.parametrize("relative", [
        ".claude/settings.json", ".claude/settings.local.json",
    ])
    def test_bad_json_names_the_file(self, ws, home, managed, relative):
        (ws / relative).write_text("{nope")
        with pytest.raises(ValueError, match="settings"):
            load(ws, home, managed)

    def test_bad_user_json(self, ws, home, managed):
        (home / ".claude" / "settings.json").write_text("[1,")
        with pytest.raises(ValueError, match=str(home)):
            load(ws, home, managed)

    def test_bad_managed_json(self, ws, home, managed):
        managed.write_text("{")
        with pytest.raises(ValueError, match="managed-settings.json"):
            load(ws, home, managed)

    def test_bad_dropin_json(self, ws, home, managed):
        d = managed.parent / "managed-settings.d"
        d.mkdir()
        (d / "x.json").write_text("{")
        with pytest.raises(ValueError, match="x.json"):
            load(ws, home, managed)

    @pytest.mark.parametrize("permissions", [
        {"deny": "Read(.env)"}, {"deny": [1]}, {"additionalDirectories": "x"},
    ])
    def test_bad_shapes(self, ws, home, managed, permissions):
        write_settings(ws / ".claude" / "settings.json", permissions)
        with pytest.raises(ValueError, match="settings.json"):
            load(ws, home, managed)

    def test_permissions_not_an_object(self, ws, home, managed):
        (ws / ".claude" / "settings.json").write_text('{"permissions": []}')
        with pytest.raises(ValueError, match="settings.json"):
            load(ws, home, managed)

    def test_bad_toml(self, ws, home, managed):
        (ws / ".lackpy" / "config.toml").write_text("[paths\n")
        with pytest.raises(ValueError, match="config.toml"):
            load(ws, home, managed)

    @pytest.mark.parametrize("body", ['deny = "Read(.env)"', 'denny = ["Read(.env)"]', "paths = 1"])
    def test_bad_paths_table(self, ws, home, managed, body):
        text = f"[paths]\n{body}\n" if body != "paths = 1" else body + "\n"
        (ws / ".lackpy" / "config.toml").write_text(text)
        with pytest.raises(ValueError, match="config.toml"):
            load(ws, home, managed)

    def test_malformed_rule_names_the_file(self, ws, home, managed):
        write_settings(ws / ".claude" / "settings.json", {"deny": ["Read(.env"]})
        with pytest.raises(ValueError, match="settings.json"):
            load(ws, home, managed)
