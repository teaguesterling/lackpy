"""PathRule parsing and PathPolicy.check (stepy100 Plan 2, Task 4).

The rule semantics come from https://code.claude.com/docs/en/permissions
(fetched 2026-10-18). Each ``TestSemanticsN`` class is one item of the brief.
Every test builds its own workspace and home under ``tmp_path``.
"""

from pathlib import Path

import pytest

from lackpy.run.paths import PathPolicy, PathRule, parse_rule
from lackpy.run.refusal import LackpyRefusal


@pytest.fixture
def ws(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    return workspace


@pytest.fixture
def home(tmp_path):
    h = tmp_path / "home"
    (h / ".claude").mkdir(parents=True)
    return h


def rule(text, effect="deny", *, anchor, home=None, source="test"):
    parsed = parse_rule(text, effect=effect, anchor=anchor, source=source, home=home)
    assert parsed is not None, text
    return parsed


def allowed(policy, path, op):
    try:
        policy.check(Path(path), op)
    except LackpyRefusal:
        return False
    return True


class TestParseRule:
    def test_read_and_edit_parse(self, ws):
        r = parse_rule("Read(./a/**)", effect="deny", anchor=ws, source="s")
        assert r == PathRule(kind="read", effect="deny", pattern="./a/**", anchor=ws, source="s",
                             home=r.home)
        assert parse_rule("Edit(b)", effect="allow", anchor=ws, source="s").kind == "edit"

    @pytest.mark.parametrize("text", ["Bash(ls)", "WebFetch(domain:x)", "Write(./a)", "mcp__x__y"])
    def test_other_tools_give_none(self, ws, text):
        # Write(...) is accepted by Claude Code but never consulted; mirror that.
        assert parse_rule(text, effect="deny", anchor=ws, source="s") is None

    def test_ask_is_parsed_as_deny(self, ws):
        assert parse_rule("Read(a)", effect="ask", anchor=ws, source="s").effect == "deny"

    @pytest.mark.parametrize("text", ["Read(", "Read()", "Edit(a"])
    def test_malformed_read_or_edit_raises(self, ws, text):
        # A deny that silently failed to parse would loosen the policy.
        with pytest.raises(ValueError):
            parse_rule(text, effect="deny", anchor=ws, source="s")

    def test_bare_read_deny_matches_everything(self, ws):
        policy = PathPolicy(ws, [rule("Read", anchor=ws)])
        assert not allowed(policy, ws / "a", "read")

    def test_bare_read_allow_is_ignored(self, ws):
        # Stricter reading: a bare allow would open the whole filesystem.
        assert parse_rule("Read", effect="allow", anchor=ws, source="s") is None


class TestSemantics1Order:
    def test_deny_beats_allow(self, ws, tmp_path):
        out = tmp_path / "data"
        out.mkdir()
        rules = [rule("Read(//" + str(out).lstrip("/") + "/**)", "allow", anchor=ws),
                 rule("Read(//" + str(out).lstrip("/") + "/secret)", "deny", anchor=ws)]
        policy = PathPolicy(ws, rules)
        assert allowed(policy, out / "x", "read")
        assert not allowed(policy, out / "secret", "read")

    def test_deny_from_any_source_wins(self, ws, home):
        rules = [rule("Read(./a)", "allow", anchor=ws, source="project"),
                 rule("Read(./a)", "deny", anchor=home / ".claude", source="user")]
        assert not allowed(PathPolicy(ws, rules), ws / "a", "read")

    def test_read_deny_blocks_reads_and_writes(self, ws):
        policy = PathPolicy(ws, [rule("Read(./a)", anchor=ws)])
        assert not allowed(policy, ws / "a", "read")
        assert not allowed(policy, ws / "a", "write")

    def test_edit_deny_blocks_writes_only(self, ws):
        policy = PathPolicy(ws, [rule("Edit(./a)", anchor=ws)])
        assert allowed(policy, ws / "a", "read")
        assert not allowed(policy, ws / "a", "write")


class TestSemantics2Anchors:
    def test_double_slash_is_absolute(self, ws, tmp_path):
        target = tmp_path / "abs.txt"
        policy = PathPolicy(ws, [rule(f"Read(/{target})", "allow", anchor=ws)])
        assert allowed(policy, target, "read")
        assert not allowed(policy, tmp_path / "other.txt", "read")

    def test_tilde_is_under_home(self, ws, home):
        policy = PathPolicy(ws, [rule("Read(~/notes/**)", "allow", anchor=ws, home=home)])
        assert allowed(policy, home / "notes" / "a", "read")
        assert not allowed(policy, home / "other", "read")

    def test_single_slash_is_relative_to_the_anchor(self, ws, home):
        anchor = home / ".claude"
        policy = PathPolicy(ws, [rule("Read(/agents/**)", "allow", anchor=anchor)])
        assert allowed(policy, anchor / "agents" / "a.md", "read")
        assert not allowed(policy, Path("/agents/a.md"), "read")

    def test_single_slash_in_workspace_settings_is_the_workspace(self, ws):
        policy = PathPolicy(ws, [rule("Read(/a)", anchor=ws)])
        assert not allowed(policy, ws / "a", "read")

    @pytest.mark.parametrize("text", ["Read(sub/a)", "Read(./sub/a)"])
    def test_plain_and_dot_slash_are_workspace_relative(self, ws, home, text):
        # Even in ~/.claude/settings.json: the anchor is for /p only.
        policy = PathPolicy(ws, [rule(text, anchor=home / ".claude")])
        assert not allowed(policy, ws / "sub" / "a", "read")
        assert allowed(policy, ws / "other" / "a", "read")


class TestSemantics3Globs:
    def test_star_stays_within_one_segment(self, ws):
        policy = PathPolicy(ws, [rule("Read(./src/*.py)", anchor=ws)])
        assert not allowed(policy, ws / "src" / "a.py", "read")
        assert allowed(policy, ws / "src" / "sub" / "a.py", "read")

    def test_double_star_crosses_segments(self, ws):
        policy = PathPolicy(ws, [rule("Read(./src/**/*.py)", anchor=ws)])
        assert not allowed(policy, ws / "src" / "a.py", "read")
        assert not allowed(policy, ws / "src" / "x" / "y" / "a.py", "read")
        assert allowed(policy, ws / "src" / "x" / "a.txt", "read")

    def test_bare_filename_matches_at_any_depth(self, ws):
        policy = PathPolicy(ws, [rule("Read(.env)", anchor=ws)])
        assert not allowed(policy, ws / ".env", "read")
        assert not allowed(policy, ws / "a" / "b" / ".env", "read")
        assert allowed(policy, ws / "a" / ".env.example", "read")

    def test_directory_pattern_deny_matches_at_any_depth(self, ws):
        policy = PathPolicy(ws, [rule("Read(secrets/**)", anchor=ws)])
        assert not allowed(policy, ws / "secrets" / "a", "read")
        assert not allowed(policy, ws / "x" / "secrets" / "a", "read")

    def test_directory_pattern_allow_matches_only_the_workspace_dir(self, ws, tmp_path):
        other = tmp_path / "other"
        policy = PathPolicy(ws, [rule("Read(src/**)", "allow", anchor=ws)])
        # Outside the workspace, so only an allow rule could admit it.
        assert not allowed(policy, other / "src" / "a", "read")
        assert allowed(policy, ws / "src" / "a", "read")

    def test_matching_is_case_sensitive(self, ws):
        # Semantics item 6: upstream does not document case; lackpy is case-sensitive.
        policy = PathPolicy(ws, [rule("Read(.env)", anchor=ws)])
        assert not allowed(policy, ws / ".env", "read")
        assert allowed(policy, ws / ".ENV", "read")


class TestSemantics4Default:
    def test_workspace_is_readable_and_writable(self, ws):
        policy = PathPolicy(ws, [])
        assert allowed(policy, ws / "a" / "b", "read")
        assert allowed(policy, ws / "a" / "b", "write")
        assert allowed(policy, ws, "read")

    def test_extra_dirs_are_readable_and_writable(self, ws, tmp_path):
        extra = tmp_path / "extra"
        extra.mkdir()
        policy = PathPolicy(ws, [], extra_dirs=[extra])
        assert allowed(policy, extra / "a", "read")
        assert allowed(policy, extra / "a", "write")

    def test_everything_else_is_refused(self, ws, tmp_path):
        policy = PathPolicy(ws, [])
        assert not allowed(policy, tmp_path / "a", "read")
        assert not allowed(policy, Path("/etc/passwd"), "read")
        assert not allowed(policy, tmp_path / "ws2", "write")  # a sibling with the same prefix

    def test_relative_path_is_against_the_workspace(self, ws, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        policy = PathPolicy(ws, [])
        assert allowed(policy, Path("a"), "read")
        assert not allowed(policy, Path("../a"), "read")

    def test_dotdot_out_of_the_workspace_refuses(self, ws):
        policy = PathPolicy(ws, [])
        assert not allowed(policy, ws / ".." / "x", "read")
        assert not allowed(policy, ws / "a" / ".." / ".." / "x", "write")

    def test_refusal_names_the_path(self, ws):
        with pytest.raises(LackpyRefusal) as info:
            PathPolicy(ws, []).check(Path("/etc/passwd"), "read")
        assert "/etc/passwd" in str(info.value)


class TestSemantics5Symlinks:
    def test_allow_must_match_the_symlink_target(self, ws, tmp_path):
        outside = tmp_path / "outside.txt"
        outside.write_text("secret")
        (ws / "link").symlink_to(outside)
        policy = PathPolicy(ws, [])
        assert not allowed(policy, ws / "link", "read")

    def test_allow_must_match_the_requested_path(self, ws, tmp_path):
        inside = ws / "a.txt"
        inside.write_text("x")
        (tmp_path / "link").symlink_to(inside)
        assert not allowed(PathPolicy(ws, []), tmp_path / "link", "read")

    def test_deny_matches_the_target(self, ws):
        (ws / ".env").write_text("k")
        (ws / "innocent").symlink_to(ws / ".env")
        policy = PathPolicy(ws, [rule("Read(.env)", anchor=ws)])
        assert not allowed(policy, ws / "innocent", "read")

    def test_deny_matches_the_link_name(self, ws):
        (ws / "data").write_text("k")
        (ws / ".env").symlink_to(ws / "data")
        policy = PathPolicy(ws, [rule("Read(.env)", anchor=ws)])
        assert not allowed(policy, ws / ".env", "read")
        assert allowed(policy, ws / "data", "read")

    def test_write_refuses_a_final_symlink(self, ws):
        (ws / "real").write_text("x")
        (ws / "link").symlink_to(ws / "real")
        policy = PathPolicy(ws, [])
        assert allowed(policy, ws / "link", "read")
        assert not allowed(policy, ws / "link", "write")

    def test_write_refuses_an_intermediate_symlink_out(self, ws, tmp_path):
        outdir = tmp_path / "outdir"
        outdir.mkdir()
        (ws / "d").symlink_to(outdir)
        assert not allowed(PathPolicy(ws, []), ws / "d" / "new.txt", "write")

    def test_write_through_link_then_dotdot_refuses(self, ws, tmp_path):
        # ws/d/../x: lexically ws/x, but the OS resolves d first, then "..".
        outdir = tmp_path / "deep" / "outdir"
        outdir.mkdir(parents=True)
        (ws / "d").symlink_to(outdir)
        assert not allowed(PathPolicy(ws, []), ws / "d" / ".." / "x", "write")

    def test_intermediate_symlink_inside_the_workspace_is_allowed(self, ws):
        (ws / "real").mkdir()
        (ws / "d").symlink_to(ws / "real")
        assert allowed(PathPolicy(ws, []), ws / "d" / "new.txt", "write")


class TestDenyLiteralPrefixIsResolved:
    def test_symlinked_home_dir_cannot_bypass_its_deny(self, ws, home, tmp_path):
        # Review M1, measured case: ~/.ssh -> R/keys; R/keys/id read directly.
        keys = tmp_path / "R" / "keys"
        keys.mkdir(parents=True)
        (keys / "id").write_text("k")
        (home / ".ssh").symlink_to(keys)
        rules = [rule("Read(~/.ssh/**)", anchor=ws, home=home),
                 rule(f"Read(/{tmp_path}/R/**)", "allow", anchor=ws)]
        policy = PathPolicy(ws, rules)
        assert not allowed(policy, keys / "id", "read")
        assert not allowed(policy, home / ".ssh" / "id", "read")
        assert allowed(policy, tmp_path / "R" / "other", "read")

    def test_symlinked_prefix_under_a_default_root(self, ws, tmp_path):
        target = ws / "real-secrets"
        target.mkdir()
        (ws / "secrets").symlink_to(target)
        policy = PathPolicy(ws, [rule("Read(./secrets/*.key)", anchor=ws)])
        assert not allowed(policy, target / "a.key", "read")
        assert allowed(policy, target / "a.txt", "read")


class TestDescribe:
    def test_one_line_per_rule_and_default(self, ws, tmp_path):
        extra = tmp_path / "extra"
        rules = [rule("Read(.env)", anchor=ws, source="a.json"),
                 rule("Edit(./out/**)", anchor=ws, source="b.json")]
        lines = PathPolicy(ws, rules, extra_dirs=[extra]).describe()
        never = [line for line in lines if line.startswith("never write:")]
        assert any(str(ws / ".claude") in line for line in never)
        assert len(lines) - len(never) == 4
        assert any("Read(.env)" in line and "a.json" in line for line in lines)
        assert any("Edit(./out/**)" in line and "b.json" in line for line in lines)
        assert any(str(ws) in line for line in lines)
        assert any(str(extra) in line for line in lines)
