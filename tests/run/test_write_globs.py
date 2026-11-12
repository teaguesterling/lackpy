"""Declared writes on a ``PathPolicy`` (spec E7, Task 7).

Under ``stepy``, a write must match a declared writes glob, on top of the
settings policy and never instead of it. The globs use the ``PathRule``
pattern semantics, relative to the workspace. ``None`` is today's behaviour;
``()`` allows no write. These tests build each policy directly: no test reads
the real ``~/.claude``.
"""

from pathlib import Path

import pytest

from lackpy.run.paths import PathPolicy, make_path_class, parse_rule, read_only_policy
from lackpy.run.paths.rules import with_write_globs
from lackpy.run.refusal import LackpyRefusal


@pytest.fixture
def ws(tmp_path):
    w = tmp_path / "ws"
    w.mkdir()
    return w


def _policy(ws, globs, rules=(), extra_dirs=()):
    return with_write_globs(PathPolicy(ws, list(rules), list(extra_dirs)), globs)


def _refused(policy, path, match=None):
    with pytest.raises(LackpyRefusal, match=match) as info:
        policy.check(path, "write")
    return info.value.what


# -- the gate ------------------------------------------------------------------

def test_a_write_that_matches_a_glob_is_allowed(ws):
    p = _policy(ws, ["out/*.txt"])
    p.check(ws / "out" / "a.txt", "write")
    p.check(Path("out/b.txt"), "write")  # relative to the workspace


def test_a_write_that_matches_no_glob_is_refused_and_names_the_globs(ws):
    p = _policy(ws, ["out/*.txt", "report.md"])
    what = _refused(p, ws / "notes.txt")
    assert str(ws / "notes.txt") in what
    assert "out/*.txt" in what and "report.md" in what
    _refused(p, ws / "out" / "a.json")  # wrong suffix
    _refused(p, ws / "out" / "sub" / "a.txt")  # * stays in one segment


def test_no_globs_refuses_every_write(ws):
    p = _policy(ws, [])
    for path in (ws / "a.txt", ws / "out" / "a.txt", ws):
        what = _refused(p, path)
        assert "no declared writes" in what


def test_reads_are_not_narrowed(ws):
    (ws / "a.txt").write_text("a")
    _policy(ws, []).check(ws / "a.txt", "read")


def test_none_keeps_todays_behaviour(ws):
    p = PathPolicy(ws, [])
    assert p.write_globs is None
    p.check(ws / "anything.txt", "write")
    p.check(ws / "deep" / "x.bin", "write")


def test_with_write_globs_does_not_change_the_given_policy(ws):
    base = PathPolicy(ws, [])
    with_write_globs(base, [])
    base.check(ws / "a.txt", "write")


def test_the_settings_deny_wins_over_a_matching_glob(ws):
    deny = parse_rule("Edit(out/secret.txt)", effect="deny", anchor=ws, source="settings.json")
    p = _policy(ws, ["out/*.txt"], rules=[deny])
    _refused(p, ws / "out" / "secret.txt", match=r"Edit\(out/secret.txt\)")
    p.check(ws / "out" / "ok.txt", "write")


def test_a_workspace_glob_does_not_cover_an_extra_directory(ws, tmp_path):
    # ``**`` is relative to the workspace: an extra directory needs its own glob.
    outside = tmp_path / "outside"
    outside.mkdir()
    _refused(_policy(ws, ["**"], extra_dirs=[outside]), outside / "x.txt", match="declared writes")


def test_a_glob_is_not_an_allow_rule(ws, tmp_path):
    # The globs are a gate on top of the settings policy: they never act as
    # an Edit allow, so a path the settings policy refuses stays refused.
    p = _policy(ws, ["out/*.txt"])
    assert p.rules == []
    _refused(p, tmp_path / "out" / "a.txt", match="outside the workspace")


def test_the_protected_locations_still_win_over_a_glob_that_matches_them(ws):
    p = _policy(ws, ["**"])
    _refused(p, ws / ".claude" / "settings.json", match="permission settings location")
    _refused(p, ws / ".Claude" / "settings.json", match="permission settings location")  # case-folded
    _refused(p, ws / ".git" / "config", match="host code")
    p.check(ws / "deep" / "x.txt", "write")  # ** is every other path in the workspace


def test_a_glob_with_no_slash_is_a_path_at_the_workspace_root(ws):
    # Teague, 2026-11-11: "Anchor at workspace". Not the PathRule bare-filename rule.
    p = _policy(ws, ["report.txt"])
    p.check(ws / "report.txt", "write")
    _refused(p, ws / "deep" / "report.txt", match="declared writes")
    _refused(p, ws / "other.txt")


def test_a_glob_with_no_slash_and_a_wildcard_is_anchored_too(ws):
    # A filled-in ``{name}.txt`` is such a glob.
    p = _policy(ws, ["*.txt"])
    p.check(ws / "r.txt", "write")
    _refused(p, ws / "deep" / "r.txt", match="declared writes")


def test_a_glob_with_only_a_trailing_slash_is_anchored_too(ws):
    # ``out/`` has no slash before its name: it must not float to any depth.
    p = _policy(ws, ["out/"])
    p.check(ws / "out", "write")
    _refused(p, ws / "deep" / "out", match="declared writes")


def test_a_double_star_glob_matches_at_any_depth(ws):
    p = _policy(ws, ["**/report.txt"])
    p.check(ws / "report.txt", "write")
    p.check(ws / "deep" / "report.txt", "write")
    p.check(ws / "a" / "b" / "report.txt", "write")


def test_mkdir_is_refused_under_a_glob_with_no_slash(ws):
    P = make_path_class(_policy(ws, ["report.txt"]))
    with pytest.raises(LackpyRefusal, match="declared writes"):
        P("deep").mkdir()
    assert not (ws / "deep").exists()


def test_a_dot_slash_glob_is_anchored_at_the_workspace(ws):
    p = _policy(ws, ["./report.txt"])
    p.check(ws / "report.txt", "write")
    _refused(p, ws / "sub" / "report.txt")


def test_matching_is_case_sensitive(ws):
    _refused(_policy(ws, ["out/*.txt"]), ws / "OUT" / "a.txt")


# -- escapes through the written path ---------------------------------------------

def test_dot_dot_in_the_written_path_is_normalised_first(ws):
    p = _policy(ws, ["out/*.txt"])
    _refused(p, ws / "out" / ".." / "x.txt")
    p.check(ws / "tmp" / ".." / "out" / "x.txt", "write")  # normalises into out/


def test_a_symlink_into_a_declared_directory_is_refused(ws):
    (ws / "out").mkdir()
    (ws / "elsewhere").symlink_to(ws / "out")
    p = _policy(ws, ["out/*.txt"])
    # The lexical form matches no glob: the allow must hold for both forms.
    _refused(p, ws / "elsewhere" / "a.txt", match="declared writes")


def test_a_declared_directory_that_is_a_symlink_out_of_the_globs_is_refused(ws):
    (ws / "real").mkdir()
    (ws / "out").symlink_to(ws / "real")
    p = _policy(ws, ["out/*.txt"])
    # The lexical form matches; the resolved target ws/real/a.txt does not.
    _refused(p, ws / "out" / "a.txt", match="declared writes")


def test_a_declared_directory_that_is_a_symlink_out_of_the_workspace_is_refused(ws, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (ws / "out").symlink_to(outside)
    _refused(_policy(ws, ["out/*.txt"]), ws / "out" / "a.txt")
    assert not (outside / "a.txt").exists()


def test_a_workspace_reached_through_a_symlink_still_matches(tmp_path):
    real = tmp_path / "real_ws"
    real.mkdir()
    link = tmp_path / "link_ws"
    link.symlink_to(real)
    p = _policy(link, ["out/*.txt"])
    p.check(link / "out" / "a.txt", "write")
    p.check(real / "out" / "a.txt", "write")
    _refused(p, link / "b.txt")


# -- escapes through the glob (with_write_globs) ------------------------------------

@pytest.mark.parametrize("glob", [
    "../x.txt", "out/../../x.txt", "../**", "//etc/*.conf", "//**", "~/notes.txt",
    "./a/../../b.txt",
])
def test_a_glob_that_leaves_the_workspace_is_refused(ws, glob):
    with pytest.raises(LackpyRefusal, match="declared writes glob") as info:
        _policy(ws, [glob])
    assert glob in info.value.what


def test_a_glob_inside_an_extra_directory_is_accepted(ws, tmp_path):
    extra = tmp_path / "extra"
    extra.mkdir()
    p = _policy(ws, [f"//{extra.as_posix().lstrip('/')}/*.txt"], extra_dirs=[extra])
    p.check(extra / "a.txt", "write")
    _refused(p, ws / "a.txt")


def test_a_leading_slash_glob_is_relative_to_the_workspace(ws):
    # PathRule: /p is relative to the rule's anchor, which is the workspace here.
    p = _policy(ws, ["/out/*.txt"])
    p.check(ws / "out" / "a.txt", "write")


def test_a_dot_dot_that_stays_in_the_workspace_is_accepted(ws):
    p = _policy(ws, ["a/../out/*.txt"])
    p.check(ws / "out" / "a.txt", "write")


def test_an_empty_glob_is_refused(ws):
    with pytest.raises(LackpyRefusal, match="declared writes glob"):
        _policy(ws, [""])


def test_a_second_narrowing_is_refused(ws):
    with pytest.raises(ValueError, match="already"):
        with_write_globs(_policy(ws, ["a/*"]), ["b/*"])


# -- the round trip and the other layers ----------------------------------------------

@pytest.mark.parametrize("globs", [None, (), ("out/*.txt", "report.md")])
def test_to_data_from_data_round_trips_write_globs(ws, globs):
    base = PathPolicy(ws, [])
    p = base if globs is None else with_write_globs(base, globs)
    back = PathPolicy.from_data(p.to_data())
    assert back.write_globs == p.write_globs
    assert back.write_globs is None if globs is None else back.write_globs == tuple(globs)


def test_from_data_keeps_no_writes_as_no_writes(ws):
    back = PathPolicy.from_data(_policy(ws, []).to_data())
    _refused(back, ws / "a.txt", match="no declared writes")


def test_from_data_checks_the_globs_again(ws):
    data = PathPolicy(ws, []).to_data()
    data["write_globs"] = ["../x.txt"]
    with pytest.raises(LackpyRefusal, match="declared writes glob"):
        PathPolicy.from_data(data)


def test_from_data_without_the_key_fails_loud(ws):
    # A producer that drops the key must not turn "no writes" into "any write".
    data = with_write_globs(PathPolicy(ws, []), []).to_data()
    del data["write_globs"]
    with pytest.raises(KeyError, match="write_globs"):
        PathPolicy.from_data(data)


def test_read_only_keeps_the_globs(ws):
    ro = read_only_policy(_policy(ws, ["out/*.txt"]))
    assert ro.write_globs == ("out/*.txt",)


def test_describe_lists_the_declared_writes(ws):
    lines = _policy(ws, ["out/*.txt"]).describe()
    assert any("out/*.txt" in line and "declared writes" in line for line in lines)


# -- the Path stand-in -------------------------------------------------------------

def test_mkdir_with_parents_is_allowed_for_the_fixed_prefix_of_a_glob(ws):
    P = make_path_class(_policy(ws, ["out/sub/*.txt"]))
    P("out/sub").mkdir(parents=True)
    assert (ws / "out" / "sub").is_dir()
    P("out/sub/a.txt").write_text("a")
    assert (ws / "out" / "sub" / "a.txt").read_text() == "a"


def test_mkdir_out_for_out_star_txt(ws):
    P = make_path_class(_policy(ws, ["out/*.txt"]))
    P("out").mkdir(parents=True)
    assert (ws / "out").is_dir()


def test_mkdir_outside_every_prefix_is_refused(ws):
    P = make_path_class(_policy(ws, ["out/*.txt"]))
    with pytest.raises(LackpyRefusal, match="declared writes"):
        P("other").mkdir()
    with pytest.raises(LackpyRefusal, match="declared writes"):
        P("out/deeper").mkdir(parents=True)
    assert not (ws / "other").exists() and not (ws / "out").exists()


def test_mkdir_of_a_literal_glob_path_is_refused(ws):
    # A glob with no wildcard declares a file: its parent may be made, not the file's own path.
    P = make_path_class(_policy(ws, ["out/r.txt"]))
    P("out").mkdir()
    with pytest.raises(LackpyRefusal, match="declared writes"):
        P("out/r.txt").mkdir()


def test_mkdir_below_a_wildcard_is_allowed_when_a_glob_can_match_under_it(ws):
    # Spec E7: an ancestor of a path some declared glob can match.
    P = make_path_class(_policy(ws, ["out/*/x.txt", "deep/**"]))
    P("out/a").mkdir(parents=True)
    P("deep/a/b").mkdir(parents=True)
    assert (ws / "out" / "a").is_dir() and (ws / "deep" / "a" / "b").is_dir()
    with pytest.raises(LackpyRefusal, match="declared writes"):
        P("out/a/b").mkdir()  # out/a/b/... can match no glob


def test_mkdir_with_no_globs_is_refused(ws):
    P = make_path_class(_policy(ws, []))
    with pytest.raises(LackpyRefusal, match="no declared writes"):
        P("out").mkdir()
    assert not (ws / "out").exists()


def test_a_refused_write_leaves_no_file(ws):
    P = make_path_class(_policy(ws, ["out/*.txt"]))
    with pytest.raises(LackpyRefusal):
        P("x.txt").write_text("x")
    assert not (ws / "x.txt").exists()


@pytest.mark.parametrize("member", ["rename", "replace", "unlink", "touch"])
def test_the_stand_in_has_no_rename_or_other_write_member(ws, member):
    # Spec E7 says a rename checks both ends. The stand-in has no rename or
    # replace (PATH_MEMBERS), so a rename into a declared glob is refused at
    # the attribute, before any policy check.
    (ws / "src.txt").write_text("s")
    P = make_path_class(_policy(ws, ["out/*.txt"]))
    with pytest.raises(LackpyRefusal):
        getattr(P("src.txt"), member)
    assert (ws / "src.txt").exists()
