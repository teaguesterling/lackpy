"""The ``Path`` stand-in and its wiring into the runtime (stepy100 Plan 2, Task 4).

The stand-in is a validated program's only route to the filesystem. Review
Focus 1 (escapes through ``..``, a symlink, or an absolute path, built at run
time) and Review Focus 2 (no real ``pathlib.Path`` in a return value) are here.
"""

import json
import pathlib

import pytest

from lackpy.lang import dictionary
from lackpy.run.modules import ModuleProxy, make_restricted_import
from lackpy.run.namespace import build_namespace
from lackpy.run.paths import PathPolicy, load_policy, make_path_class
from lackpy.run.refusal import LackpyRefusal
from lackpy.run.runner import RestrictedRunner


@pytest.fixture
def ws(tmp_path, monkeypatch):
    w = tmp_path / "ws"
    w.mkdir()
    # The process cwd is elsewhere: relative paths must use the workspace.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    return w


@pytest.fixture
def P(ws):
    return make_path_class(PathPolicy(ws, []))


def run(program, policy, params=None):
    return RestrictedRunner().run(program, params=params, path_policy=policy)


class TestShape:
    def test_class_takes_the_platform_class_name(self, P):
        # Ruling I2: type(p).__name__ and repr(type(p)) match plain Python.
        real = type(pathlib.Path())
        assert P.__name__ == real.__name__
        assert P.__qualname__ == real.__qualname__
        assert repr(P) == repr(real)

    def test_type_name_in_a_program(self, ws):
        result = run("from pathlib import Path\ntype(Path('a')).__name__", PathPolicy(ws, []))
        assert result.success, result.error
        assert result.output == type(pathlib.Path("a")).__name__

    def test_members_are_exactly_the_dictionary(self, P):
        public = {n for n in vars(P) if not n.startswith("_")}
        assert public == set(dictionary.PATH_MEMBERS)

    def test_str_is_as_built(self, P):
        assert str(P("src") / "a.cpp") == "src/a.cpp"
        assert str(P("src", "b")) == "src/b"
        assert str(P()) == "."

    def test_fspath_and_repr_match_pathlib(self, P):
        import os
        p = P("src") / "a.cpp"
        assert os.fspath(p) == "src/a.cpp"
        assert repr(p) == repr(pathlib.Path("src/a.cpp"))

    def test_pure_properties(self, P):
        p = P("src/a.tar.gz")
        assert (p.name, p.stem, p.suffix, p.parts) == ("a.tar.gz", "a.tar", ".gz", ("src", "a.tar.gz"))
        assert all(type(v) in (str, tuple) for v in (p.name, p.stem, p.suffix, p.parts))

    @pytest.mark.parametrize("make", [
        lambda P: P("x").parent,
        lambda P: P("x") / "y",
        lambda P: P("x").with_suffix(".py"),
        lambda P: P("a/b").relative_to("a"),
        lambda P: P("a/b").relative_to(P("a")),
        lambda P: P("a").joinpath("b", P("c")),
    ])
    def test_path_results_are_stand_ins(self, P, make):
        # Review Focus 2: never a real pathlib.Path.
        result = make(P)
        assert isinstance(result, P)
        assert not isinstance(result, pathlib.PurePath)

    def test_values(self, P):
        assert str(P("a/b").relative_to("a")) == "b"
        assert str(P("a").joinpath("b", P("c"))) == "a/b/c"
        assert str(P("x.txt").with_suffix(".py")) == "x.py"
        assert str(P("x").parent) == "."

    def test_eq_hash_lt(self, P):
        assert P("a") == P("a")
        assert P("a") == P("./a")
        assert P("a") != P("b")
        assert P("a") != "a"  # plain pathlib: Path == str is False
        assert hash(P("a")) == hash(P("a"))
        assert sorted([P("b"), P("a")]) == [P("a"), P("b")]
        assert len({P("a"), P("a"), P("b")}) == 2

    def test_rtruediv(self, P):
        # Ruling I3.
        p = "a" / P("b")
        assert isinstance(p, P)
        assert str(p) == "a/b"
        with pytest.raises(TypeError):
            1 / P("b")

    @pytest.mark.parametrize("a, b", [("a", "b"), ("b", "a"), ("a", "a")])
    def test_full_ordering_matches_pathlib(self, P, a, b):
        ra, rb = pathlib.Path(a), pathlib.Path(b)
        assert (P(a) < P(b), P(a) <= P(b), P(a) > P(b), P(a) >= P(b)) == (ra < rb, ra <= rb, ra > rb, ra >= rb)

    def test_ordering_against_str_is_type_error(self, P):
        for op in ("<", "<=", ">", ">="):
            with pytest.raises(TypeError):
                eval(f"P('a') {op} 'a'", {"P": P})  # noqa: S307

    def test_eq_with_a_real_path_follows_pathlib(self, P):
        assert P("a") == pathlib.Path("a")
        assert P("a") != pathlib.Path("b")


class TestRefusals:
    @pytest.mark.parametrize("attr", ["unlink", "open", "parser", "resolve", "absolute",
                                      "chmod", "rename", "touch", "with_name", "_raw"])
    def test_other_instance_attributes_refuse(self, P, attr):
        with pytest.raises(LackpyRefusal):
            getattr(P("x"), attr)

    def test_open_refusal_carries_the_hint(self, P):
        with pytest.raises(LackpyRefusal) as info:
            P("x").open()
        assert info.value.hint == dictionary.UNAPPROVED["open"]

    @pytest.mark.parametrize("attr", ["home", "cwd", "read_text", "_flavour"])
    def test_class_attributes_refuse(self, P, attr):
        with pytest.raises(LackpyRefusal):
            getattr(P, attr)

    def test_home_refuses(self, P):
        with pytest.raises(LackpyRefusal):
            P.home()

    def test_setattr_and_delattr_refuse(self, P):
        p = P("x")
        with pytest.raises(LackpyRefusal):
            p.name = "y"
        with pytest.raises(LackpyRefusal):
            p.anything = 1
        with pytest.raises(LackpyRefusal):
            del p.name
        with pytest.raises(LackpyRefusal):
            P.read_text = len

    def test_no_instance_state(self, P):
        # As for ModuleProxy (Task 3 review M1): no attribute leads to the real path.
        p = P("x")
        for slot in P.__slots__:
            if slot != "__weakref__":
                assert not hasattr(p, slot)

    def test_bad_argument_is_type_error(self, P):
        with pytest.raises(TypeError):
            P(1)


class TestIO:
    def test_read_write_relative_to_the_workspace(self, ws, P):
        P("a.txt").write_text("hi")
        assert (ws / "a.txt").read_text() == "hi"
        assert P("a.txt").read_text() == "hi"
        P("b.bin").write_bytes(b"\x00")
        assert P("b.bin").read_bytes() == b"\x00"
        assert P("a.txt").exists() is True
        assert P("a.txt").is_file() is True
        assert P(".").is_dir() is True
        assert P("nope").exists() is False

    def test_mkdir(self, ws, P):
        P("d/e/f").mkdir(parents=True)
        assert (ws / "d" / "e" / "f").is_dir()
        P("d").mkdir(exist_ok=True)
        with pytest.raises(FileExistsError):
            P("d").mkdir()

    def test_glob_rglob_iterdir_return_as_built_stand_ins(self, ws, P):
        (ws / "src" / "sub").mkdir(parents=True)
        (ws / "src" / "a.py").write_text("")
        (ws / "src" / "sub" / "b.py").write_text("")
        got = sorted(P("src").glob("*.py"))
        assert [str(p) for p in got] == ["src/a.py"]
        assert all(isinstance(p, P) for p in got)
        assert sorted(str(p) for p in P("src").rglob("*.py")) == ["src/a.py", "src/sub/b.py"]
        assert sorted(str(p) for p in P(".").glob("src/**/*.py")) == ["src/a.py", "src/sub/b.py"]
        assert sorted(str(p) for p in P("src").iterdir()) == ["src/a.py", "src/sub"]
        assert all(isinstance(p, P) for p in P("src").iterdir())

    def test_glob_omits_denied_entries(self, ws, tmp_path):
        (ws / ".env").write_text("k")
        (ws / "a.txt").write_text("")
        (ws / "out").symlink_to(tmp_path / "elsewhere")
        from lackpy.run.paths import parse_rule
        deny = parse_rule("Read(.env)", effect="deny", anchor=ws, source="t")
        P = make_path_class(PathPolicy(ws, [deny]))
        assert sorted(str(p) for p in P(".").iterdir()) == ["a.txt"]
        assert sorted(str(p) for p in P(".").glob("*")) == ["a.txt"]
        assert sorted(str(p) for p in P(".").rglob("*")) == ["a.txt"]

    def test_dotdot_inside_a_name_is_a_plain_glob(self, ws, P):
        # Review M2: only a ".." part refuses; "a..txt" is an ordinary name.
        (ws / "a..txt").write_text("")
        assert [str(p) for p in P(".").glob("a..txt")] == ["a..txt"]
        assert [str(p) for p in P(".").glob("x..y")] == []

    @pytest.mark.parametrize("pattern", ["/etc/*", "../*", "a/../../*", "..", "**/../*"])
    def test_glob_absolute_or_dotdot_pattern_refuses(self, P, pattern):
        with pytest.raises(LackpyRefusal):
            P(".").glob(pattern)
        with pytest.raises(LackpyRefusal):
            P(".").rglob(pattern)


def _plain_error(ws, monkeypatch, fn):
    """Run ``fn`` with plain pathlib from the workspace, and return the OSError."""
    monkeypatch.chdir(ws)
    try:
        fn(pathlib.Path)
    except OSError as e:
        return e
    pytest.fail("plain pathlib did not raise")


class TestOSErrorFilename:
    """Review I1: an OSError names the path as the program built it."""

    @pytest.mark.parametrize("call", [
        lambda P: P("nope.txt").read_text(),
        lambda P: P("nope.bin").read_bytes(),
        lambda P: P("missing/x.txt").write_text("x"),
        lambda P: P("missing/x.bin").write_bytes(b"x"),
        lambda P: P("d").mkdir(),
        lambda P: P("missing/e").mkdir(),
        lambda P: list(P("nodir").iterdir()),
        lambda P: P("f").mkdir(parents=True),
    ])
    def test_matches_plain_python(self, ws, P, monkeypatch, call):
        (ws / "d").mkdir()
        (ws / "f").write_text("")
        try:
            call(P)
        except OSError as e:
            ours = (type(e), e.filename, str(e))
        else:
            pytest.fail("the stand-in did not raise")
        plain = _plain_error(ws, monkeypatch, call)
        assert ours == (type(plain), plain.filename, str(plain))

    @pytest.mark.parametrize("method, call", [
        ("iterdir", lambda P: P("nodir").iterdir()),
        ("glob", lambda P: P("nodir").glob("*")),
        ("rglob", lambda P: P("nodir").rglob("*")),
    ])
    def test_eager_listing_error_is_rewritten(self, ws, P, monkeypatch, method, call):
        # Review N1: Python 3.13+ scandirs inside iterdir() itself, before
        # iteration. Make every version raise at call time to cover that.
        def eager(self, *args, **kwargs):
            raise FileNotFoundError(2, "No such file or directory", str(self))
        monkeypatch.setattr(pathlib.Path, method, eager)
        with pytest.raises(FileNotFoundError) as info:
            list(call(P))
        assert info.value.filename == "nodir"
        assert str(ws) not in str(info.value)

    def test_program_prints_the_built_path(self, ws):
        program = ("from pathlib import Path\n"
                   "try:\n    Path('nope.txt').read_text()\nexcept FileNotFoundError as e:\n    print(e)\n")
        result = run(program, PathPolicy(ws, []))
        assert result.success, result.error
        assert result.stdout == "[Errno 2] No such file or directory: 'nope.txt'\n"
        assert str(ws) not in result.stdout


class TestPolicySourcesAreNotWritable:
    """Security ruling: a program must not rewrite the policy for the next run."""

    @pytest.fixture
    def home(self, tmp_path):
        h = tmp_path / "home"
        (h / ".claude").mkdir(parents=True)
        return h

    @pytest.fixture
    def managed(self, tmp_path):
        d = tmp_path / "etc" / "claude-code"
        (d / "managed-settings.d").mkdir(parents=True)
        return d / "managed-settings.json"

    @pytest.fixture
    def P(self, ws, home, managed, tmp_path):
        # A broad Edit allow and Read allow must not lift the refusal.
        (ws / ".claude").mkdir()
        (ws / ".claude" / "settings.json").write_text(json.dumps({"permissions": {
            "allow": ["Edit(//**)", "Read(//**)", "Edit(./.claude/**)", "Edit(~/.claude/**)"],
            "additionalDirectories": [str(home), str(managed.parent)],
        }}))
        return make_path_class(load_policy(ws, home=home, managed=managed))

    def _targets(self, ws, home, managed):
        return [ws / ".claude" / "settings.json", ws / ".claude" / "settings.local.json",
                ws / ".lackpy" / "config.toml", home / ".claude" / "settings.json",
                managed, managed.parent / "managed-settings.d" / "z.json"]

    def test_direct_write_refuses(self, ws, home, managed, P):
        for target in self._targets(ws, home, managed):
            for write in (lambda p: p.write_text("{}"), lambda p: p.write_bytes(b"{}")):
                with pytest.raises(LackpyRefusal):
                    write(P(str(target)))
            assert not target.exists() or target == ws / ".claude" / "settings.json"
        # Relative to the workspace, too.
        with pytest.raises(LackpyRefusal):
            P(".claude/settings.local.json").write_text("{}")
        with pytest.raises(LackpyRefusal):
            P(".lackpy/config.toml").write_text("")

    def test_write_through_a_symlink_refuses(self, ws, home, managed, P):
        for i, target in enumerate(self._targets(ws, home, managed)):
            target.parent.mkdir(parents=True, exist_ok=True)
            (ws / f"dirlink{i}").symlink_to(target.parent)
            with pytest.raises(LackpyRefusal):
                P(f"dirlink{i}/{target.name}").write_text("{}")

    def test_mkdir_refuses(self, ws, home, managed, P):
        for target in [ws / ".lackpy" / "kits", home / ".claude" / "agents" / "x",
                       managed.parent / "managed-settings.d" / "sub",
                       ws / ".claude" / "commands" / "deep"]:
            with pytest.raises(LackpyRefusal):
                P(str(target)).mkdir(parents=True)
            assert not target.exists()
        with pytest.raises(LackpyRefusal):
            P(".lackpy").mkdir(parents=True)
        assert not (ws / ".lackpy").exists()

    def test_case_variants_refuse(self, ws, P):
        # A case-insensitive filesystem would reach the real file.
        with pytest.raises(LackpyRefusal):
            P(".Claude/settings.json").write_text("{}")

    def test_reads_are_unaffected(self, ws, P):
        assert P(".claude/settings.json").read_text().startswith("{")

    @pytest.mark.parametrize("relative", [
        "sub/.claude/settings.json", "sub/.lackpy/config.toml", "a/b/.CLAUDE/x", "deep/.LackPy/k",
    ])
    def test_nested_sources_refuse(self, ws, P, relative):
        # Ruling N2: any .claude/.lackpy component, so a later run rooted in sub/ is safe.
        with pytest.raises(LackpyRefusal):
            P(relative).write_text("{}")
        with pytest.raises(LackpyRefusal):
            P(relative).parent.mkdir(parents=True)
        assert not (ws / relative).exists()

    def test_extra_dir_sources_refuse(self, home, managed, tmp_path):
        ws2, extra = tmp_path / "ws2", tmp_path / "extra2"
        ws2.mkdir()
        extra.mkdir()
        P = make_path_class(PathPolicy(ws2, [], extra_dirs=[extra]))
        assert P(str(extra / "x")).write_text("ok") == 2  # the extra dir itself is writable
        with pytest.raises(LackpyRefusal):
            P(str(extra / ".claude" / "x")).write_text("{}")
        with pytest.raises(LackpyRefusal):
            P(str(extra / ".lackpy" / "config.toml")).write_text("")

    def test_home_claude_json_refuses(self, home, P):
        with pytest.raises(LackpyRefusal):
            P(str(home / ".claude.json")).write_text("{}")
        assert not (home / ".claude.json").exists()

    def test_symlink_into_a_nested_source_refuses(self, ws, P):
        (ws / "sub" / ".claude").mkdir(parents=True)
        (ws / "innocent").symlink_to(ws / "sub" / ".claude")
        with pytest.raises(LackpyRefusal):
            P("innocent/settings.json").write_text("{}")

    def test_nested_reads_still_work(self, ws, home, P):
        (ws / "sub" / ".claude").mkdir(parents=True)
        (ws / "sub" / ".claude" / "settings.json").write_text("{}")
        (home / ".claude.json").write_text("{}")
        assert P("sub/.claude/settings.json").read_text() == "{}"
        assert P(str(home / ".claude.json")).read_text() == "{}"

    def test_direct_policy_protects_the_workspace_sources(self, ws):
        P = make_path_class(PathPolicy(ws, []))
        with pytest.raises(LackpyRefusal):
            P(".claude/settings.json").write_text("{}")
        with pytest.raises(LackpyRefusal):
            P(".lackpy/config.toml").write_text("")


class TestReviewFocus1:
    """Escapes built at run time. Static checks cannot see them."""

    def test_dotdot(self, P):
        with pytest.raises(LackpyRefusal):
            P("../x").read_text()
        with pytest.raises(LackpyRefusal):
            P("../x").write_text("x")
        with pytest.raises(LackpyRefusal):
            P("../x").exists()

    def test_absolute(self, P):
        with pytest.raises(LackpyRefusal):
            P("/etc/passwd").read_text()

    def test_symlink_out(self, ws, tmp_path, P):
        secret = tmp_path / "secret.txt"
        secret.write_text("s")
        (ws / "link").symlink_to(secret)
        with pytest.raises(LackpyRefusal):
            P("link").read_text()
        with pytest.raises(LackpyRefusal):
            P("link").write_text("x")
        assert secret.read_text() == "s"

    def test_symlinked_dir_out(self, ws, tmp_path, P):
        outdir = tmp_path / "outdir"
        outdir.mkdir()
        (ws / "d").symlink_to(outdir)
        with pytest.raises(LackpyRefusal):
            P("d/new.txt").write_text("x")
        with pytest.raises(LackpyRefusal):
            P("d/sub").mkdir()
        assert list(outdir.iterdir()) == []

    def test_mkdir_parents_checks_every_created_dir(self, ws, tmp_path, P):
        with pytest.raises(LackpyRefusal):
            P("../made/here").mkdir(parents=True)
        assert not (tmp_path / "made").exists()

    def test_params_built_dotdot(self, ws):
        result = run("from pathlib import Path\nparams = {'root': 'sub'}\n"
                     "(Path(params['root']) / '..' / 'x').read_text()",
                     PathPolicy(ws, []), params={"root": "."})
        assert not result.success
        assert "not allowed in lackpy" in (result.error or ""), result.error

    @pytest.mark.parametrize("handler", ["OSError", "Exception", "ValueError"])
    def test_refusal_is_not_caught(self, ws, handler):
        program = ("from pathlib import Path\n"
                   f"try:\n    x = Path('/etc/passwd').read_text()\nexcept {handler}:\n    x = 'caught'\nx")
        result = run(program, PathPolicy(ws, []))
        assert not result.success
        assert result.output != "caught"


class TestSettingsDriven:
    def test_read_deny_in_claude_settings(self, ws, tmp_path):
        (ws / ".claude").mkdir()
        (ws / ".claude" / "settings.json").write_text(
            json.dumps({"permissions": {"deny": ["Read(./secrets/**)"]}}))
        (ws / "secrets").mkdir()
        (ws / "secrets" / "a").write_text("s")
        P = make_path_class(load_policy(ws, home=tmp_path / "home", managed=tmp_path / "none.json"))
        with pytest.raises(LackpyRefusal):
            P("secrets/a").read_text()
        with pytest.raises(LackpyRefusal):
            P("secrets/b").write_text("x")
        assert not (ws / "secrets" / "b").exists()

    def test_edit_deny_allows_reads_refuses_writes(self, ws, tmp_path):
        (ws / ".claude").mkdir()
        (ws / ".claude" / "settings.json").write_text(
            json.dumps({"permissions": {"deny": ["Edit(./out/**)"]}}))
        (ws / "out").mkdir()
        (ws / "out" / "a").write_text("r")
        P = make_path_class(load_policy(ws, home=tmp_path / "home", managed=tmp_path / "none.json"))
        assert P("out/a").read_text() == "r"
        with pytest.raises(LackpyRefusal):
            P("out/a").write_text("w")
        with pytest.raises(LackpyRefusal):
            P("out/sub").mkdir()
        assert (ws / "out" / "a").read_text() == "r"


class TestWiring:
    def test_no_policy_keeps_refusing(self):
        # Ruling C, fail closed: without a PathPolicy, pathlib refuses.
        with pytest.raises(LackpyRefusal) as info:
            make_restricted_import()("pathlib")
        assert info.value.hint == "Path arrives with the PathPolicy"

    def test_policy_serves_a_pathlib_proxy(self, ws):
        imp = make_restricted_import(path_policy=PathPolicy(ws, []))
        proxy = imp("pathlib")
        assert isinstance(proxy, ModuleProxy)
        assert proxy.Path is not pathlib.Path
        assert proxy.Path.__name__ == type(pathlib.Path()).__name__  # ruling I2
        assert imp("pathlib").Path is proxy.Path
        with pytest.raises(LackpyRefusal):
            proxy.PurePath

    def test_namespace_binds_it(self, ws):
        ns = build_namespace(program_file=None, path_policy=PathPolicy(ws, []), print_fn=print)
        exec("from pathlib import Path\nx = str(Path('a') / 'b')", ns)  # noqa: S102
        assert ns["x"] == "a/b"

    def test_program_reads_and_writes(self, ws):
        program = ("from pathlib import Path\n"
                   "Path('out.txt').write_text('hello')\n"
                   "import pathlib\n"
                   "pathlib.Path('out.txt').read_text()")
        result = run(program, PathPolicy(ws, []))
        assert result.success, result.error
        assert result.output == "hello"
        assert (ws / "out.txt").read_text() == "hello"

    def test_program_cannot_reach_real_path(self, ws):
        result = run("from pathlib import Path\nPath('x').parent", PathPolicy(ws, []))
        assert result.success, result.error
        assert not isinstance(result.output, pathlib.PurePath)


class TestHostCodeFilesAreNotWritable:
    """Ruling I5 (final review): a write to ``.git`` internals, ``.mcp.json`` or
    ``.envrc`` makes a later host command run code the program chose
    (``core.fsmonitor`` and hooks, MCP server commands, direnv). Writes refuse
    there; reads stay allowed."""

    @pytest.mark.parametrize("relative", [
        ".git/config", ".git/hooks/pre-commit", "sub/.git/config", "a/b/.git/hooks/post-checkout",
        ".mcp.json", "sub/.mcp.json", ".envrc", "deep/er/.envrc",
    ])
    def test_direct_write_refuses(self, ws, P, relative):
        for write in (lambda p: p.write_text("x"), lambda p: p.write_bytes(b"x")):
            with pytest.raises(LackpyRefusal):
                write(P(relative))
        assert not (ws / relative).exists()

    def test_existing_git_config_is_not_rewritten(self, ws, P):
        (ws / ".git").mkdir()
        (ws / ".git" / "config").write_text("[core]\n")
        with pytest.raises(LackpyRefusal):
            P(".git/config").write_text("[core]\n\tfsmonitor = touch PWNED\n")
        assert (ws / ".git" / "config").read_text() == "[core]\n"

    def test_write_through_a_symlink_refuses(self, ws, P):
        (ws / ".git" / "hooks").mkdir(parents=True)
        (ws / "innocent").symlink_to(ws / ".git")
        with pytest.raises(LackpyRefusal):
            P("innocent/config").write_text("x")
        with pytest.raises(LackpyRefusal):
            P("innocent/hooks/pre-commit").write_text("x")
        (ws / "sub").mkdir()
        (ws / "dirlink").symlink_to(ws / "sub")
        (ws / "sub" / ".mcp.json").write_text("{}")
        (ws / "mcp-link.json").symlink_to(ws / "sub" / ".mcp.json")
        with pytest.raises(LackpyRefusal):
            P("mcp-link.json").write_text("{}")  # the final component is a symlink
        assert (ws / "sub" / ".mcp.json").read_text() == "{}"
        assert not (ws / ".git" / "config").exists()

    @pytest.mark.parametrize("relative", [".git/hooks", "a/.git/x", ".mcp.json", "sub/.envrc"])
    def test_mkdir_parents_refuses(self, ws, P, relative):
        with pytest.raises(LackpyRefusal):
            P(relative).mkdir(parents=True)
        assert not (ws / relative).exists()

    def test_mkdir_parents_refuses_below_the_git_dir(self, ws, P):
        with pytest.raises(LackpyRefusal):
            P(".git/hooks/deep").mkdir(parents=True)
        assert not (ws / ".git").exists()

    @pytest.mark.parametrize("relative", [".GIT/config", ".Git/hooks/x", ".MCP.json", ".EnvRC"])
    def test_case_variants_refuse(self, ws, P, relative):
        # A case-insensitive filesystem would reach the real file.
        with pytest.raises(LackpyRefusal):
            P(relative).write_text("x")

    def test_reads_are_unaffected(self, ws, P):
        (ws / ".git").mkdir()
        (ws / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
        (ws / ".mcp.json").write_text("{}")
        (ws / ".envrc").write_text("export X=1\n")
        assert P(".git/HEAD").read_text().startswith("ref:")
        assert P(".mcp.json").read_text() == "{}"
        assert P(".envrc").read_text() == "export X=1\n"

    @pytest.mark.parametrize("relative", [".gitignore", ".github/workflows/ci.yml", "git/config",
                                          "mcp.json", "envrc", ".git-blame-ignore-revs"])
    def test_lookalike_names_stay_writable(self, ws, P, relative):
        P(relative).parent.mkdir(parents=True, exist_ok=True)
        assert P(relative).write_text("ok") == 2

    def test_refusal_names_the_reason(self, ws, P):
        with pytest.raises(LackpyRefusal, match="runs as host code"):
            P(".git/config").write_text("x")
