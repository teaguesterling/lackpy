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
    return RestrictedRunner().run(program, namespace={}, params=params, path_policy=policy)


class TestShape:
    def test_class_is_named_path(self, P):
        assert P.__name__ == "Path"
        assert repr(P) == "<class 'pathlib.Path'>"

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

    @pytest.mark.parametrize("pattern", ["/etc/*", "../*", "a/../../*", "..", "x..y"])
    def test_glob_absolute_or_dotdot_pattern_refuses(self, P, pattern):
        with pytest.raises(LackpyRefusal):
            P(".").glob(pattern)
        with pytest.raises(LackpyRefusal):
            P(".").rglob(pattern)


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
        result = run("from pathlib import Path\n(Path(params['root']) / '..' / 'x').read_text()",
                     PathPolicy(ws, []), params={"params": {"root": "."}})
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
        assert proxy.Path.__name__ == "Path"
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
