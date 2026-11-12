"""The ``stepy`` command for programs (spec E8-E11, Task 8).

Each test calls ``main`` in a temporary workspace (the current directory).
The autouse ``_isolated_path_policy`` fixture in ``tests/conftest.py`` gives
every service policy load an empty temporary home, so no test reads the real
``~/.claude``.
"""

import pytest

from lackpy.stepy import main

_COUNT_TODOS = '''\
# /// intent
# goal   = "Count the TODO markers in each C++ file under a directory."
# reads  = ["{root}/*.cpp"]
# writes = []
# output = "printed"
#
# [params]
# root  = { type = "str", doc = "directory to scan, relative to the workspace" }
# limit = { type = "int", doc = "most files to report" }
# ///
from pathlib import Path

params = {"root": "src", "limit": 20}

counts = []
for p in Path(params["root"]).glob("*.cpp"):
    text = p.read_text()
    counts.append((text.count("TODO"), p.name))
top = sorted(counts, reverse=True)[: params["limit"]]
for n, name in top:
    print(name, n)
'''


@pytest.fixture
def ws(tmp_path, monkeypatch):
    w = tmp_path / "ws"
    w.mkdir()
    monkeypatch.chdir(w)
    return w


def _script(ws, text, name="s.py"):
    path = ws / name
    path.write_text(text)
    return str(path)


def _writer(writes, path, params=None):
    """A program that declares ``writes`` (None: no block) and writes ``path``."""
    lines = []
    if writes is not None or params:
        lines.append("# /// intent")
        if writes is not None:
            lines.append(f"# writes = {writes!r}".replace("'", '"'))
        if params:
            lines.append("# [params]")
            lines += [f'# {k} = {{ type = "str", doc = "a name" }}' for k in params]
        lines.append("# ///")
    lines.append("from pathlib import Path")
    if params:
        lines.append(f"params = {params!r}")
    lines.append(f"p = Path({path})")
    if "/" in path:  # a mkdir of the workspace itself is a write too
        lines.append("p.parent.mkdir(parents=True, exist_ok=True)")
    lines += ["p.write_text('x')", "print('wrote')"]
    return "\n".join(lines) + "\n"


def test_runs_a_program_and_prints_only_its_prints(ws, capsys):
    assert main([_script(ws, "print('hi')\n42\n")]) == 0
    out = capsys.readouterr()
    assert out.out == "hi\n"
    assert out.err == ""


def test_named_options_set_params(ws, capsys):
    (ws / "src").mkdir()
    (ws / "src" / "a.cpp").write_text("TODO TODO")
    (ws / "src" / "b.cpp").write_text("TODO")
    (ws / "lib").mkdir()
    (ws / "lib" / "c.cpp").write_text("TODO TODO TODO")
    script = _script(ws, _COUNT_TODOS)
    assert main([script, "--root", "src", "--limit", "1"]) == 0
    assert capsys.readouterr().out == "a.cpp 2\n"
    assert main([script, "--root", "lib"]) == 0
    assert capsys.readouterr().out == "c.cpp 3\n"


_FLAGS = '''\
# /// intent
# [params]
# loud    = { type = "bool", doc = "shout" }
# names   = { type = "list[str]", doc = "the names" }
# dry_run = { type = "bool", doc = "write nothing" }
# ///
params = {"loud": False, "names": ["x"], "dry_run": True}
print(params["loud"], params["names"], params["dry_run"])
'''


def test_bool_and_list_options(ws, capsys):
    script = _script(ws, _FLAGS)
    assert main([script]) == 0
    assert capsys.readouterr().out == "False ['x'] True\n"
    assert main([script, "--loud", "--names", "a", "--names", "b", "--no-dry_run"]) == 0
    assert capsys.readouterr().out == "True ['a', 'b'] False\n"


@pytest.mark.parametrize("args,named", [
    (["--nope", "1"], "--nope"),
    (["--lou"], "--lou"),           # no prefix matching: one way to set each parameter
    (["--names"], "--names"),       # a missing value
    (["--loud", "yes"], "yes"),     # no positional arguments
    (["--loud", "--loud"], "--loud"),
    (["--dry-run"], "--dry-run"),   # an underscore stays an underscore
])
def test_unknown_option_exits_2_and_runs_nothing(ws, capsys, args, named):
    assert main([_script(ws, _FLAGS), *args]) == 2
    out = capsys.readouterr()
    assert out.out == ""
    assert out.err.startswith("stepy: ") and named in out.err


def test_a_block_after_code_exits_2_and_runs_nothing(ws, capsys):
    # Final review 9: a misplaced block fails loud; it is never read as "no block".
    script = _script(ws, 'print("ran")\n# /// intent\n# writes = ["a.txt"]\n# ///\n')
    assert main([script]) == 2
    out = capsys.readouterr()
    assert out.out == ""
    assert out.err.startswith("stepy: intent block: ") and "at the top" in out.err


def test_wrong_type_exits_2(ws, capsys):
    assert main([_script(ws, _COUNT_TODOS), "--limit", "many"]) == 2
    out = capsys.readouterr()
    assert out.out == ""
    assert "--limit" in out.err and "int" in out.err


def test_no_block_means_no_writes(ws, capsys):
    assert main([_script(ws, _writer(None, "'a.txt'"))]) == 1
    out = capsys.readouterr()
    assert out.out == ""
    assert out.err.startswith("stepy: ") and "a.txt" in out.err
    assert not (ws / "a.txt").exists()


def test_a_block_with_no_writes_key_means_no_writes(ws, capsys):
    script = _script(ws, '# /// intent\n# goal = "write"\n# ///\n' + _writer(None, "'a.txt'"))
    assert main([script]) == 1
    assert "no declared writes" in capsys.readouterr().err
    assert not (ws / "a.txt").exists()


def test_declared_write_is_allowed(ws, capsys):
    assert main([_script(ws, _writer(["out/*.txt"], "'out/a.txt'"))]) == 0
    assert capsys.readouterr().out == "wrote\n"
    assert (ws / "out" / "a.txt").read_text() == "x"


def test_write_outside_the_globs_is_refused(ws, capsys):
    assert main([_script(ws, _writer(["out/*.txt"], "'other.txt'"))]) == 1
    err = capsys.readouterr().err
    assert "other.txt" in err and "out/*.txt" in err
    assert not (ws / "other.txt").exists()


def test_param_in_write_glob_is_filled_in(ws, capsys):
    script = _script(ws, _writer(["out/{name}.txt"], "'out/' + params['name'] + '.txt'",
                                 params={"name": "a"}))
    assert main([script, "--name", "r"]) == 0
    assert (ws / "out" / "r.txt").exists()
    assert main([script]) == 0  # the default fills it in too
    assert (ws / "out" / "a.txt").exists()
    capsys.readouterr()
    # The filled glob allows only its own path: out/a.txt is not out/r.txt.
    other = _script(ws, _writer(["out/{name}.txt"], "'out/a.txt'", params={"name": "a"}),
                    name="other.py")
    assert main([other, "--name", "r"]) == 1
    assert "out/r.txt" in capsys.readouterr().err
    # Review Focus 3: a value cannot move the glob out (more values in the next test).
    assert main([script, "--name", "../x"]) == 2
    assert not (ws.parent / "x.txt").exists()


@pytest.mark.parametrize("value", ["*", "a?", "[ab]", ".", "..", "../x", "a/../b", ""])
def test_a_bad_value_in_a_write_glob_exits_2_and_runs_nothing(ws, capsys, value):
    # Review Focus 3: a value cannot widen a glob or move it out of the workspace.
    script = _script(ws, _writer(["out/{name}.txt"], "'out/x.txt'", params={"name": "a"}))
    assert main([script, "--name", value]) == 2
    out = capsys.readouterr()
    assert out.out == ""
    assert "name" in out.err
    assert not (ws / "out").exists()


def test_a_value_used_only_in_reads_may_hold_a_glob_character(ws, capsys):
    (ws / "src").mkdir()
    (ws / "src" / "a.cpp").write_text("TODO")
    assert main([_script(ws, _COUNT_TODOS), "--root", "s*"]) == 0
    capsys.readouterr()


def test_a_glob_that_leaves_the_workspace_exits_2(ws, tmp_path, capsys):
    assert main([_script(ws, _writer(["../*.txt"], "'../escaped.txt'"))]) == 2
    out = capsys.readouterr()
    assert out.out == ""
    assert "declared writes glob" in out.err
    assert not (tmp_path / "escaped.txt").exists()


def test_a_placeholder_that_names_no_param_exits_2(ws, capsys):
    assert main([_script(ws, _writer(["out/{nope}.txt"], "'out/x.txt'"))]) == 2
    assert "nope" in capsys.readouterr().err


def test_timeout_exits_124(ws, capsys):
    assert main([_script(ws, "for i in iter(int, 1):\n    pass\n"), "--timeout", "0.5"]) == 124
    err = capsys.readouterr().err
    # The service's text: "timeout: the run took longer than 0.5 s and was stopped".
    assert err.startswith("stepy: timeout: ") and "took longer than 0.5 s" in err


_PRINT_AND_WRITE = _writer(["out/*.txt"], "'out/a.txt'")


@pytest.mark.parametrize("check", [False, True], ids=["run", "check"])
@pytest.mark.parametrize("args,named", [
    (["--timeout", "100000"], "max_timeout"),   # over the profile's cap
    (["--timeout", "0"], "timeout"),
    (["--timeout", "nan"], "timeout"),
    (["--workspace", "/nonexistent"], "/nonexistent"),
    (["--profile", "nope"], "nope"),
])
def test_a_bad_own_option_exits_2_and_runs_nothing(ws, capsys, args, named, check):
    # Fix round 1: --check and a run make the same pre-run checks.
    assert main([_script(ws, _PRINT_AND_WRITE), *args, *(["--check"] if check else [])]) == 2
    out = capsys.readouterr()
    assert out.out == ""
    assert out.err.startswith("stepy: ") and named in out.err
    assert not (ws / "out").exists()


@pytest.mark.parametrize("check", [False, True], ids=["run", "check"])
def test_a_float_param_with_an_int_default_exits_2(ws, capsys, check):
    script = _script(ws, '# /// intent\n# [params]\n# ratio = { type = "float", doc = "r" }\n'
                         '# ///\nparams = {"ratio": 1}\nprint(params["ratio"])\n')
    assert main([script, "--ratio", "0.5", *(["--check"] if check else [])]) == 2
    out = capsys.readouterr()
    assert out.out == ""
    assert "ratio" in out.err and "float" in out.err


def test_a_policy_that_cannot_be_loaded_fails_the_run(ws, capsys):
    # _prepare leaves this to the run: exit 1 with the settings error (fix round 1).
    (ws / ".claude").mkdir()
    (ws / ".claude" / "settings.json").write_text("{not json")
    assert main([_script(ws, "print('hi')\n")]) == 1
    out = capsys.readouterr()
    assert out.out == ""
    assert out.err.startswith("stepy: ") and "settings.json" in out.err


def test_a_file_that_is_not_utf8_exits_2(ws, capsys):
    path = ws / "bad.py"
    path.write_bytes(b"print('\xff\xfe')\n")
    assert main([str(path)]) == 2
    out = capsys.readouterr()
    assert out.out == ""
    assert out.err.startswith("stepy: ") and "bad.py" in out.err


def test_a_failure_prints_what_ran_then_the_error(ws, capsys):
    assert main([_script(ws, "print('before')\nx = 1 / 0\nprint('after')\n")]) == 1
    out = capsys.readouterr()
    assert out.out == "before\n"
    assert out.err.startswith("stepy: ") and "division by zero" in out.err


def test_a_validation_error_exits_2_and_runs_nothing(ws, capsys):
    assert main([_script(ws, "print('hi')\nimport os\n")]) == 2
    out = capsys.readouterr()
    assert out.out == ""
    assert "import" in out.err.lower()


def test_check_runs_nothing(ws, capsys):
    assert main([_script(ws, _writer(["out/*.txt"], "'out/a.txt'")), "--check"]) == 0
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""
    assert not (ws / "out").exists()


def test_check_reports_the_errors(ws, capsys):
    bad = _COUNT_TODOS.replace('"limit": 20', '"limit": "20"')
    assert main([_script(ws, bad), "--check"]) == 2
    assert "limit" in capsys.readouterr().err
    assert main([_script(ws, "import os\n", name="v.py"), "--check"]) == 2
    assert capsys.readouterr().err


def test_a_param_may_not_use_a_stepy_option_name(ws, capsys):
    script = _script(ws, '# /// intent\n# [params]\n# check = { type = "bool", doc = "c" }\n'
                         '# ///\nparams = {"check": False}\n')
    assert main([script, "--check"]) == 2
    assert "check" in capsys.readouterr().err


def test_help_lists_the_interface(ws, capsys):
    assert main([_script(ws, _COUNT_TODOS), "--help"]) == 0
    out = capsys.readouterr().out
    assert "Count the TODO markers in each C++ file under a directory." in out
    assert "--root str (default: 'src') directory to scan, relative to the workspace" in out
    assert "--limit int (default: 20) most files to report" in out
    assert "writes: declared, enforced" in out
    assert "reads: declared, not checked" in out and "{root}/*.cpp" in out
    assert "checks" in out


def test_help_with_no_block(ws, capsys):
    assert main([_script(ws, "print('ran')\n"), "--help"]) == 0
    out = capsys.readouterr().out
    assert "this script declares nothing, so it may write nothing" in out
    assert "ran" not in out


def test_dash_c_runs_program_text(ws, capsys):
    assert main(["-c", "print(6 * 7)"]) == 0
    assert capsys.readouterr().out == "42\n"


def test_dash_c_takes_options(ws, capsys):
    program = '# /// intent\n# [params]\n# n = { type = "int", doc = "n" }\n# ///\n' \
              'params = {"n": 1}\nprint(params["n"] * 2)\n'
    assert main(["-c", program, "--n", "21"]) == 0
    assert capsys.readouterr().out == "42\n"


@pytest.mark.parametrize("name", ["s.txt", "s"])
def test_other_extension_exits_2(ws, capsys, name):
    assert main([_script(ws, "print('hi')\n", name=name)]) == 2
    out = capsys.readouterr()
    assert out.out == ""
    assert ".py" in out.err and ".md" in out.err


def test_a_missing_file_exits_2(ws, capsys):
    assert main([str(ws / "nope.py")]) == 2
    assert "nope.py" in capsys.readouterr().err


def test_no_script_exits_2(ws, capsys):
    assert main([]) == 2
    assert "usage" in capsys.readouterr().err


def test_shebang_file_runs(ws, capsys):
    script = _script(ws, "#!/usr/bin/env stepy\n" + _COUNT_TODOS)
    (ws / "src").mkdir()
    (ws / "src" / "a.cpp").write_text("TODO")
    assert main([script, "--limit", "5"]) == 0
    assert capsys.readouterr().out == "a.cpp 1\n"


def test_workspace_option(ws, tmp_path, capsys):
    other = tmp_path / "other"
    other.mkdir()
    script = _script(ws, _writer(["out/*.txt"], "'out/a.txt'"))
    assert main([script, "--workspace", str(other)]) == 0
    assert (other / "out" / "a.txt").exists()
    assert not (ws / "out").exists()


# --- Task 9: stepy runs a literate document (spec E8, E10, E11) ---

def _doc(ws, *cells, prose="# Title\n\nSome prose.\n", name="d.md"):
    """A document: ``prose``, then each cell as a ```lackpy fence with prose between.

    The run counts every cell, prose too, so code cell k is document cell 2k + 1.
    """
    parts = [prose] + [f"```lackpy\n{c}```\n\nBetween.\n" for c in cells]
    path = ws / name
    path.write_text("\n".join(parts))
    return str(path)


def _block(writes=None, params=None):
    """An intent block for a document's first cell, then its params literal."""
    lines = ["# /// intent"]
    if writes is not None:
        lines.append(f"# writes = {writes!r}".replace("'", '"'))
    if params:
        lines.append("# [params]")
        lines += [f'# {k} = {{ type = "str", doc = "a name" }}' for k in params]
    lines.append("# ///")
    if params:
        lines.append(f"params = {params!r}")
    return "\n".join(lines) + "\n"


def test_document_prints_only_cell_output(ws, capsys):
    assert main([_doc(ws, "print('a')\nx = 2\n", "print('b')\nx\n")]) == 0
    out = capsys.readouterr()
    assert out.out == "a\nb\n"  # no prose, no rendered value
    assert out.err == ""


def test_document_stops_at_the_first_failure(ws, capsys):
    assert main([_doc(ws, "print('one')\n", "print('two')\ny = 1 / 0\n", "print('three')\n")]) == 1
    out = capsys.readouterr()
    assert out.out == "one\ntwo\n"  # the failed cell's print, then nothing after it
    assert out.err.startswith("stepy: cell 3 (print('two')): ")
    assert "ZeroDivisionError" in out.err


def test_document_params_from_options(ws, capsys):
    doc = _doc(ws, _block(params={"name": "world"}), "print('hello', params['name'])\n")
    assert main([doc]) == 0
    assert capsys.readouterr().out == "hello world\n"
    assert main([doc, "--name", "you"]) == 0
    assert capsys.readouterr().out == "hello you\n"
    assert main([doc, "--nope", "x"]) == 2
    assert "--nope" in capsys.readouterr().err


def test_document_params_are_checked_against_the_literal(ws, capsys):
    # Ruling P3: the literal may be in any code cell; [params] is checked against it.
    block = '# /// intent\n# [params]\n# n = { type = "int", doc = "n" }\n# ///\nprint(1)\n'
    doc = _doc(ws, block, "params = {'n': 'one'}\nprint(params['n'])\n")
    assert main([doc]) == 2
    out = capsys.readouterr()
    assert out.out == "" and "'n'" in out.err and "int" in out.err


_DOC_WRITE = "from pathlib import Path\nPath('out').mkdir(exist_ok=True)\nPath({!r}).write_text('x')\nprint('wrote')\n"


def test_document_declared_writes_are_enforced(ws, capsys):
    assert main([_doc(ws, _block(writes=["out/*.txt"]), _DOC_WRITE.format("out/a.txt"))]) == 0
    assert capsys.readouterr().out == "wrote\n"
    assert (ws / "out" / "a.txt").exists()
    assert main([_doc(ws, _block(writes=["out/*.txt"]), _DOC_WRITE.format("out/b.md"))]) == 1
    err = capsys.readouterr().err
    # A refusal's text names the path and the globs, not the cell (E1: it is unchanged).
    assert err.startswith("stepy: refused: ") and "out/b.md" in err and "out/*.txt" in err
    assert not (ws / "out" / "b.md").exists()


def test_document_with_no_block_may_write_nothing(ws, capsys):
    assert main([_doc(ws, "from pathlib import Path\nPath('a.txt').write_text('x')\n")]) == 1
    assert "a.txt" in capsys.readouterr().err
    assert not (ws / "a.txt").exists()


def test_document_timeout_is_per_cell_and_exits_124(ws, capsys):
    doc = _doc(ws, "print('one')\n", "print('two')\n", "for i in iter(int, 1): pass\n",
               "print('after')\n")
    assert main([doc, "--timeout", "3"]) == 124
    out = capsys.readouterr()
    assert out.out == "one\ntwo\n"  # Task 5 review ruling: the earlier cells' prints, in order
    assert out.err.startswith("stepy: cell 5 (for i in iter(int, 1): pass): timeout: ")
    assert "the cell took longer than 3 s" in out.err


def test_document_check_runs_nothing(ws, capsys):
    assert main([_doc(ws, _block(writes=["out/*.txt"]), _DOC_WRITE.format("out/a.txt")),
                 "--check"]) == 0
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""
    assert not (ws / "out").exists()


def test_document_validation_error_exits_2_and_runs_nothing(ws, capsys):
    assert main([_doc(ws, "print('hi')\n", "import os\n")]) == 2
    out = capsys.readouterr()
    assert out.out == ""
    # Final review 6: the run errors' form, "cell {i} ({first line})", index from 0.
    assert "cell 3 (import os): " in out.err, out.err


def test_document_help_says_the_timeout_is_per_cell(ws, capsys):
    assert main([_doc(ws, _block(params={"name": "world"}), "print(1)\n"), "--help"]) == 0
    out = capsys.readouterr().out
    assert "--name str (default: 'world') a name" in out
    assert "--timeout SECONDS is the time limit for each cell" in out
    assert main([_doc(ws, "print('ran')\n", name="e.md"), "--help"]) == 0
    out = capsys.readouterr().out
    assert "this document declares nothing, so it may write nothing" in out
    assert "ran" not in out


def _profiles(ws, default):
    (ws / ".lackpy").mkdir()
    (ws / ".lackpy" / "config.toml").write_text(
        f'[profile]\ndefault = "{default}"\n\n'
        '[profiles.ro]\npolicy = { read_only = true }\n\n[profiles.rw]\n')


def test_document_profile_is_the_configured_default_run_as_literate(ws, capsys):
    # As render --execute resolves it: the named profile, else the default, as literate.
    _profiles(ws, "ro")
    doc = _doc(ws, _block(writes=["out/*.txt"]), _DOC_WRITE.format("out/a.txt"))
    assert main([doc]) == 1  # the read-only default refuses the write
    capsys.readouterr()
    assert not (ws / "out" / "a.txt").exists()
    assert main([doc, "--profile", "rw"]) == 0
    assert capsys.readouterr().out == "wrote\n"
    assert main([doc, "--profile", "nope"]) == 2
    assert "nope" in capsys.readouterr().err


def test_render_execute_is_unchanged(ws, capsys):
    # The shared profile helper must not change render --execute: it still
    # prints the rendered document, prose included.
    from lackpy.ctl import main as ctl_main
    doc = ws / "r.md"
    doc.write_text("# Title\n\n```lackpy\nprint('a')\nx = 6 * 7\n```\n\nX={x}\n")
    assert ctl_main(["render", "--execute", str(doc)]) == 0
    out = capsys.readouterr().out
    assert "# Title" in out and "X=42" in out
