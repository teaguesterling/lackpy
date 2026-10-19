"""Literate stepy gates 1, 2 and 4 (design §Gates), through the real entry points.

* Gate 1: the Plan 2 Task 7 C1 escape document refuses through every entry
  point. That document reached host ``os`` through ``(lambda: 0).__globals__``
  and ``object.__subclasses__()``, and read a file outside the workspace. Each
  shape below is its own document. Each one must fail (``success=False`` or a
  nonzero exit), must not show the outside file's text, and must leave every
  file in the workspace and outside it unchanged.
* Gate 2: a cell that writes ``.claude/settings.json`` refuses.
* Gate 4: a refusal inside a ``try``, in a cell, stops the document.

Gate 3 is ``test_equivalence.py``; gate 5 is ``test_no_host_exec.py``.

No test reads the real ``~/.claude`` or ``/etc/claude-code``: every policy is
loaded with a temporary home and ``managed=None``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from lackpy import service as service_module
from lackpy.profiles import Profile
from lackpy.run.paths import load_policy
from lackpy.service import LackpyService

SECRET = "SECRET-OUTSIDE-THE-WORKSPACE"
_LIT = Profile(execution="literate")


def _escape_docs(outside: Path) -> dict[str, str]:
    secret = outside / "secret.txt"
    return {
        "lambda-globals": "```lackpy\ng = (lambda: 0).__globals__\nb = g['__builtins__']\n```\n\nB={b}\n",
        "globals-read-file": ("```lackpy\ng = (lambda: 0).__globals__\n"
                              f"t = g['read_file']('{secret}')\n```\n\nT={{t}}\n"),
        "object-subclasses": "```lackpy\nn = len(object.__subclasses__())\n```\n\nN={n}\n",
        "base-subclasses": "```lackpy\nn = len(().__class__.__base__.__subclasses__())\n```\n\nN={n}\n",
        "import-os": "```lackpy\nimport os\nt = os.listdir('/')\n```\n\nT={t}\n",
        "dunder-import-os": "```lackpy\nm = __import__('os')\nt = m.listdir('/')\n```\n\nT={t}\n",
        "read-outside-absolute": ("```lackpy\nfrom pathlib import Path\n"
                                  f"t = Path('{secret}').read_text()\n```\n\nT={{t}}\n"),
        "read-outside-relative": ("```lackpy\nfrom pathlib import Path\n"
                                  "t = Path('../outside/secret.txt').read_text()\n```\n\nT={t}\n"),
        "read-annotation-outside": f"```lackpy @read({secret})\n```\n",
        "write-outside": ("```lackpy\nfrom pathlib import Path\n"
                          f"Path('{outside / 'new.txt'}').write_text('x')\n```\n"),
        # A prose interpolation starts with a name (``{(`` is literal text).
        "prose-globals": "```lackpy\nx = 1\n```\n\nG={len((lambda: 0).__globals__)}\n",
        "prose-subclasses": "```lackpy\nx = 1\n```\n\nS={len(().__class__.__base__.__subclasses__())}\n",
        "prose-object-subclasses": "```lackpy\nx = 1\n```\n\nS={object.__subclasses__()}\n",
        "prose-read-outside": ("```lackpy\nfrom pathlib import Path\n```\n\n"
                               f"T={{Path('{secret}').read_text()}}\n"),
    }


_NAMES = list(_escape_docs(Path("/x")))


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A workspace with a configured ``lit`` profile, a secret outside it, and a
    temporary home that every service policy load uses."""
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    ws = tmp_path / "ws"
    (ws / ".lackpy").mkdir(parents=True)
    (ws / ".lackpy" / "config.toml").write_text('[profiles.lit]\nexecution = "literate"\n')
    (ws / "test.txt").write_text("hello")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text(SECRET)

    def loader(workspace):
        return load_policy(workspace, home=home, managed=None)

    # The CLI and lackpyctl build their own service: route its default loader too.
    monkeypatch.setattr(service_module, "load_policy", loader)
    return {"tmp": tmp_path, "home": home, "ws": ws, "outside": outside, "loader": loader}


def _files(*roots: Path) -> dict[str, bytes]:
    return {str(p): p.read_bytes() for root in roots for p in sorted(root.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts}


def _service(env) -> LackpyService:
    return LackpyService(workspace=env["ws"], path_policy_loader=env["loader"])


def _run_program(env, doc):
    r = asyncio.run(_service(env).run_program(doc, profile=_LIT))
    return r.success, repr(r)


def _delegate(env, doc):
    r = asyncio.run(_service(env).delegate("x", profile=_LIT, _program_override=doc))
    return r["success"], repr(r)


def _mcp_run_program(env, doc):
    pytest.importorskip("mcp", reason="mcp required for the MCP entry point")
    from lackpy.mcp import server
    server._service = _service(env)
    try:
        r = asyncio.run(server.run_program(doc, profile="lit"))
    finally:
        server._service = None
    return r["success"], json.dumps(r)


def _ctl_render_execute(env, doc):
    from lackpy.ctl import main
    path = env["tmp"] / "docs" / "doc.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text(doc)
    code = main(["render", "--execute", str(path), "--base-dir", str(env["ws"])])
    return code == 0, ""


def _cli_profile(env, doc):
    from lackpy.cli import main
    path = env["tmp"] / "docs" / "doc.py"
    path.parent.mkdir(exist_ok=True)
    path.write_text(doc)
    code = main([str(path), "--workspace", str(env["ws"]), "--profile", "lit"])
    return code == 0, ""


ENTRY_POINTS = {
    "run_program": _run_program,
    "delegate": _delegate,
    "mcp.run_program": _mcp_run_program,
    "lackpyctl render --execute": _ctl_render_execute,
    "lackpy --profile": _cli_profile,
}


@pytest.mark.parametrize("entry", list(ENTRY_POINTS))
@pytest.mark.parametrize("name", _NAMES)
def test_gate1_escape_refuses_on_every_entry_point(env, capsys, entry, name):
    doc = _escape_docs(env["outside"])[name]
    before = _files(env["ws"], env["outside"])
    ok, shown = ENTRY_POINTS[entry](env, doc)
    captured = capsys.readouterr()
    assert not ok, f"{name} via {entry} succeeded"
    assert SECRET not in shown + captured.out + captured.err
    assert _files(env["ws"], env["outside"]) == before


@pytest.mark.parametrize("name", _NAMES)
def test_gate1_each_escape_fails_for_the_right_reason(env, name):
    # Not a parse error or a config error: the validator or the policy stops it.
    r = asyncio.run(_service(env).run_program(_escape_docs(env["outside"])[name], profile=_LIT))
    assert "Validation failed" in r.error or r.error.startswith("refused"), r.error


def test_gate1_a_good_document_runs_on_every_entry_point(env, capsys):
    # The gate is not "everything fails": the same entry points run a good document.
    doc = "```lackpy\nfrom pathlib import Path\nt = Path('test.txt').read_text()\n```\n\nT={t}\n"
    import importlib.util
    has_mcp = importlib.util.find_spec("mcp") is not None
    for entry, run in ENTRY_POINTS.items():
        if entry == "mcp.run_program" and not has_mcp:
            continue  # the MCP leg runs under blq test-mcp
        ok, _ = run(env, doc)
        assert ok, entry
    assert "T=hello" in capsys.readouterr().out


def test_gate2_a_cell_that_writes_claude_settings_refuses(env):
    doc = ("```lackpy\nfrom pathlib import Path\n"
           "Path('.claude/settings.json').write_text('{}')\n```\n\n"
           "```lackpy\nPath('after.txt').write_text('x')\n```\n")
    r = asyncio.run(_service(env).run_program(doc, profile=_LIT))
    assert not r.success
    assert "refused" in r.error
    assert not (env["ws"] / ".claude").exists()
    assert not (env["ws"] / "after.txt").exists()


def test_gate2_the_write_annotation_refuses_too(env):
    doc = "```lackpy @write(.claude/settings.json)\n{}\n```\n"
    r = asyncio.run(_service(env).run_program(doc, profile=_LIT))
    assert not r.success
    assert not (env["ws"] / ".claude").exists()


def test_gate4_a_refusal_inside_try_stops_the_document(env):
    secret = env["outside"] / "secret.txt"
    doc = ("```lackpy\nfrom pathlib import Path\n"
           "try:\n"
           f"    t = Path('{secret}').read_text()\n"
           "except OSError:\n"
           "    t = 'caught'\n"
           "```\n\n"
           "T={t}\n\n"
           "```lackpy\nPath('after.txt').write_text('x')\n```\n")
    r = asyncio.run(_service(env).run_program(doc, profile=_LIT))
    assert not r.success
    assert r.error.startswith("refused")
    assert "caught" not in r.stdout
    assert SECRET not in repr(r)
    assert not (env["ws"] / "after.txt").exists()
