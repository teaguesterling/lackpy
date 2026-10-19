"""Tests for lackpyctl CLI."""

import pytest

from lackpy.ctl import build_parser


def test_parser_init():
    parser = build_parser()
    args = parser.parse_args(["init", "--ollama-url", "http://localhost:11435"])
    assert args.command == "init"
    assert args.ollama_url == "http://localhost:11435"


def test_parser_status():
    parser = build_parser()
    args = parser.parse_args(["status"])
    assert args.command == "status"


def test_parser_kit_list():
    parser = build_parser()
    args = parser.parse_args(["profile", "list"])
    assert args.command == "profile"
    assert args.profile_command == "list"


def test_parser_toolbox_list():
    parser = build_parser()
    args = parser.parse_args(["toolbox", "list"])
    assert args.command == "toolbox"
    assert args.toolbox_command == "list"


def test_parser_spec():
    parser = build_parser()
    args = parser.parse_args(["spec"])
    assert args.command == "spec"


def test_parser_mcp_serve():
    parser = build_parser()
    args = parser.parse_args(["mcp", "serve"])
    assert args.command == "mcp"
    assert args.mcp_command == "serve"


def test_parser_mcp_init():
    parser = build_parser()
    args = parser.parse_args(["mcp", "init"])
    assert args.command == "mcp"
    assert args.mcp_command == "init"


def test_parser_mcp_init_force():
    parser = build_parser()
    args = parser.parse_args(["mcp", "init", "--force"])
    assert args.force is True


def test_parser_mcp_init_name():
    parser = build_parser()
    args = parser.parse_args(["mcp", "init", "--name", "my-lackpy"])
    assert args.name == "my-lackpy"


def test_parser_render():
    parser = build_parser()
    args = parser.parse_args(["render", "doc.md"])
    assert args.command == "render"
    assert args.path == "doc.md"
    # Rendering must not run the document by default: a literate document
    # writes files and shells out, so execution is opt-in.
    assert args.execute is False


def test_parser_render_execute_and_output():
    parser = build_parser()
    args = parser.parse_args(["render", "doc.md", "--execute", "-o", "out.md"])
    assert args.execute is True
    assert args.output == "out.md"


# --- literate stepy Task 4: `render --execute` runs through the service ---
# (Task 7 fix round 2, N1 refused it with LITERATE_PENDING until the literate
# interpreter ran on the stepy100 runtime.) The escape document is gate 1 in
# tests/literate/test_gate.py.

@pytest.fixture
def tmp_policy_home(tmp_path, monkeypatch):
    """Every service policy load uses a temporary home and no managed settings."""
    from lackpy import service
    from lackpy.run.paths import load_policy
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    monkeypatch.setattr(service, "load_policy",
                        lambda ws: load_policy(ws, home=home, managed=None))
    return home


def _doc(tmp_path, text):
    doc = tmp_path / "doc" / "doc.md"
    doc.parent.mkdir(exist_ok=True)
    doc.write_text(text)
    return doc


def test_render_execute_renders_the_run(tmp_path, tmp_policy_home, capsys):
    from lackpy.ctl import main
    doc = _doc(tmp_path, "```lackpy\nx = 6 * 7\n```\n\nX={x}\n")
    assert main(["render", "--execute", str(doc)]) == 0
    assert "X=42" in capsys.readouterr().out


def test_render_execute_resolves_paths_against_the_base_dir(tmp_path, tmp_policy_home, capsys):
    from lackpy.ctl import main
    base = tmp_path / "base"
    base.mkdir()
    doc = _doc(tmp_path, "```lackpy @write(out.txt)\nkept\n```\n")
    out = tmp_path / "out.md"
    assert main(["render", "--execute", str(doc), "--base-dir", str(base), "-o", str(out)]) == 0
    assert (base / "out.txt").read_text() == "kept"
    assert not (doc.parent / "out.txt").exists()
    assert out.exists()


def test_render_execute_failure_exits_nonzero_and_writes_no_output_file(
        tmp_path, tmp_policy_home, capsys):
    from lackpy.ctl import main
    doc = _doc(tmp_path, "```lackpy\nimport os\n```\n")
    out = tmp_path / "out.md"
    assert main(["render", "--execute", str(doc), "-o", str(out)]) == 1
    assert "Import" in capsys.readouterr().err
    assert not out.exists()


# Fix round 1, M6: render --execute honours the configured default profile and --profile.

def _ws_with_profiles(tmp_path, default=None):
    ws = tmp_path / "ws"
    (ws / ".lackpy").mkdir(parents=True)
    text = ""
    if default:
        text += f'[profile]\ndefault = "{default}"\n\n'
    text += '[profiles.ro]\nexecution = "literate"\npolicy = { read_only = true }\n\n'
    text += '[profiles.rw]\nexecution = "literate"\n'
    (ws / ".lackpy" / "config.toml").write_text(text)
    return ws


_WRITE = "```lackpy\nfrom pathlib import Path\nPath('o.txt').write_text('x')\n```\n"


def test_render_execute_honours_a_read_only_default_profile(tmp_path, tmp_policy_home, capsys):
    from lackpy.ctl import main
    ws = _ws_with_profiles(tmp_path, default="ro")
    doc = _doc(tmp_path, _WRITE)
    assert main(["render", "--execute", str(doc), "--base-dir", str(ws)]) == 1
    assert not (ws / "o.txt").exists()


def test_render_execute_takes_a_profile(tmp_path, tmp_policy_home, capsys):
    from lackpy.ctl import main
    ws = _ws_with_profiles(tmp_path, default="rw")
    doc = _doc(tmp_path, _WRITE)
    assert main(["render", "--execute", "--profile", "ro", str(doc), "--base-dir", str(ws)]) == 1
    assert not (ws / "o.txt").exists()
    assert main(["render", "--execute", "--profile", "rw", str(doc), "--base-dir", str(ws)]) == 0
    assert (ws / "o.txt").read_text() == "x"


def test_render_execute_with_an_unknown_profile_exits_1(tmp_path, tmp_policy_home, capsys):
    from lackpy.ctl import main
    ws = _ws_with_profiles(tmp_path)
    doc = _doc(tmp_path, _WRITE)
    assert main(["render", "--execute", "--profile", "nope", str(doc), "--base-dir", str(ws)]) == 1
    assert "nope" in capsys.readouterr().err
    assert not (ws / "o.txt").exists()
