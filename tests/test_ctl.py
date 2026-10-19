"""Tests for lackpyctl CLI."""

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


# --- Task 7 fix round 2, N1: `render --execute` ran the literate axis unvalidated ---

def _escape_doc(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("SECRET-RENDER")
    doc = tmp_path / "doc" / "doc.md"
    doc.parent.mkdir()
    doc.write_text("```lackpy\ng = (lambda: 0).__globals__\n"
                   f"t = g['read_file']('{outside}')\n```\n\nT={{t}}\n")
    return doc


def test_render_execute_is_refused(tmp_path, capsys):
    from lackpy.ctl import main
    from lackpy.profiles import LITERATE_PENDING
    code = main(["render", "--execute", str(_escape_doc(tmp_path))])
    captured = capsys.readouterr()
    assert code == 1
    assert LITERATE_PENDING in captured.err
    assert "SECRET-RENDER" not in captured.out + captured.err


def test_render_execute_writes_no_output_file(tmp_path, capsys):
    from lackpy.ctl import main
    out = tmp_path / "out.md"
    assert main(["render", "--execute", str(_escape_doc(tmp_path)), "-o", str(out)]) == 1
    assert not out.exists()


def test_render_without_execute_still_renders_and_runs_nothing(tmp_path, capsys):
    from lackpy.ctl import main
    assert main(["render", str(_escape_doc(tmp_path))]) == 0
    out = capsys.readouterr().out
    assert "g['read_file']" in out          # the source is shown
    assert "SECRET-RENDER" not in out        # nothing ran
