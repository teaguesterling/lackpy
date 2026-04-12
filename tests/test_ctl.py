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
    args = parser.parse_args(["kit", "list"])
    assert args.command == "kit"
    assert args.kit_command == "list"


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
