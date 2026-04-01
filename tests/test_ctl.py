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
