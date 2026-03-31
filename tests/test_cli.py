"""Tests for the CLI."""

from lackpy.cli import build_parser


def test_parser_delegate():
    parser = build_parser()
    args = parser.parse_args(["delegate", "find callers of foo", "--kit", "debug"])
    assert args.command == "delegate"
    assert args.intent == "find callers of foo"
    assert args.kit == "debug"


def test_parser_generate():
    parser = build_parser()
    args = parser.parse_args(["generate", "read file main.py", "--kit", "debug"])
    assert args.command == "generate"
    assert args.intent == "read file main.py"


def test_parser_validate():
    parser = build_parser()
    args = parser.parse_args(["validate", "program.py", "--kit", "debug"])
    assert args.command == "validate"
    assert args.file == "program.py"


def test_parser_kit_list():
    parser = build_parser()
    args = parser.parse_args(["kit", "list"])
    assert args.command == "kit"
    assert args.kit_command == "list"


def test_parser_spec():
    parser = build_parser()
    args = parser.parse_args(["spec"])
    assert args.command == "spec"


def test_parser_status():
    parser = build_parser()
    args = parser.parse_args(["status"])
    assert args.command == "status"
