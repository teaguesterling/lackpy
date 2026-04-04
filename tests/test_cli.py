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


def test_parser_create_flag():
    parser = build_parser()
    args = parser.parse_args(["-c", "find python files", "--create", "--name", "FindPy", "--kit", "read,glob"])
    assert args.intent == "find python files"
    assert args.create is True
    assert args.name == "FindPy"


def test_parser_generate_flag():
    parser = build_parser()
    args = parser.parse_args(["-c", "read file main.py", "--generate", "--kit", "read_file"])
    assert args.intent == "read file main.py"
    assert args.generate is True


def test_parser_param_flag():
    parser = build_parser()
    args = parser.parse_args(["-c", "read file test.txt", "--param", "x=1", "--param", "y=2"])
    assert args.param == ["x=1", "y=2"]


def test_parser_validate_flag():
    parser = build_parser()
    args = parser.parse_args(["--validate", "-c", "x = 1"])
    assert args.validate is True
    assert args.intent == "x = 1"
