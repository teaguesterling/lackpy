"""Tests for the lackpy CLI."""

from lackpy.cli import build_parser, _parse_tools


class TestParserFlags:
    """Test the flag-based interface (the kept functionality)."""

    def test_intent_flag(self):
        parser = build_parser()
        args = parser.parse_args(["-c", "read file main.py"])
        assert args.intent == "read file main.py"

    def test_create_flag(self):
        parser = build_parser()
        args = parser.parse_args(["-c", "find python files", "--create", "--name", "FindPy", "--kit", "read,glob"])
        assert args.intent == "find python files"
        assert args.create is True
        assert args.name == "FindPy"

    def test_generate_flag(self):
        parser = build_parser()
        args = parser.parse_args(["-c", "read file main.py", "--generate", "--kit", "read_file"])
        assert args.intent == "read file main.py"
        assert args.generate is True

    def test_param_flag(self):
        parser = build_parser()
        args = parser.parse_args(["-c", "read file test.txt", "--param", "x=1", "--param", "y=2"])
        assert args.param == ["x=1", "y=2"]

    def test_validate_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--validate", "-c", "x = 1"])
        assert args.validate is True
        assert args.intent == "x = 1"

    def test_mode_flag(self):
        parser = build_parser()
        args = parser.parse_args(["-c", "read file", "--mode", "spm"])
        assert args.mode == "spm"

    def test_workspace_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--workspace", "/tmp/test", "-c", "hello"])
        assert str(args.workspace) == "/tmp/test"


    def test_tools_flag(self):
        parser = build_parser()
        args = parser.parse_args(["-c", "do something", "--tools", "read_file,edit_file"])
        assert args.tools == "read_file,edit_file"

    def test_tools_flag_with_kit(self):
        parser = build_parser()
        args = parser.parse_args(["-c", "do something", "--kit", "debug", "--tools", "edit_file"])
        assert args.kit == "debug"
        assert args.tools == "edit_file"

    def test_tools_flag_without_kit(self):
        parser = build_parser()
        args = parser.parse_args(["-c", "do something", "--tools", "read_file"])
        assert args.kit is None
        assert args.tools == "read_file"


class TestParseTools:
    def test_single_tool(self):
        assert _parse_tools("read_file") == ["read_file"]

    def test_multiple_tools(self):
        assert _parse_tools("read_file,edit_file") == ["read_file", "edit_file"]

    def test_strips_whitespace(self):
        assert _parse_tools(" read_file , edit_file ") == ["read_file", "edit_file"]

    def test_skips_empty(self):
        assert _parse_tools("read_file,,edit_file") == ["read_file", "edit_file"]


class TestNoDeprecatedSubcommands:
    """Verify all deprecated subcommands are removed."""

    def test_no_subcommands_in_parser(self):
        parser = build_parser()
        # Parser should have no subparsers at all
        args = parser.parse_args([])
        assert not hasattr(args, "command") or getattr(args, "command", None) is None

    def test_help_mentions_lackpyctl(self):
        parser = build_parser()
        help_text = parser.format_help()
        assert "lackpyctl" in help_text
