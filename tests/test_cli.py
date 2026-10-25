"""Tests for the lackpy CLI."""

import json

import pytest

from lackpy.cli import build_parser, main
from lackpy.service import KIT_REMOVED


class TestParserFlags:
    """Test the flag-based interface (the kept functionality)."""

    def test_intent_flag(self):
        parser = build_parser()
        args = parser.parse_args(["-c", "read file main.py"])
        assert args.intent == "read file main.py"

    def test_create_flag(self):
        parser = build_parser()
        args = parser.parse_args(["-c", "find python files", "--create", "--name", "FindPy", "--profile", "fast"])
        assert args.intent == "find python files"
        assert args.create is True
        assert args.name == "FindPy"

    def test_generate_flag(self):
        parser = build_parser()
        args = parser.parse_args(["-c", "read file main.py", "--generate", "--profile", "fast"])
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



    def test_tools_flag_is_gone(self, capsys):
        # Retired kit model: --tools added extra tool names to the profile.
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["-c", "do something", "--tools", "read_file"])

    def test_profile_help_names_a_profile_only(self):
        help_text = build_parser().format_help()
        assert "tool list" not in help_text
        assert "--tools" not in help_text


class TestPlainProgramFile:
    """A plain program file runs without --profile (stepy100 Task 7)."""

    def test_plain_file_runs_without_a_profile(self, tmp_path, capsys):
        (tmp_path / "t.txt").write_text("abc")
        prog = tmp_path / "prog.py"
        prog.write_text("from pathlib import Path\nprint(Path('t.txt').read_text())\n")
        code = main([str(prog), "--workspace", str(tmp_path)])
        out = json.loads(capsys.readouterr().out)
        assert code == 0, out
        assert out["stdout"] == "abc\n"

    def test_tool_profile_is_an_error(self, tmp_path, capsys):
        prog = tmp_path / "prog.py"
        prog.write_text("x = 1\n")
        code = main([str(prog), "--workspace", str(tmp_path), "--profile", "read_file"])
        out = json.loads(capsys.readouterr().out)
        assert code == 1
        assert KIT_REMOVED in out["error"]

    def test_file_entrypoint_rejects_tools(self, tmp_path):
        prog = tmp_path / "prog.py"
        prog.write_text("x = 1\n")
        with pytest.raises(SystemExit):
            main([str(prog), "--workspace", str(tmp_path), "--tools", "read_file"])


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


class TestTypedParams:
    """Review I6: ``--param k=v`` is typed like the program's default."""

    def _run(self, tmp_path, capsys, program, *params):
        prog = tmp_path / "prog.py"
        prog.write_text(program)
        argv = [str(prog), "--workspace", str(tmp_path)]
        for p in params:
            argv += ["--param", p]
        code = main(argv)
        return code, json.loads(capsys.readouterr().out)

    def test_int_param_against_an_int_default(self, tmp_path, capsys):
        code, out = self._run(tmp_path, capsys, 'params = {"n": 3}\nparams["n"] * 2', "n=5")
        assert code == 0, out
        assert out["output"] == 10

    def test_literal_params_of_other_types(self, tmp_path, capsys):
        program = 'params = {"xs": [0], "flag": False, "ratio": 0.5}\n[params["xs"], params["flag"], params["ratio"]]'
        code, out = self._run(tmp_path, capsys, program, "xs=[1, 2]", "flag=True", "ratio=2")
        assert code == 0, out
        assert out["output"] == [[1, 2], True, 2]

    def test_str_default_keeps_the_raw_text(self, tmp_path, capsys):
        # "5" and "[1]" stay strings when the default is a str.
        program = 'params = {"name": "x", "tag": "y"}\n[params["name"], params["tag"]]'
        code, out = self._run(tmp_path, capsys, program, "name=5", "tag=[1]")
        assert code == 0, out
        assert out["output"] == ["5", "[1]"]

    def test_text_that_is_not_a_literal_stays_a_string(self, tmp_path, capsys):
        code, out = self._run(tmp_path, capsys, 'params = {"p": None}\nparams["p"]', "p=src/*.py")
        assert code == 0, out
        assert out["output"] == "src/*.py"

    def test_lackey_int_param(self, tmp_path, capsys):
        lackey = tmp_path / "double.py"
        lackey.write_text(
            "from lackpy.lackey import Lackey\n\n"
            "class Double(Lackey):\n"
            "    n: int = 3\n\n"
            "    def run(self) -> int:\n"
            "        return self.n * 2\n")
        code = main([str(lackey), "--workspace", str(tmp_path), "--param", "n=5"])
        out = json.loads(capsys.readouterr().out)
        assert code == 0, out
        assert out["output"] == 10


class TestTimeout:
    """Plan 3 Task 5 (D8): `lackpy <file>` takes --timeout SECONDS."""

    def test_file_timeout_stops_the_run(self, tmp_path, capsys):
        prog = tmp_path / "prog.py"
        prog.write_text('import re\nre.search(r"(a+)+$", "a" * 26 + "b")\n')
        code = main([str(prog), "--workspace", str(tmp_path), "--timeout", "1"])
        out = json.loads(capsys.readouterr().out)
        assert code == 1
        assert out["error_phase"] == "timeout"
        assert out["error"] == "timeout: the run took longer than 1 s and was stopped"

    def test_file_timeout_over_the_cap_refuses(self, tmp_path, capsys):
        prog = tmp_path / "prog.py"
        prog.write_text("1\n")
        code = main([str(prog), "--workspace", str(tmp_path), "--timeout", "500"])
        out = json.loads(capsys.readouterr().out)
        assert code == 1
        assert out["error_phase"] == "refused"
        assert out["error"] == "timeout 500 s is over the profile's max_timeout (120 s, profile (none))"
