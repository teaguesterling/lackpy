"""Tests for parsing Lackey files."""

import pytest
from pathlib import Path
from lackpy.lackey.parser import parse_lackey, LackeyInfo


@pytest.fixture
def sample_lackey(tmp_path):
    code = '''
from lackpy.lackey import Lackey, Tool, Log, System, User, Assistant

class CountLines(Lackey):
    """Count lines in files matching a pattern."""

    read = Tool()
    glob = Tool()

    pattern: str = "**/*.py"
    max_depth: int = 10

    returns: int

    creation_log = Log([
        System("You are a cell generator."),
        User("count lines in files"),
        Assistant("files = glob(pattern)\\nlen(files)", accepted=True),
    ])

    def run(self) -> int:
        files = self.glob(self.pattern)
        for f in files:
            content = self.read(f)
            print(f"{f}: {len(content.splitlines())}")
        return len(files)
'''
    path = tmp_path / "count_lines.py"
    path.write_text(code)
    return path


class TestParseLackey:
    def test_parses_class_name(self, sample_lackey):
        info = parse_lackey(sample_lackey)
        assert info.class_name == "CountLines"

    def test_parses_name(self, sample_lackey):
        info = parse_lackey(sample_lackey)
        assert info.name == "count_lines"

    def test_parses_description(self, sample_lackey):
        info = parse_lackey(sample_lackey)
        assert "Count lines" in info.description

    def test_parses_tools(self, sample_lackey):
        info = parse_lackey(sample_lackey)
        assert "read" in info.tools
        assert "glob" in info.tools

    def test_parses_params(self, sample_lackey):
        info = parse_lackey(sample_lackey)
        assert "pattern" in info.params
        assert info.params["pattern"]["type"] == "str"
        assert info.params["pattern"]["default"] == "**/*.py"
        assert "max_depth" in info.params

    def test_parses_returns(self, sample_lackey):
        info = parse_lackey(sample_lackey)
        assert info.returns == "int"

    def test_extracts_run_body(self, sample_lackey):
        info = parse_lackey(sample_lackey)
        assert "glob" in info.run_body
        assert "read" in info.run_body
        assert "self." not in info.run_body

    def test_has_creation_log(self, sample_lackey):
        info = parse_lackey(sample_lackey)
        assert info.has_creation_log is True

    def test_stores_path(self, sample_lackey):
        info = parse_lackey(sample_lackey)
        assert info.path == sample_lackey


class TestParseLackeyMinimal:
    def test_minimal_lackey(self, tmp_path):
        code = '''
from lackpy.lackey import Lackey, Tool

class Simple(Lackey):
    read = Tool()
    def run(self):
        return self.read("test.txt")
'''
        path = tmp_path / "simple.py"
        path.write_text(code)
        info = parse_lackey(path)
        assert info.class_name == "Simple"
        assert "read" in info.tools
        assert info.params == {}
        assert info.has_creation_log is False
