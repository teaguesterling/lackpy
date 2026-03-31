"""Tests for creating Lackey files from generated programs."""

import pytest
from pathlib import Path
from lackpy.lackey.creator import create_lackey_source, save_lackey
from lackpy.lackey.parser import parse_lackey


class TestCreateLackeySource:
    def test_basic_creation(self):
        source = create_lackey_source(
            program="files = glob(pattern)\nlen(files)",
            name="CountFiles", tools=["glob"],
            params={"pattern": {"type": "str", "default": "**/*.py"}},
            returns="int",
        )
        assert "class CountFiles(Lackey):" in source
        assert "glob = Tool()" in source
        assert 'pattern: str = "**/*.py"' in source
        assert "returns: int" in source
        assert "self.glob(self.pattern)" in source

    def test_no_params(self):
        source = create_lackey_source(
            program="content = read('test.py')\ncontent",
            name="ReadTest", tools=["read"],
        )
        assert "class ReadTest(Lackey):" in source
        assert "read = Tool()" in source
        assert "self.read(" in source

    def test_with_creation_log(self):
        source = create_lackey_source(
            program="files = glob('*.py')\nfiles",
            name="FindPy", tools=["glob"],
            creation_log=[
                {"role": "system", "content": "prompt"},
                {"role": "user", "content": "find python files"},
                {"role": "assistant", "content": "files = glob('*.py')\nfiles", "accepted": True},
            ],
        )
        assert "creation_log = Log(" in source
        assert "System(" in source
        assert "User(" in source
        assert "Assistant(" in source

    def test_roundtrip(self, tmp_path):
        source = create_lackey_source(
            program="files = glob(pattern)\nfor f in files:\n    content = read(f)\n    print(f)\nlen(files)",
            name="CountLines", tools=["read", "glob"],
            params={"pattern": {"type": "str", "default": "**/*.py"}},
            returns="int",
        )
        path = tmp_path / "count_lines.py"
        path.write_text(source)

        info = parse_lackey(path)
        assert info.class_name == "CountLines"
        assert "read" in info.tools
        assert "glob" in info.tools
        assert "pattern" in info.params
        assert info.returns == "int"
        assert "glob(pattern)" in info.run_body


class TestSaveLackey:
    def test_saves_file(self, tmp_path):
        path = save_lackey(
            program="files = glob('*.py')\nfiles",
            name="FindPy", tools=["glob"], output_dir=tmp_path,
        )
        assert path.exists()
        assert path.name == "find_py.py"
        assert "class FindPy(Lackey):" in path.read_text()

    def test_filename_from_class_name(self, tmp_path):
        path = save_lackey(
            program="x = 1", name="MyComplexTask",
            tools=[], output_dir=tmp_path,
        )
        assert path.name == "my_complex_task.py"
