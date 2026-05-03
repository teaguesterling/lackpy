"""Tests for literate tool functions."""

import os
from pathlib import Path

import pytest

from lackpy.interpreters.literate.tools import (
    apply_diff,
    make_tool_namespace,
    read_file,
    search_content,
    write_file,
)


@pytest.fixture
def tmp_workspace(tmp_path):
    """Create a temporary workspace with some files."""
    (tmp_path / "hello.txt").write_text("Hello World\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def main():\n    print('hi')\n")
    prev = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(prev)


class TestReadFile:
    def test_read_existing_file(self, tmp_workspace):
        content = read_file("hello.txt")
        assert content == "Hello World\n"

    def test_read_nested_file(self, tmp_workspace):
        content = read_file("src/main.py")
        assert "def main():" in content

    def test_read_missing_file(self, tmp_workspace):
        with pytest.raises(FileNotFoundError):
            read_file("missing.txt")


class TestWriteFile:
    def test_write_new_file(self, tmp_workspace):
        write_file("output.txt", "written content")
        assert (tmp_workspace / "output.txt").read_text() == "written content"

    def test_write_creates_parents(self, tmp_workspace):
        write_file("deep/nested/file.txt", "deep content")
        assert (tmp_workspace / "deep/nested/file.txt").read_text() == "deep content"

    def test_write_overwrites_existing(self, tmp_workspace):
        write_file("hello.txt", "new content")
        assert (tmp_workspace / "hello.txt").read_text() == "new content"


class TestApplyDiff:
    def test_simple_replacement(self, tmp_workspace):
        diff = """\
--- a/hello.txt
+++ b/hello.txt
@@ -1 +1 @@
-Hello World
+Hello Universe"""
        result = apply_diff("hello.txt", diff)
        assert "Hello Universe" in result
        assert "Hello World" not in result

    def test_missing_file_raises(self, tmp_workspace):
        diff = "@@ -1 +1 @@\n-old\n+new"
        with pytest.raises(FileNotFoundError):
            apply_diff("nonexistent.py", diff)

    def test_addition(self, tmp_workspace):
        diff = """\
--- a/src/main.py
+++ b/src/main.py
@@ -1,2 +1,3 @@
 def main():
     print('hi')
+    return 0"""
        result = apply_diff("src/main.py", diff)
        assert "return 0" in result
        assert "print('hi')" in result


class TestSearchContent:
    def test_finds_pattern(self, tmp_workspace):
        result = search_content("def main", "src/")
        assert "main.py" in result
        assert "def main" in result

    def test_no_matches(self, tmp_workspace):
        result = search_content("nonexistent_symbol", "src/")
        assert "no matches" in result


class TestMakeToolNamespace:
    def test_has_all_tools(self):
        ns = make_tool_namespace()
        expected = {"read_file", "write_file", "apply_diff", "search_content", "run_command", "run_tests"}
        assert expected == set(ns.keys())

    def test_tools_are_callable(self):
        ns = make_tool_namespace()
        for tool in ns.values():
            assert callable(tool)
