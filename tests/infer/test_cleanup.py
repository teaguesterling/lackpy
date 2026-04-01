"""Tests for deterministic cleanup transforms."""

import pytest

from lackpy.infer.cleanup import deterministic_cleanup


class TestStripImports:
    def test_strips_import(self):
        result = deterministic_cleanup("import os\nx = 1")
        assert "import os" not in result
        assert "x = 1" in result

    def test_strips_from_import(self):
        result = deterministic_cleanup("from os.path import join\nx = 2")
        assert "from os.path" not in result
        assert "x = 2" in result

    def test_strips_multiple_imports(self):
        program = "import os\nimport sys\nfrom pathlib import Path\ny = 3"
        result = deterministic_cleanup(program)
        assert "import" not in result
        assert "y = 3" in result

    def test_preserves_non_imports(self):
        program = "x = 1\ny = x + 2"
        result = deterministic_cleanup(program)
        assert "x = 1" in result or "x" in result  # ast.unparse may reformat


class TestRewriteOpen:
    def test_rewrites_open_read(self):
        result = deterministic_cleanup("data = open(path).read()")
        assert "open" not in result
        assert "read(path)" in result

    def test_rewrites_open_readlines(self):
        result = deterministic_cleanup("lines = open(path).readlines()")
        assert "open" not in result
        assert "read(path)" in result
        assert "splitlines()" in result

    def test_open_in_loop(self):
        program = "for f in files:\n    data = open(f).read()"
        result = deterministic_cleanup(program)
        assert "open" not in result
        assert "read(f)" in result

    def test_leaves_read_alone(self):
        result = deterministic_cleanup("data = read(path)")
        assert "read(path)" in result
        assert "open" not in result


class TestRewriteWithOpen:
    def test_with_open_read(self):
        program = "with open(f) as fh:\n    content = fh.read()"
        result = deterministic_cleanup(program)
        assert "open" not in result
        assert "with" not in result
        assert "read(f)" in result

    def test_with_open_read_mode(self):
        program = "with open(f, 'r') as fh:\n    content = fh.read()"
        result = deterministic_cleanup(program)
        assert "open" not in result
        assert "read(f)" in result

    def test_with_open_readlines(self):
        program = "with open(f) as fh:\n    lines = fh.readlines()"
        result = deterministic_cleanup(program)
        assert "open" not in result
        assert "read(f)" in result
        assert "splitlines()" in result

    def test_with_open_in_loop(self):
        program = "for f in files:\n    with open(f) as fh:\n        content = fh.read()\n    print(content)"
        result = deterministic_cleanup(program)
        assert "open" not in result
        assert "with" not in result
        assert "read(f)" in result

    def test_full_model_pattern(self):
        """The exact pattern qwen2.5-coder generates for multi-file reads."""
        program = (
            "files = glob('src/*.py')\n"
            "for f in files:\n"
            "    with open(f, 'r') as fh:\n"
            "        content = fh.read()\n"
            "    name = f.rsplit('/', 1)[-1]\n"
            "    print(f'{name}: {len(content.splitlines())}')"
        )
        result = deterministic_cleanup(program)
        assert "open" not in result
        assert "with" not in result
        assert "read(f)" in result
        assert "glob(" in result

    def test_preserves_non_open_with(self):
        """with statements that aren't open() should be left alone."""
        program = "with ctx() as c:\n    c.do_thing()"
        result = deterministic_cleanup(program)
        assert "with" in result

    def test_with_open_iterate_lines(self):
        program = "with open(f) as fh:\n    for line in fh:\n        print(line)"
        result = deterministic_cleanup(program)
        assert "open" not in result
        assert "read(f)" in result
        assert "splitlines()" in result


class TestRewritePathCalls:
    def test_rewrites_os_path_basename(self):
        program = "import os\nname = os.path.basename(filepath)"
        result = deterministic_cleanup(program)
        assert "os.path.basename" not in result
        assert "rsplit" in result
        assert "-1" in result or "[-1]" in result

    def test_rewrites_os_path_join(self):
        program = "import os\nfull = os.path.join(base, name)"
        result = deterministic_cleanup(program)
        assert "os.path.join" not in result
        # Result should be an f-string containing base and name
        assert "base" in result
        assert "name" in result

    def test_basename_no_import(self):
        # Without import, the AST rewriter still transforms os.path.basename
        program = "name = os.path.basename(p)"
        result = deterministic_cleanup(program)
        assert "os.path.basename" not in result
        assert "rsplit" in result


class TestCombined:
    def test_full_cleanup(self):
        program = (
            "import os\n"
            "from pathlib import Path\n"
            "data = open(filepath).read()\n"
            "name = os.path.basename(filepath)\n"
        )
        result = deterministic_cleanup(program)
        assert "import" not in result
        assert "open" not in result
        assert "os.path.basename" not in result
        assert "read(filepath)" in result
        assert "rsplit" in result

    def test_empty_after_cleanup(self):
        program = "import os\nfrom sys import path"
        result = deterministic_cleanup(program)
        assert result.strip() == ""

    def test_already_clean(self):
        program = "data = read(path)\nname = path.rsplit('/', 1)[-1]"
        result = deterministic_cleanup(program)
        assert "read(path)" in result
        assert "rsplit" in result

    def test_syntax_error_returns_text_cleaned(self):
        program = "import os\ndef invalid syntax here:::"
        result = deterministic_cleanup(program)
        # Should not raise; imports stripped, text-level fallback used
        assert "import os" not in result
