"""Tests for deterministic cleanup transforms.

stepy100 Task 8: a lackpy program imports from the dictionary, so cleanup keeps
approved imports and strips only unapproved ones. ``open(p).read()`` becomes
``Path(p).read_text()`` (adding ``from pathlib import Path``), never the
retired ``read_file(p)``. The kit-era tests asserted that every import went and
that ``read_file(...)`` appeared.
"""

from lackpy.infer.cleanup import deterministic_cleanup
from lackpy.lang.validator import validate

PATH_IMPORT = "from pathlib import Path"


class TestImports:
    def test_strips_unapproved_import(self):
        result = deterministic_cleanup("import os\nx = 1")
        assert "import os" not in result
        assert "x = 1" in result

    def test_strips_unapproved_from_import(self):
        result = deterministic_cleanup("from os.path import join\nx = 2")
        assert "from os.path" not in result
        assert "x = 2" in result

    def test_keeps_approved_imports(self):
        program = "import json\nfrom collections import Counter\nfrom pathlib import Path\ny = 3"
        result = deterministic_cleanup(program)
        assert result == program

    def test_mixed_imports(self):
        program = "import os\nimport sys\nfrom pathlib import Path\ny = 3"
        result = deterministic_cleanup(program)
        assert "import os" not in result
        assert "import sys" not in result
        assert PATH_IMPORT in result

    def test_preserves_non_imports(self):
        program = "x = 1\ny = x + 2"
        assert deterministic_cleanup(program) == program


class TestImportLinesAreNodes:
    """Fix round 1, review I1: imports are removed by their AST line ranges, so
    text inside a string literal is never touched."""

    def test_import_lines_inside_a_string_survive(self):
        program = (
            "from pathlib import Path\n"
            "\n"
            "Path('gen.py').write_text(\'\'\'import os\n"
            "from sys import argv\n"
            "from the start\n"
            "print(1)\n"
            "\'\'\')\n"
        ).rstrip()
        assert validate(program).valid, validate(program).errors
        assert deterministic_cleanup(program) == program

    def test_mixed_import_statement_is_kept_for_the_validator(self):
        # Removing the line would drop json too (a NameError later).
        program = "import os, json\nx = json.dumps(1)"
        assert deterministic_cleanup(program) == program

    def test_import_sharing_a_line_is_kept(self):
        program = "import os; x = 1"
        assert "x = 1" in deterministic_cleanup(program)

    def test_multiline_from_import_removed_whole(self):
        program = "from os.path import (\n    join,\n    basename,\n)\nx = 1"
        assert deterministic_cleanup(program) == "x = 1"


class TestCommentsSurvive:
    def test_intent_block_and_waiver_kept_when_nothing_is_rewritten(self):
        program = (
            "# /// intent\n"
            '# goal   = "count"\n'
            "# ///\n"
            "from pathlib import Path\n"
            "\n"
            'text = Path("a.txt").read_text()  # ste: style.io-position, measured\n'
            "print(len(text))"
        )
        assert deterministic_cleanup(program) == program


class TestRewriteOpen:
    def test_rewrites_open_read(self):
        result = deterministic_cleanup("data = open(path).read()")
        assert "open" not in result
        assert "Path(path).read_text()" in result
        assert PATH_IMPORT in result
        assert "read_file" not in result
        assert validate(result, allowed_names=set()).valid, validate(result).errors

    def test_readlines_is_not_rewritten(self):
        # Fix round 1, review M1: str.splitlines also splits on \x0c, \x85 and
        # others; file lines split on \n only. Left for the validator to report.
        program = "lines = open(path).readlines()"
        assert deterministic_cleanup(program) == program

    def test_open_read_mode_is_rewritten(self):
        result = deterministic_cleanup("data = open(path, 'r').read()")
        assert "Path(path).read_text()" in result

    def test_open_encoding_is_kept(self):
        result = deterministic_cleanup("data = open(path, encoding='utf-8').read()")
        assert "Path(path).read_text(encoding='utf-8')" in result

    def test_binary_open_is_not_rewritten(self):
        # read_text() would return str where the program expects bytes.
        result = deterministic_cleanup("data = open(path, 'rb').read()")
        assert "read_text" not in result

    def test_open_in_loop(self):
        program = "for f in files:\n    data = open(f).read()"
        result = deterministic_cleanup(program)
        assert "open" not in result
        assert "Path(f).read_text()" in result

    def test_path_import_not_duplicated(self):
        program = "from pathlib import Path\ndata = open(p).read()"
        result = deterministic_cleanup(program)
        assert result.count(PATH_IMPORT) == 1

    def test_path_import_comes_first(self):
        result = deterministic_cleanup("x = 1\ndata = open(p).read()")
        assert result.splitlines()[0] == PATH_IMPORT

    def test_leaves_read_alone(self):
        result = deterministic_cleanup("data = read(path)")
        assert "read(path)" in result
        assert "open" not in result
        assert PATH_IMPORT not in result


class TestRewriteWithOpen:
    def test_with_open_read(self):
        program = "with open(f) as fh:\n    content = fh.read()"
        result = deterministic_cleanup(program)
        assert "open" not in result
        assert "with" not in result
        assert "content = Path(f).read_text()" in result
        assert PATH_IMPORT in result

    def test_with_open_read_mode(self):
        program = "with open(f, 'r') as fh:\n    content = fh.read()"
        result = deterministic_cleanup(program)
        assert "open" not in result
        assert "Path(f).read_text()" in result

    def test_with_open_write_mode_is_not_rewritten(self):
        program = "with open(f, 'w') as fh:\n    content = fh.read()"
        result = deterministic_cleanup(program)
        assert "read_text" not in result

    def test_with_open_readlines_is_not_rewritten(self):
        # Fix round 1, review M1.
        program = "with open(f) as fh:\n    lines = fh.readlines()"
        assert deterministic_cleanup(program) == program

    def test_with_open_two_reads_is_not_rewritten(self):
        # Fix round 1, review M2: the second read gives '' under python3.
        program = "with open(f) as fh:\n    a = fh.read()\n    b = fh.read()"
        assert deterministic_cleanup(program) == program

    def test_with_open_other_handle_use_is_not_rewritten(self):
        # Fix round 1, review M2.
        program = "with open(f) as fh:\n    a = fh.read()\n    fh.close()"
        assert deterministic_cleanup(program) == program

    def test_with_open_in_loop(self):
        program = "for f in files:\n    with open(f) as fh:\n        content = fh.read()\n    print(content)"
        result = deterministic_cleanup(program)
        assert "open" not in result
        assert "with" not in result
        assert "Path(f).read_text()" in result

    def test_full_model_pattern(self):
        """The pattern qwen2.5-coder generates for multi-file reads, in Path form."""
        program = (
            "import glob\n"
            "files = sorted(Path('src').glob('*.py'))\n"
            "for f in files:\n"
            "    with open(f, 'r') as fh:\n"
            "        content = fh.read()\n"
            "    print(f'{f.name}: {len(content.splitlines())}')"
        )
        result = deterministic_cleanup(program)
        assert "open" not in result
        assert "with" not in result
        assert "import glob" not in result
        assert "Path(f).read_text()" in result
        assert validate(result, allowed_names=set()).valid, validate(result).errors

    def test_preserves_non_open_with(self):
        """with statements that aren't open() should be left alone."""
        program = "with ctx() as c:\n    c.do_thing()"
        result = deterministic_cleanup(program)
        assert "with" in result

    def test_with_open_iterate_lines_is_not_rewritten(self):
        # Fix round 1, review M1.
        program = "with open(f) as fh:\n    for line in fh:\n        print(line)"
        assert deterministic_cleanup(program) == program


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
        assert "base" in result
        assert "name" in result

    def test_basename_no_import(self):
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
        assert "import os" not in result
        assert result.count(PATH_IMPORT) == 1
        assert "open" not in result
        assert "os.path.basename" not in result
        assert "Path(filepath).read_text()" in result
        assert "rsplit" in result
        assert validate(result, allowed_names=set()).valid, validate(result).errors

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
        # Should not raise; the unapproved import is stripped, text-level fallback used
        assert "import os" not in result
