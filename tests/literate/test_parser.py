"""Tests for the literate document parser."""

from lackpy.interpreters.literate.parser import Cell, Frontmatter, parse


class TestFrontmatter:
    def test_no_frontmatter(self):
        result = parse("Hello world")
        assert result.frontmatter == Frontmatter()
        assert not result.errors

    def test_basic_frontmatter(self):
        doc = "---\necho: false\noutput: all\ninterpreter: python\n---\nHello"
        result = parse(doc)
        assert result.frontmatter.echo == "false"
        assert result.frontmatter.output == "all"
        assert result.frontmatter.interpreter == "python"

    def test_partial_frontmatter(self):
        doc = "---\necho: auto\n---\nHello"
        result = parse(doc)
        assert result.frontmatter.echo == "auto"
        assert result.frontmatter.output == "auto"  # default

    def test_unclosed_frontmatter_treated_as_prose(self):
        doc = "---\necho: false\nHello"
        result = parse(doc)
        assert result.frontmatter == Frontmatter()  # no frontmatter parsed


class TestProseCells:
    def test_prose_only(self):
        result = parse("Hello world\nSecond line")
        assert len(result.cells) == 1
        assert result.cells[0].cell_type == "prose"
        assert result.cells[0].content == "Hello world\nSecond line"

    def test_whitespace_only_not_emitted(self):
        result = parse("   \n\n  ")
        assert len(result.cells) == 0

    def test_prose_between_blocks(self):
        doc = "Before\n\n```lackpy\nx = 1\n```\n\nAfter"
        result = parse(doc)
        types = [c.cell_type for c in result.cells]
        assert types == ["prose", "code", "prose"]
        assert result.cells[0].content == "Before\n"
        assert result.cells[2].content == "\nAfter"


class TestCodeCells:
    def test_basic_code_block(self):
        doc = "```lackpy\nx = 42\n```"
        result = parse(doc)
        assert len(result.cells) == 1
        assert result.cells[0].cell_type == "code"
        assert result.cells[0].content == "x = 42"

    def test_multiline_code_block(self):
        doc = "```lackpy\nx = 1\ny = 2\nz = x + y\n```"
        result = parse(doc)
        assert result.cells[0].content == "x = 1\ny = 2\nz = x + y"

    def test_empty_code_block(self):
        doc = "```lackpy\n```"
        result = parse(doc)
        assert result.cells[0].content == ""

    def test_code_with_options(self):
        doc = "```lackpy echo=false output=all\nx = 1\n```"
        result = parse(doc)
        assert result.cells[0].options == {"echo": "false", "output": "all"}

    def test_unclosed_block(self):
        doc = "```lackpy\nx = 1"
        result = parse(doc)
        assert len(result.cells) == 1
        assert result.cells[0].cell_type == "code"
        assert result.cells[0].content == "x = 1"


class TestAnnotations:
    def test_hidden(self):
        doc = "```lackpy @hidden\nsetup = True\n```"
        result = parse(doc)
        assert result.cells[0].cell_type == "hidden"

    def test_gather(self):
        doc = "```lackpy @gather\ndata = fetch()\n```"
        result = parse(doc)
        assert result.cells[0].cell_type == "gather"

    def test_continue(self):
        doc = "```lackpy @continue\n```"
        result = parse(doc)
        assert result.cells[0].cell_type == "continue"

    def test_scratch(self):
        doc = "```lackpy @scratch\nx = complex_calc()\n```"
        result = parse(doc)
        assert result.cells[0].cell_type == "scratch"

    def test_read_with_path(self):
        doc = '```lackpy @read(src/main.py)\n```'
        result = parse(doc)
        assert result.cells[0].cell_type == "read"
        assert result.cells[0].annotation_args == {"path": "src/main.py"}

    def test_write_with_path(self):
        doc = '```lackpy @write(output.txt)\nfile contents here\n```'
        result = parse(doc)
        assert result.cells[0].cell_type == "write"
        assert result.cells[0].annotation_args == {"path": "output.txt"}
        assert result.cells[0].content == "file contents here"

    def test_diff_with_path(self):
        doc = '```lackpy @diff(src/main.py)\n--- a/src/main.py\n+++ b/src/main.py\n```'
        result = parse(doc)
        assert result.cells[0].cell_type == "diff"
        assert result.cells[0].annotation_args == {"path": "src/main.py"}

    def test_path_annotation_missing_path(self):
        doc = "```lackpy @read\n```"
        result = parse(doc)
        assert len(result.errors) == 1
        assert "requires a path" in result.errors[0]

    def test_read_path_in_body(self):
        """Models often put the path on the first line of the body."""
        doc = "```lackpy @read\nsrc/main.py\n```"
        result = parse(doc)
        assert not result.errors
        assert result.cells[0].cell_type == "read"
        assert result.cells[0].annotation_args == {"path": "src/main.py"}
        assert result.cells[0].content == ""

    def test_write_path_in_body(self):
        doc = "```lackpy @write\noutput.txt\nfile contents here\n```"
        result = parse(doc)
        assert not result.errors
        assert result.cells[0].annotation_args == {"path": "output.txt"}
        assert result.cells[0].content == "file contents here"

    def test_unknown_annotation(self):
        doc = "```lackpy @bogus\nx = 1\n```"
        result = parse(doc)
        assert len(result.errors) == 1
        assert "unknown annotation" in result.errors[0]
        assert result.cells[0].cell_type == "code"  # falls back

    def test_annotation_with_options(self):
        doc = "```lackpy @hidden echo=true\nsetup = True\n```"
        result = parse(doc)
        assert result.cells[0].cell_type == "hidden"
        assert result.cells[0].options == {"echo": "true"}

    def test_annotation_in_body_produces_error(self):
        doc = "```lackpy\n@hidden\nx = 1\n```"
        result = parse(doc)
        assert len(result.errors) == 1
        assert "inside code body" in result.errors[0]
        assert "```lackpy @hidden" in result.errors[0]

    def test_annotation_in_body_with_path(self):
        doc = "```lackpy\n@read(src/main.py)\n```"
        result = parse(doc)
        assert len(result.errors) == 1
        assert "inside code body" in result.errors[0]
        assert "```lackpy @read" in result.errors[0]

    def test_annotation_in_body_not_triggered_for_decorators(self):
        doc = "```lackpy\n@decorator\ndef func(): pass\n```"
        result = parse(doc)
        assert not result.errors


class TestLineNumbers:
    def test_prose_line_numbers(self):
        doc = "Line one\n\n```lackpy\nx = 1\n```\n\nLine seven"
        result = parse(doc)
        assert result.cells[0].line_number == 1  # prose starts at line 1
        assert result.cells[1].line_number == 3  # code fence at line 3

    def test_frontmatter_offset(self):
        doc = "---\necho: true\n---\nProse here\n\n```lackpy\nx = 1\n```"
        result = parse(doc)
        assert result.cells[0].line_number == 4  # prose after frontmatter
        assert result.cells[1].line_number == 6  # code fence


class TestFullDocument:
    def test_mixed_document(self):
        doc = """\
---
echo: true
output: auto
---
# Analysis Report

```lackpy @hidden
data = load_data("input.csv")
```

The dataset has {len(data)} rows.

```lackpy
summary = analyze(data)
summary
```

## Results

The analysis shows: {summary}"""

        result = parse(doc)
        assert not result.errors
        assert result.frontmatter.echo == "true"
        types = [c.cell_type for c in result.cells]
        assert types == ["prose", "hidden", "prose", "code", "prose"]

    def test_gather_continue_pattern(self):
        doc = """\
```lackpy @gather
files = list_files("src/")
```

```lackpy @gather
tests = run_tests()
```

```lackpy @continue
```

Based on the gathered information..."""

        result = parse(doc)
        assert not result.errors
        types = [c.cell_type for c in result.cells]
        assert types == ["gather", "gather", "continue", "prose"]
