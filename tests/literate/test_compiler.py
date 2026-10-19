"""Tests for the literate document compiler."""

import pytest

from lackpy.interpreters.literate.compiler import (
    CONTINUE_SENTINEL,
    ProseSegment,
    compile_cells,
    compile_document,
    prose_segments,
)
from lackpy.interpreters.literate.parser import Cell, Frontmatter, ParseResult


def _make_result(cells: list[Cell]) -> ParseResult:
    return ParseResult(frontmatter=Frontmatter(), cells=cells)


def _segs(content: str):
    return prose_segments(Cell(cell_type="prose", content=content))


class TestProseSegments:
    """D3: prose compiles to text and expression segments, not to code.

    The kernel renders a text segment as it is and an expression segment
    through ``CellSession.eval_expr``; nothing here is executed."""

    def test_simple_prose(self):
        assert _segs("Hello world") == [ProseSegment(text="Hello world")]

    def test_prose_with_interpolation(self):
        assert _segs("Value is {x}") == [
            ProseSegment(text="Value is "), ProseSegment(expr="x", display=True)]

    def test_empty_prose(self):
        assert _segs("   \n  ") == []

    def test_prose_with_triple_quotes(self):
        assert _segs('Has """quotes"""') == [ProseSegment(text='Has """quotes"""')]

    def test_prose_with_json_braces_not_interpolated(self):
        assert _segs('Config: {"key": "value"}') == [
            ProseSegment(text='Config: {"key": "value"}')]

    def test_prose_with_attribute_interpolation(self):
        assert _segs("Type: {obj.name}")[1] == ProseSegment(expr="obj.name", display=True)

    def test_prose_with_function_call_interpolation(self):
        assert _segs("Count: {len(items)}")[1] == ProseSegment(expr="len(items)", display=True)

    def test_prose_with_nested_call_interpolation(self):
        assert _segs("Count: {len(inv.items())}")[1] == ProseSegment(
            expr="len(inv.items())", display=True)

    def test_prose_with_format_spec_interpolation(self):
        # A format spec or conversion is applied as Python does: the segment is
        # an f-string, evaluated in the session, and its text is not truncated.
        seg = _segs("Price: {total:.2f}")[1]
        assert seg.display is False
        assert seg.expr.startswith("f") and "total:.2f" in seg.expr

    def test_prose_with_conversion_interpolation(self):
        seg = _segs("S: {s!r}")[1]
        assert seg.display is False and "s!r" in seg.expr

    def test_prose_with_subscript_interpolation(self):
        assert _segs("First: {items[0]}")[1] == ProseSegment(expr="items[0]", display=True)

    def test_every_expression_is_one_line(self):
        for content in ("{a +\n b}", "{chr(10).join([f'{k}\\t{v}' for k, v in d.items()])}"):
            for seg in _segs(content):
                if seg.expr is not None:
                    assert "\n" not in seg.expr

    def test_a_malformed_expression_is_kept_for_the_error_path(self):
        seg = _segs("Bad: {x +}")[1]
        assert seg.expr == "x +"

    def test_compiled_prose_is_analysis_source_only(self):
        # compile_cell(prose) gives the expressions, one per line, for static
        # analysis (references, the dependency graph). It has no print and no
        # display helper: the kernel never execs it.
        code = compile_cells(_make_result([Cell(cell_type="prose", content="A {x} b {len(ys)}")]))
        assert code == "x\nlen(ys)"
        assert "print" not in code and "__literate" not in code

    def test_prose_without_interpolation_compiles_to_nothing(self):
        assert compile_cells(_make_result([Cell(cell_type="prose", content="Hello")])) == ""


class TestSplitInterpolation:
    """Test the brace-matching interpolation splitter directly."""

    def test_no_interpolation(self):
        from lackpy.interpreters.literate.compiler import _split_interpolation
        parts = _split_interpolation("Hello world")
        assert parts == [("Hello world", False)]

    def test_simple_variable(self):
        from lackpy.interpreters.literate.compiler import _split_interpolation
        parts = _split_interpolation("Hello {name}")
        assert parts == [("Hello ", False), ("name", True)]

    def test_nested_braces(self):
        from lackpy.interpreters.literate.compiler import _split_interpolation
        parts = _split_interpolation("Items: {chr(10).join([f'- {x}' for x in items])}")
        assert len(parts) == 2
        assert parts[0] == ("Items: ", False)
        assert parts[1][1] is True
        assert "items" in parts[1][0]
        assert parts[1][0].startswith("chr(10)")

    def test_multiple_expressions(self):
        from lackpy.interpreters.literate.compiler import _split_interpolation
        parts = _split_interpolation("{a} and {b}")
        assert len(parts) == 3
        assert parts[0] == ("a", True)
        assert parts[1] == (" and ", False)
        assert parts[2] == ("b", True)

    def test_json_braces_not_matched(self):
        from lackpy.interpreters.literate.compiler import _split_interpolation
        parts = _split_interpolation('Config: {"key": "value"}')
        assert len(parts) == 1
        assert parts[0][1] is False

    def test_unmatched_brace_treated_as_literal(self):
        from lackpy.interpreters.literate.compiler import _split_interpolation
        parts = _split_interpolation("Incomplete {expr")
        assert all(not is_expr for _, is_expr in parts)

    def test_double_brace_escape(self):
        from lackpy.interpreters.literate.compiler import _split_interpolation
        parts = _split_interpolation("Use {{variable}} in prose")
        assert parts == [("Use {variable} in prose", False)]

    def test_double_brace_escape_with_interpolation(self):
        from lackpy.interpreters.literate.compiler import _split_interpolation
        parts = _split_interpolation("{{escaped}} and {real}")
        assert len(parts) == 2
        assert parts[0] == ("{escaped} and ", False)
        assert parts[1] == ("real", True)

    def test_double_close_brace_escape(self):
        from lackpy.interpreters.literate.compiler import _split_interpolation
        parts = _split_interpolation("Result: }} done")
        assert parts == [("Result: } done", False)]

    def test_unmatched_doubled_open_brace(self):
        from lackpy.interpreters.literate.compiler import _split_interpolation
        parts = _split_interpolation("Text {{}")
        assert len(parts) == 1
        assert parts[0] == ("Text {}", False)

    def test_triple_close_brace(self):
        from lackpy.interpreters.literate.compiler import _split_interpolation
        parts = _split_interpolation("Result: }}}")
        assert all(not is_expr for _, is_expr in parts)
        resolved = "".join(text for text, _ in parts)
        assert resolved == "Result: }}"

    def test_escaped_brace_adjacent_to_expression(self):
        from lackpy.interpreters.literate.compiler import _split_interpolation
        parts = _split_interpolation("{{{x}}}")
        assert parts == [("{", False), ("x", True), ("}", False)]


class TestCodeCompilation:
    def test_code_passthrough(self):
        cells = [Cell(cell_type="code", content="x = 42")]
        code = compile_cells(_make_result(cells))
        assert code.strip() == "x = 42"

    def test_multiline_code(self):
        cells = [Cell(cell_type="code", content="x = 1\ny = 2\nz = x + y")]
        code = compile_cells(_make_result(cells))
        assert "x = 1\ny = 2\nz = x + y" in code


class TestHiddenCompilation:
    def test_hidden_passthrough(self):
        cells = [Cell(cell_type="hidden", content="setup = True")]
        code = compile_cells(_make_result(cells))
        assert code.strip() == "setup = True"


class TestGatherCompilation:
    def test_gather_passthrough(self):
        cells = [Cell(cell_type="gather", content="data = fetch()")]
        code = compile_cells(_make_result(cells))
        assert code.strip() == "data = fetch()"


class TestContinueCompilation:
    def test_continue_emits_sentinel(self):
        cells = [Cell(cell_type="continue", content="")]
        code = compile_cells(_make_result(cells))
        assert CONTINUE_SENTINEL in code


class TestReadCompilation:
    def test_read_compiles_to_print(self):
        cells = [Cell(
            cell_type="read",
            content="",
            annotation_args={"path": "src/main.py"},
        )]
        code = compile_cells(_make_result(cells))
        assert "print(Path('src/main.py').read_text())" in code
        assert "from pathlib import Path" in code


class TestWriteCompilation:
    def test_write_compiles_to_path_write_text(self):
        cells = [Cell(
            cell_type="write",
            content="print('hello')",
            annotation_args={"path": "output.py"},
        )]
        code = compile_cells(_make_result(cells))
        assert code == "from pathlib import Path\nPath('output.py').write_text(\"print('hello')\")"


class TestDiffCompilation:
    def test_diff_compiles_to_a_comment(self):
        # Ruling I1/I2: the kernel applies a diff on the host side (D6). The
        # joined program says so; it binds and uses nothing.
        diff_content = "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new"
        cells = [Cell(
            cell_type="diff",
            content=diff_content,
            annotation_args={"path": "file.py"},
        )]
        code = compile_cells(_make_result(cells))
        assert code == "# @diff(file.py): applied by lackpy, not by this program"


class TestScratchCompilation:
    def test_scratch_compiles_to_its_body(self):
        # The kernel runs the body through the session and writes the
        # [scratch: ...] summary on the host side (no locals(), which the
        # validator rejects).
        cells = [Cell(cell_type="scratch", content="x = 42\ny = 'hello'")]
        assert compile_cells(_make_result(cells)) == "x = 42\ny = 'hello'"


class TestFullDocumentCompilation:
    def test_compile_document_string(self):
        doc = "Hello {name}\n\n```lackpy\nname = 'World'\n```\n\nGoodbye"
        code = compile_document(doc)
        assert code == "name\nname = 'World'"

    def test_compile_document_with_errors_raises(self):
        doc = "```lackpy @read\n```"
        with pytest.raises(ValueError, match="Parse errors"):
            compile_document(doc)

    def test_mixed_document(self):
        doc = """\
---
echo: true
---
# Report

```lackpy @hidden
data = [1, 2, 3]
```

The data has {len(data)} items.

```lackpy
total = sum(data)
total
```

Total: {total}"""

        code = compile_document(doc)
        lines = code.split("\n")
        assert not any("Report" in l for l in lines)  # prose text is not code
        assert any("data = [1, 2, 3]" in l for l in lines)
        assert "len(data)" in lines
        assert any("total = sum(data)" in l for l in lines)
        assert lines[-1] == "total"


class TestCompiledDocumentsRunInTheKernel:
    """The compiled output is never exec'd by the host (anti-goal); these
    documents run through the kernel and its CellSession instead."""

    @staticmethod
    def _render(doc: str) -> str:
        from lackpy.interpreters.literate.parser import parse

        from .conftest import make_kernel

        kernel = make_kernel()
        out = []
        for i, cell in enumerate(parse(doc).cells):
            result = kernel.execute_cell(cell, i)
            assert result.success, result.error
            out.append(result.output or "")
        return "".join(out)

    def test_prose_only_executes(self):
        assert self._render("Hello world") == "Hello world\n"

    def test_code_with_interpolation_executes(self):
        out = self._render("```lackpy @hidden\nx = 42\n```\n\nThe answer is {x}.")
        assert "The answer is 42." in out

    def test_scratch_executes(self):
        out = self._render("```lackpy @scratch\na = 10\nb = 'hi'\n```")
        assert "[scratch: a=int, b=str]" in out

    def test_multiline_prose_with_interpolation(self):
        out = self._render("```lackpy\ncount = 5\n```\n\n# Title\n\nFound {count} items.\n\nDone.")
        assert "Found 5 items." in out

    def test_backslash_in_nested_fstring(self):
        out = self._render("```lackpy\ndata = {'a': 1}\n```\n\n"
                           r"Rows: {chr(10).join([f'{k}\t{v}' for k, v in data.items()])}")
        assert "Rows: a\t1" in out

    def test_prose_with_literal_backslash(self):
        out = self._render("```lackpy\nname = 'me'\n```\n\n" r"Path: C:\Users\{name}")
        assert r"Path: C:\Users\me" in out
