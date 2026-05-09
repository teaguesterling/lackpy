"""Tests for the literate document compiler."""

import pytest

from lackpy.interpreters.literate.compiler import (
    CONTINUE_SENTINEL,
    compile_cells,
    compile_document,
)
from lackpy.interpreters.literate.parser import Cell, Frontmatter, ParseResult


def _make_result(cells: list[Cell]) -> ParseResult:
    return ParseResult(frontmatter=Frontmatter(), cells=cells)


class TestProseCompilation:
    def test_simple_prose(self):
        cells = [Cell(cell_type="prose", content="Hello world")]
        code = compile_cells(_make_result(cells))
        assert "print(" in code and "Hello world" in code

    def test_prose_with_interpolation(self):
        cells = [Cell(cell_type="prose", content="Value is {x}")]
        code = compile_cells(_make_result(cells))
        assert "print(" in code and "{x}" in code

    def test_empty_prose(self):
        cells = [Cell(cell_type="prose", content="   \n  ")]
        code = compile_cells(_make_result(cells))
        assert "print()" in code

    def test_prose_with_triple_quotes(self):
        cells = [Cell(cell_type="prose", content='Has """quotes"""')]
        code = compile_cells(_make_result(cells))
        assert "print(" in code

    def test_prose_with_json_braces_not_interpolated(self):
        cells = [Cell(cell_type="prose", content='Config: {"key": "value"}')]
        code = compile_cells(_make_result(cells))
        assert 'f"""' not in code
        assert "print(" in code

    def test_prose_with_attribute_interpolation(self):
        cells = [Cell(cell_type="prose", content="Type: {obj.name}")]
        code = compile_cells(_make_result(cells))
        assert "{obj.name}" in code

    def test_prose_with_function_call_interpolation(self):
        cells = [Cell(cell_type="prose", content="Count: {len(items)}")]
        code = compile_cells(_make_result(cells))
        assert "{len(items)}" in code

    def test_prose_with_nested_call_interpolation(self):
        cells = [Cell(cell_type="prose", content="Count: {len(inv.items())}")]
        code = compile_cells(_make_result(cells))
        assert "{len(inv.items())}" in code

    def test_prose_with_format_spec_interpolation(self):
        cells = [Cell(cell_type="prose", content="Price: {total:.2f}")]
        code = compile_cells(_make_result(cells))
        assert "{total:.2f}" in code

    def test_prose_with_subscript_interpolation(self):
        cells = [Cell(cell_type="prose", content="First: {items[0]}")]
        code = compile_cells(_make_result(cells))
        assert "{items[0]}" in code


class TestProseCompilationEdgeCases:
    """Edge cases that previously caused SyntaxError with repr()-based compilation."""

    def test_multiline_prose_with_interpolation(self):
        content = "# Title\n\nFound {count} items.\n\nDone."
        cells = [Cell(cell_type="prose", content=content)]
        code = compile_cells(_make_result(cells))
        ns = {"count": 5}
        compiled = compile(code, "<test>", "exec")
        _safe_exec(compiled, ns)

    def test_dict_subscript_with_quotes_in_interpolation(self):
        content = "Value: {data['key']}"
        cells = [Cell(cell_type="prose", content=content)]
        code = compile_cells(_make_result(cells))
        ns = {"data": {"key": 42}, "print": print}
        compiled = compile(code, "<test>", "exec")
        _safe_exec(compiled, ns)

    def test_nested_braces_in_interpolation(self):
        content = "Items: {chr(10).join([f'- {x}' for x in items])}"
        cells = [Cell(cell_type="prose", content=content)]
        code = compile_cells(_make_result(cells))
        ns = {"items": ["a", "b"], "chr": chr, "print": print}
        compiled = compile(code, "<test>", "exec")
        _safe_exec(compiled, ns)

    def test_backslash_in_nested_fstring(self):
        r"""Backslash-n in nested f-string expression should not cause SyntaxError."""
        content = r"Rows: {chr(10).join([f'{k}\t{v}' for k, v in data.items()])}"
        cells = [Cell(cell_type="prose", content=content)]
        code = compile_cells(_make_result(cells))
        ns = {"data": {"a": 1}, "chr": chr, "print": print}
        compiled = compile(code, "<test>", "exec")
        _safe_exec(compiled, ns)

    def test_multiple_interpolations_in_multiline_prose(self):
        content = "Found {total} files totaling {loc} lines.\n\nAverage: {avg} lines per file."
        cells = [Cell(cell_type="prose", content=content)]
        code = compile_cells(_make_result(cells))
        ns = {"total": 100, "loc": 5000, "avg": 50}
        compiled = compile(code, "<test>", "exec")
        _safe_exec(compiled, ns)

    def test_triple_quotes_in_prose_with_interpolation(self):
        content = 'Result is {x} with """emphasis"""'
        cells = [Cell(cell_type="prose", content=content)]
        code = compile_cells(_make_result(cells))
        assert "print(" in code
        compile(code, "<test>", "exec")

    def test_prose_with_literal_backslash(self):
        content = r"Path: C:\Users\{name}"
        cells = [Cell(cell_type="prose", content=content)]
        code = compile_cells(_make_result(cells))
        compile(code, "<test>", "exec")


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
        assert "print(read_file(" in code and "src/main.py" in code


class TestWriteCompilation:
    def test_write_compiles_to_write_file(self):
        cells = [Cell(
            cell_type="write",
            content="print('hello')",
            annotation_args={"path": "output.py"},
        )]
        code = compile_cells(_make_result(cells))
        assert "write_file(" in code and "output.py" in code
        assert "print('hello')" in code


class TestDiffCompilation:
    def test_diff_compiles_to_apply_diff(self):
        diff_content = "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new"
        cells = [Cell(
            cell_type="diff",
            content=diff_content,
            annotation_args={"path": "file.py"},
        )]
        code = compile_cells(_make_result(cells))
        assert "apply_diff(" in code and "file.py" in code
        assert "-old" in code
        assert "+new" in code


class TestScratchCompilation:
    def test_scratch_captures_variables(self):
        cells = [Cell(cell_type="scratch", content="x = 42\ny = 'hello'")]
        code = compile_cells(_make_result(cells))
        assert "_scratch_names_before" in code
        assert "_scratch_names_after" in code
        assert "x = 42" in code
        assert "[scratch:" in code

    def test_scratch_empty_body(self):
        cells = [Cell(cell_type="scratch", content="")]
        code = compile_cells(_make_result(cells))
        assert "pass" in code


class TestFullDocumentCompilation:
    def test_compile_document_string(self):
        doc = "Hello {name}\n\n```lackpy\nname = 'World'\n```\n\nGoodbye"
        code = compile_document(doc)
        assert "print(" in code
        assert "name = 'World'" in code

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
        assert any("print(" in l and "Report" in l for l in lines)
        assert any("data = [1, 2, 3]" in l for l in lines)
        assert any("len(data)" in l for l in lines)
        assert any("total = sum(data)" in l for l in lines)
        assert any("{total}" in l for l in lines)


class TestCompilationExecutable:
    """Verify compiled code actually runs.

    These tests use exec() intentionally to validate that the compiler
    output is syntactically valid and semantically correct Python.
    """

    def _run(self, code: str, namespace: dict | None = None) -> list[str]:
        """Run compiled code, capturing print output."""
        output: list[str] = []
        ns = namespace or {}
        ns["print"] = lambda *a, **kw: output.append(" ".join(str(x) for x in a))
        compiled = compile(code, "<test>", "exec")
        _safe_exec(compiled, ns)
        return output


def _safe_exec(code: object, ns: dict) -> None:
    """Wrapper for exec used in tests — validates compiled literate output."""
    exec(code, ns)  # noqa: S102


class TestCompilationRuns(TestCompilationExecutable):
    def test_prose_only_executes(self):
        doc = "Hello world"
        code = compile_document(doc)
        output = self._run(code)
        assert output == ["Hello world"]

    def test_code_with_interpolation_executes(self):
        doc = "```lackpy @hidden\nx = 42\n```\n\nThe answer is {x}."
        code = compile_document(doc)
        output = self._run(code)
        assert any("42" in line for line in output)

    def test_scratch_executes(self):
        doc = "```lackpy @scratch\na = 10\nb = 'hi'\n```"
        code = compile_document(doc)
        output = self._run(code)
        assert any("[scratch:" in line for line in output)
        assert any("a=int" in line for line in output)
        assert any("b=str" in line for line in output)
