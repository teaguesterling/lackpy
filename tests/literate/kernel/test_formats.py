"""Tests for format converters - markdown / Cell / .ipynb."""

import pytest

from lackpy.interpreters.literate.kernel.driver import CellExecutionEvent
from lackpy.interpreters.literate.kernel.formats import (
    from_notebook,
    render_markdown,
    to_notebook,
)
from lackpy.interpreters.literate.kernel.interface import CellResult
from lackpy.interpreters.literate.parser import Cell, Frontmatter


def _event(cell: Cell, index: int, output: str = "", status: str = "executed") -> CellExecutionEvent:
    return CellExecutionEvent(
        cell=cell,
        cell_index=index,
        result=CellResult(
            success=True, output=output, error=None,
            error_phase=None, namespace_delta={}, cell_index=index,
        ),
        status=status,
        generation=0,
    )


class TestToNotebook:
    def test_basic_structure(self):
        log = [
            _event(Cell(cell_type="prose", content="Hello"), 0, output="Hello\n"),
            _event(Cell(cell_type="hidden", content="x = 1"), 1),
        ]
        nb = to_notebook(log, Frontmatter())
        assert nb["nbformat"] == 4
        assert len(nb["cells"]) == 2

    def test_prose_becomes_markdown_cell(self):
        log = [_event(Cell(cell_type="prose", content="# Title"), 0)]
        nb = to_notebook(log, Frontmatter())
        assert nb["cells"][0]["cell_type"] == "markdown"
        assert nb["cells"][0]["source"] == "# Title"

    def test_code_becomes_code_cell(self):
        log = [_event(Cell(cell_type="code", content="x = 42"), 0)]
        nb = to_notebook(log, Frontmatter())
        assert nb["cells"][0]["cell_type"] == "code"
        assert nb["cells"][0]["source"] == "x = 42"

    def test_hidden_becomes_code_with_metadata(self):
        log = [_event(Cell(cell_type="hidden", content="y = 1"), 0)]
        nb = to_notebook(log, Frontmatter())
        assert nb["cells"][0]["cell_type"] == "code"
        assert nb["cells"][0]["metadata"]["lackpy"]["cell_type"] == "hidden"

    def test_output_captured(self):
        log = [_event(Cell(cell_type="code", content="print(42)"), 0, output="42\n")]
        nb = to_notebook(log, Frontmatter())
        assert nb["cells"][0]["outputs"][0]["text"] == "42\n"

    def test_frontmatter_in_metadata(self):
        fm = Frontmatter(echo="true", output="auto")
        nb = to_notebook([], fm)
        assert nb["metadata"]["lackpy"]["frontmatter"]["echo"] == "true"


class TestFromNotebook:
    def test_roundtrip(self):
        log = [
            _event(Cell(cell_type="prose", content="Hello"), 0),
            _event(Cell(cell_type="hidden", content="x = 1"), 1),
            _event(Cell(cell_type="code", content="print(x)"), 2, output="1\n"),
        ]
        fm = Frontmatter(echo="true")
        nb = to_notebook(log, fm)
        recovered_fm, recovered_cells = from_notebook(nb)
        assert recovered_fm.echo == "true"
        assert len(recovered_cells) == 3
        assert recovered_cells[0].cell_type == "prose"
        assert recovered_cells[0].content == "Hello"
        assert recovered_cells[1].cell_type == "hidden"
        assert recovered_cells[2].cell_type == "code"


class TestRenderMarkdown:
    def test_prose_passthrough(self):
        log = [_event(Cell(cell_type="prose", content="Hello world"), 0)]
        md = render_markdown(log, Frontmatter())
        assert "Hello world" in md

    def test_code_in_fence(self):
        log = [_event(Cell(cell_type="code", content="x = 42"), 0)]
        md = render_markdown(log, Frontmatter())
        assert "```lackpy" in md
        assert "x = 42" in md

    def test_hidden_annotated(self):
        log = [_event(Cell(cell_type="hidden", content="y = 1"), 0)]
        md = render_markdown(log, Frontmatter())
        assert "```lackpy @hidden" in md

    def test_write_annotated_with_path(self):
        cell = Cell(cell_type="write", content="code", annotation_args={"path": "out.py"})
        log = [_event(cell, 0)]
        md = render_markdown(log, Frontmatter())
        assert "```lackpy @write(out.py)" in md

    def test_frontmatter_included(self):
        fm = Frontmatter(echo="false", output="json")
        log = [_event(Cell(cell_type="prose", content="Hi"), 0)]
        md = render_markdown(log, fm)
        assert "---" in md
        assert "echo: false" in md

    def test_skipped_cells_omitted(self):
        log = [
            _event(Cell(cell_type="code", content="bad"), 0, status="skipped"),
            _event(Cell(cell_type="prose", content="Good"), 1),
        ]
        md = render_markdown(log, Frontmatter())
        assert "bad" not in md
        assert "Good" in md
