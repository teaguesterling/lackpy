"""Tests for StreamingCellParser - incremental fence detection."""

import pytest

from lackpy.interpreters.literate.kernel.streaming_parser import StreamingCellParser


@pytest.fixture
def parser():
    return StreamingCellParser()


class TestBasicParsing:
    def test_prose_only_on_flush(self, parser):
        cells = parser.feed("Hello world")
        assert cells == []
        cells = parser.flush()
        assert len(cells) == 1
        assert cells[0].cell_type == "prose"
        assert cells[0].content == "Hello world"

    def test_code_fence_yields_prose_then_code(self, parser):
        text = "Some prose\n\n```lackpy\nx = 42\n```\n"
        cells = parser.feed(text)
        assert len(cells) == 2
        assert cells[0].cell_type == "prose"
        assert "Some prose" in cells[0].content
        assert cells[1].cell_type == "code"
        assert cells[1].content == "x = 42"

    def test_hidden_annotation(self, parser):
        text = "```lackpy @hidden\nsecret = 1\n```\n"
        cells = parser.feed(text)
        assert len(cells) == 1
        assert cells[0].cell_type == "hidden"
        assert cells[0].content == "secret = 1"

    def test_write_annotation_with_path(self, parser):
        text = "```lackpy @write(out.py)\nprint('hi')\n```\n"
        cells = parser.feed(text)
        assert len(cells) == 1
        assert cells[0].cell_type == "write"
        assert cells[0].annotation_args == {"path": "out.py"}
        assert cells[0].content == "print('hi')"


class TestStreamingBehavior:
    def test_chunked_input(self, parser):
        cells1 = parser.feed("Hello\n\n```lack")
        assert cells1 == []
        cells2 = parser.feed("py\nx = 1\n``")
        assert cells2 == []
        cells3 = parser.feed("`\n")
        assert len(cells3) == 2
        assert cells3[0].cell_type == "prose"
        assert cells3[1].cell_type == "code"
        assert cells3[1].content == "x = 1"

    def test_multiple_fences(self, parser):
        text = (
            "Intro\n\n"
            "```lackpy @hidden\na = 1\n```\n\n"
            "Middle\n\n"
            "```lackpy\nb = a + 1\n```\n"
        )
        cells = parser.feed(text)
        assert len(cells) == 4
        assert cells[0].cell_type == "prose"
        assert cells[1].cell_type == "hidden"
        assert cells[2].cell_type == "prose"
        assert cells[3].cell_type == "code"

    def test_non_lackpy_fence_is_prose(self, parser):
        text = "```python\nx = 1\n```\n"
        cells = parser.feed(text)
        flush = parser.flush()
        all_cells = cells + flush
        assert len(all_cells) == 1
        assert all_cells[0].cell_type == "prose"

    def test_unclosed_fence_on_flush(self, parser):
        parser.feed("```lackpy\nx = 1\n")
        cells = parser.flush()
        assert len(cells) == 1
        assert cells[0].cell_type == "code"
        assert cells[0].content == "x = 1"

    def test_unclosed_fence_on_flush_marked_truncated(self, parser):
        parser.feed("```lackpy\nx = 1\n")
        cells = parser.flush()
        assert len(cells) == 1
        assert cells[0].truncated is True

    def test_closed_fence_not_truncated(self, parser):
        cells = parser.feed("```lackpy\nx = 1\n```\n")
        assert len(cells) == 1
        assert cells[0].truncated is False


class TestFrontmatter:
    def test_frontmatter_consumed(self, parser):
        text = "---\necho: true\noutput: auto\n---\n\nHello"
        parser.feed(text)
        cells = parser.flush()
        assert len(cells) == 1
        assert cells[0].cell_type == "prose"
        assert "Hello" in cells[0].content
        assert parser.frontmatter.echo == "true"

    def test_no_frontmatter(self, parser):
        text = "Just prose"
        parser.feed(text)
        parser.flush()
        assert parser.frontmatter.echo == "true"


class TestAnnotationAutoCorrection:
    def test_hidden_in_body_auto_corrected(self, parser):
        cells = parser.feed("```lackpy\n@hidden\nx = 1\n```\n")
        assert len(cells) == 1
        assert cells[0].cell_type == "hidden"
        assert cells[0].content == "x = 1"

    def test_gather_in_body_auto_corrected(self, parser):
        cells = parser.feed("```lackpy\n@gather\ndata = 42\n```\n")
        assert len(cells) == 1
        assert cells[0].cell_type == "gather"
        assert cells[0].content == "data = 42"

    def test_decorator_not_auto_corrected(self, parser):
        cells = parser.feed("```lackpy\n@decorator\ndef f(): pass\n```\n")
        assert len(cells) == 1
        assert cells[0].cell_type == "code"

    def test_fence_line_annotation_unchanged(self, parser):
        cells = parser.feed("```lackpy @hidden\nx = 1\n```\n")
        assert len(cells) == 1
        assert cells[0].cell_type == "hidden"
        assert cells[0].content == "x = 1"


class TestReset:
    def test_reset_clears_state(self, parser):
        parser.feed("```lackpy\nx = 1\n")
        parser.reset()
        cells = parser.flush()
        assert cells == []
