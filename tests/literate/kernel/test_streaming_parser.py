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


class TestComputeTagsStreaming:
    """<compute> is one syntax across both parsers.

    Delimiter knowledge lives in parser.normalize_compute_tags; this parser
    calls it rather than re-implementing tag scanning. What is specific here is
    *partiality*: an unterminated tag must NOT be converted, because the outer
    fence length depends on the whole body (a ``` arriving later needs a longer
    fence), so an early conversion would be frozen wrong.
    """

    def test_complete_compute_block_yields_a_cell(self, parser):
        cells = parser.feed("<compute hidden>\nx = 1\n</compute>\n")
        code = [c for c in cells if c.cell_type != "prose"]
        assert len(code) == 1
        assert code[0].cell_type == "hidden"
        assert code[0].content.strip() == "x = 1"

    def test_tag_split_across_chunks(self, parser):
        assert parser.feed("<comp") == []
        assert parser.feed("ute hidden>\nx = ") == []
        cells = parser.feed("1\n</compute>\n")
        code = [c for c in cells if c.cell_type != "prose"]
        assert len(code) == 1 and code[0].cell_type == "hidden"

    def test_unterminated_tag_is_held_not_emitted(self, parser):
        # Nothing may execute until the block is complete.
        assert parser.feed("Prose.\n\n<compute hidden>\nx = 1\n") == []

    def test_flush_treats_an_unterminated_tag_as_complete(self, parser):
        parser.feed("<compute hidden>\nx = 1\n")
        cells = parser.flush()
        code = [c for c in cells if c.cell_type != "prose"]
        assert len(code) == 1 and code[0].cell_type == "hidden"

    def test_fenced_payload_inside_a_write_block_survives_streaming(self, parser):
        doc = (
            '<compute write="notes.md">\n# Notes\n\n```python\n'
            "def add(a, b):\n    return a + b\n```\n</compute>\n"
        )
        cells = [c for c in parser.feed(doc) if c.cell_type != "prose"]
        assert len(cells) == 1
        assert cells[0].cell_type == "write"
        assert cells[0].annotation_args["path"] == "notes.md"
        assert "def add(a, b):" in cells[0].content
        assert cells[0].content.count("```") == 2

    def test_fences_still_stream(self, parser):
        cells = parser.feed("```lackpy @hidden\nx = 1\n```\n")
        code = [c for c in cells if c.cell_type != "prose"]
        assert len(code) == 1 and code[0].cell_type == "hidden"
