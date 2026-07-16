"""L2 — round-trip law + annotation-channel grammar.

The round-trip law: *anything the kernel renders MUST be legal parser input.*
Kernel-generated notes travel through the ``[kernel]…[/kernel]`` annotation
channel so a rendered document, fed back to a stateless writer and re-emitted,
re-parses cleanly instead of re-printing and stacking.

Coverage:
  * channel spans are inert on reparse — batch parser, streaming parser, and
    end-to-end through the interpreter (mechanisms a + c);
  * ``render_markdown`` (the canonical, source-preserving render) emits notes
    through the channel and strips stale annotations before re-emitting
    (mechanism b), so re-rendering is a fixpoint — no stacking;
  * a HISTORICAL poisoned transcript (a writer that imitated the kernel's bare
    ``[warning: …]`` / ``[scratch: …]`` output) round-trips clean.

Imports are deliberately limited to stable public symbols and the note text is
inlined, so these tests FAIL (not error) against the pre-L2 code — the red→green
proof for the poisoned-transcript regression.
"""

import pytest

from lackpy.interpreters.base import ExecutionContext
from lackpy.interpreters.literate import LiterateInterpreter
from lackpy.interpreters.literate.kernel.driver import CellExecutionEvent
from lackpy.interpreters.literate.kernel.formats import render_markdown
from lackpy.interpreters.literate.kernel.interface import CellResult
from lackpy.interpreters.literate.kernel.streaming_parser import StreamingCellParser
from lackpy.interpreters.literate.parser import Cell, Frontmatter, parse

# The exact literals the kernel historically emitted, inlined so the poisoned
# fixture depends on no new symbol (fails, not errors, on pre-L2 code). The
# em-dash is intentional — it must match the driver's truncation warning byte
# for byte, which is exactly what makes the scoped legacy strip safe.
TRUNCATION_NOTE = "cell may be truncated — model hit token limit"


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


def _log(cells: list[Cell]) -> list[CellExecutionEvent]:
    return [_event(c, i) for i, c in enumerate(cells)]


@pytest.fixture
def interpreter():
    return LiterateInterpreter()


@pytest.fixture
def context(tmp_path):
    return ExecutionContext(base_dir=tmp_path)


class TestChannelInertOnReparse:
    """[kernel]…[/kernel] spans are ignored by binding parsers (mechanisms a+c)."""

    def test_batch_parser_drops_inline_kernel_span(self):
        doc = "Intro line\n\n[kernel] warning: something happened [/kernel]\n\nMore prose\n"
        cells = parse(doc).cells
        joined = "\n".join(c.content for c in cells)
        assert "[kernel]" not in joined
        assert "warning: something happened" not in joined
        # Authored prose survives — the channel strip is scoped, not greedy.
        assert any("Intro line" in c.content for c in cells)
        assert any("More prose" in c.content for c in cells)

    def test_batch_parser_drops_multiline_kernel_block(self):
        doc = (
            "Before\n\n"
            "[kernel]\n"
            "ledger: 2 holes bound this round\n"
            "budget: 3/5 rounds used\n"
            "[/kernel]\n\n"
            "After\n"
        )
        cells = parse(doc).cells
        joined = "\n".join(c.content for c in cells)
        assert "[kernel]" not in joined
        assert "ledger:" not in joined
        assert "budget:" not in joined
        assert any("Before" in c.content for c in cells)
        assert any("After" in c.content for c in cells)

    def test_kernel_span_does_not_split_code(self):
        doc = "```lackpy\nx = 1\n```\n\n[kernel] note [/kernel]\n"
        cells = parse(doc).cells
        code = [c for c in cells if c.cell_type == "code"]
        assert len(code) == 1
        assert code[0].content == "x = 1"
        assert all("note" not in c.content for c in cells)

    def test_streaming_parser_drops_kernel_span(self):
        p = StreamingCellParser()
        cells = p.feed("Hello\n\n[kernel] scratch: x=int [/kernel]\n\n```lackpy\ny = 2\n```\n")
        cells += p.flush()
        joined = "\n".join(c.content for c in cells)
        assert "[kernel]" not in joined
        assert "scratch: x=int" not in joined
        assert any("Hello" in c.content for c in cells)
        assert any(c.cell_type == "code" and c.content == "y = 2" for c in cells)

    @pytest.mark.asyncio
    async def test_interpreter_ignores_kernel_block(self, interpreter, context):
        # End-to-end: a [kernel] block must NOT re-print as prose. Pre-L2 it
        # parses as prose and compiles to print(...) — this is the inertness the
        # channel guarantees.
        doc = "Real output here\n\n[kernel] warning: injected note [/kernel]\n"
        result = await interpreter.execute(doc, context)
        assert result.success
        assert "Real output here" in result.output
        assert "injected note" not in result.output
        assert "[kernel]" not in result.output


class TestRoundTripLaw:
    """Canonical render → reparse → re-render is a fixpoint (no stacking)."""

    def test_render_emits_truncation_through_channel(self):
        cell = Cell(cell_type="code", content="x = 1", truncated=True)
        md = render_markdown(_log([cell]), Frontmatter())
        # The note rides the channel, not bare prose.
        assert "[kernel]" in md
        assert TRUNCATION_NOTE in md
        assert "x = 1" in md

    def test_channel_note_is_inert_when_rendered_doc_is_reparsed(self):
        cell = Cell(cell_type="code", content="x = 1", truncated=True)
        md1 = render_markdown(_log([cell]), Frontmatter())

        reparsed = parse(md1).cells
        # The kernel note left no prose cell behind.
        assert all(TRUNCATION_NOTE not in c.content for c in reparsed)
        assert any(c.cell_type == "code" and c.content == "x = 1" for c in reparsed)

        # Re-rendering the reparsed doc does NOT re-emit or stack the note
        # (the reparsed code cell is no longer flagged truncated).
        md2 = render_markdown(_log(reparsed), Frontmatter())
        assert md2.count("[kernel]") == 0
        assert TRUNCATION_NOTE not in md2
        assert "x = 1" in md2

    def test_authored_prose_survives_round_trip(self):
        cells = [
            Cell(cell_type="prose", content="An explanation the author wrote."),
            Cell(cell_type="code", content="z = 3"),
        ]
        md = render_markdown(_log(cells), Frontmatter())
        reparsed = parse(md).cells
        joined = "\n".join(c.content for c in reparsed)
        assert "An explanation the author wrote." in joined
        assert "z = 3" in joined


class TestPoisonedTranscript:
    """Historical exp1 failure: a writer imitated the kernel's bare output.

    A stateless writer, shown the kernel's ``[warning: …]`` / ``[scratch: …]``
    stdout, echoed those literals back as prose. Fed forward, they re-printed
    and STACKED every round. The channel + scoped strip make the transcript
    round-trip clean while leaving the author's real content intact.
    """

    POISONED = (
        "Let me analyse the data.\n\n"
        f"[warning: {TRUNCATION_NOTE}]\n\n"
        "```lackpy\n"
        "total = sum(range(10))\n"
        "print(total)\n"
        "```\n\n"
        "[scratch: total=int]\n\n"
        "```lackpy\n"
        "total += 1\n"
        "```\n\n"
        "[scratch: ]\n\n"  # empty form: a scratch cell that bound no NEW name
        "That gives the running total.\n"
    )

    def test_poisoned_transcript_renders_clean(self):
        # Parse the poisoned doc, then render it as the canonical document.
        cells = parse(self.POISONED).cells
        rendered = render_markdown(_log(cells), Frontmatter())

        # The kernel-imitation literals are gone from the rendered document...
        assert f"[warning: {TRUNCATION_NOTE}]" not in rendered
        assert "[scratch: total=int]" not in rendered
        assert "[scratch: ]" not in rendered  # empty-scratch form stripped too

        # ...but every authored line survives.
        assert "Let me analyse the data." in rendered
        assert "total = sum(range(10))" in rendered
        assert "That gives the running total." in rendered

    def test_poisoned_transcript_is_a_render_fixpoint(self):
        # Render, feed back, render again: the literals must not resurrect or
        # stack. This is the regression that proves exp1's poisoning is fixed
        # for the canonical render.
        cells1 = parse(self.POISONED).cells
        rendered1 = render_markdown(_log(cells1), Frontmatter())

        cells2 = parse(rendered1).cells
        rendered2 = render_markdown(_log(cells2), Frontmatter())

        assert rendered2.count("[warning:") == 0
        assert rendered2.count("[scratch:") == 0
        assert rendered1 == rendered2  # fixpoint

    def test_user_authored_bracket_prose_is_not_eaten(self):
        # Scope guard (document is sole source of truth): text that merely
        # resembles a kernel literal but is authored prose must be preserved.
        doc = (
            "[warning: do not delete the config file]\n\n"
            "[scratch: this is my note, not a kernel summary]\n"
        )
        cells = parse(doc).cells
        rendered = render_markdown(_log(cells), Frontmatter())
        assert "do not delete the config file" in rendered
        assert "this is my note, not a kernel summary" in rendered
