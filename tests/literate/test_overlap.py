"""Tests for the stateless-writer overlap-strip guard (L8).

A stateless writer is re-prompted each round with the current document view.
It may begin its emission by re-echoing the tail of that view; without a
guard the echo is re-parsed and re-executed (worse for non-idempotent cells)
and re-printed into the rendered document. ``strip_overlap`` cuts the echoed
prefix before parsing. These tests cover the batch/session execution path
(LiterateSession); the streaming driver consumes one live generation and has
no re-shown view per feed.
"""

import pytest

from lackpy.interpreters.base import ExecutionContext
from lackpy.interpreters.literate import LiterateSession
from lackpy.interpreters.literate.session import strip_overlap


@pytest.fixture
def context(tmp_path):
    return ExecutionContext(base_dir=tmp_path)


class TestStripOverlapUnit:
    def test_exact_overlap_whole_view_reechoed(self):
        # The writer re-emits the ENTIRE document it was shown, then continues.
        # (The view's trailing newline is ignored when matching, so the cut
        # lands after "...42." and the echoed newlines lead the remainder.)
        shown = "# Report\n\nThe count is 42.\n"
        emission = shown + "\nNew content here."
        assert strip_overlap(shown, emission) == "\n\nNew content here."

    def test_partial_overlap_tail_reechoed(self):
        # Only the tail of the view is re-echoed.
        shown = "# Report\n\nThe count is 42.\nDone with part one."
        emission = "Done with part one.\n\nPart two begins."
        assert strip_overlap(shown, emission) == "\n\nPart two begins."

    def test_no_overlap_is_untouched(self):
        shown = "# Report\n\nThe count is 42.\n"
        emission = "Entirely new content."
        assert strip_overlap(shown, emission) == emission

    def test_short_overlap_below_threshold_is_untouched(self):
        # A tiny coincidental match (below min_overlap) must not strip.
        shown = "...ends with the.\n"
        emission = ".\nNew sentence."  # 2-char match "." / ".\n" etc.
        assert strip_overlap(shown, emission) == emission

    def test_whitespace_only_overlap_is_untouched(self):
        shown = "content" + "\n" * 12
        emission = "\n" * 12 + "next"
        assert strip_overlap(shown, emission) == emission

    def test_longest_overlap_wins(self):
        # "b\n" is a suffix of shown AND a shorter "\n" also matches; the
        # longest suffix-of-shown that prefixes the emission must be cut.
        shown = "aaa\nrepeated tail\nrepeated tail\n"
        emission = "repeated tail\nrepeated tail\nfresh"
        assert strip_overlap(shown, emission) == "\nfresh"

    def test_empty_shown_is_untouched(self):
        assert strip_overlap("", "anything at all") == "anything at all"

    def test_full_echo_strips_to_whitespace_only(self):
        shown = "# Report\n\nAll of it.\n"
        assert not strip_overlap(shown, shown).strip()


class TestSessionOverlapGuard:
    @pytest.mark.asyncio
    async def test_reechoed_prose_not_rendered_twice(self, context):
        # exp1 close (L2 conflict #5): the writer is shown the canonical
        # SOURCE-PRESERVING render (session.rendered), so its re-echo is of the
        # source template "The count is {n}." — NOT the interpolated flat stdout
        # "The count is 42." (the old fed-back form). The overlap guard cuts the
        # echo either way; what changed is the fed-back document is now
        # round-trippable. Interpolated values live in clean_doc, per round.
        session = LiterateSession(context)
        r1 = await session.step("```lackpy @hidden\nn = 42\n```\n\nThe count is {n}.")
        assert r1.ok, r1.errors
        assert "The count is 42." in r1.clean_doc                  # interpolated
        assert session.rendered.count("The count is {n}.") == 1    # source, once

        # Round 2: the writer re-echoes the tail of the rendered view (the
        # source form it was shown), then adds new content.
        r2 = await session.step("The count is {n}.\n\nAnd n squared is {n * n}.")
        assert r2.ok, r2.errors
        assert "1764" in r2.clean_doc                              # interpolated
        assert "And n squared is {n * n}." in session.rendered
        assert session.rendered.count("The count is {n}.") == 1    # not stacked

    @pytest.mark.asyncio
    async def test_non_idempotent_cell_not_reexecuted(self, context):
        # Regression: a client that shows the writer the RAW document source
        # (fences included) risks the writer re-echoing a non-idempotent cell.
        # Without the guard the echoed cell re-executes (counter bumps twice).
        session = LiterateSession(context)
        doc1 = (
            "```lackpy @hidden\ncounter = 1\n```\n\n"
            "```lackpy @hidden\ncounter = counter + 1\n```"
        )
        r1 = await session.step(doc1)
        assert r1.ok, r1.errors

        # Writer was shown doc1 verbatim; it re-echoes the tail cell.
        doc2 = (
            "```lackpy @hidden\ncounter = counter + 1\n```\n\n"
            "Counter: {counter}"
        )
        r2 = await session.step(doc2, shown=doc1)
        assert r2.ok, r2.errors
        assert "Counter: 2" in r2.clean_doc  # would be 3 without the guard

    @pytest.mark.asyncio
    async def test_pure_echo_with_no_new_content_is_a_left(self, context):
        session = LiterateSession(context)
        r1 = await session.step("Plain prose round one.")
        assert r1.ok
        r2 = await session.step("Plain prose round one.")
        assert not r2.ok
        assert any("re-echo" in e for e in r2.errors)

    @pytest.mark.asyncio
    async def test_first_round_has_no_view_to_strip(self, context):
        session = LiterateSession(context)
        result = await session.step("First round content.")
        assert result.ok
        assert "First round content." in result.clean_doc
