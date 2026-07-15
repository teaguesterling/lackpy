"""Tests for writer-controlled pause interruption (L3).

The fenced ``@continue`` cell -> compiler sentinel -> kernel flag path already
exists and is tested elsewhere. These tests cover the two interruption paths
that catch a pause BEFORE a complete fence exists:

  1. stop-sequence: a client-side StopScanner cuts a *streaming* emission at
     the first ``@continue`` marker (keeping the marker, suppressing matches
     inside <think> blocks) and aborts generation;
  2. textual fallback: ``split_at_continue`` cuts a complete emission at the
     first bare ``@continue`` marker and DISCARDS the remainder
     (reasoning-without-values is protocol-correct to drop).

Path touched: batch/session (LiterateSession.step) plus the thin-client
plumbing the agent script uses. The streaming driver's fenced-@continue pause
is unchanged.
"""

import pytest

from lackpy.interpreters.base import ExecutionContext
from lackpy.interpreters.literate import LiterateSession
from lackpy.interpreters.literate.session import StopScanner, split_at_continue


@pytest.fixture
def context(tmp_path):
    return ExecutionContext(base_dir=tmp_path)


class TestSplitAtContinueUnit:
    def test_no_marker_is_untouched(self):
        doc = "Prose.\n\n```lackpy\nx = 1\n```\n"
        assert split_at_continue(doc) == (doc, False)

    def test_bare_marker_cuts_and_discards_remainder(self):
        doc = "Before.\n\n@continue\n\nAfter is discarded."
        kept, cont = split_at_continue(doc)
        assert cont
        assert "Before." in kept
        assert "@continue" not in kept
        assert "After" not in kept

    def test_fenced_continue_cell_is_left_for_the_sentinel_path(self):
        # A COMPLETE fence is the existing compiler-sentinel path; the textual
        # fallback must not consume it.
        doc = "Before.\n\n```lackpy @continue\n```\n\nAfter."
        assert split_at_continue(doc) == (doc, False)

    def test_trailing_unclosed_continue_fence_pauses(self):
        # Stop-scanner cut shape: generation aborted right after the marker on
        # the fence-open line -- the dangling open line is dropped.
        doc = "Before.\n\n```lackpy @continue"
        kept, cont = split_at_continue(doc)
        assert cont
        assert kept.rstrip() == "Before."

    def test_bare_marker_at_tail_of_unclosed_fence_discards_partial_cell(self):
        # Stop-scanner cut inside an open code fence: the partial cell is
        # reasoning-without-values -- discard it whole rather than auto-close
        # and execute a half-written cell.
        doc = "Before.\n\n```lackpy\nx = compute_something(\n@continue"
        kept, cont = split_at_continue(doc)
        assert cont
        assert kept.rstrip() == "Before."
        assert "compute_something" not in kept

    def test_marker_inside_closed_fence_body_is_not_a_pause(self):
        # A non-first body line saying @continue inside a CLOSED fence is code
        # (it will fail static analysis); the fallback must not eat the doc.
        doc = "```lackpy\nx = 1\n@continue\n```\n\nAfter."
        assert split_at_continue(doc) == (doc, False)

    def test_only_marker_cuts_to_empty(self):
        kept, cont = split_at_continue("@continue")
        assert cont and not kept.strip()


class TestStopScannerUnit:
    def test_no_stop_passes_everything_through(self):
        s = StopScanner(["@continue"])
        assert not s.feed("Hello ")
        assert not s.feed("world")
        assert s.text == "Hello world"
        assert not s.stopped

    def test_stop_cuts_at_marker_and_keeps_it(self):
        s = StopScanner(["@continue"])
        assert s.feed("Before\n\n@continue\n\nJunk after")
        assert s.stopped
        assert s.text == "Before\n\n@continue"

    def test_marker_split_across_chunks(self):
        s = StopScanner(["@continue"])
        assert not s.feed("Doc body\n\n@cont")
        assert s.feed("inue and trailing junk")
        assert s.text == "Doc body\n\n@continue"

    def test_marker_inside_think_block_is_suppressed(self):
        s = StopScanner(["@continue"])
        assert not s.feed("<think>maybe I should @continue here?</think>")
        assert not s.feed("Real doc.")
        assert not s.stopped
        assert s.feed(" @continue")
        assert s.text.endswith("@continue")

    def test_open_think_block_suppresses_until_closed(self):
        s = StopScanner(["@continue"])
        # While the think block is open, nothing is scannable.
        assert not s.feed("<think>pondering @continue")
        # Once it closes, the post-think region is scanned and the marker fires.
        assert s.feed(" more pondering</think>doc @continue tail")
        assert s.stopped
        assert s.text.endswith("doc @continue")

    def test_feed_after_stop_is_ignored(self):
        s = StopScanner(["@continue"])
        s.feed("x@continue")
        assert s.feed("more")
        assert s.text == "x@continue"


class TestSessionPauseFallback:
    @pytest.mark.asyncio
    async def test_bare_marker_pauses_and_discards_remainder(self, context):
        session = LiterateSession(context)
        result = await session.step(
            "```lackpy @hidden\nx = 1\n```\n\nValue: {x}\n\n"
            "@continue\n\n"
            "Discarded: {undefined_name}"
        )
        assert result.ok, result.errors
        assert result.continue_requested
        assert "Value: 1" in result.clean_doc
        assert "@continue" not in result.clean_doc
        assert "Discarded" not in result.clean_doc

    @pytest.mark.asyncio
    async def test_remainder_cells_never_execute(self, context):
        # Non-idempotent discard proof: the cell after the marker must not run.
        session = LiterateSession(context)
        r1 = await session.step(
            "```lackpy @hidden\ncounter = 1\n```\n\n"
            "@continue\n\n"
            "```lackpy @hidden\ncounter = counter + 100\n```"
        )
        assert r1.ok and r1.continue_requested
        r2 = await session.step("Counter: {counter}")
        assert r2.ok, r2.errors
        assert "Counter: 1" in r2.clean_doc

    @pytest.mark.asyncio
    async def test_marker_only_emission_is_a_pause_not_a_left(self, context):
        session = LiterateSession(context)
        result = await session.step("@continue")
        assert result.ok
        assert result.continue_requested

    @pytest.mark.asyncio
    async def test_fenced_continue_still_works(self, context):
        session = LiterateSession(context)
        result = await session.step("```lackpy @continue\n```")
        assert result.ok and result.continue_requested


class TestSegmentLoop:
    """Simulate the thin-client segment loop over both interruption paths."""

    @pytest.mark.asyncio
    async def test_stop_sequence_then_textual_fallback_then_finish(self, context):
        session = LiterateSession(context)

        # --- Segment 1: streaming emission, cut by the stop-sequence path.
        # The 'model' streams a gather + pause, then would ramble on; the
        # scanner aborts at the marker (junk after is never even received).
        scanner = StopScanner(["@continue"])
        chunks = [
            "<think>let me gather first, then @continue</think>",
            "```lackpy @hidden\ndata = [1, 2, 3]\n```\n\n",
            "Gathered {len(data)} items.\n\n@cont",
            "inue",
        ]
        for chunk in chunks:
            if scanner.feed(chunk):
                break
        assert scanner.stopped
        r1 = await session.step(scanner.text)
        assert r1.ok, r1.errors
        assert r1.continue_requested
        assert "Gathered 3 items." in r1.clean_doc

        # --- Segment 2: non-streaming emission, bare marker mid-text; the
        # textual fallback cuts and DISCARDS the remainder.
        r2 = await session.step(
            "The total is {sum(data)}.\n\n@continue\n\nBogus: {sum(data) * 999}"
        )
        assert r2.ok, r2.errors
        assert r2.continue_requested
        assert "The total is 6." in r2.clean_doc
        assert "Bogus" not in r2.clean_doc
        assert "5994" not in r2.clean_doc

        # --- Segment 3: finishes with no pause.
        r3 = await session.step("Done: {max(data)} was the largest.")
        assert r3.ok, r3.errors
        assert not r3.continue_requested

        rendered = session.rendered
        assert "Gathered 3 items." in rendered
        assert "The total is 6." in rendered
        assert "Done: 3 was the largest." in rendered
        assert "@continue" not in rendered
