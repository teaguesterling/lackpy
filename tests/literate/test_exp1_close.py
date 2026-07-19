"""L2 exp1 close — the feedback loop renders from the ledger.

The CAPSTONE of the L1/L2 arc. L2 (#46) built the ``[kernel]…[/kernel]``
annotation channel + the round-trip law on the canonical source-preserving
render, but DEFERRED closing exp1 end-to-end: the live session feedback loop
still fed the writer FLAT STDOUT (interpolated prose, printed values, fences
gone), which re-prints and STACKS kernel-imitable strings across rounds — the
exp1 poisoning. L1 (#47–#52) then landed the queryable ledger + forgiveness
values, so there is now an execution log to render from.

This chunk wires them together: ``LiterateSession.rendered`` is now the
canonical source-preserving render, with each round's forgiveness state drawn
FROM THE LEDGER and routed through the SAME L2 ``[kernel]`` channel. These
tests prove the loop is closed:

* :class:`TestExp1FeedbackLoopClosed` — a multi-round session whose rendered
  doc is fed back as the next writer's input; kernel notes do not re-print,
  stack, or poison later rounds.
* :class:`TestFlatStdoutWouldStack` — the red→green: the pre-wiring flat-stdout
  view bakes a printed value in as bare prose (would stack when echoed); the
  source-preserving render feeds back the CODE, never the baked value.
* :class:`TestRoundTripRegression` — the ledger-annotated render reparses with
  zero note-derived cells/bindings; forgiveness reprs never bind.
* :class:`TestComposeWithL1` — a hole filled in a later round shows the
  supersede, not a stacked/stale hole.
"""

import pytest

from lackpy.interpreters.base import ExecutionContext
from lackpy.interpreters.literate import LiterateSession
from lackpy.interpreters.literate.kernel.forgiveness import is_forgiving
from lackpy.interpreters.literate.parser import parse


@pytest.fixture
def context(tmp_path):
    return ExecutionContext(base_dir=tmp_path)


class TestExp1FeedbackLoopClosed:
    """Feed each round's canonical render back as the next writer's input;
    the ledger-derived [kernel] notes must not re-print, stack, or poison."""

    @pytest.mark.asyncio
    async def test_exp1_feedback_loop_closed(self, context):
        session = LiterateSession(context)

        # Round 1: a cell over an unbound name reifies a typed hole (L1.1) →
        # the ledger records it → the canonical render carries an INERT
        # [kernel] hole note (not flat stdout).
        r1 = await session.step(
            "Let me compute the ratio.\n\n```lackpy\nratio = numerator / 100\n```"
        )
        assert not r1.ok  # aggregate Left: a hole was reified this round
        doc1 = session.rendered
        assert "[kernel]" in doc1
        assert "numerator" in doc1  # the hole names its unbound cause
        # L4: each splice also OPENS with one manifest [kernel] block, so the
        # no-accretion invariant is stated on NOTE TEXT (must never stack) and
        # manifests (exactly one per splice), not on a raw [kernel] count.
        assert doc1.count("hole: 'numerator' unbound") == 1
        assert doc1.count("manifest: segment") == 1

        # The fed-back render re-parses with NO [kernel] prose cells (the parser
        # strips the channel span) — the note is inert, it cannot re-print.
        assert all("[kernel]" not in c.content for c in parse(doc1).cells)
        # The authored source survives the round trip (source-preserving).
        assert any(
            c.cell_type != "prose" and "ratio = numerator / 100" in c.content
            for c in parse(doc1).cells
        )

        # Round 2: a STATELESS writer, shown doc1, re-echoes it VERBATIM then
        # adds a new line. The overlap guard cuts the echo; the holed cell is
        # not re-executed and the [kernel] note is not re-printed or stacked.
        await session.step(doc1 + "\n\nStill working on it.")
        assert "Still working on it." in session.rendered
        # No accretion: the hole note still appears exactly once; the only new
        # [kernel] block is round 2's own manifest (one per splice, L4).
        assert session.rendered.count("hole: 'numerator' unbound") == 1
        assert session.rendered.count("manifest: segment") == 2

        # Round 3: the writer IMITATES the kernel — reproducing a [kernel] line
        # MID-document (not a clean prefix, so overlap-strip won't catch it).
        # The parser's channel strip makes it inert regardless.
        imitation = (
            "Here is my summary.\n\n"
            "[kernel] hole: 'numerator' unbound [/kernel]\n\n"
            "Final note."
        )
        await session.step(imitation)
        final = session.rendered
        assert "Here is my summary." in final
        assert "Final note." in final
        # STILL no accretion: the imitated note text did not STACK — it appears
        # exactly once (round 1's real ledger note), the round-3 imitation
        # added no second copy — and manifests stay one-per-splice.
        assert final.count("hole: 'numerator' unbound") == 1
        assert final.count("manifest: segment") == 3

    @pytest.mark.asyncio
    async def test_concatenated_multiround_render_reparses_cleanly(self, context):
        # The claim tested at the boundary we actually ship: the CONCATENATED
        # multi-round `session.rendered` (not a single-round proxy) re-parses to
        # the same cell structure — every authored code fence survives, no
        # [kernel] channel text becomes a cell, and no note re-prints as prose.
        session = LiterateSession(context)
        await session.step("Round one prose.\n\n```lackpy\na = missing + 1\n```")
        await session.step("Round two prose.\n\n```lackpy\nmissing = 5\n```")
        doc = session.rendered

        cells = parse(doc).cells
        code = [c for c in cells if c.cell_type != "prose"]
        # Both authored fences survive as code cells (no gluing / no fence loss).
        assert [c.content for c in code] == ["a = missing + 1", "missing = 5"]
        # Round boundary did not glue the two authored prose lines into one cell.
        prose = [c.content for c in cells if c.cell_type == "prose"]
        assert any("Round one prose." in p and "Round two prose." not in p
                   for p in prose)
        # No channel text and no note text leaked into any cell (never re-prints).
        assert all("[kernel]" not in c.content for c in cells)
        assert all("hole:" not in c.content and "superseded:" not in c.content
                   for c in cells)

        # Re-running the whole rendered doc in a FRESH session: no [kernel] note
        # STACKS (each distinct note appears at most once) and the holed cell's
        # SOURCE re-parses as code — it re-holes / gets reactively filled, it
        # never re-prints the prior note as prose. (The count may LEGITIMATELY
        # rise: collapsing rounds lets L1.4 dirty re-exec fill `a` — that is the
        # reactive engine resolving more, not poisoning.)
        fresh = LiterateSession(context)
        await fresh.step(doc)
        rerun = fresh.rendered
        for note in ("hole: 'missing' unbound",
                     "hole: 'a' blocked by missing"):
            assert rerun.count(note) <= 1  # no duplicate stacking of any note
        # The holed cell's source survived as code (not re-printed as prose).
        assert any(c.cell_type != "prose" and "a = missing + 1" in c.content
                   for c in parse(rerun).cells)


class TestFlatStdoutWouldStack:
    """Red→green: the pre-wiring flat-stdout feedback bakes printed output in
    as bare prose (stacks when echoed); the source-preserving render does not."""

    @pytest.mark.asyncio
    async def test_flat_stdout_bakes_value_but_render_preserves_source(self, context):
        session = LiterateSession(context)
        r1 = await session.step("```lackpy\nprint('TALLY', 6 * 7)\n```")
        assert r1.ok, r1.errors

        # PRE-WIRING (the exp1 bug): the fed-back doc was clean_doc == flat
        # stdout. The printed value is baked in as bare text — fed forward, a
        # writer echoes it and it re-prints / stacks.
        assert "TALLY 42" in r1.clean_doc

        # POST-WIRING: rendered is source-preserving. The CODE is fed back, not
        # the value — so the value is (re)produced by execution, once, and is
        # never accreted into the document as stacked prose.
        doc = session.rendered
        assert "print('TALLY', 6 * 7)" in doc  # source preserved
        assert "TALLY 42" not in doc  # value NOT baked into the document

        # Reparsing the render yields the code cell and no bare-prose value.
        cells = parse(doc).cells
        assert any(
            c.cell_type != "prose" and "print('TALLY', 6 * 7)" in c.content
            for c in cells
        )
        assert all("TALLY 42" not in c.content for c in cells)


class TestRoundTripRegression:
    """The ledger-annotated canonical render reparses cleanly: annotations add
    zero cells/bindings, and forgiveness reprs never reparse as real bindings."""

    @pytest.mark.asyncio
    async def test_ledger_annotations_add_no_cells_or_bindings(self, context):
        session = LiterateSession(context)
        r1 = await session.step("Intro.\n\n```lackpy\nout = undefined_thing + 1\n```")
        assert not r1.ok
        doc = session.rendered
        assert "[kernel]" in doc
        # No forgiveness display repr (⟨…⟩) leaks into the source-preserving
        # render — prose is not interpolated, so a hole's repr never appears.
        assert "⟨" not in doc  # ⟨

        cells = parse(doc).cells
        # The channel notes contributed NO cell content …
        assert all("[kernel]" not in c.content for c in cells)
        # … and NO code: the only code cell is the AUTHORED one.
        code_cells = [c for c in cells if c.cell_type != "prose"]
        assert len(code_cells) == 1
        assert "undefined_thing" in code_cells[0].content

        # Re-run the reparsed render in a FRESH session: the annotations bound
        # nothing — the only forgiving values come from the authored cell, and
        # the unbound name stays a hole (not resurrected by any note repr).
        session2 = LiterateSession(context)
        await session2.step(doc)
        assert is_forgiving(session2._kernel.lookup("undefined_thing"))
        assert is_forgiving(session2._kernel.lookup("out"))

    def test_forgiveness_repr_literal_does_not_bind(self):
        # L2's refusal law, re-verified for the ledger surface: a display repr
        # ⟨name: unbound⟩ pasted into a doc is not parser input that binds — it
        # stays inert prose, defining no code cell.
        doc = "The value is ⟨x: unbound⟩ here.\n"
        cells = parse(doc).cells
        assert all(c.cell_type == "prose" for c in cells)
        assert all("x = " not in c.content for c in cells)


class TestComposeWithL1:
    """A hole in round 1 filled in round 2 shows the supersede correctly in the
    fed-back doc — not a stacked/stale hole annotation."""

    @pytest.mark.asyncio
    async def test_hole_filled_next_round_shows_supersede_not_stacked_hole(
        self, context
    ):
        session = LiterateSession(context)

        # Round 1: reference an unbound name → a hole on the referencing cell.
        r1 = await session.step("```lackpy\nz = base * 2\n```")
        assert not r1.ok
        assert session.ledger.query(entry_type="hole_opened", name="base")
        assert "hole: 'base' unbound" in session.rendered

        # Round 2: DEFINE base — the prior hole binding is superseded (L1.3,
        # hole → real value) and ledgered.
        r2 = await session.step("```lackpy\nbase = 21\n```")
        assert r2.ok, r2.errors
        assert session.ledger.query(entry_type="superseded", name="base")

        final = session.rendered
        # The fed-back doc shows the SUPERSEDE on round 2's cell …
        assert "superseded: 'base'" in final
        # … and does NOT re-stack the stale hole: it appears exactly once (in
        # round 1's section), never re-emitted by round 2.
        assert final.count("hole: 'base' unbound") == 1
