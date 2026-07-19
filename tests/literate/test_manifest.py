"""L4 — session manifest in every kernel block (batch/session path).

Every rendered segment the session splices into the fed-back document OPENS
with a manifest, emitted through L2's ``[kernel]`` channel (block form — no
new block type). The four fields and their REAL sources:

* segment index — ``LiterateSession._segments``: writer emissions folded so
  far. One ``step()`` = one model call = one budget unit, mirroring the client
  loop (``scripts/literate_agent.py`` spends one ``max_iterations`` iteration
  per call, Left or Right).
* pause budget remaining — ``max_rounds - segments_consumed``, where
  ``max_rounds`` is the client's real ``max_iterations`` cap the writer never
  otherwise sees. Marker-only pause rounds and Lefts consume budget but splice
  nothing, so the printed remaining can drop by more than 1 between splices —
  that is the client loop's true arithmetic, not an off-by-one.
* observations delivered — ``len(session.ledger)`` at splice time: the
  queryable L1 ledger is the source (journal ≈ ledger is WRONG — there is no
  separate journal counter).
* "full history retained by kernel" — fixed affirmation line
  (:data:`annotations.RETENTION_NOTE`).

Because the manifest rides the L2 channel it inherits the round-trip law:
inert on reparse (parser strips channel spans from prose regions) and covered
by strip-stale (render strips prior spans from authored prose before
re-emitting) — so a fed-back or writer-imitated manifest never stacks.
"""

import re

import pytest

from lackpy.interpreters.base import ExecutionContext
from lackpy.interpreters.literate import LiterateSession
from lackpy.interpreters.literate.annotations import RETENTION_NOTE
from lackpy.interpreters.literate.parser import parse


@pytest.fixture
def context(tmp_path):
    return ExecutionContext(base_dir=tmp_path)


MANIFEST_FIELD_RES = (
    r"manifest: segment \d+",
    r"pause budget remaining: ",
    r"observations delivered: \d+ ledger entries",
    re.escape(RETENTION_NOTE),
)


def manifest_blocks(doc: str) -> list[str]:
    """All [kernel] blocks in ``doc`` that contain a manifest header."""
    spans = re.findall(r"\[kernel\].*?\[/kernel\]", doc, re.DOTALL)
    return [s for s in spans if "manifest: segment" in s]


class TestManifestPresentEverySplice:
    @pytest.mark.asyncio
    async def test_manifest_present_every_splice(self, context):
        session = LiterateSession(context, max_rounds=5)
        r1 = await session.step("One.\n\n```lackpy\na = 1\n```\n\n@continue")
        assert r1.ok and r1.continue_requested
        r2 = await session.step("Two.\n\n```lackpy\nb = a + 1\n```\n\n@continue")
        assert r2.ok and r2.continue_requested
        r3 = await session.step("Three.\n\n```lackpy\nc = b + 1\n```")
        assert r3.ok and not r3.continue_requested

        # Every spliced fragment OPENS with a manifest block …
        assert len(session._rendered_parts) == 3
        for fragment in session._rendered_parts:
            assert fragment.startswith("[kernel]\nmanifest: segment ")

        # … and every manifest block carries all four fields.
        blocks = manifest_blocks(session.rendered)
        assert len(blocks) == 3
        for block in blocks:
            for field_re in MANIFEST_FIELD_RES:
                assert re.search(field_re, block), (field_re, block)

        # Segment indices appear in order, one per splice.
        assert re.findall(r"manifest: segment (\d+)", session.rendered) == [
            "1", "2", "3",
        ]


class TestPauseBudgetArithmetic:
    @pytest.mark.asyncio
    async def test_pause_budget_arithmetic_across_multipause(self, context):
        session = LiterateSession(context, max_rounds=4)

        # Segment 1 pauses: 4 - 1 = 3 remaining at the splice.
        r1 = await session.step("R1.\n\n```lackpy\na = 1\n```\n\n@continue")
        assert r1.continue_requested
        assert "pause budget remaining: 3 of 4" in session.rendered
        assert session.budget_remaining == 3

        # Segment 2 is a MARKER-ONLY pause: it consumed a model call (the
        # client loop spends an iteration on it) but splices nothing — so no
        # new manifest, yet the budget dropped.
        r2 = await session.step("@continue")
        assert r2.ok and r2.continue_requested
        assert len(manifest_blocks(session.rendered)) == 1
        assert session.segments_consumed == 2
        assert session.budget_remaining == 2

        # Segment 3 pauses again: 4 - 3 = 1 remaining, shown at its splice.
        r3 = await session.step("R3.\n\n```lackpy\nb = a + 1\n```\n\n@continue")
        assert r3.continue_requested
        assert "manifest: segment 3" in session.rendered
        assert "pause budget remaining: 1 of 4" in session.rendered

        # Segment 4 finishes: 0 remaining — the writer can SEE the budget is
        # spent (surfacing max_iterations is the point of the field).
        r4 = await session.step("R4.\n\n```lackpy\nc = b + 1\n```")
        assert r4.ok and not r4.continue_requested
        assert "pause budget remaining: 0 of 4" in session.rendered

        # The exact decreasing sequence, in document order. Segment 2 (the
        # marker-only pause) spliced nothing, hence 3 -> 1 between splices:
        # correct client-loop arithmetic, every model call consumes.
        remaining = re.findall(
            r"pause budget remaining: (\d+) of 4", session.rendered
        )
        assert remaining == ["3", "1", "0"]

    @pytest.mark.asyncio
    async def test_uncapped_session_says_so_instead_of_inventing_numbers(
        self, context
    ):
        session = LiterateSession(context)  # no max_rounds configured
        await session.step("```lackpy\nx = 1\n```")
        assert (
            "pause budget remaining: unbounded (no round cap configured)"
            in session.rendered
        )
        assert session.budget_remaining is None
        assert session.max_rounds is None


class TestManifestObservationsFromLedger:
    @pytest.mark.asyncio
    async def test_manifest_observations_from_ledger(self, context):
        session = LiterateSession(context, max_rounds=9)

        await session.step("```lackpy\na = 1\n```\n\n@continue")
        n1 = len(session.ledger)
        assert n1 > 0
        assert f"observations delivered: {n1} ledger entries" in session.rendered

        # Round 2 delivers more (an executed cell + prints + a reified hole for
        # the unknown name) — the count GROWS with the ledger, because it IS
        # the ledger's length at each splice, not a separate journal tally.
        await session.step(
            "```lackpy\nb = a + 1\nprint(b)\n```\n\n```lackpy\nq = mystery + 1\n```"
        )
        n2 = len(session.ledger)
        assert n2 > n1

        counts = [
            int(m)
            for m in re.findall(
                r"observations delivered: (\d+) ledger entries", session.rendered
            )
        ]
        assert counts == [n1, n2]

        # Source-of-truth cross-check: the printed counts match direct ledger
        # queries (the same queryable surface AIDR mirrors), including the
        # round-2 hole — a delivered observation, not a swallowed failure.
        assert session.ledger.query(entry_type="hole_opened", name="mystery")
        assert counts[-1] == len(session.ledger.entries())


class TestManifestInertAndNonStacking:
    @pytest.mark.asyncio
    async def test_manifest_inert_and_non_stacking(self, context):
        session = LiterateSession(context, max_rounds=3)
        await session.step("First.\n\n```lackpy\nx = 10\n```\n\n@continue")
        doc1 = session.rendered
        assert len(manifest_blocks(doc1)) == 1

        # INERT: the parser strips the channel span, so no manifest text
        # reaches any cell — it can never re-print as prose or parse as code.
        cells = parse(doc1).cells
        assert all("[kernel]" not in c.content for c in cells)
        assert all("manifest" not in c.content for c in cells)
        assert all(RETENTION_NOTE not in c.content for c in cells)

        # INERT: re-running the manifest-bearing render in a FRESH session
        # binds nothing from the manifest — only the authored binding exists.
        fresh = LiterateSession(context)
        r = await fresh.step(doc1)
        assert r.ok, r.errors
        assert set(fresh.scope) == {"x"}

        # NON-STACKING: round 2's writer both re-echoes the shown doc (overlap
        # guard cuts it) AND imitates round 1's manifest mid-document (parser
        # strip + render strip-stale make it inert). Round 2's fragment must
        # carry ONLY its own manifest.
        imitation = (
            doc1
            + "\nEchoing my notes.\n\n"
            + "[kernel]\nmanifest: segment 1\n"
            + "pause budget remaining: 99 of 100\n"
            + "observations delivered: 12345 ledger entries\n"
            + f"{RETENTION_NOTE}\n[/kernel]\n\n"
            + "Moving on.\n\n```lackpy\ny = x * 2\n```"
        )
        r2 = await session.step(imitation)
        assert r2.ok, r2.errors

        final = session.rendered
        blocks = manifest_blocks(final)
        # Exactly one manifest per splice — round 1's (real) + round 2's own;
        # the imitated segment-1 manifest did NOT survive into round 2, and its
        # fabricated numbers are nowhere in the document.
        assert len(blocks) == 2
        assert final.count("manifest: segment 1") == 1
        assert final.count("manifest: segment 2") == 1
        assert "99 of 100" not in final
        assert "12345" not in final
        assert "manifest: segment 1" not in session._rendered_parts[1]
        assert session._rendered_parts[1].startswith(
            "[kernel]\nmanifest: segment 2"
        )
        assert session._rendered_parts[1].count("manifest: segment") == 1
        # Authored content around the imitation survived.
        assert "Echoing my notes." in final
        assert "Moving on." in final
