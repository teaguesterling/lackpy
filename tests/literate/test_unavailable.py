"""L1.5 acceptance: pending / unavailable-source forgiveness.

The brief's fifth semantic: "unavailable source → pending".  In THIS kernel
the concrete "unavailable source" is the EFFECT CEILING GATE
(:mod:`..effects`): a cell whose statically-classified effects exceed the
profile's grade ceiling requires an effect the kernel refuses to run *right
now* — the source is not gone (a raised ceiling / different profile would run
it) and not unknown (that's a Hole) and not a runtime error (that's an
ErrorValue): it is genuinely unavailable.  Under the two-layer contract
(design conflict #2, option (c)) the gate's document-level pre-execution
refusal becomes per-cell binding-layer forgiveness:

* the over-ceiling cell is never executed (the gate's refuse-before-running
  guarantee is kept per cell — its effect never happens);
* the names it would have bound bind an :class:`Unavailable` value, one
  ``source_unavailable`` ledger entry each (nothing silent);
* execution CONTINUES — within-ceiling cells run, no abort, no rollback;
* the round still reports Left (aggregate layer): ``source_unavailable`` is a
  reified failure like ``hole_opened`` / ``error_reified``.

Conflict #4 (the "pending" name collision, design default DECIDED): the
driver's ``pending`` = "cell not executed due to @continue pause"; the L1.5
concept deliberately uses the word *unavailable* — value kind
:class:`Unavailable`, entry type ``source_unavailable`` — so the two can
never be confused in the ledger.

These tests cover the batch/session path only; the streaming driver (and its
pause-``pending`` status) is deliberately untouched.
"""

import pytest

from lackpy.interpreters.base import ExecutionContext
from lackpy.interpreters.literate import LiterateInterpreter, LiterateSession
from lackpy.lang.grader import Grade


@pytest.fixture
def interpreter():
    return LiterateInterpreter()


class TestUnavailableSource:
    @pytest.mark.asyncio
    async def test_unavailable_source_binds_pending(self, interpreter, tmp_path):
        """A cell whose source is unavailable (over the effect ceiling) binds
        an Unavailable value + a source_unavailable ledger entry; the round
        continues — no abort, no document-level refusal."""
        from lackpy.interpreters.literate.kernel.forgiveness import (
            SOURCE_UNAVAILABLE,
            Unavailable,
            is_forgiving,
        )

        ctx = ExecutionContext(
            base_dir=tmp_path, config={"grade_ceiling": Grade(1, 1)}
        )
        doc = (
            "```lackpy\ndata = write_file('out.txt', 'x')\n```\n\n"
            "```lackpy\nok = 1 + 1\n```\n"
        )
        result = await interpreter.execute(doc, ctx)

        # Binding layer: data is reified as Unavailable and ledgered.
        ledger = result.metadata["ledger"]
        entries = ledger.query(entry_type=SOURCE_UNAVAILABLE, name="data")
        assert len(entries) == 1
        assert "ceiling" in entries[0].detail["reason"]

        value = result.metadata["variables"]["data"]
        assert isinstance(value, Unavailable)
        assert is_forgiving(value)
        assert "unavailable" in repr(value)

        # Refuse-before-running is kept per cell: the effect never happened.
        assert not (tmp_path / "out.txt").exists()

        # Execution CONTINUED: the within-ceiling cell ran, no abort.
        assert result.metadata["completed"] is True
        assert result.metadata["variables"]["ok"] == 2

        # Aggregate layer: the round still reports Left.
        assert not result.success
        assert "effect ceiling exceeded" in result.error

    @pytest.mark.asyncio
    async def test_pending_distinct_from_pause_pending(self, interpreter, tmp_path):
        """Conflict #4: the L1.5 concept must not collide with the driver's
        pause status ``"pending"`` ("not executed due to @continue pause") —
        distinct names at both the value-kind and ledger-entry level."""
        from lackpy.interpreters.literate.kernel.forgiveness import (
            FORGIVENESS_ENTRY_TYPES,
            SOURCE_UNAVAILABLE,
            Unavailable,
        )
        from lackpy.interpreters.literate.kernel.versions import value_kind

        # The names themselves are distinct from the driver's pause status.
        assert SOURCE_UNAVAILABLE == "source_unavailable"
        assert SOURCE_UNAVAILABLE != "pending"
        assert "pending" not in FORGIVENESS_ENTRY_TYPES
        assert value_kind(Unavailable(name="x", reason="r")) == "unavailable"

        # And in a real gated run the ledger records source_unavailable rows,
        # never the driver's "pending" status.
        ctx = ExecutionContext(
            base_dir=tmp_path, config={"grade_ceiling": Grade(1, 1)}
        )
        result = await interpreter.execute(
            "```lackpy\ndata = write_file('o.txt', 'x')\n```\n", ctx
        )
        ledger = result.metadata["ledger"]
        assert ledger.query(entry_type=SOURCE_UNAVAILABLE)
        assert ledger.query(entry_type="pending") == []

    @pytest.mark.asyncio
    async def test_pending_chains_and_versions(self, tmp_path):
        """Composition: downstream references to an Unavailable value chain as
        blocked holes (L1.1); a later within-ceiling assertion of the name
        supersedes the Unavailable version with a real value (L1.3)."""
        from lackpy.interpreters.literate.kernel.forgiveness import (
            SOURCE_UNAVAILABLE,
            Unavailable,
        )

        ctx = ExecutionContext(
            base_dir=tmp_path, config={"grade_ceiling": Grade(1, 1)}
        )
        session = LiterateSession(ctx)
        result = await session.step(
            "```lackpy\ndata = write_file('out.txt', 'x')\n```\n\n"
            "```lackpy\nlength = len(data)\n```\n\n"
            "```lackpy\ndata = 'now available'\n```\n"
        )

        ledger = session.ledger

        # Cell 1: gated → data bound Unavailable, ledgered.
        assert ledger.query(entry_type=SOURCE_UNAVAILABLE, name="data")

        # Cell 2: chains — length is a hole blocked by data (L1.1 machinery).
        blocked = ledger.query(entry_type="hole_opened", name="length")
        assert blocked and "data" in blocked[0].detail["blocked_by"]

        # Cell 3: the source became available → versioned supersede (L1.3):
        # v1 (unavailable) superseded by v2 (real value).
        history = session._versions.history("data")
        assert len(history) == 2
        assert isinstance(history[0].value, Unavailable)
        assert history[0].superseded_by == 2
        assert history[1].value == "now available"

        superseded = ledger.query(entry_type="superseded", name="data")
        assert len(superseded) == 1
        assert superseded[0].detail["prior"] == "unavailable"
        assert superseded[0].detail["from_version"] == 1
        assert superseded[0].detail["to_version"] == 2

        # The round is Left (the unavailability + chain are reified failures),
        # but the final binding stands.
        assert not result.ok
        assert session._kernel.lookup("data") == "now available"
