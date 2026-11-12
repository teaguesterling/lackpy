"""L1.5 acceptance: the effect ceiling gate refuses a cell before it runs.

In THIS kernel the concrete "unavailable source" is the EFFECT CEILING GATE
(:mod:`..effects`): a cell whose statically-classified effects exceed the
profile's grade ceiling requires an effect the kernel refuses to run *right
now* — the source is not gone (a raised ceiling / different profile would run
it). Under spec E1 the gate's verdict is a failure of that cell:

* the over-ceiling cell is never executed (the gate's refuse-before-running
  guarantee is kept per cell — its effect never happens);
* one ``source_unavailable`` ledger entry records it (nothing silent), and it
  binds nothing;
* the document stops at that cell, and the round reports Left:
  ``source_unavailable`` is one of the failure entry types.

Conflict #4 (the "pending" name collision, design default DECIDED): the
driver's ``pending`` = "cell not executed due to @continue pause"; the L1.5
concept deliberately uses the word *unavailable* — entry type
``source_unavailable`` — so the two can never be confused in the ledger.

These tests cover the batch/session path only; the streaming driver (and its
pause-``pending`` status) is deliberately untouched.
"""

import pytest

from lackpy.interpreters.base import ExecutionContext
from lackpy.interpreters.literate import LiterateInterpreter
from lackpy.lang.grader import Grade
from lackpy.run.paths import load_policy


def _ctx(base, tmp_home):
    """A read-only ceiling (w=1), and a policy, so a cell that the gate lets
    through could really run."""
    policy = load_policy(base, home=tmp_home, managed=None)
    return ExecutionContext(base_dir=base, config={
        "grade_ceiling": Grade(1, 1), "path_policy": policy})


@pytest.fixture
def interpreter():
    return LiterateInterpreter()


class TestUnavailableSource:
    @pytest.mark.asyncio
    async def test_unavailable_source_is_a_ledgered_failure(self, interpreter, tmp_path, tmp_home):
        """A cell whose source is unavailable (over the effect ceiling) is a
        failure (spec E1): one source_unavailable ledger entry, nothing bound,
        and the document stops there. It is not a document-level refusal."""
        from lackpy.interpreters.literate.kernel.forgiveness import SOURCE_UNAVAILABLE

        ctx = _ctx(tmp_path, tmp_home)
        doc = (
            "```lackpy\nfrom pathlib import Path\ndata = Path('out.txt').write_text('x')\n```\n\n"
            "```lackpy\nok = 1 + 1\n```\n"
        )
        result = await interpreter.execute(doc, ctx)

        ledger = result.metadata["ledger"]
        entries = ledger.query(entry_type=SOURCE_UNAVAILABLE)
        assert len(entries) == 1 and entries[0].name is None
        assert "ceiling" in entries[0].detail["reason"]
        assert "data" not in result.metadata["variables"]

        # Refuse-before-running is kept per cell: the effect never happened.
        assert not (tmp_path / "out.txt").exists()

        # The document stopped: the within-ceiling cell after it did not run.
        assert result.metadata["completed"] is True
        assert "ok" not in result.metadata["variables"]

        assert not result.success
        assert result.error.startswith("cell 0 (from pathlib import Path): "
                                       "effect ceiling exceeded")

    @pytest.mark.asyncio
    async def test_pending_distinct_from_pause_pending(self, interpreter, tmp_path, tmp_home):
        """Conflict #4: the L1.5 concept must not collide with the driver's
        pause status ``"pending"`` ("not executed due to @continue pause") —
        distinct from the ledger entry type."""
        from lackpy.interpreters.literate.kernel.forgiveness import (
            FORGIVENESS_ENTRY_TYPES,
            SOURCE_UNAVAILABLE,
        )

        # The names themselves are distinct from the driver's pause status.
        assert SOURCE_UNAVAILABLE == "source_unavailable"
        assert SOURCE_UNAVAILABLE != "pending"
        assert "pending" not in FORGIVENESS_ENTRY_TYPES

        # And in a real gated run the ledger records source_unavailable rows,
        # never the driver's "pending" status.
        ctx = _ctx(tmp_path, tmp_home)
        result = await interpreter.execute(
            "```lackpy\nfrom pathlib import Path\ndata = Path('o.txt').write_text('x')\n```\n", ctx
        )
        ledger = result.metadata["ledger"]
        assert ledger.query(entry_type=SOURCE_UNAVAILABLE)
        assert ledger.query(entry_type="pending") == []
