"""L1 CONFORMANCE SUITE — the ledger semantics as a formal contract.

This module is the L1 acceptance artifact. Under spec E1 and E3 a failure
stops the document at its cell, binds nothing, and is ledgered; the suite
proves the ledger side holds as one contract, in one place:

  1. unbound name      — ``hole_opened``
  2. runtime error     — ``error_reified``
  3. re-assertion      — re-bind      → new version,      ``superseded``
  4. over the ceiling  — ``source_unavailable``

Every semantic is a LEDGER ENTRY — never a control-flow abort — and each
conformance test also asserts what the failure did not bind.

The per-round Either/Left is a VIEW DERIVED FROM THE LEDGER
(``reified_failures`` / ``FORGIVENESS_ENTRY_TYPES``): a round containing a
failure reports Left while the bindings from before the failure stand.

Execution path: batch/session ONLY — ``LiterateSession.step`` and
``LiterateInterpreter.execute`` both funnel into
``LiterateInterpreter._run_document``, which is where L1 lives.  The
streaming driver (and its pause-``pending`` status, and its model-driven
recovery) is a different layer, deliberately untouched by L1.

"""

import pytest

from lackpy.interpreters.base import ExecutionContext
from lackpy.interpreters.literate import LiterateInterpreter, LiterateSession
from lackpy.interpreters.literate.kernel.forgiveness import (
    ERROR_REIFIED,
    FORGIVENESS_ENTRY_TYPES,
    HOLE_OPENED,
    SOURCE_UNAVAILABLE,
    TIMED_OUT,
    reified_failures,
)
from lackpy.interpreters.literate.kernel.versions import SUPERSEDED
from lackpy.lang.grader import Grade
from lackpy.run.paths import load_policy


@pytest.fixture
def context(tmp_path, tmp_home):
    # Literate stepy: cells write through Path under a policy whose workspace
    # is tmp_path (the kit's write_file and host os are gone).
    return ExecutionContext(base_dir=tmp_path, config={
        "path_policy": load_policy(tmp_path, home=tmp_home, managed=None)})


def lackpy_doc(*cells: str) -> str:
    """A literate document of ```lackpy fenced cells, one per argument."""
    return "\n\n".join(f"```lackpy\n{body}\n```" for body in cells)


async def run_batch(context, doc):
    """One batch run (fresh kernel) through the L1-instrumented path."""
    return await LiterateInterpreter().execute(doc, context)


class TestSemantic1UnboundName:
    """Semantic 1 — an unbound name is a failure of its cell.

    Contract: referencing an undefined name does not raise; one
    ``hole_opened`` ledger entry records it, nothing is bound for the cell,
    and the round reports Left through the aggregate view.
    """

    @pytest.mark.asyncio
    async def test_undefined_name_is_a_ledgered_failure(self, context):
        session = LiterateSession(context)

        # No StaticAnalysisError: step returns normally.
        result = await session.step(lackpy_doc("y = x + 1"))

        # Spec E1: the unbound name is a failure, and nothing is bound for it.
        ns = session._kernel.get_namespace()
        assert "x" not in ns and "y" not in ns

        # Ledger entry: exactly one hole_opened for the origin hole.
        entries = session.ledger.query(entry_type=HOLE_OPENED, name="x")
        assert len(entries) == 1
        assert entries[0].detail["reason"] == "unbound"

        # Aggregate layer: the round is Left — derived from the ledger, not
        # thrown; the errors name the name for the caller.
        assert result.ok is False
        assert reified_failures(session.ledger.entries())
        assert any("x" in err for err in result.errors)


class TestSemantic2RuntimeError:
    """Semantic 2 — a runtime error is a failure of its cell.

    Contract: a raising cell stops the run and nothing rolls back — bindings
    made BEFORE the error stand, cells AFTER the error do not run, the failed
    cell binds nothing, and one ``error_reified`` ledger entry records it.
    """

    @pytest.mark.asyncio
    async def test_runtime_error_is_ledgered_and_the_run_stops(self, context):
        session = LiterateSession(context)
        result = await session.step(
            lackpy_doc(
                "before = 'kept'",      # bound before the failure
                "v = 1 / 0",            # raises ZeroDivisionError
                "after = 2",            # spec E1: never runs
            )
        )

        # Nothing is bound for the failed cell.
        ns = session._kernel.get_namespace()
        assert "v" not in ns

        # Ledger entry: the failure is recorded as error_reified, with no name.
        entries = session.ledger.query(entry_type=ERROR_REIFIED)
        assert len(entries) == 1 and entries[0].name is None
        assert "ZeroDivisionError" in entries[0].detail["error"]
        assert entries[0].detail["error_phase"] == "runtime"

        # No rollback: prior bindings kept. The run stopped: later cells did
        # not run.
        assert ns["before"] == "kept"
        assert "after" not in ns

        # Aggregate layer: Left, derived — the state above all stands.
        assert result.ok is False
        assert reified_failures(session.ledger.entries())


class TestSemantic3ReassertionVersioning:
    """Semantic 3 — re-assertion: re-binding a name is a VERSIONED supersession.

    Contract: binding a name that already has a recorded version creates a
    NEW immutable version, marks the prior version ``superseded_by`` it, and
    writes one ``superseded`` ledger entry carrying the transition.  The
    namespace remains the latest-wins view (current == latest).  A
    supersession is a legitimate transition, NOT a failure: the
    round stays Right.
    """

    @pytest.mark.asyncio
    async def test_rebind_versions_and_supersedes(self, context):
        session = LiterateSession(context)
        result = await session.step(lackpy_doc("n = 1", "n = 2"))

        # Version history: two immutable versions, prior marked superseded.
        history = session.versions.history("n")
        assert [v.value for v in history] == [1, 2]
        assert [v.version for v in history] == [1, 2]
        assert history[0].superseded_by == 2
        assert history[0].superseded is True
        assert history[1].superseded_by is None

        # Current = latest, and the namespace agrees (latest-wins view).
        assert session.versions.current("n").value == 2
        assert session._kernel.get_namespace()["n"] == 2

        # Ledger entry: one superseded row naming the transition.
        entries = session.ledger.query(entry_type=SUPERSEDED, name="n")
        assert len(entries) == 1
        assert entries[0].detail["from_version"] == 1
        assert entries[0].detail["to_version"] == 2
        assert entries[0].detail["prior"] == "value"

        # A supersession is not a failure: the round is Right.
        assert result.ok is True
        assert not reified_failures(session.ledger.entries())


class TestSemantic4OverTheCeiling:
    """Semantic 4 — an over-ceiling cell is a failure of its cell.

    Contract: a cell whose statically-classified effects exceed the effect
    ceiling is NEVER executed (refuse-before-running, kept per cell — the
    effect does not happen), it binds nothing, one ``source_unavailable``
    ledger entry records it with the gate's reason, and the document stops
    there.  Deliberately DISTINCT from the streaming driver's
    pause-``pending`` (design conflict #4): the L1.5 vocabulary never uses
    the word "pending", and a ``@continue`` pause on this same path ledgers
    ``continue_requested`` — the two can never be confused in ledger queries.
    """

    @pytest.mark.asyncio
    async def test_over_ceiling_cell_is_a_ledgered_failure(self, context, tmp_path):
        ctx = ExecutionContext(
            base_dir=tmp_path, config={**context.config, "grade_ceiling": Grade(1, 1)}
        )
        result = await run_batch(
            ctx,
            lackpy_doc(
                "from pathlib import Path\ndata = Path('out.txt').write_text('x')",  # over
                "ok = 1 + 1",                          # within — never runs
            ),
        )

        # Spec E1: nothing is bound for the gated cell.
        assert "data" not in result.metadata["variables"]

        # Ledger entry: one source_unavailable row, with the gate's reason.
        ledger = result.metadata["ledger"]
        entries = ledger.query(entry_type=SOURCE_UNAVAILABLE)
        assert len(entries) == 1 and entries[0].name is None
        assert "ceiling" in entries[0].detail["reason"]

        # Refuse-before-running held per cell: the effect never happened,
        # and the run stopped at the gated cell.
        assert not (tmp_path / "out.txt").exists()
        assert "ok" not in result.metadata["variables"]
        assert result.metadata["completed"] is True

        # Aggregate layer: unavailable is a failure → Left.
        assert result.success is False

    @pytest.mark.asyncio
    async def test_distinct_from_driver_pause_pending(self, context):
        # Conflict #4: the unavailable vocabulary never says "pending" ...
        assert "pending" not in SOURCE_UNAVAILABLE
        assert all("pending" not in t for t in FORGIVENESS_ENTRY_TYPES)

        # ... and a @continue pause on this same path is a DIFFERENT ledger
        # concept (continue_requested), a Right round, and no failure.
        session = LiterateSession(context)
        result = await session.step("```lackpy @continue\n```")
        assert result.ok is True and result.continue_requested is True
        assert session.ledger.query(entry_type="continue_requested")
        assert not session.ledger.query(entry_type=SOURCE_UNAVAILABLE)


class TestTwoLayerComposition:
    """Cross-semantic checks: the contract holds as a WHOLE.

    State from before a failure is kept, the failure is ledgered, and Left is
    a derived aggregate signal — these tests pin the seams between semantics.
    """

    @pytest.mark.asyncio
    async def test_left_round_keeps_real_bindings(self, context):
        """A round mixing a real binding and an unbound name reports Left AND
        keeps the real binding — state kept, not rolled back."""
        session = LiterateSession(context)
        result = await session.step(
            lackpy_doc("real = 42", "h = missing_thing")
        )

        # Aggregate layer says Left ...
        assert result.ok is False
        assert reified_failures(session.ledger.entries())

        # ... while the real value stands, versioned. The failed cell binds
        # nothing (spec E1).
        ns = session._kernel.get_namespace()
        assert ns["real"] == 42
        assert session.versions.current("real").value == 42
        assert "h" not in ns

    @pytest.mark.asyncio
    async def test_left_is_exactly_the_failure_types(self, context):
        """The aggregate view's trigger set is pinned: hole_opened /
        error_reified / source_unavailable / timed_out make a round Left
        (timed_out: Plan 3 D7, a cell stopped at its limit);
        superseded, executed, and continue_requested never do."""
        assert FORGIVENESS_ENTRY_TYPES == {
            HOLE_OPENED,
            ERROR_REIFIED,
            SOURCE_UNAVAILABLE,
            TIMED_OUT,
        }
        for benign in (SUPERSEDED, "executed", "continue_requested"):
            assert benign not in FORGIVENESS_ENTRY_TYPES
