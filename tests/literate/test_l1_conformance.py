"""L1 CONFORMANCE SUITE — the forgiveness semantics as a formal contract.

This module is the L1 acceptance artifact (design-lackpy-L1: "Acceptance:
conformance suite (one test per semantic) + chained-hole").  The five
forgiveness semantics are implemented across L1.0–L1.5; this suite proves
they hold TOGETHER, as one contract, in one place:

  1. typed hole        — unknown name → ``Hole`` bound,   ``hole_opened``
  2. error-as-value    — runtime error → ``ErrorValue``,  ``error_reified``
  3. re-assertion      — re-bind      → new version,      ``superseded``
  4. unavailable       — over ceiling → ``Unavailable``,  ``source_unavailable``
  5. dirty-subgraph    — upstream re-assert → dependents re-run, ``dirty``

Every semantic is a FIRST-CLASS VALUE **plus** a LEDGER ENTRY — never a
control-flow abort — so each conformance test asserts BOTH sides: the reified
value in the namespace/version history AND the queryable ledger row.

The two-layer contract (design conflict #2, DECIDED option (c), Teague
2026-07-17) binds the suite together:

* **Binding layer (default mechanism):** failures reify as bound values,
  ledgered; execution continues; nothing rolls back.
* **Aggregate layer (preserved signal):** the per-round Either/Left survives
  as a VIEW DERIVED FROM THE LEDGER (``round_is_left`` /
  ``FORGIVENESS_ENTRY_TYPES``) — a round containing any reified failure still
  reports Left while all its bindings stand.

Execution path: batch/session ONLY — ``LiterateSession.step`` and
``LiterateInterpreter.execute`` both funnel into
``LiterateInterpreter._run_document``, which is where L1 lives.  The
streaming driver (and its pause-``pending`` status, and its model-driven
recovery) is a different layer, deliberately untouched by L1.

The headline acceptance is :class:`TestChainedHoleAcceptance` — a hole in
cell A blocks dependent cell B, B becomes a blocked value too, and BOTH are
ledgered.
"""

import pytest

from lackpy.interpreters.base import ExecutionContext
from lackpy.interpreters.literate import LiterateInterpreter, LiterateSession
from lackpy.interpreters.literate.kernel.forgiveness import (
    ERROR_REIFIED,
    FORGIVENESS_ENTRY_TYPES,
    HOLE_OPENED,
    SOURCE_UNAVAILABLE,
    ErrorValue,
    Hole,
    Unavailable,
    is_forgiving,
    round_is_left,
)
from lackpy.interpreters.literate.kernel.reactive import DIRTY
from lackpy.interpreters.literate.kernel.versions import SUPERSEDED
from lackpy.lang.grader import Grade


@pytest.fixture
def context(tmp_path):
    return ExecutionContext(base_dir=tmp_path)


def lackpy_doc(*cells: str) -> str:
    """A literate document of ```lackpy fenced cells, one per argument."""
    return "\n\n".join(f"```lackpy\n{body}\n```" for body in cells)


async def run_batch(context, doc):
    """One batch run (fresh kernel) through the L1-instrumented path."""
    return await LiterateInterpreter().execute(doc, context)


class TestSemantic1TypedHole:
    """Semantic 1 — typed hole: an unknown name BINDS ⟨name: unbound⟩.

    Contract: referencing an undefined name does not raise and does not
    block the round; the name binds a ``Hole``, one ``hole_opened`` ledger
    entry records it, and the round reports Left through the aggregate view
    only.
    """

    @pytest.mark.asyncio
    async def test_undefined_name_binds_hole_and_is_ledgered(self, context):
        session = LiterateSession(context)

        # No StaticAnalysisError, no abort: step returns normally.
        result = await session.step(lackpy_doc("y = x + 1"))

        # Reified value: x is a typed hole, bound in the live namespace.
        ns = session._kernel.get_namespace()
        assert ns["x"] == Hole("x")
        assert repr(ns["x"]) == "⟨x: unbound⟩"
        assert is_forgiving(ns["x"])

        # Ledger entry: exactly one hole_opened for the origin hole.
        entries = session.ledger.query(entry_type=HOLE_OPENED, name="x")
        assert len(entries) == 1
        assert entries[0].detail["reason"] == "unbound"

        # Aggregate layer: the round is Left — derived from the ledger, not
        # thrown; the errors name the hole for the caller.
        assert result.ok is False
        assert round_is_left(session.ledger.entries())
        assert any("x" in err for err in result.errors)


class TestSemantic2ErrorAsValue:
    """Semantic 2 — error-as-value: a runtime error BINDS an ``ErrorValue``.

    Contract: a raising cell does not abort the run and nothing rolls back —
    bindings made BEFORE the error stand, cells AFTER the error still run,
    the failed cell's unreached names bind ``ErrorValue``s carrying the
    exception, and one ``error_reified`` ledger entry records each.
    """

    @pytest.mark.asyncio
    async def test_runtime_error_reifies_and_run_continues(self, context):
        session = LiterateSession(context)
        result = await session.step(
            lackpy_doc(
                "before = 'kept'",      # bound before the failure
                "v = 1 / 0",            # raises ZeroDivisionError
                "after = 2",            # must still execute — no abort
            )
        )

        # Reified value: v is an ErrorValue carrying the exception info.
        ns = session._kernel.get_namespace()
        assert isinstance(ns["v"], ErrorValue)
        assert "ZeroDivisionError" in ns["v"].error
        assert ns["v"].error_phase == "runtime"

        # Ledger entry: the failure is recorded as error_reified.
        entries = session.ledger.query(entry_type=ERROR_REIFIED, name="v")
        assert len(entries) == 1
        assert "ZeroDivisionError" in entries[0].detail["error"]

        # No rollback, no abort: prior bindings kept, later cells ran.
        assert ns["before"] == "kept"
        assert ns["after"] == 2

        # Aggregate layer: Left, derived — the state above all stands.
        assert result.ok is False
        assert round_is_left(session.ledger.entries())


class TestSemantic3ReassertionVersioning:
    """Semantic 3 — re-assertion: re-binding a name is a VERSIONED supersession.

    Contract: binding a name that already has a recorded version creates a
    NEW immutable version, marks the prior version ``superseded_by`` it, and
    writes one ``superseded`` ledger entry carrying the transition.  The
    namespace remains the latest-wins view (current == latest).  A
    supersession is a legitimate transition, NOT a reified failure: the
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
        assert not round_is_left(session.ledger.entries())


class TestSemantic4UnavailableSource:
    """Semantic 4 — unavailable source: an over-ceiling cell BINDS ``Unavailable``.

    Contract: a cell whose statically-classified effects exceed the effect
    ceiling is NEVER executed (refuse-before-running, kept per cell — the
    effect does not happen), the names it would have bound bind
    ``Unavailable`` values carrying the gate's reason, one
    ``source_unavailable`` ledger entry records each, and the rest of the
    document still runs.  Deliberately DISTINCT from the streaming driver's
    pause-``pending`` (design conflict #4): the L1.5 vocabulary never uses
    the word "pending", and a ``@continue`` pause on this same path ledgers
    ``continue_requested`` — the two can never be confused in ledger queries.
    """

    @pytest.mark.asyncio
    async def test_over_ceiling_cell_binds_unavailable(self, context, tmp_path):
        ctx = ExecutionContext(
            base_dir=tmp_path, config={"grade_ceiling": Grade(1, 1)}
        )
        result = await run_batch(
            ctx,
            lackpy_doc(
                "data = write_file('out.txt', 'x')",  # over the ceiling
                "ok = 1 + 1",                          # within — must run
            ),
        )

        # Reified value: data is Unavailable, carrying the gate's reason.
        data = result.metadata["variables"]["data"]
        assert isinstance(data, Unavailable)
        assert is_forgiving(data)
        assert "ceiling" in data.reason

        # Ledger entry: one source_unavailable row for the name.
        ledger = result.metadata["ledger"]
        entries = ledger.query(entry_type=SOURCE_UNAVAILABLE, name="data")
        assert len(entries) == 1
        assert "ceiling" in entries[0].detail["reason"]

        # Refuse-before-running held per cell: the effect never happened,
        # and execution continued past the gated cell.
        assert not (tmp_path / "out.txt").exists()
        assert result.metadata["variables"]["ok"] == 2
        assert result.metadata["completed"] is True

        # Aggregate layer: unavailable is a reified failure → Left.
        assert result.success is False

    @pytest.mark.asyncio
    async def test_distinct_from_driver_pause_pending(self, context):
        # Conflict #4: the unavailable vocabulary never says "pending" ...
        assert "pending" not in SOURCE_UNAVAILABLE
        assert all("pending" not in t for t in FORGIVENESS_ENTRY_TYPES)

        # ... and a @continue pause on this same path is a DIFFERENT ledger
        # concept (continue_requested), a Right round, and no reified value.
        session = LiterateSession(context)
        result = await session.step("```lackpy @continue\n```")
        assert result.ok is True and result.continue_requested is True
        assert session.ledger.query(entry_type="continue_requested")
        assert not session.ledger.query(entry_type=SOURCE_UNAVAILABLE)


class TestSemantic5DirtySubgraph:
    """Semantic 5 — dirty-subgraph: re-asserting upstream re-runs ONLY dependents.

    Contract: a re-assertion (an L1.3 ``superseded`` entry) marks the
    transitive dependents of the old value stale; each is ledgered with a
    ``dirty`` entry (nothing silent).  An EFFECT-FREE dependent is
    re-executed against the updated namespace (``reexecuted=True``, new
    version); an EFFECTFUL dependent is withheld (``reexecuted=False`` + a
    reason) so re-execution can never replay a world effect; an INDEPENDENT
    cell is untouched — never dirtied, never re-run.  ``dirty`` is
    bookkeeping, not a failure: it does not make the run Left.
    """

    @pytest.mark.asyncio
    async def test_reassert_reexecutes_pure_withholds_effectful(
        self, context, tmp_path
    ):
        result = await run_batch(
            context,
            lackpy_doc(
                "a = 1",                                  # 0: upstream
                "b = a + 1",                              # 1: pure dependent
                "w = write_file('log.txt', str(a))",      # 2: effectful dependent
                "c = 99",                                 # 3: independent
                "a = 2",                                  # 4: re-assertion
            ),
        )
        ledger = result.metadata["ledger"]
        versions = result.metadata["versions"]

        # The re-assertion itself was versioned (semantic 3 feeds semantic 5).
        assert ledger.query(entry_type=SUPERSEDED, name="a")

        # Ledger: both dependents dirtied, in dependency order — and ONLY them.
        dirty = ledger.query(entry_type=DIRTY)
        by_cell = {e.detail["cell_index"]: e for e in dirty}
        assert sorted(by_cell) == [1, 2]

        # Pure dependent re-executed: b refreshed 2 → 3, a new version.
        assert by_cell[1].detail["reexecuted"] is True
        assert by_cell[1].detail["triggered_by"] == ["a"]
        assert result.metadata["variables"]["b"] == 3
        assert [v.value for v in versions.history("b")] == [2, 3]

        # Effectful dependent withheld: ledgered but NOT re-run, with reason —
        # the file was written exactly once, with the ORIGINAL upstream value.
        assert by_cell[2].detail["reexecuted"] is False
        assert "effect" in by_cell[2].detail["reason"]
        assert (tmp_path / "log.txt").read_text() == "1"

        # Independent cell untouched: never dirtied, single version stands.
        assert result.metadata["variables"]["c"] == 99
        assert len(versions.history("c")) == 1

        # dirty/superseded are bookkeeping, not failures: the run is Right.
        assert result.success is True
        assert not round_is_left(ledger.entries())


class TestChainedHoleAcceptance:
    """THE L1 acceptance (design-lackpy-L1): the chained hole.

    A hole in cell A must BLOCK dependent cell B — B's binding becomes a
    blocked (chained) hole rather than a crash or a bogus real value — and
    BOTH A and B must be ledgered.  This is the composite that proves the
    binding layer propagates forgiveness through the document instead of
    merely tolerating one failure.
    """

    @pytest.mark.asyncio
    async def test_hole_in_a_blocks_dependent_b_both_ledgered(self, context):
        session = LiterateSession(context)
        result = await session.step(
            lackpy_doc(
                "a = missing + 1",   # cell A: 'missing' is an origin hole
                "b = a * 2",         # cell B: depends on A → chained hole
            )
        )
        ns = session._kernel.get_namespace()
        ledger = session.ledger

        # The origin: 'missing' is a typed hole, ledgered.
        assert ns["missing"] == Hole("missing")
        assert ledger.query(entry_type=HOLE_OPENED, name="missing")

        # Cell A's binding is a CHAINED hole naming its upstream cause ...
        assert isinstance(ns["a"], Hole)
        assert ns["a"].blocked_by == ("missing",)
        a_rows = ledger.query(entry_type=HOLE_OPENED, name="a")
        assert len(a_rows) == 1
        assert a_rows[0].detail["blocked_by"] == ["missing"]

        # ... and cell B chains in turn: blocked by a, never executed —
        # a hole was statically skipped, it never reached `*`.
        assert isinstance(ns["b"], Hole)
        assert ns["b"].blocked_by == ("a",)
        b_rows = ledger.query(entry_type=HOLE_OPENED, name="b")
        assert len(b_rows) == 1
        assert b_rows[0].detail["blocked_by"] == ["a"]

        # BOTH cells are in the ledger (distinct cell indices), nothing silent.
        assert a_rows[0].detail["cell_index"] == 0
        assert b_rows[0].detail["cell_index"] == 1

        # No crash; the round reports Left through the aggregate view.
        assert result.ok is False
        assert round_is_left(ledger.entries())


class TestTwoLayerComposition:
    """Cross-semantic checks: the two-layer contract holds as a WHOLE.

    Conflict #2 option (c) in one sentence: forgiveness is the default
    mechanism (state kept, failures reified + ledgered) and Left is a
    derived aggregate signal — these tests pin the seams between semantics.
    """

    @pytest.mark.asyncio
    async def test_left_round_keeps_real_bindings(self, context):
        """A round mixing a real binding and a hole reports Left AND keeps
        the real binding — state kept, not rolled back (the two layers are
        independent: the signal does not undo the mechanism)."""
        session = LiterateSession(context)
        result = await session.step(
            lackpy_doc("real = 42", "h = missing_thing")
        )

        # Aggregate layer says Left ...
        assert result.ok is False
        assert round_is_left(session.ledger.entries())

        # ... while the binding layer kept everything: the real value stands,
        # versioned, alongside the reified hole.
        ns = session._kernel.get_namespace()
        assert ns["real"] == 42
        assert session.versions.current("real").value == 42
        assert isinstance(ns["h"], Hole)

    @pytest.mark.asyncio
    async def test_hole_filled_later_is_versioned_supersede(self, context):
        """L1.1 × L1.3: filling a hole is a versioned transition — the hole
        is version 1, the real value version 2, the prior kind is recorded
        as "hole" in the superseded entry, and the filling round is Right."""
        session = LiterateSession(context)

        left = await session.step(lackpy_doc("y = x + 1"))  # x: origin hole
        assert left.ok is False

        mark = len(session.ledger)
        right = await session.step(lackpy_doc("x = 5"))     # fill the hole

        # The filling round itself is clean: Right, per-round slice not Left.
        assert right.ok is True
        assert not round_is_left(session.ledger.entries()[mark:])

        # Version history shows hole → value as an immutable supersession.
        history = session.versions.history("x")
        assert isinstance(history[0].value, Hole)
        assert history[1].value == 5
        assert history[0].superseded_by == 2

        # The superseded ledger entry names the transition and the prior kind.
        rows = session.ledger.query(entry_type=SUPERSEDED, name="x")
        assert len(rows) == 1
        assert rows[0].detail["prior"] == "hole"
        assert rows[0].detail["from_version"] == 1
        assert rows[0].detail["to_version"] == 2

        # Current = the real value, in both the history and the namespace.
        assert session.versions.current("x").value == 5
        assert session._kernel.get_namespace()["x"] == 5

    @pytest.mark.asyncio
    async def test_left_is_exactly_the_reified_failure_types(self, context):
        """The aggregate view's trigger set is pinned: hole_opened /
        error_reified / source_unavailable make a round Left; superseded,
        dirty, executed, and continue_requested never do."""
        assert FORGIVENESS_ENTRY_TYPES == {
            HOLE_OPENED,
            ERROR_REIFIED,
            SOURCE_UNAVAILABLE,
        }
        for benign in (SUPERSEDED, DIRTY, "executed", "continue_requested"):
            assert benign not in FORGIVENESS_ENTRY_TYPES


class TestReassertionFailureConformance:
    """Failure on RE-assertion + failing/withheld re-exec (review findings 1–3).

    Adversarial-review coverage (2026-07 review): the green suite never
    exercised a cell that FAILS while re-binding an already-known name, nor a
    dirty re-exec that fails.  The contract under the two-layer design:

    * a failed (re)bind reifies as an :class:`ErrorValue` that SUPERSEDES the
      prior value/hole (L1.3 — value→error / hole→error is a versioned
      transition, never a silent stale keep), with a NAMED ``error_reified``
      entry, and downstream cells chain off it (L1.1);
    * work the failed cell completed before raising is still kept;
    * the ``dirty`` entry reports the re-exec's ACTUAL outcome (``clean`` /
      ``reified``), never a clean-refresh claim for a failed re-run;
    * a dirtied cell is re-executed only when PROVABLY effect-free — an
      attribute-call write (invisible to the bare-name effect classifier) is
      withheld, so a world effect can never fire twice.
    """

    @pytest.mark.asyncio
    async def test_error_on_reasserted_known_name_reifies_and_chains(
        self, context
    ):
        # Linear path: x = 5, then a failing re-assert, then a dependent.
        result = await run_batch(
            context, lackpy_doc("x = 5", "x = 1 // 0", "z = x + 1")
        )
        ledger = result.metadata["ledger"]
        versions = result.metadata["versions"]
        variables = result.metadata["variables"]

        # x reifies as an ErrorValue — the stale 5 must not survive.
        assert isinstance(variables["x"], ErrorValue)
        assert "ZeroDivisionError" in variables["x"].error

        # L1.3 composition: value→error is a versioned supersession.
        hist = versions.history("x")
        assert [type(v.value).__name__ for v in hist] == ["int", "ErrorValue"]
        assert hist[0].superseded_by == hist[1].version
        sup = ledger.query(entry_type=SUPERSEDED, name="x")
        assert sup and sup[-1].detail["prior"] == "value"

        # The error_reified entry NAMES x — never name=None for a real bind.
        err = ledger.query(entry_type=ERROR_REIFIED, name="x")
        assert err and err[-1].detail["kept"] == []

        # L1.1 composition: downstream chains off the reified error.
        assert isinstance(variables["z"], Hole)
        assert "x" in variables["z"].blocked_by
        assert round_is_left(ledger.entries())

    @pytest.mark.asyncio
    async def test_hole_then_failing_reassert_supersedes_hole_with_error(
        self, context
    ):
        # hole→error: x holed in round 1, failing re-bind in round 2.
        session = LiterateSession(context)
        await session.step(lackpy_doc("y = x + 1"))    # x: origin hole
        await session.step(lackpy_doc("x = 1 // 0"))   # x: failing re-bind
        ns = session._kernel.get_namespace()
        assert isinstance(ns["x"], ErrorValue)
        hist = session.versions.history("x")
        assert [type(v.value).__name__ for v in hist] == ["Hole", "ErrorValue"]
        sup = session.ledger.query(entry_type=SUPERSEDED, name="x")
        assert sup and sup[-1].detail["prior"] == "hole"

    @pytest.mark.asyncio
    async def test_partial_bindings_before_failure_still_kept(self, context):
        # The OTHER side of the partition: work the failed cell completed
        # before raising is kept (and versioned); only the unfinished
        # (re)bind reifies.
        result = await run_batch(
            context, lackpy_doc("x = 5", "x = 6\nboom = 1 // 0")
        )
        variables = result.metadata["variables"]
        ledger = result.metadata["ledger"]
        assert variables["x"] == 6                # rebound pre-failure: kept
        assert isinstance(variables["boom"], ErrorValue)
        err = ledger.query(entry_type=ERROR_REIFIED, name="boom")
        assert err and err[-1].detail["kept"] == ["x"]
        assert [
            v.value for v in result.metadata["versions"].history("x")
        ] == [5, 6]

    @pytest.mark.asyncio
    async def test_failing_dirty_reexec_reifies_and_ledger_tells_truth(
        self, context
    ):
        # Reactive path: re-assert a → re-exec of b = 10 // a raises.
        result = await run_batch(
            context, lackpy_doc("a = 1", "b = 10 // a", "c = b + 1", "a = 0")
        )
        ledger = result.metadata["ledger"]
        versions = result.metadata["versions"]
        variables = result.metadata["variables"]
        dirty = {
            e.detail["cell_index"]: e for e in ledger.query(entry_type=DIRTY)
        }

        # b reifies (no stale 10) and the transition is versioned.
        assert isinstance(variables["b"], ErrorValue)
        b_hist = versions.history("b")
        assert [type(v.value).__name__ for v in b_hist] == ["int", "ErrorValue"]

        # The dirty entry reports the ACTUAL outcome — no clean-refresh claim.
        assert dirty[1].detail["reexecuted"] is True
        assert dirty[1].detail["outcome"] == "reified"
        assert dirty[1].detail["reified"] == ["b"]

        # Downstream c re-execs into a chained hole off the reified b —
        # truthfully recorded too.
        assert isinstance(variables["c"], Hole)
        assert "b" in variables["c"].blocked_by
        assert dirty[2].detail["reexecuted"] is True
        assert dirty[2].detail["outcome"] == "reified"
        assert dirty[2].detail["reified"] == ["c"]
        assert round_is_left(ledger.entries())

    @pytest.mark.asyncio
    async def test_clean_dirty_reexec_outcome_recorded(self, context):
        result = await run_batch(
            context, lackpy_doc("a = 1", "b = a + 1", "a = 2")
        )
        dirty = result.metadata["ledger"].query(entry_type=DIRTY)
        assert dirty[0].detail["reexecuted"] is True
        assert dirty[0].detail["outcome"] == "clean"
        assert result.metadata["variables"]["b"] == 3

    @pytest.mark.asyncio
    async def test_attribute_call_write_withheld_from_reexec(
        self, context, tmp_path
    ):
        # World-effect withhold: the effect classifier attributes effects to
        # BARE-NAME calls only, so `target.write_text(...)` grades "pure".
        # The conservative gate must withhold it anyway: the file is written
        # exactly once, with the ORIGINAL upstream value.
        out = tmp_path / "tick.txt"
        result = await run_batch(
            context,
            lackpy_doc(
                "import pathlib\ntarget = pathlib.Path(%r)" % str(out),
                "a = 1",
                "n = target.write_text(f'tick{a}')",
                "a = 0",
            ),
        )
        ledger = result.metadata["ledger"]
        dirty = {
            e.detail["cell_index"]: e for e in ledger.query(entry_type=DIRTY)
        }
        assert dirty[2].detail["reexecuted"] is False
        assert "cannot prove effect-free" in dirty[2].detail["reason"]
        # Written exactly once — a double-fire would have left 'tick0'.
        assert out.read_text() == "tick1"
