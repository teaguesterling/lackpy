"""L1.4 acceptance: dirty-subgraph re-execution (minimal derived graph).

The last forgiveness semantic (sem 3): re-asserting an UPSTREAM name re-executes
ONLY the cells that transitively depend on it — not the whole document, not the
independent cells.  The mechanism is a *minimal derived dependency graph* built
by reusing the def/ref sets :mod:`static_analysis` already computes (DECIDED
design option (a); NO plan IR — the "IR core" is deferred debt).  When a name is
re-asserted (an L1.3 ``superseded`` entry), its transitive dependents are marked
DIRTY (a ``dirty`` ledger entry each) and re-run in dependency order against the
updated namespace, re-versioning (L1.3) and re-forgiving (L1.1/1.2/1.5) as
needed.

Effect-replay safety: only EFFECT-FREE (pure) dependents are re-run; a dependent
with a world effect is marked dirty but withheld (``reexecuted=False``), so
re-execution can never write a file twice.

These tests cover the batch/session path (``LiterateInterpreter._run_document``);
the streaming driver is untouched.
"""

import pytest

from lackpy.interpreters.base import ExecutionContext
from lackpy.interpreters.literate import LiterateInterpreter, LiterateSession
from lackpy.interpreters.literate.kernel import (
    DIRTY,
    DependencyGraph,
    ErrorValue,
    Hole,
)
from lackpy.interpreters.literate.kernel.forgiveness import round_is_left


@pytest.fixture
def context(tmp_path):
    return ExecutionContext(base_dir=tmp_path)


async def _run(context, doc):
    return await LiterateInterpreter().execute(doc, context)


def _dirty(ledger):
    return ledger.query(entry_type=DIRTY)


class TestDependencyGraphUnit:
    """The derived graph's closure logic (unit level, no execution)."""

    def test_direct_dependent_before_binder_is_dirty(self):
        # cell0 binds a; cell1 reads a; cell2 re-binds a (current binder).
        g = DependencyGraph()
        g.add_cell(0, refs=set(), binds={"a"})
        g.add_cell(1, refs={"a"}, binds={"b"})
        g.add_cell(2, refs=set(), binds={"a"})
        # Re-asserting a: reader cell1 (index < the current binder 2) is stale.
        assert g.dirty_closure({"a"}) == [(1, {"a"})]

    def test_reader_after_current_binder_not_dirty(self):
        # A reader at an index >= the current binder already saw the new value.
        g = DependencyGraph()
        g.add_cell(0, refs=set(), binds={"a"})  # a = 41 (current binder)
        g.add_cell(1, refs={"a"}, binds={"b"})  # b = a + 1 — already fresh
        # a's only binder is cell0; no reader sits before it → nothing dirty.
        assert g.dirty_closure({"a"}) == []

    def test_transitive_forward_propagation(self):
        # a -> b -> c chain; sibling d off the path.
        g = DependencyGraph()
        g.add_cell(0, refs=set(), binds={"a"})
        g.add_cell(1, refs={"a"}, binds={"b"})
        g.add_cell(2, refs={"b"}, binds={"c"})
        g.add_cell(3, refs=set(), binds={"d"})
        g.add_cell(4, refs=set(), binds={"a"})  # re-assert a (current binder)
        closure = g.dirty_closure({"a"})
        assert [i for i, _ in closure] == [1, 2]  # b then c, in order; not d

    def test_self_reassertion_does_not_dirty_itself(self):
        # `a = a + 1`: the cell references a AND is a's current binder, so it
        # must not re-run itself (the index gate handles this — it is the
        # cycle/termination guard the brief demands).
        g = DependencyGraph()
        g.add_cell(0, refs=set(), binds={"a"})
        g.add_cell(1, refs={"a"}, binds={"a"})  # a = a + 1, current binder
        assert g.dirty_closure({"a"}) == []


class TestDirtySubgraphReexecution:
    """L1.4 acceptance — the execution path re-runs only the dependents."""

    @pytest.mark.asyncio
    async def test_reassertion_reexecutes_only_dependents(self, context):
        # Doc binds A, then B uses A, then independent C; finally re-assert A.
        # B must RE-RUN with the new value; C must NOT re-run.
        result = await _run(
            context,
            "```lackpy\na = 1\n```\n\n"
            "```lackpy\nb = a + 1\n```\n\n"
            "```lackpy\nc = 99\n```\n\n"
            "```lackpy\na = 2\n```",
        )
        ledger = result.metadata["ledger"]
        versions = result.metadata["versions"]

        # B re-ran with a = 2 → b = 3 (was 2 before the re-assertion).
        assert result.metadata["variables"]["b"] == 3
        b_hist = versions.history("b")
        assert [v.value for v in b_hist] == [2, 3]  # versioned on re-exec

        # Exactly one dirty cell — B (cell index 1) — triggered by a; C absent.
        dirty = _dirty(ledger)
        assert len(dirty) == 1
        assert dirty[0].detail["cell_index"] == 1
        assert dirty[0].detail["triggered_by"] == ["a"]
        assert dirty[0].detail["reexecuted"] is True

        # C never re-ran: its binding stands and it was never dirtied.
        assert result.metadata["variables"]["c"] == 99
        assert versions.history("c") and len(versions.history("c")) == 1

    @pytest.mark.asyncio
    async def test_dirty_subgraph_transitive(self, context):
        # A -> B -> C chain; a sibling D off A's path.  Re-assert A → B and C
        # both re-run, in order; D does not.
        result = await _run(
            context,
            "```lackpy\na = 1\n```\n\n"
            "```lackpy\nb = a + 1\n```\n\n"
            "```lackpy\nc = b + 1\n```\n\n"
            "```lackpy\nd = 100\n```\n\n"
            "```lackpy\na = 10\n```",
        )
        ledger = result.metadata["ledger"]
        vs = result.metadata["variables"]

        # b = a + 1 = 11 ; c = b + 1 = 12 after A re-asserted to 10.
        assert vs["b"] == 11 and vs["c"] == 12
        # D untouched.
        assert vs["d"] == 100

        dirty = _dirty(ledger)
        # Dependents dirtied in dependency (ascending index) order: B then C.
        assert [e.detail["cell_index"] for e in dirty] == [1, 2]
        assert all(e.detail["reexecuted"] for e in dirty)
        # D (cell index 3) is never among the dirtied cells.
        assert 3 not in {e.detail["cell_index"] for e in dirty}

    @pytest.mark.asyncio
    async def test_no_dependency_no_rerun(self, context):
        # Re-asserting a name nothing depends on marks nothing dirty and
        # re-runs nothing.
        result = await _run(
            context,
            "```lackpy\na = 1\n```\n\n"
            "```lackpy\nb = 50\n```\n\n"   # independent of a
            "```lackpy\na = 2\n```",
        )
        ledger = result.metadata["ledger"]
        assert _dirty(ledger) == []
        # b never recomputed; still its single original version.
        assert len(result.metadata["versions"].history("b")) == 1
        assert result.metadata["variables"]["b"] == 50

    @pytest.mark.asyncio
    async def test_dirty_reexec_composes_with_forgiveness(self, context):
        # A dependent that re-execs into a now-forgiving upstream reifies and
        # versions — no crash, ledgered.  base -> derived; then base is
        # re-asserted as a hole (an unknown reference), so the dependent
        # `derived` re-execs into a chained hole.
        result = await _run(
            context,
            "```lackpy\nbase = 10\n```\n\n"
            "```lackpy\nderived = base + 1\n```\n\n"
            "```lackpy\nbase = broken_ref\n```",  # broken_ref undefined
        )
        ledger = result.metadata["ledger"]
        versions = result.metadata["versions"]

        # No crash: the run completed and reports Left (a failure was reified).
        assert result.metadata["completed"] is True
        assert result.success is False
        assert round_is_left(ledger.entries())

        # derived was dirtied by base and re-executed.
        dirty = _dirty(ledger)
        assert [e.detail["cell_index"] for e in dirty] == [1]
        assert dirty[0].detail["triggered_by"] == ["base"]
        assert dirty[0].detail["reexecuted"] is True

        # On re-exec base is now a Hole, so derived re-reifies as a chained
        # hole AND versions (v1 real value → v2 hole).
        d_hist = versions.history("derived")
        assert d_hist[0].value == 11
        assert isinstance(d_hist[-1].value, Hole)
        assert d_hist[0].superseded_by == d_hist[-1].version
        # The re-exec reification is ledgered (a hole_opened for derived).
        assert ledger.query(entry_type="hole_opened", name="derived")

    @pytest.mark.asyncio
    async def test_reexec_into_runtime_error_is_ledgered_not_crash(self, context):
        # A dependent whose re-exec now RAISES is LEDGERED as a failure rather
        # than crashing the run (composition with L1.2) — but note the honest
        # limit: because the dependent was ALREADY bound on the first pass, the
        # raise leaves that (now-stale) value in place, so L1.2's kept/missing
        # logic records a `name=None` error_reified (q in `kept`) instead of
        # rebinding q as an ErrorValue or re-versioning it.  The run still
        # completes and reports Left — no crash, nothing silent.
        result = await _run(
            context,
            "```lackpy\na = 1\n```\n\n"
            "```lackpy\nq = 1 / a\n```\n\n"
            "```lackpy\na = 0\n```",  # re-assert a → q re-execs into 1/0
        )
        ledger = result.metadata["ledger"]
        versions = result.metadata["versions"]
        # Completed, no crash; q's cell was dirtied and re-run.
        assert result.metadata["completed"] is True
        dirty = _dirty(ledger)
        assert [e.detail["cell_index"] for e in dirty] == [1]
        # The re-exec's ZeroDivisionError is ledgered (error_reified), Left.
        err = ledger.query(entry_type="error_reified")
        assert err and err[-1].name is None and err[-1].detail["kept"] == ["q"]
        assert round_is_left(ledger.entries())
        # Honest limit: q keeps its stale first-pass value; it is NOT rebound
        # as an ErrorValue and NOT re-versioned (single version).
        assert result.metadata["variables"]["q"] == 1.0
        assert len(versions.history("q")) == 1

    @pytest.mark.asyncio
    async def test_effectful_dependent_is_dirtied_but_not_reexecuted(
        self, tmp_path
    ):
        # Effect-replay guard: a dependent that WRITES a file is marked dirty
        # but NOT re-run (its effect must not fire twice).
        context = ExecutionContext(base_dir=tmp_path)
        result = await _run(
            context,
            "```lackpy\nname = 'a.txt'\n```\n\n"
            "```lackpy\nwrote = write_file(name, 'hi')\n```\n\n"
            "```lackpy\nname = 'b.txt'\n```",  # re-assert name → dependent write
        )
        ledger = result.metadata["ledger"]
        dirty = _dirty(ledger)
        assert len(dirty) == 1
        assert dirty[0].detail["cell_index"] == 1
        assert dirty[0].detail["reexecuted"] is False
        assert "effectful" in dirty[0].detail["reason"]
        # The write happened exactly once (only a.txt exists — never b.txt).
        assert (tmp_path / "a.txt").exists()
        assert not (tmp_path / "b.txt").exists()

    @pytest.mark.asyncio
    async def test_session_reexec_within_round(self, context):
        # The session path shares _run_document, so dirty re-exec works there
        # too — within a single round's cells.
        session = LiterateSession(context)
        await session.step(
            "```lackpy\na = 1\n```\n\n"
            "```lackpy\nb = a + 1\n```\n\n"
            "```lackpy\na = 5\n```"
        )
        assert session._kernel.get_namespace()["b"] == 6
        assert session.ledger.query(entry_type=DIRTY)
