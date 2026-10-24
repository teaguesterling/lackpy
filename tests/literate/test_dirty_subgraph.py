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

Effect-replay safety (strict mode, the default): each cell's first execution
is OBSERVED through its slice of the session trace (literate stepy Task 3: a
``Path`` write entry, ``effects.path_writes``); a dependent observed writing is
marked dirty but withheld (``reexecuted=False`` + the observed writes), so an
observed write is not replayed.

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
from lackpy.run.paths import load_policy


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
    async def test_heal_reexecutes_only_dependents(self, context):
        # Teague's C1 decision: only a heal re-runs. B uses A while A is an
        # unbound hole, then independent C; finally A heals. B must RE-RUN
        # with the value; C must NOT re-run.
        result = await _run(
            context,
            "```lackpy\nb = a + 1\n```\n\n"
            "```lackpy\nc = 99\n```\n\n"
            "```lackpy\na = 2\n```",
        )
        ledger = result.metadata["ledger"]
        versions = result.metadata["versions"]

        # B re-ran with a = 2 → b = 3 (a chained hole before the heal).
        assert result.metadata["variables"]["b"] == 3
        b_hist = versions.history("b")
        assert isinstance(b_hist[0].value, Hole) and b_hist[-1].value == 3

        # Exactly one dirty cell — B (cell index 0) — triggered by a; C absent.
        dirty = _dirty(ledger)
        assert len(dirty) == 1
        assert dirty[0].detail["cell_index"] == 0
        assert dirty[0].detail["triggered_by"] == ["a"]
        assert dirty[0].detail["reexecuted"] is True

        # C never re-ran: its binding stands and it was never dirtied.
        assert result.metadata["variables"]["c"] == 99
        assert versions.history("c") and len(versions.history("c")) == 1

    @pytest.mark.asyncio
    async def test_dirty_subgraph_transitive(self, context):
        # A -> B -> C chain; a sibling D off A's path.  A heals (C1) → B and C
        # both re-run, in order; D does not.
        result = await _run(
            context,
            "```lackpy\nb = a + 1\n```\n\n"
            "```lackpy\nc = b + 1\n```\n\n"
            "```lackpy\nd = 100\n```\n\n"
            "```lackpy\na = 10\n```",
        )
        ledger = result.metadata["ledger"]
        vs = result.metadata["variables"]

        # b = a + 1 = 11 ; c = b + 1 = 12 after A healed to 10.
        assert vs["b"] == 11 and vs["c"] == 12
        # D untouched.
        assert vs["d"] == 100

        dirty = _dirty(ledger)
        # Dependents dirtied in dependency (ascending index) order: B then C.
        assert [e.detail["cell_index"] for e in dirty] == [0, 1]
        assert all(e.detail["reexecuted"] for e in dirty)
        # D (cell index 2) is never among the dirtied cells.
        assert 2 not in {e.detail["cell_index"] for e in dirty}

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
        # A dependent re-run by a heal that hits ANOTHER unresolved name
        # reifies and versions — no crash, ledgered.  (C1: it was "base
        # re-asserted as a hole", a value-to-hole change, which no longer
        # re-runs anything.)  derived reads base (unbound, then healed) and
        # missing (never bound), so its re-exec re-reifies a chained hole.
        result = await _run(
            context,
            "```lackpy\nderived = base + missing\n```\n\n"
            "```lackpy\nbase = 10\n```",
        )
        ledger = result.metadata["ledger"]
        versions = result.metadata["versions"]

        # No crash: the run completed and reports Left (a failure was reified).
        assert result.metadata["completed"] is True
        assert result.success is False
        assert round_is_left(ledger.entries())

        # derived was dirtied by base and re-executed.
        dirty = _dirty(ledger)
        assert [e.detail["cell_index"] for e in dirty] == [0]
        assert dirty[0].detail["triggered_by"] == ["base"]
        assert dirty[0].detail["reexecuted"] is True
        assert dirty[0].detail["outcome"] == "reified"

        # On re-exec missing is still a Hole, so derived re-reifies as a
        # chained hole AND versions (v1 hole → v2 hole).
        d_hist = versions.history("derived")
        assert len(d_hist) == 2
        assert all(isinstance(v.value, Hole) for v in d_hist)
        assert d_hist[0].superseded_by == d_hist[-1].version
        # The re-exec reification is ledgered (a hole_opened for derived).
        assert ledger.query(entry_type="hole_opened", name="derived")

    @pytest.mark.asyncio
    async def test_reexec_into_runtime_error_is_ledgered_not_crash(self, context):
        # A dependent whose re-exec now RAISES is LEDGERED as a failure rather
        # than crashing the run (composition with L1.2).  The dependent was
        # already bound on the first pass, and the failed re-bind must NOT
        # silently keep that stale value: q reifies as an ErrorValue,
        # superseding the first-pass value (value→error, a new version), with
        # a NAMED error_reified entry and a truthful dirty entry.  (This test
        # previously pinned the pre-fix stale-keep behavior as an "honest
        # limit"; that was the Finding-1 defect — updated to the contract.)
        # C1: q is blocked on the unbound a; a heals to 0 → q re-execs into 1/0.
        result = await _run(
            context,
            "```lackpy\nq = 1 / a\n```\n\n"
            "```lackpy\na = 0\n```",
        )
        ledger = result.metadata["ledger"]
        versions = result.metadata["versions"]
        # Completed, no crash; q's cell was dirtied and re-run.
        assert result.metadata["completed"] is True
        dirty = _dirty(ledger)
        assert [e.detail["cell_index"] for e in dirty] == [0]
        # The dirty entry tells the truth: re-executed, but the re-run failed
        # and was reified — it does not claim a clean refresh.
        assert dirty[0].detail["reexecuted"] is True
        assert dirty[0].detail["outcome"] == "reified"
        assert dirty[0].detail["reified"] == ["q"]
        # The re-exec's ZeroDivisionError is ledgered (error_reified) WITH the
        # name — q was the intended re-bind and did not complete — and Left.
        err = ledger.query(entry_type="error_reified")
        assert err and err[-1].name == "q" and err[-1].detail["kept"] == []
        assert round_is_left(ledger.entries())
        # q is an ErrorValue, and the transition is versioned:
        # v1 (chained hole, superseded) → v2 (error).
        assert isinstance(result.metadata["variables"]["q"], ErrorValue)
        q_hist = versions.history("q")
        assert [type(v.value).__name__ for v in q_hist] == ["Hole", "ErrorValue"]
        assert q_hist[0].superseded_by == q_hist[1].version
        assert ledger.query(entry_type="superseded", name="q")

    @pytest.mark.asyncio
    async def test_effectful_dependent_is_dirtied_but_not_reexecuted(
        self, tmp_path, tmp_home
    ):
        # Effect-replay guard (strict mode): a dependent OBSERVED writing a
        # file on its first run is marked dirty but NOT re-run. (Literate
        # stepy: the write goes through Path under a policy, not write_file.)
        context = ExecutionContext(base_dir=tmp_path, config={
            "path_policy": load_policy(tmp_path, home=tmp_home, managed=None)})
        result = await _run(
            context,
            "```lackpy\nfrom pathlib import Path\nname = 'a.txt'\n```\n\n"
            # C1: the dependent writes, then fails; name becomes an ErrorValue
            # and then heals, which dirties the failed dependent.
            "```lackpy\nwrote = Path(name).write_text('hi')\nboom = 1 // 0\n```\n\n"
            "```lackpy\nname = 1 // 0\n```\n\n"
            "```lackpy\nname = 'b.txt'\n```",
        )
        ledger = result.metadata["ledger"]
        dirty = _dirty(ledger)
        assert len(dirty) == 1
        assert dirty[0].detail["cell_index"] == 1
        assert dirty[0].detail["reexecuted"] is False
        assert dirty[0].detail["mode"] == "strict"
        assert "observed to perform a world effect" in dirty[0].detail["reason"]
        # The observed evidence is on the entry: the Path write in the trace.
        assert dirty[0].detail["observed_effects"] == ["Path.write_text"]
        # The write happened exactly once (only a.txt exists — never b.txt).
        assert (tmp_path / "a.txt").exists()
        assert not (tmp_path / "b.txt").exists()

    @pytest.mark.asyncio
    async def test_ceiling_refused_dependent_stays_refused_on_reexec(
        self, tmp_path, tmp_home
    ):
        # Ceiling-gate soundness on the REACTIVE path: a cell the effect
        # ceiling refused on the linear pass has no observed_effects entry
        # (it never ran), but that absence means REFUSED, not "safe first
        # execution" — the re-exec pass must keep it refused, not run it.
        # (Regression: the observation-gated re-exec ran refused cells,
        # executing the over-ceiling effect the gate had refused.)
        from lackpy.interpreters.literate.kernel.forgiveness import Unavailable
        from lackpy.lang.grader import Grade

        context = ExecutionContext(base_dir=tmp_path, config={
            "grade_ceiling": Grade(1, 1),
            "path_policy": load_policy(tmp_path, home=tmp_home, managed=None)})
        result = await _run(
            context,
            "```lackpy\nfrom pathlib import Path\nname = 'out1.txt'\n```\n\n"
            "```lackpy\nw = Path(name).write_text('DANGER')\n```\n\n"
            "```lackpy\nname = 1 // 0\n```\n\n"  # C1: an ErrorValue, then a heal
            "```lackpy\nname = 'out2.txt'\n```",  # heal → dirties cell 1
        )
        # The refused effect never fires — on either pass.
        assert not (tmp_path / "out1.txt").exists()
        assert not (tmp_path / "out2.txt").exists()
        ledger = result.metadata["ledger"]
        dirty = _dirty(ledger)
        assert [e.detail["cell_index"] for e in dirty] == [1]
        # Withheld for the ceiling — honestly ledgered as such, not claimed
        # as a clean re-execution.
        assert dirty[0].detail["reexecuted"] is False
        assert "effect ceiling" in dirty[0].detail["reason"]
        assert "gate_violation" in dirty[0].detail
        # w stays Unavailable — its single reified version, never re-bound.
        w_hist = result.metadata["versions"].history("w")
        assert len(w_hist) == 1
        assert isinstance(w_hist[0].value, Unavailable)

    @pytest.mark.asyncio
    async def test_ceiling_refused_stays_refused_in_permissive_mode(
        self, tmp_path, tmp_home
    ):
        # The ceiling refusal is POLICY, not replay caution: permissive mode
        # (which accepts effect replay) still may not run a refused cell.
        from lackpy.lang.grader import Grade

        context = ExecutionContext(
            base_dir=tmp_path,
            config={
                "grade_ceiling": Grade(1, 1),
                "reactive_mode": "permissive",
                "path_policy": load_policy(tmp_path, home=tmp_home, managed=None),
            },
        )
        result = await _run(
            context,
            "```lackpy\nfrom pathlib import Path\nname = 'out1.txt'\n```\n\n"
            "```lackpy\nw = Path(name).write_text('DANGER')\n```\n\n"
            "```lackpy\nname = 1 // 0\n```\n\n"  # C1: an ErrorValue, then a heal
            "```lackpy\nname = 'out2.txt'\n```",
        )
        assert not (tmp_path / "out1.txt").exists()
        assert not (tmp_path / "out2.txt").exists()
        dirty = _dirty(result.metadata["ledger"])
        assert [e.detail["cell_index"] for e in dirty] == [1]
        assert dirty[0].detail["reexecuted"] is False
        assert "effect ceiling" in dirty[0].detail["reason"]

    @pytest.mark.asyncio
    async def test_hole_skipped_dependent_reruns_when_hole_fills(self, context):
        # CONTRAST with the ceiling case: a cell skipped because a referenced
        # name was UNRESOLVED (hole-skipped, also absent from observed_effects)
        # SHOULD run when the hole fills — that absence really is a safe first
        # execution.  Distinguishing these two absences is the point of the
        # gate_violations check.
        result = await _run(
            context,
            "```lackpy\ny = x + 1\n```\n\n"  # x unknown → hole-skipped
            "```lackpy\nx = 5\n```",  # fills the hole → supersedes Hole(x)
        )
        dirty = _dirty(result.metadata["ledger"])
        assert [e.detail["cell_index"] for e in dirty] == [0]
        assert dirty[0].detail["reexecuted"] is True
        assert dirty[0].detail["outcome"] == "clean"
        # The re-run was its FIRST execution: y is now real, hole superseded.
        assert result.metadata["variables"]["y"] == 6
        y_hist = result.metadata["versions"].history("y")
        assert isinstance(y_hist[0].value, Hole)
        assert y_hist[-1].value == 6

    @pytest.mark.asyncio
    async def test_session_reexec_within_round(self, context):
        # The session path shares _run_document, so dirty re-exec works there
        # too — within a single round's cells.
        session = LiterateSession(context)
        await session.step(
            "```lackpy\nb = a + 1\n```\n\n"  # C1: a heals from a hole
            "```lackpy\na = 5\n```"
        )
        assert session._kernel.get_namespace()["b"] == 6
        assert session.ledger.query(entry_type=DIRTY)


class TestReexecutionThroughTheSession:
    """Literate stepy Task 2: a dirty dependent re-runs its validated code with
    ``CellSession.rerun_cell``; the rebind that dirtied it is a new cell."""

    @pytest.mark.asyncio
    async def test_a_rerun_does_not_append_to_the_program(self, context):
        # C1: b's cell joined the program and failed at runtime; a becomes an
        # ErrorValue, then heals, so b's cell re-runs with rerun_cell.
        session = LiterateSession(context)
        await session.step(
            "```lackpy\na = 0\n```\n\n```lackpy\nb = 1 / a\n```\n\n"
            "```lackpy\na = 1 // 0\n```\n\n```lackpy\na = 2\n```")
        assert session._kernel.get_namespace()["b"] == 0.5
        assert session._kernel.session._program == "a = 0\nb = 1 / a\na = 1 // 0\na = 2"

    @pytest.mark.asyncio
    async def test_a_refusal_during_reexecution_stops_the_document(
            self, tmp_path, tmp_home):
        (tmp_path / "in.txt").write_text("ok")
        context = ExecutionContext(base_dir=tmp_path, config={
            "path_policy": load_policy(tmp_path, home=tmp_home, managed=None)})
        result = await _run(
            context,
            "```lackpy\nfrom pathlib import Path\np = 'in.txt'\n```\n\n"
            "```lackpy\nt = Path(p).read_text()\nboom = 1 // 0\n```\n\n"
            "```lackpy\np = 1 // 0\n```\n\n"  # C1: an ErrorValue, then a heal
            "```lackpy\np = '/etc/hostname'\n```\n\n"
            "```lackpy\nafter = 1\n```\n",
            )
        # The linear pass ran every cell; the re-run of cell 1 refused.
        assert result.success is False
        refused = result.metadata["ledger"].query(entry_type="refused")
        assert [e.detail["cell_index"] for e in refused] == [1]
        assert "/etc/hostname" in result.error
        assert result.metadata["variables"]["t"] == "ok"  # never reified


class TestNonDataBindingsAreTrackedByName:
    """Review I2: a binding to a non-data value (a def, a Path) supersedes and
    dirties by name; a hole it fills heals."""

    @pytest.mark.asyncio
    async def test_a_hole_filled_by_a_def_heals(self, context):
        session = LiterateSession(context)
        await session.step(
            "```lackpy\ny = f(1)\n```\n\n```lackpy\ndef f(n):\n    return n * 2\n```\n\n"
            "Y is {y}\n")
        ns = session._kernel.get_namespace()
        assert ns["y"] == 2
        assert not isinstance(session.versions.current("f").value, Hole)
        assert session.ledger.query(entry_type="superseded", name="f")
        later = await session.step("Later y={y}\n")
        assert later.ok and "Later y=2" in later.clean_doc

    @pytest.mark.asyncio
    async def test_a_hole_filled_by_a_path_heals(self, tmp_path, tmp_home):
        context = ExecutionContext(base_dir=tmp_path, config={
            "path_policy": load_policy(tmp_path, home=tmp_home, managed=None)})
        result = await _run(
            context,
            "```lackpy\nn = p.name\n```\n\n"
            "```lackpy\nfrom pathlib import Path\np = Path('a.txt')\n```\n")
        assert result.metadata["variables"]["n"] == "a.txt"
        versions = result.metadata["versions"]
        assert not isinstance(versions.current("p").value, Hole)
        assert [e.detail["cell_index"] for e in _dirty(result.metadata["ledger"])] == [0]

    @pytest.mark.asyncio
    async def test_data_then_a_def_supersedes_and_does_not_rerun(self, context):
        # Teague's C1 decision: a value-to-value rebind (data to a def) is
        # versioned (superseded) but re-runs nothing; y keeps python3's 4.
        result = await _run(
            context,
            "```lackpy\nf = 3\n```\n\n```lackpy\ny = f + 1\n```\n\n"
            "```lackpy\ndef f(n):\n    return n\n```\n")
        ledger = result.metadata["ledger"]
        sup = ledger.query(entry_type="superseded", name="f")
        assert sup and sup[-1].detail["cell_index"] == 2
        assert _dirty(ledger) == []
        assert result.metadata["variables"]["y"] == 4


class TestPlainRebindDoesNotReexecute:
    """Teague's C1 decision: a dependent cell re-runs only when a name it reads
    heals from Hole, ErrorValue or Unavailable to a value. A plain rebind never
    re-runs earlier cells, so a document gives python3's answer (goal 4)."""

    CSV = ("```lackpy\nrows = 'name,n\\nann,3\\nbob,5\\ncat,7\\n'.splitlines()\n```\n\n"
           "```lackpy\nrows = rows[1:]\n```\n\n"
           "```lackpy\nrows = [r for r in rows if r]\n```\n\n"
           "There are {len(rows)} people.\n\n"
           "```lackpy\nlen(rows)\n```\n")

    @pytest.mark.asyncio
    async def test_the_reviewers_csv_document_gives_python3s_answer(self, context):
        # The final review measured output=2 and rows=['bob,5','cat,7'] (ann lost)
        # while the prose said 3; python3 on the joined cells gives 3.
        result = await _run(context, self.CSV)
        assert result.success, result.error
        assert result.metadata["value"] == 3
        assert result.metadata["variables"]["rows"] == ["ann,3", "bob,5", "cat,7"]
        assert "There are 3 people." in result.output
        assert _dirty(result.metadata["ledger"]) == []

    @pytest.mark.asyncio
    async def test_two_slices_of_one_name(self, context):
        doc = ("```lackpy\nitems = ['a', 'b', 'c', 'd']\n```\n\n"
               "```lackpy\nitems = items[1:]\n```\n\n"
               "```lackpy\nitems = items[1:]\nitems\n```\n")
        result = await _run(context, doc)
        assert result.success, result.error
        assert result.metadata["value"] == ["c", "d"]

    @pytest.mark.asyncio
    async def test_an_earlier_reader_keeps_the_value_it_saw(self, context):
        doc = ("```lackpy\na = 1\n```\n\n```lackpy\nb = a + 1\n```\n\n"
               "```lackpy\na = 2\n```\n")
        result = await _run(context, doc)
        assert result.metadata["variables"]["b"] == 2  # python3: b = 2
        assert _dirty(result.metadata["ledger"]) == []

    @pytest.mark.asyncio
    async def test_a_heal_still_reruns_the_blocked_reader_but_not_a_reader_that_ran(self, context):
        # x heals (hole -> 5); cell 0 was blocked on it and re-runs. z read the
        # healed x and ran; the later plain rebind of x must not re-run it.
        doc = ("```lackpy\ny = x + 1\n```\n\n```lackpy\nx = 5\n```\n\n"
               "```lackpy\nz = x * 2\n```\n\n```lackpy\nx = x + 1\n```\n")
        result = await _run(context, doc)
        vs = result.metadata["variables"]
        assert vs["z"] == 10
        assert [e.detail["cell_index"] for e in _dirty(result.metadata["ledger"])] == [0]
