"""L1.1 + L1.2 acceptance: two-layer forgiveness — typed holes + error-as-value.

The decided contract (design conflict #2, option (c), Teague 2026-07-17):

* **Binding layer (default):** a failure is REIFIED AS A BOUND VALUE and
  ledgered; execution continues; NO rollback.  An unknown name binds a typed
  hole ``⟨name: unbound⟩`` (``hole_opened``); a runtime error binds an
  :class:`ErrorValue` (``error_reified``).  A cell that references a hole or
  an error value cannot produce a real value, so its bindings become *chained*
  holes — detected statically (referenced name bound to a forgiving value),
  so the cell is skipped, never executed: a hole never flows into arithmetic.
* **Aggregate layer (preserved):** the per-round Either/Left signal survives
  as a VIEW DERIVED FROM THE LEDGER (:func:`round_is_left`) — a round
  containing any reified failure still reports Left to callers who ask; it is
  no longer a control-flow abort.

These tests cover the batch/session execution path
(``LiterateSession.step`` → ``LiterateInterpreter._run_document`` →
``LightweightKernel``) — the path this PR instruments with the L1.0 ledger.
The streaming path (``StreamingDriver`` + recovery) is deliberately untouched.
"""

import pytest

from lackpy.interpreters.base import ExecutionContext
from lackpy.interpreters.literate import LiterateInterpreter, LiterateSession
from lackpy.interpreters.literate.kernel import ErrorValue, Hole
from lackpy.interpreters.literate.kernel.forgiveness import (
    FORGIVENESS_ENTRY_TYPES,
    reified_failures,
    round_is_left,
)


@pytest.fixture
def context(tmp_path):
    return ExecutionContext(base_dir=tmp_path)


@pytest.fixture
def interpreter():
    return LiterateInterpreter()


class TestTypedHoles:
    """L1.1 — an unknown name binds a typed hole instead of raising/blocking."""

    @pytest.mark.asyncio
    async def test_undefined_name_binds_hole(self, context):
        session = LiterateSession(context)
        result = await session.step("```lackpy\ny = x + 1\n```")

        # Binding layer: x is reified as a typed hole and ledgered — the round
        # completed (no raised StaticAnalysisError, no abort).
        entries = session.ledger.query(entry_type="hole_opened", name="x")
        assert len(entries) == 1
        assert entries[0].detail["reason"] == "unbound"

        ns = session._kernel.get_namespace()
        assert ns["x"] == Hole("x")
        assert repr(ns["x"]) == "⟨x: unbound⟩"

        # Aggregate layer: the round still reports Left.
        assert not result.ok
        assert any("x" in e for e in result.errors)

    @pytest.mark.asyncio
    async def test_chained_hole(self, context):
        # Hole in cell A blocks dependent cell B; BOTH are ledgered.
        session = LiterateSession(context)
        await session.step(
            "```lackpy\na = missing + 1\n```\n\n```lackpy\nb = a * 2\n```"
        )
        ledger = session.ledger
        assert ledger.query(entry_type="hole_opened", name="missing")  # origin
        a_entries = ledger.query(entry_type="hole_opened", name="a")
        b_entries = ledger.query(entry_type="hole_opened", name="b")
        assert a_entries and b_entries
        assert "missing" in a_entries[0].detail["blocked_by"]
        assert "a" in b_entries[0].detail["blocked_by"]

        ns = session._kernel.get_namespace()
        assert isinstance(ns["a"], Hole) and isinstance(ns["b"], Hole)
        assert ns["a"].blocked_by == ("missing",)
        assert ns["b"].blocked_by == ("a",)

    @pytest.mark.asyncio
    async def test_aggregate_either_view_from_ledger(self, context):
        session = LiterateSession(context)

        clean = await session.step("```lackpy\nk = 1\n```")
        assert clean.ok
        assert not round_is_left(session.ledger.entries())

        mark = len(session.ledger)
        left = await session.step("```lackpy\nv = unknown_name\n```")
        round_entries = session.ledger.entries()[mark:]

        # The Left is a view DERIVED FROM THE LEDGER, not a control-flow abort.
        assert round_is_left(round_entries)
        failures = reified_failures(round_entries)
        assert failures
        assert all(e.entry_type in FORGIVENESS_ENTRY_TYPES for e in failures)
        assert not left.ok

        # ...while the forgiving values remain bound: state NOT erased.
        ns = session._kernel.get_namespace()
        assert ns["k"] == 1
        assert isinstance(ns["unknown_name"], Hole)
        assert isinstance(ns["v"], Hole)

    @pytest.mark.asyncio
    async def test_hole_round_keeps_output_and_annotates(self, context):
        # Document is sole source of truth; renders append/annotate only —
        # the failure is annotated through the L2 kernel channel, not silent.
        session = LiterateSession(context)
        result = await session.step(
            "Before the hole.\n\n```lackpy\nw = absent\n```"
        )
        assert not result.ok
        assert "Before the hole." in session.rendered
        assert "[kernel]" in session.rendered

    @pytest.mark.asyncio
    async def test_filling_a_hole_rebinds_the_real_value(self, context):
        # The forgiveness UX: a later round defines the missing name and can
        # then use it — the hole is an invitation, not a poisoned namespace.
        session = LiterateSession(context)
        assert not (await session.step("```lackpy\ny = x + 1\n```")).ok
        fixed = await session.step(
            "```lackpy\nx = 41\ny = x + 1\n```\n\ny={y}"
        )
        assert fixed.ok, fixed.errors
        assert "y=42" in fixed.clean_doc
        ns = session._kernel.get_namespace()
        assert ns["x"] == 41 and ns["y"] == 42

    @pytest.mark.asyncio
    async def test_batch_path_binds_hole_and_ledgers(self, interpreter, context):
        # The batch execute() runs the same instrumented path.
        result = await interpreter.execute("```lackpy\ny = x + 1\n```", context)
        assert not result.success                     # aggregate Left preserved
        assert "x" in (result.error or "")
        ledger = result.metadata["ledger"]
        assert ledger.query(entry_type="hole_opened", name="x")
        assert ledger.query(entry_type="hole_opened", name="y")


class TestErrorAsValue:
    """L1.2 — a runtime error binds an error value; no abort, no rollback."""

    @pytest.mark.asyncio
    async def test_runtime_error_binds_error_value(self, context):
        session = LiterateSession(context)
        result = await session.step("```lackpy\nx = 1 / 0\n```")

        # Binding layer: the failure is reified — bound + ledgered.
        entries = session.ledger.query(entry_type="error_reified", name="x")
        assert len(entries) == 1
        assert "ZeroDivisionError" in entries[0].detail["error"]

        ns = session._kernel.get_namespace()
        assert isinstance(ns["x"], ErrorValue)
        assert "ZeroDivisionError" in repr(ns["x"])

        # Aggregate layer: Left preserved, with the exception info surfaced.
        assert not result.ok
        assert "ZeroDivisionError" in result.errors[0]

    @pytest.mark.asyncio
    async def test_round_continues_after_error_no_rollback(self, context):
        # The load-bearing L1.2 shape: the round does NOT abort at the failed
        # cell — later cells run, earlier bindings stay, nothing rolls back.
        session = LiterateSession(context)
        result = await session.step(
            "```lackpy @hidden\nbefore = 1\n```\n\n"
            "```lackpy\nboom = 1 / 0\n```\n\n"
            "```lackpy\nafter = before + 1\n```"
        )
        assert not result.ok                     # aggregate Left
        ns = session._kernel.get_namespace()
        assert ns["before"] == 1                 # kept, not rolled back
        assert ns["after"] == 2                  # later cell RAN
        assert isinstance(ns["boom"], ErrorValue)

    @pytest.mark.asyncio
    async def test_partial_bindings_keep_their_real_values(self, context):
        # `x = 2` completed before the raise: completed work is kept as its
        # real value — the failure itself is what gets ledgered.
        session = LiterateSession(context)
        result = await session.step(
            "```lackpy\nx = 2\nraise ValueError('boom')\n```"
        )
        assert not result.ok
        assert session._kernel.get_namespace()["x"] == 2
        errs = session.ledger.query(entry_type="error_reified")
        assert errs and "ValueError" in errs[0].detail["error"]

    @pytest.mark.asyncio
    async def test_error_value_chains_to_dependent_cell(self, context):
        # An error value blocks a dependent cell the same way a hole does.
        session = LiterateSession(context)
        await session.step(
            "```lackpy\ne = 1 / 0\n```\n\n```lackpy\nf = e + 1\n```"
        )
        ns = session._kernel.get_namespace()
        assert isinstance(ns["e"], ErrorValue)
        assert isinstance(ns["f"], Hole)
        assert ns["f"].blocked_by == ("e",)
        assert session.ledger.query(entry_type="hole_opened", name="f")

    @pytest.mark.asyncio
    async def test_aggregate_view_covers_errors(self, context):
        session = LiterateSession(context)
        mark = len(session.ledger)
        await session.step("```lackpy\nz = 1 / 0\n```")
        round_entries = session.ledger.entries()[mark:]
        assert round_is_left(round_entries)
        assert reified_failures(round_entries)[0].entry_type == "error_reified"

    @pytest.mark.asyncio
    async def test_batch_path_reifies_error_and_ledgers(self, interpreter, context):
        result = await interpreter.execute("```lackpy\nx = 1 / 0\n```", context)
        assert not result.success                     # aggregate Left
        assert "ZeroDivisionError" in (result.error or "")
        assert result.metadata["ledger"].query(entry_type="error_reified", name="x")
