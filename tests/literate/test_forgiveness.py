"""L1.1 + L1.2 acceptance: a failure is ledgered and stops the document.

Spec E1 and E3: a failure (an unbound name, a runtime error) stops the
document at its cell. It is ledgered (``hole_opened`` for an unbound name,
``error_reified`` for a runtime error), it binds nothing, and cells after it
do not run. Nothing rolls back: the bindings from before the cell stand.

The per-round Either/Left signal is a VIEW DERIVED FROM THE LEDGER
(:func:`reified_failures`) — a round containing a failure reports Left to
callers who ask; it is not a control-flow abort.

These tests cover the batch/session execution path
(``LiterateSession.step`` → ``LiterateInterpreter._run_document`` →
``LightweightKernel``) — the path this PR instruments with the L1.0 ledger.
The streaming path (``StreamingDriver`` + recovery) is deliberately untouched.
"""

import pytest

from lackpy.interpreters.base import ExecutionContext
from lackpy.interpreters.literate import LiterateInterpreter, LiterateSession
from lackpy.interpreters.literate.kernel.forgiveness import (
    FORGIVENESS_ENTRY_TYPES,
    reified_failures,
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
    async def test_undefined_name_is_a_ledgered_failure(self, context):
        session = LiterateSession(context)
        result = await session.step("```lackpy\ny = x + 1\n```")

        # Spec E1: the unbound name is ledgered, and nothing is bound for it
        # (no raised StaticAnalysisError either).
        entries = session.ledger.query(entry_type="hole_opened", name="x")
        assert len(entries) == 1
        assert entries[0].detail["reason"] == "unbound"

        ns = session._kernel.get_namespace()
        assert "x" not in ns and "y" not in ns

        # Aggregate layer: the round still reports Left.
        assert not result.ok
        assert any("x" in e for e in result.errors)

    @pytest.mark.asyncio
    async def test_aggregate_either_view_from_ledger(self, context):
        session = LiterateSession(context)

        clean = await session.step("```lackpy\nk = 1\n```")
        assert clean.ok
        assert not reified_failures(session.ledger.entries())

        mark = len(session.ledger)
        left = await session.step("```lackpy\nv = unknown_name\n```")
        round_entries = session.ledger.entries()[mark:]

        # The Left is a view DERIVED FROM THE LEDGER, not a control-flow abort.
        assert reified_failures(round_entries)
        failures = reified_failures(round_entries)
        assert failures
        assert all(e.entry_type in FORGIVENESS_ENTRY_TYPES for e in failures)
        assert not left.ok

        # ...while the earlier state stays, and the failure binds nothing (E1).
        ns = session._kernel.get_namespace()
        assert ns["k"] == 1
        assert "unknown_name" not in ns and "v" not in ns

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
        # Spec E1: ``y`` is not a hole; the cell failed and bound nothing.
        assert not ledger.query(entry_type="hole_opened", name="y")


class TestErrorAsValue:
    """L1.2 — a runtime error binds an error value; no abort, no rollback."""

    @pytest.mark.asyncio
    async def test_runtime_error_is_a_ledgered_failure(self, context):
        session = LiterateSession(context)
        result = await session.step("```lackpy\nx = 1 / 0\n```")

        # Spec E1: the failure is ledgered, and nothing is bound for it.
        entries = session.ledger.query(entry_type="error_reified")
        assert len(entries) == 1 and entries[0].name is None
        assert "ZeroDivisionError" in entries[0].detail["error"]
        assert "x" not in session._kernel.get_namespace()

        # Aggregate layer: Left preserved, with the exception info surfaced.
        assert not result.ok
        assert "ZeroDivisionError" in result.errors[0]

    @pytest.mark.asyncio
    async def test_round_stops_at_the_error(self, context):
        # Spec E1: the round stops at the failed cell. Later cells do not run,
        # and earlier bindings stay.
        session = LiterateSession(context)
        result = await session.step(
            "```lackpy @hidden\nbefore = 1\n```\n\n"
            "```lackpy\nboom = 1 / 0\n```\n\n"
            "```lackpy\nafter = before + 1\n```"
        )
        assert not result.ok                     # aggregate Left
        ns = session._kernel.get_namespace()
        assert ns["before"] == 1                 # kept, not rolled back
        assert "after" not in ns                 # the later cell did not run
        assert "boom" not in ns

    @pytest.mark.asyncio
    async def test_partial_bindings_keep_their_real_values(self, context):
        # `x = 2` completed before the raise. The in-process fold keeps no
        # checkpoint, so its live kernel keeps x == 2 (spec E1: E2 applies only
        # through LiterateRun). The result's variables leave x out (E2), and
        # the failure itself is what gets ledgered.
        session = LiterateSession(context)
        result = await session.step(
            "```lackpy\nx = 2\nfail = {}['boom']\n```"
        )
        assert not result.ok
        assert session._kernel.get_namespace()["x"] == 2
        errs = session.ledger.query(entry_type="error_reified")
        assert errs and "KeyError" in errs[0].detail["error"]

    @pytest.mark.asyncio
    async def test_aggregate_view_covers_errors(self, context):
        session = LiterateSession(context)
        mark = len(session.ledger)
        await session.step("```lackpy\nz = 1 / 0\n```")
        round_entries = session.ledger.entries()[mark:]
        assert reified_failures(round_entries)
        assert reified_failures(round_entries)[0].entry_type == "error_reified"

    @pytest.mark.asyncio
    async def test_batch_path_reifies_error_and_ledgers(self, interpreter, context):
        result = await interpreter.execute("```lackpy\nx = 1 / 0\n```", context)
        assert not result.success                     # aggregate Left
        assert "ZeroDivisionError" in (result.error or "")
        # Spec E1: one entry for the failure, with no name bound.
        entries = result.metadata["ledger"].query(entry_type="error_reified")
        assert len(entries) == 1 and entries[0].name is None
        assert "x" not in result.metadata["variables"]
