"""L7 acceptance tests: kernel self-test, fail-closed startup probe.

The evidence case behind the feature: a host eval-restriction once silently
zeroed every eval — the kernel produced garbage instead of refusing.  These
tests prove the self-test catches exactly that (induced breakage → structured
refusal naming the failed probe, NOT a silent wrong answer), covers each
semantic class the kernel supports, and fails legibly (no tracebacks).
"""

from __future__ import annotations

import pytest

from lackpy.interpreters.base import ExecutionContext
from lackpy.interpreters.literate import LiterateInterpreter, LiterateSession
from lackpy.interpreters.literate.kernel import lightweight
from lackpy.interpreters.literate.kernel.lightweight import LightweightKernel
from lackpy.interpreters.literate.parser import Cell
from lackpy.interpreters.literate.selftest import (
    REFUSAL_HEADLINE,
    SELFTEST_FAILED,
    SELFTEST_PASSED,
    run_selftest,
)

DOC = "```lackpy\nanswer = 2 + 2\nprint(answer)\n```\n"


@pytest.fixture
def context(tmp_path):
    return ExecutionContext(base_dir=tmp_path)


def _zeroing_exec(monkeypatch):
    """Induce the real-world failure mode: exec 'succeeds' but every new
    binding is silently zeroed (the host eval-restriction evidence case)."""
    real = lightweight._do_exec

    def zeroed(code, ns):
        before = set(ns)
        real(code, ns)
        for name in set(ns) - before:
            ns[name] = 0

    monkeypatch.setattr(lightweight, "_do_exec", zeroed)


class TestSelftestPassesOnHealthyKernel:
    @pytest.mark.asyncio
    async def test_selftest_passes_on_healthy_kernel(self, context):
        session = LiterateSession(context)
        assert session.selftest.ok
        assert all(p.ok for p in session.selftest.probes)
        # The outcome is ledgered (observable, queryable) ...
        assert session.ledger.query(entry_type=SELFTEST_PASSED)
        # ... and the session accepts work normally.
        result = await session.step(DOC)
        assert result.ok
        assert "4" in result.clean_doc

    @pytest.mark.asyncio
    async def test_batch_execute_passes_and_reports_status(self, context):
        result = await LiterateInterpreter().execute(DOC, context)
        assert result.success
        assert "4" in result.output
        assert result.metadata["selftest"]["ok"] is True


class TestSelftestCoversEachSemanticClass:
    def test_selftest_covers_each_semantic_class(self, context):
        session = LiterateSession(context)
        probes = {p.probe: p for p in session.selftest.probes}
        # One probe per semantic class the kernel supports + the round-trip law.
        assert set(probes) == {"value", "hole", "error", "unavailable", "round_trip"}
        assert all(p.ok for p in probes.values())
        # Each probe demonstrably exercised its class (the report says so).
        assert "4" in probes["value"].got
        assert "unbound" in probes["hole"].got  # ⟨name: unbound⟩ Hole repr
        assert "ZeroDivisionError" in probes["error"].got  # ErrorValue repr
        assert "unavailable" in probes["unavailable"].got  # Unavailable repr
        assert "round trip clean" in probes["round_trip"].got

    def test_probes_leave_no_state_behind(self, context):
        """Probe hygiene: no probe binding leaks into scope, and the session
        ledger holds only the one summary entry (a probe's hole_opened must
        never make the first real round Left)."""
        session = LiterateSession(context)
        assert not any(name.startswith("lackpy_selftest_") for name in session.scope)
        assert [e.entry_type for e in session.ledger.entries()] == [SELFTEST_PASSED]


class TestInducedBreakageRefusesNotGarbage:
    @pytest.mark.asyncio
    async def test_induced_breakage_refuses_not_garbage(self, context, monkeypatch):
        """THE headline: eval silently zeroed -> refusal naming the probe,
        never the silent wrong answer the old kernel produced."""
        _zeroing_exec(monkeypatch)
        session = LiterateSession(context)

        assert not session.selftest.ok
        failed = [p.probe for p in session.selftest.probes if not p.ok]
        assert "value" in failed
        assert session.ledger.query(entry_type=SELFTEST_FAILED)

        result = await session.step(DOC)
        assert result.ok is False
        # The refusal names the failed probe ...
        assert any("'value'" in e for e in result.errors)
        assert any("self-test" in e for e in result.errors)
        # ... and NOTHING ran: no garbage output, no executed ledger entries.
        assert result.clean_doc == ""
        assert "0" not in result.clean_doc
        assert session.ledger.query(entry_type="executed") == []

        # Fail-closed is sticky: every subsequent attempt refuses too.
        again = await session.step(DOC)
        assert again.ok is False

    @pytest.mark.asyncio
    async def test_batch_execute_refuses_on_induced_breakage(
        self, context, monkeypatch
    ):
        _zeroing_exec(monkeypatch)
        result = await LiterateInterpreter().execute(DOC, context)
        assert result.success is False
        assert result.output is None  # no garbage — nothing executed
        assert result.metadata["completed"] is False
        assert result.metadata["selftest"]["ok"] is False
        assert "'value'" in result.error

    def test_disabled_kernel_backstop(self, context, monkeypatch):
        """A failed self-test hard-disables the kernel itself: even a caller
        that ignores the report gets a structured refusal from execute_cell
        (and inspect), on any path holding this kernel."""
        _zeroing_exec(monkeypatch)
        kernel = LightweightKernel()
        report = run_selftest(kernel, LiterateInterpreter())
        assert not report.ok
        assert kernel.disabled is not None

        result = kernel.execute_cell(Cell(cell_type="code", content="x = 1"), 0)
        assert result.success is False
        assert result.error_phase == "disabled"
        assert "self-test failed" in result.error
        assert "kernel disabled" in kernel.inspect("2 + 2")

        # restart() cannot un-break evaluation — the disable is sticky.
        kernel.restart()
        assert kernel.disabled is not None

    def test_raising_probe_is_failed_probe_not_traceback(self, context, monkeypatch):
        """A probe that raises is a FAILED probe with the exception captured
        legibly — the self-test never escapes as an unhandled traceback."""
        def boom(*args, **kwargs):
            raise RuntimeError("host denied exec")

        monkeypatch.setattr(LiterateInterpreter, "_run_resolved_cell", boom)
        session = LiterateSession(context)  # must not raise
        assert not session.selftest.ok
        value = next(p for p in session.selftest.probes if p.probe == "value")
        assert "RuntimeError: host denied exec" in value.got


class TestRefusalIsLegible:
    def test_refusal_is_legible(self, context, monkeypatch):
        _zeroing_exec(monkeypatch)
        session = LiterateSession(context)
        report = session.selftest

        text = report.describe()
        # Structured and readable: headline, per-probe verdicts, expected vs got.
        assert "kernel self-test: FAILED" in text
        assert "execution disabled" in text
        assert "probe 'value'" in text
        assert "expected:" in text
        assert "got:" in text
        # Healthy probes are reported too (full diagnostics, not first-failure).
        assert "probe 'round_trip'" in text
        # Not a raw traceback.
        assert "Traceback" not in text

        lines = report.refusal_lines()
        assert lines[0] == REFUSAL_HEADLINE
        assert any("expected" in ln and "got" in ln for ln in lines[1:])

    def test_healthy_report_is_legible_too(self, context):
        report = LiterateSession(context).selftest
        text = report.describe()
        assert "PASSED" in text
        assert "value" in text and "round_trip" in text
