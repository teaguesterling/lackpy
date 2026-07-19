"""L7 kernel self-test — fail-closed startup probe battery.

Before the batch/session path accepts real work, the kernel must PROVE it can
evaluate: a host eval-restriction once silently zeroed every eval, and the
kernel produced garbage instead of refusing.  This module is the refusal.

The battery runs one probe per semantic class the kernel supports, each
through the REAL surface (nothing reimplemented):

* ``value`` — a known evaluation (``2 + 2``) actually computes ``4``, through
  the shared per-cell pipeline (:meth:`LiterateInterpreter._run_resolved_cell`)
  AND the raw ``inspect`` eval surface.  Catches the "eval silently zeroed"
  failure directly.
* ``hole`` — an unknown name reifies a typed :class:`~.kernel.forgiveness.Hole`
  (L1.1) and is ledgered, rather than crashing or silently succeeding.
* ``error`` — a runtime error reifies an
  :class:`~.kernel.forgiveness.ErrorValue` (L1.2) and is ledgered, rather than
  aborting.
* ``unavailable`` — a gate-refused source reifies an
  :class:`~.kernel.forgiveness.Unavailable` (L1.5) and is ledgered.  The probe
  drives the reification surface with a synthetic gate reason: the ceiling
  gate's *classification* is policy, the semantic class is the reified value.
* ``round_trip`` — a rendered probe document (prose + code + a
  ``[kernel]``-channel note) reparses cleanly (the L2 round-trip law): source
  preserved, channel note inert.

Probe hygiene: probes run against the REAL kernel (the one about to accept
work) but with a THROWAWAY ledger/version-history, and the kernel namespace is
``snapshot()``-ed before and ``restore()``-d after — a probe's ``hole_opened``
entry must never make the session's first real round Left, and probe names
must never leak into scope.

FAIL-CLOSED: any failed probe disables the kernel
(:meth:`~.kernel.lightweight.LightweightKernel.disable`) — every subsequent
work attempt returns a legible structured refusal naming the failed probe(s),
modeled on the effect-ceiling gate's refuse-before-running precedent.  A probe
that RAISES is a failed probe with the exception as its ``got`` — the
self-test itself never escapes as a traceback (failing legibly is the
feature).

This guards the batch/session execution path
(:meth:`LiterateInterpreter.execute` and :class:`~.session.LiterateSession`),
which is where the probed forgiveness semantics live; the streaming driver
does not construct its own kernel, but any kernel a failed self-test disabled
refuses work on every path that holds it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

from .annotations import KERNEL_OPEN
from .kernel.forgiveness import (
    ERROR_REIFIED,
    HOLE_OPENED,
    SOURCE_UNAVAILABLE,
    ErrorValue,
    Hole,
    Unavailable,
)
from .kernel.formats import render_markdown_from_cells
from .kernel.ledger import Ledger
from .kernel.lightweight import LightweightKernel
from .kernel.versions import BindingVersions
from .parser import Cell, Frontmatter, parse

#: Ledger entry types for the observable self-test outcome (recorded once, at
#: session construction).  Like every entry type, queryable:
#: ``ledger.query(entry_type=SELFTEST_FAILED)``.
SELFTEST_PASSED = "selftest_passed"
SELFTEST_FAILED = "selftest_failed"

#: The refusal headline — the structured refusal's first line everywhere the
#: disabled kernel/session/interpreter refuses work.
REFUSAL_HEADLINE = (
    "kernel self-test failed — execution disabled: "
    "refusing work instead of risking garbage"
)

#: Reserved name prefix for probe bindings.  Probes bind under these names and
#: the namespace is restored afterward; user documents should not rely on them.
PROBE_NAME_PREFIX = "lackpy_selftest_"


@dataclass(frozen=True)
class ProbeResult:
    """The outcome of one probe: what was expected vs what the kernel did."""

    probe: str
    semantic_class: str
    ok: bool
    expected: str
    got: str


@dataclass(frozen=True)
class SelfTestReport:
    """The observable self-test status: overall verdict + per-probe results."""

    ok: bool
    probes: tuple[ProbeResult, ...]

    def failed(self) -> tuple[ProbeResult, ...]:
        return tuple(p for p in self.probes if not p.ok)

    def refusal_lines(self) -> list[str]:
        """The structured refusal a refused work attempt returns: the headline
        plus one line per failed probe (which probe, expected vs got)."""
        lines = [REFUSAL_HEADLINE]
        for p in self.failed():
            lines.append(
                f"probe '{p.probe}' ({p.semantic_class}) failed: "
                f"expected {p.expected}; got {p.got}"
            )
        return lines

    def describe(self) -> str:
        """The legible multi-line report (human/model-readable, no traceback)."""
        if self.ok:
            return (
                f"kernel self-test: PASSED — "
                f"{len(self.probes)}/{len(self.probes)} probes ok "
                f"({', '.join(p.probe for p in self.probes)})"
            )
        lines = ["kernel self-test: FAILED — execution disabled"]
        for p in self.probes:
            if p.ok:
                lines.append(f"  probe '{p.probe}' ({p.semantic_class}): ok")
            else:
                lines.append(f"  probe '{p.probe}' ({p.semantic_class}): FAIL")
                lines.append(f"    expected: {p.expected}")
                lines.append(f"    got: {p.got}")
        lines.append(
            "refusing all work: a kernel that cannot pass its own probes "
            "would produce garbage, not results"
        )
        return "\n".join(lines)

    def summary(self) -> dict[str, Any]:
        """Plain-data view for ledger detail / result metadata."""
        return {"ok": self.ok, "probes": [asdict(p) for p in self.probes]}


# --- the probe battery ------------------------------------------------------
#
# Each probe returns (ok, expected, got).  ``interpreter`` is the
# LiterateInterpreter whose _run_resolved_cell IS the real per-cell pipeline
# (forgiveness pre-pass -> execute -> error reification -> versioning) — the
# probes invoke it rather than reimplementing any semantics.

_ProbeFn = Callable[[LightweightKernel, Any, Ledger, BindingVersions],
                    tuple[bool, str, str]]


def _run_cell(
    interpreter: Any,
    source: str,
    kernel: LightweightKernel,
    ledger: Ledger,
    versions: BindingVersions,
) -> None:
    cell = Cell(cell_type="code", content=source)
    interpreter._run_resolved_cell(cell, 0, kernel, ledger, versions, set())


def _probe_value(
    kernel: LightweightKernel, interpreter: Any,
    ledger: Ledger, versions: BindingVersions,
) -> tuple[bool, str, str]:
    expected = "lackpy_selftest_value = 2 + 2 binds 4 (and inspect('2 + 2') == '4')"
    _run_cell(interpreter, "lackpy_selftest_value = 2 + 2", kernel, ledger, versions)
    bound = kernel.lookup("lackpy_selftest_value", default="<unbound>")
    if bound != 4:
        return False, expected, f"bound {bound!r}"
    inspected = kernel.inspect("2 + 2")
    if inspected != "4":
        return False, expected, f"inspect('2 + 2') returned {inspected!r}"
    return True, expected, "bound 4"


def _probe_hole(
    kernel: LightweightKernel, interpreter: Any,
    ledger: Ledger, versions: BindingVersions,
) -> tuple[bool, str, str]:
    expected = (
        "an unknown referenced name reifies a typed Hole "
        "and is ledgered (hole_opened)"
    )
    _run_cell(
        interpreter,
        "lackpy_selftest_chained = lackpy_selftest_missing",
        kernel, ledger, versions,
    )
    origin = kernel.lookup("lackpy_selftest_missing", default="<unbound>")
    chained = kernel.lookup("lackpy_selftest_chained", default="<unbound>")
    if not isinstance(origin, Hole):
        return False, expected, f"'lackpy_selftest_missing' bound {origin!r}"
    if not isinstance(chained, Hole) or not chained.blocked_by:
        return False, expected, f"chained binding was {chained!r}"
    if not ledger.query(entry_type=HOLE_OPENED):
        return False, expected, "hole bound but no hole_opened ledger entry"
    return True, expected, f"{origin!r}"


def _probe_error(
    kernel: LightweightKernel, interpreter: Any,
    ledger: Ledger, versions: BindingVersions,
) -> tuple[bool, str, str]:
    expected = (
        "a runtime error (1 // 0) reifies an ErrorValue "
        "and is ledgered (error_reified)"
    )
    _run_cell(interpreter, "lackpy_selftest_err = 1 // 0", kernel, ledger, versions)
    bound = kernel.lookup("lackpy_selftest_err", default="<unbound>")
    if not isinstance(bound, ErrorValue):
        return False, expected, f"'lackpy_selftest_err' bound {bound!r}"
    if "ZeroDivisionError" not in bound.error:
        return False, expected, f"error value carries {bound.error!r}"
    if not ledger.query(entry_type=ERROR_REIFIED):
        return False, expected, "error reified but no error_reified ledger entry"
    return True, expected, f"{bound!r}"


def _probe_unavailable(
    kernel: LightweightKernel, interpreter: Any,
    ledger: Ledger, versions: BindingVersions,
) -> tuple[bool, str, str]:
    expected = (
        "a gate-refused source reifies an Unavailable "
        "and is ledgered (source_unavailable)"
    )
    # The reification surface is the semantic class; the ceiling gate's
    # classification (policy) is exercised by its own tests.  Lazy import:
    # the reifier lives in the package __init__, which imports lazily itself.
    from . import _reify_unavailable_for_cell

    cell = Cell(cell_type="code", content="lackpy_selftest_gated = 1")
    _reify_unavailable_for_cell(
        cell, 0, "effect ceiling exceeded: selftest probe (synthetic)",
        kernel, ledger, versions, set(),
    )
    bound = kernel.lookup("lackpy_selftest_gated", default="<unbound>")
    if not isinstance(bound, Unavailable):
        return False, expected, f"'lackpy_selftest_gated' bound {bound!r}"
    if not ledger.query(entry_type=SOURCE_UNAVAILABLE):
        return False, expected, "value bound but no source_unavailable ledger entry"
    return True, expected, f"{bound!r}"


def _probe_round_trip(
    kernel: LightweightKernel, interpreter: Any,
    ledger: Ledger, versions: BindingVersions,
) -> tuple[bool, str, str]:
    expected = (
        "a rendered probe doc (prose + code + [kernel] note) reparses cleanly: "
        "source preserved, channel note inert"
    )
    cells = [
        Cell(cell_type="prose", content="Selftest round-trip probe prose."),
        Cell(cell_type="code", content="lackpy_selftest_rt = 1"),
    ]
    rendered = render_markdown_from_cells(
        cells, Frontmatter(), {1: ["selftest: round-trip probe note"]}
    )
    reparsed = parse(rendered)
    if reparsed.errors:
        return False, expected, f"reparse errors: {'; '.join(reparsed.errors)}"
    code_ok = any(
        c.cell_type == "code" and c.content.strip() == "lackpy_selftest_rt = 1"
        for c in reparsed.cells
    )
    if not code_ok:
        return False, expected, (
            "code source did not survive the round trip: "
            f"{[(c.cell_type, c.content) for c in reparsed.cells]!r}"
        )
    leaked = [c for c in reparsed.cells if KERNEL_OPEN in c.content]
    if leaked:
        return False, expected, (
            f"[kernel] channel note re-parsed as content: {leaked[0].content!r}"
        )
    return True, expected, "round trip clean"


#: The battery: (probe name, semantic class it covers, probe fn).  Order is
#: report order; ALL probes run (even after a failure) so the report carries
#: full diagnostics before the kernel is disabled once at the end.
_PROBES: tuple[tuple[str, str, _ProbeFn], ...] = (
    ("value", "known evaluation computes the right value", _probe_value),
    ("hole", "unknown name reifies a typed Hole (L1.1)", _probe_hole),
    ("error", "runtime error reifies an ErrorValue (L1.2)", _probe_error),
    ("unavailable", "refused source reifies an Unavailable (L1.5)",
     _probe_unavailable),
    ("round_trip", "rendered document reparses cleanly (L2)", _probe_round_trip),
)


def run_selftest(kernel: LightweightKernel, interpreter: Any) -> SelfTestReport:
    """Run the probe battery against ``kernel`` BEFORE it accepts work.

    Returns the report; on any failure the kernel is DISABLED
    (fail-closed) so even a caller that ignores the report cannot execute
    through it.  Never raises: a probe that raises is a failed probe.
    """
    snapshot = kernel.snapshot()
    results: list[ProbeResult] = []
    try:
        for name, semantic_class, fn in _PROBES:
            ledger = Ledger(session_id="selftest")
            versions = BindingVersions()
            try:
                ok, expected, got = fn(kernel, interpreter, ledger, versions)
            except Exception as e:  # a raising probe is a FAILED probe
                ok, expected, got = (
                    False,
                    "probe completes without raising",
                    f"probe raised {type(e).__name__}: {e}",
                )
            results.append(
                ProbeResult(
                    probe=name, semantic_class=semantic_class,
                    ok=ok, expected=expected, got=got,
                )
            )
    finally:
        kernel.restore(snapshot)

    report = SelfTestReport(ok=all(p.ok for p in results), probes=tuple(results))
    if not report.ok:
        kernel.disable(
            "self-test failed: probe(s) "
            + ", ".join(p.probe for p in report.failed())
        )
    return report
