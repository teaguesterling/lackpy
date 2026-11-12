"""Literate programming interpreter — top-level entry point.

Pipeline: document → parser.parse() → [Cell] → kernel.execute_cell() → stdout

The kernel compiles each cell type to Python via compiler._COMPILERS,
runs static analysis (compile check + AST name resolution), then runs
the cell through the document's CellSession (lackpy.run.cells): each
cell is validated as an extension of the program so far and runs in the
restricted namespace. Prose is rendered from text and expression
segments, each expression evaluated in the session. Captured stdout IS
the rendered document. A refusal ends the document (design D4), and so
does the first failure (spec E1).

This module owns the batch execution path (LiterateInterpreter.execute).
The streaming path uses kernel.StreamingDriver instead — it adds
recovery, plugin orchestration, and parse-as-you-generate execution.
"""

from __future__ import annotations

import ast
import time
from dataclasses import dataclass
from typing import Any

from ...lang import ValidationResult
from ..base import (
    ExecutionContext,
    InterpreterExecutionResult,
    InterpreterValidationResult,
)
from .annotations import kernel_note
from ...run.cells import CellSession
from ...run.trace import START_SUFFIX, Trace, TraceEntry
from .compiler import compile_cell, prose_segments
from .display import DEFAULT_DISPLAY_THRESHOLD
from .effects import as_grade, classify_cell, exceeds_ceiling, path_writes
from .kernel import LightweightKernel
from .kernel.driver import REFUSED
from .kernel.forgiveness import (
    ERROR_REIFIED,
    HOLE_OPENED,
    SOURCE_UNAVAILABLE,
    TIMED_OUT,
    describe_failure,
    reified_failures,
)
from .kernel.formats import render_markdown_from_cells
from .kernel.ledger import Ledger, LedgerEntry
from .kernel.static_analysis import BUILTIN_NAMES, collect_names
from .kernel.versions import SUPERSEDED, BindingVersions, value_kind
from .parser import Cell, ParseResult, parse
from .prompt import LITERATE_HINT


class LiterateInterpreter:
    """Literate programming interpreter.

    Takes a markdown document with ```lackpy code blocks, parses it
    into cells, and executes each cell incrementally through the kernel.
    The captured stdout is the rendered document — prose becomes print()
    calls, code executes and its output joins the stream.
    """

    name = "literate"
    description = "Literate programming — markdown with embedded lackpy code blocks"

    def system_prompt_hint(self) -> str:
        return LITERATE_HINT

    def validate(
        self,
        program: str,
        context: ExecutionContext,
    ) -> InterpreterValidationResult:
        """Validate a literate document by parsing it."""
        result = parse(program)
        if result.errors:
            return InterpreterValidationResult(
                valid=False,
                errors=result.errors,
            )
        return InterpreterValidationResult(valid=True)

    async def execute(
        self,
        program: str,
        context: ExecutionContext,
    ) -> InterpreterExecutionResult:
        """Execute a literate document and return the rendered output.

        Uses LightweightKernel directly (not StreamingDriver) because the
        batch path doesn't need streaming, recovery, or plugin orchestration.

        L7: before the fresh kernel accepts this document, it must PASS the
        startup self-test (:func:`.selftest.run_selftest`) — prove it can
        evaluate a known value, stop at each kind of failure, and that a
        render reparses.  A failed self-test disables the kernel and returns
        the legible structured refusal instead of running anything (fail
        closed: refusal, not garbage).
        """
        start = time.perf_counter()
        try:
            kernel = _build_kernel(context)
        except ValueError as exc:
            return InterpreterExecutionResult(
                success=False,
                error=f"Configuration error: {exc}",
                output_format="none",
                duration_ms=(time.perf_counter() - start) * 1000,
                metadata={"completed": False},
            )
        from .selftest import run_selftest

        report = run_selftest(kernel, self)
        if not report.ok:
            return InterpreterExecutionResult(
                success=False,
                error=report.describe(),
                output_format="none",
                duration_ms=(time.perf_counter() - start) * 1000,
                metadata={"completed": False, "selftest": report.summary()},
            )
        parsed = parse(program)
        if context.params and not parsed.errors:
            params_error = _apply_document_params(parsed, context.params)
            if params_error is not None:
                return InterpreterExecutionResult(
                    success=False,
                    error=f"Configuration error: {params_error}",
                    output_format="none",
                    duration_ms=(time.perf_counter() - start) * 1000,
                    metadata={"completed": False, "selftest": report.summary()},
                )
        result = await self._run_document(program, context, kernel, parsed=parsed)
        result.metadata["selftest"] = report.summary()
        return result

    async def _run_document(
        self,
        program: str,
        context: ExecutionContext,
        kernel: LightweightKernel,
        ledger: Ledger | None = None,
        versions: BindingVersions | None = None,
        parsed: ParseResult | None = None,
    ) -> InterpreterExecutionResult:
        """Run a literate document through an ALREADY-BUILT kernel: classify ->
        ceiling gate -> run cells (unbound-name check + execute) -> render.

        Shared by the batch ``execute()`` (a fresh kernel each call) and
        ``LiterateSession.step()`` (one persistent kernel threaded across rounds).
        The kernel owns the namespace, so this must NOT rebuild it -- doing so
        would wipe state threaded from prior rounds.

        ``ledger`` is the L1.0 event ledger this path records into (the batch
        path previously had no ledger wiring at all — only the streaming
        driver did).  The session threads ONE ledger across rounds; the batch
        ``execute()`` creates one per call.  Every cell outcome is recorded:
        ``executed`` / ``continue_requested`` on success; on failure,
        ``hole_opened`` (an unbound name), ``error_reified`` (an exception),
        ``source_unavailable`` (the ceiling gate) or ``timed_out`` (nothing
        silent). A failure binds nothing.

        Failure (spec E1): the run stops at its first failure (an exception,
        an unbound name, the ceiling gate, a timeout, or a prose expression
        that raises). The failure is ledgered and binds nothing; later cells
        do not run. ``success`` is False, and ``error`` is ``cell {index}
        ({first line}): {cause}``. The output and bindings from before the
        failed cell stand.

        ``versions`` is the L1.3 binding version history (ground rule 3:
        bindings are immutable versions).  Every name a cell binds is recorded as a new
        :class:`~.kernel.versions.BindingVersion`; re-asserting a name marks
        the prior version superseded and writes one ``superseded`` ledger
        entry with the version transition.  The exec-namespace dict stays the
        mutable latest-wins surface; immutability lives in this recorded
        layer.  The session threads ONE history across rounds (like the
        ledger); the batch path creates one per call.
        """
        start = time.perf_counter()
        ledger = ledger if ledger is not None else Ledger()
        versions = versions if versions is not None else BindingVersions()
        run_mark = len(ledger)

        # ``parsed``: the batch ``execute()`` passes the document it parsed and
        # applied the params to (final review I4); else parse it here.
        parsed = parsed if parsed is not None else parse(program)
        if parsed.errors:
            return InterpreterExecutionResult(
                success=False,
                error="Parse errors: " + "; ".join(parsed.errors),
                output_format="none",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        # Ceiling gate (conditional), PER CELL since L1.5: classify each cell
        # statically and record which cells exceed the ceiling -- those cells'
        # sources are UNAVAILABLE (the gate refuses to run their required
        # effect right now). The execution loop below ledgers the first such
        # cell as a failure and stops there (spec E1). The gate's
        # refuse-before-running guarantee is kept per cell: an over-ceiling
        # cell is never executed, so its effect never happens.
        #
        # Profile -> ceiling wiring unchanged: an explicit
        # context.config["grade_ceiling"] wins; otherwise the ceiling DEFAULTS
        # to the granted toolset's grade (ResolvedProfile.grade == its tools'
        # grade) -- a document may not exceed the effect grade of the tools its
        # profile granted. No ceiling and no toolset => no gate. Cells that
        # don't compile/parse are skipped here so the kernel reports the
        # precise syntax error rather than a misleading refusal. (This gates
        # only the batch/session path; the StreamingDriver path is untouched.)
        gate_violations = _gate_violations(parsed, context)

        # NOTE (L1.2): there is no write journal on this path. A failure
        # stops the document (spec E1) and is ledgered; a file the cells wrote
        # before it stays.
        output_parts: list[str] = []
        continue_requested = False
        assigned_names: set[str] = set()

        # D4: the text of the refusal that ended this run, if one did.
        refusal: str | None = None
        # Spec E1: the index and cause of the failure that ended this run.
        failure: tuple[int, str] | None = None
        # Each cell's latest value, for the cells whose latest run succeeded.
        values: dict[int, Any] = {}

        # Plan 3 Task 6 (D6): in the session worker, ``checkpoint(index)`` forks
        # at the top of each cell. The cell runs in the child (the hook returns
        # None there); the parent is the checkpoint, and returns a Resume only
        # when the child is gone. In-process there is no hook.
        #
        # The rule for the code after the hook (Task 6 fix round 1 ruling): the
        # child of the fork NEVER awaits. asyncio clears the running loop at
        # fork (``asyncio.events.on_fork``), and the loop's epoll fd is shared
        # with every ancestor checkpoint, so the cell path from here on must be
        # plain synchronous code. ``test_cell_path_after_the_hook_never_awaits``
        # pins it.
        #
        # ``cell_done()`` marks the end of a cell's run when no hook follows it
        # (after the last cell): the parent then stops the
        # cell's deadline, so host work is not charged to the cell.
        #
        # Task 7: a cell that runs no code (prose with no interpolation,
        # ``runs_no_code``) takes no checkpoint of its own. The hook runs at the
        # start of each segment: the first cell, and each cell after one that
        # can run code. A segment is a run of no-code cells and the one cell
        # after them that can run code (a code cell, or prose with an
        # interpolation), so its checkpoint is also the one before each of its
        # no-code cells (a resume from it renders only plain text again). A
        # timeout returned there belongs to that one cell, the only one in the
        # segment that runs code (fix round 1, C1): it is a failure, and the
        # plain text before it renders as usual.
        checkpoint = (context.config or {}).get("checkpoint")
        cell_done = (context.config or {}).get("cell_done")
        timed_out = False
        pending_timeout: Any = None
        index = 0
        while index < len(parsed.cells):
            cell = parsed.cells[index]
            segment_start = index == 0 or not runs_no_code(parsed.cells[index - 1])
            resume = (checkpoint(index) if checkpoint is not None and kernel.refusal is None
                      and segment_start else None)
            if resume is not None and resume.kind == "resume":
                # Task 7: continue at ``index`` over the new document's cells,
                # with the params the first run applied. The document is
                # validated as the fresh route validates it (the session
                # worker's ``_submit``), and refused with the same text (fix
                # round 1, I2).
                validation = self.validate(resume.document or "", context)
                if not validation.valid:
                    return InterpreterExecutionResult(
                        success=False,
                        error=f"Validation failed: {'; '.join(validation.errors)}",
                        output_format="none",
                        duration_ms=(time.perf_counter() - start) * 1000,
                    )
                reparsed = parse(resume.document or "")
                params_error = None
                if context.params:
                    params_error = _apply_document_params(reparsed, context.params)
                if params_error is not None:
                    return InterpreterExecutionResult(
                        success=False,
                        error=f"Configuration error: {params_error}",
                        output_format="none",
                        duration_ms=(time.perf_counter() - start) * 1000,
                    )
                parsed = reparsed
                gate_violations = _gate_violations(parsed, context)
                continue  # the hook returns None for the new cell at ``index``
            if resume is not None:  # "timeout": the child was stopped at the limit
                timed_out = True
                pending_timeout = resume
            if pending_timeout is not None and not runs_no_code(cell):
                # Spec E1: a timeout is a failure of the segment's one cell that
                # runs code. The document stops there.
                note = _record_timeout(cell, index, pending_timeout, kernel, ledger)
                output_parts.append(kernel_note(note) + "\n")
                failure = (index, note)
                break

            # The ceiling gate (L1.5): a cell over the ceiling is a failure. It
            # is ledgered (`source_unavailable`) and never executed, so the
            # refused effect never happens, and the document stops (spec E1).
            # Gate-first: the refusal is about the cell's required effects,
            # independent of whether its names would resolve.
            if index in gate_violations and kernel.refusal is None:
                reason = gate_violations[index]
                ledger.record(SOURCE_UNAVAILABLE, name=None, detail={
                    "cell_index": index, "cell_type": cell.cell_type, "reason": reason,
                })
                output_parts.append(kernel_note(reason) + "\n")
                failure = (index, reason)
                break

            outcome = self._run_resolved_cell(
                cell, index, kernel, ledger, versions, assigned_names
            )
            _keep_value(values, index, outcome)
            output_parts.append(outcome.rendered)
            if outcome.continue_requested:
                continue_requested = True
            if outcome.refusal is not None:
                # D4: a refusal ends the document. Later cells do not run (the
                # session is closed).
                refusal = outcome.refusal
                break
            if outcome.failure is not None:
                # Spec E1: the first failure ends the document. Later cells
                # do not run.
                failure = (index, outcome.failure)
                break
            index += 1

        # ``cell_done()`` marks the end of the last cell's run (see above).
        if cell_done is not None:
            cell_done()

        elapsed = (time.perf_counter() - start) * 1000

        raw_ns = kernel.get_namespace(assigned_names)
        variables = {
            k: v for k, v in raw_ns.items()
            if k in assigned_names
            and k not in _INTERNAL_NAMES
            and not callable(v)
        }
        if failure is not None:
            # Spec E2: the bindings are those from before the failed cell. The
            # failed cell recorded no version, so a name it rebound before it
            # failed keeps its last recorded value.
            for k in variables:
                current = versions.current(k)
                if current is not None:
                    variables[k] = current.value

        rendered = "".join(output_parts)

        # Aggregate layer: the run's Either is a VIEW DERIVED FROM THE LEDGER —
        # Left iff the run ledgered a failure — while the output, bindings, and
        # writes before the failure stand.  ``completed`` distinguishes a
        # finished run (Right or Left) from a pre-execution refusal (parse
        # errors), which returns earlier and never touches state.
        round_entries = ledger.entries()[run_mark:]
        failures = reified_failures(round_entries)

        # Canonical source-preserving render (conflict #5, Teague-approved): the
        # "document" the round-trip law holds for — authored prose + code SOURCE
        # preserved, with this round's LEDGER-derived failure notes routed
        # through the L2 [kernel] channel (inert on reparse). This — NOT the flat
        # stdout in `output` — is what the session feeds back to a stateless
        # writer, closing the exp1 loop. `output` stays flat stdout for the
        # Left-correction path and existing callers.
        rendered_markdown = render_markdown_from_cells(
            parsed.cells, parsed.frontmatter, _annotations_from_ledger(round_entries)
        )

        errors = [describe_failure(e) for e in failures]
        if failure is not None:
            # Ruling P2: the error names the failed cell, then the cause.
            failed_index, cause = failure
            errors = [f"cell {failed_index} ({_first_line(parsed.cells[failed_index])}): "
                      f"{cause}"]
        if refusal is not None:
            errors.insert(0, f"refused: {refusal}")
        return InterpreterExecutionResult(
            success=not failures and failure is None and refusal is None,
            output=rendered,
            error="; ".join(errors) or None,
            output_format="markdown",
            duration_ms=elapsed,
            metadata={
                "refused": refusal,
                # Plan 3 D7: a refusal outranks a timeout (it latches the
                # session; a timeout does not).
                "error_phase": ("refused" if refusal is not None
                                else "timeout" if timed_out else None),
                # Spec E2: the index of the failed cell, which the session
                # worker sends so the next resume starts no later than it.
                "failed_cell": failure[0] if failure is not None else None,
                "variables": variables,
                "continue_requested": continue_requested,
                "cell_count": len(parsed.cells),
                "completed": True,
                "ledger": ledger,
                "versions": versions,
                "rendered_markdown": rendered_markdown,
                # The session's trace: every Path call the cells made (D9).
                "trace": kernel.session.trace,
                # The last code cell's trailing-expression value, when its
                # latest run succeeded (Task 4 fix round 1, M1); else None.
                "value": _last_code_value(parsed.cells, values),
                "frontmatter": {
                    "echo": parsed.frontmatter.echo,
                    "output": parsed.frontmatter.output,
                    "interpreter": parsed.frontmatter.interpreter,
                },
            },
        )

    def _run_resolved_cell(
        self,
        cell: Cell,
        index: int,
        kernel: LightweightKernel,
        ledger: Ledger,
        versions: BindingVersions,
        assigned_names: set[str],
    ) -> _CellOutcome:
        """Run one cell through the per-cell pipeline: unbound-name check (spec E1)
        -> execute -> version every binding (L1.3).

        Returns the rendered fragment for this cell, whether it requested a
        ``@continue`` pause, and how it ended: a refusal, a failure (the
        document stops at either, spec E1) or a success with its value.  A
        failure binds nothing. Mutates ``kernel`` / ``ledger`` / ``versions`` /
        ``assigned_names`` in place.  The ceiling gate (L1.5) stays in the
        caller: a gated cell never reaches this body.
        """
        # Spec E1: a cell that reads a name no earlier cell bound is a failure.
        # It is ledgered, binds nothing, and is not executed. Not after a
        # refusal (review I1): a closed session refuses every cell, so the
        # kernel's refused result below is the outcome.
        if kernel.refusal is None:
            unbound = sorted(_unbound_names(cell, kernel))
            if unbound:
                for name in unbound:
                    ledger.record(HOLE_OPENED, name=name, detail={
                        "reason": "unbound", "cell_index": index,
                        "cell_type": cell.cell_type,
                    })
                names = ", ".join(f"'{n}'" for n in unbound)
                cause = (f"the name {names} is not bound. Bind it in an earlier cell"
                         if len(unbound) == 1 else
                         f"the names {names} are not bound. Bind them in earlier cells")
                return _CellOutcome(rendered=kernel_note(cause) + "\n",
                                    continue_requested=False, failure=cause)

        # Observe the execution we are doing anyway (D7): the Path writes in
        # this cell's slice of the session trace, recorded on the ``executed``
        # entry. The Path stand-in is a cell's only route to the filesystem.
        trace = kernel.session.trace.entries
        mark = len(trace)
        result = kernel.execute_cell(cell, index)
        observed = path_writes(trace[mark:])

        if result.error_phase == REFUSED:
            # D4: a refusal is not a failure to recover from. It is ledgered,
            # noted through the [kernel] channel, and the caller stops the
            # document.
            error = result.error or "refused"
            ledger.record(REFUSED, detail={
                "cell_index": index, "cell_type": cell.cell_type,
                "error": error, "error_phase": REFUSED,
            })
            parts = [result.output] if result.output else []
            parts.append(kernel_note(f"refused: {error}") + "\n")
            return _CellOutcome(
                rendered="".join(parts), continue_requested=False, refusal=error,
            )

        if not result.success:
            # Spec E1: the failure is ledgered and the document stops at this
            # cell. Nothing is bound for it. What the cell printed before it
            # failed stays in the output.
            error = result.error or "unknown error"
            ledger.record(ERROR_REIFIED, name=None, detail={
                "cell_index": index, "cell_type": cell.cell_type,
                "error": error, "error_phase": result.error_phase or "runtime",
            })
            parts: list[str] = []
            if result.output:
                parts.append(result.output)
            parts.append(kernel_note(f"error: {error}") + "\n")
            return _CellOutcome(
                rendered="".join(parts), continue_requested=False, failure=error,
            )

        parts = []
        if result.output:
            parts.append(result.output)

        assigned_names.update(result.namespace_delta.keys())

        # L1.3: every name this cell bound is a new immutable version; a name
        # with a prior version is a RE-ASSERTION — the prior version is marked
        # superseded and one ``superseded`` ledger entry records the transition
        # (nothing silent).  namespace_delta is identity-based, so rebinding the
        # identical object is unobservable and not versioned.
        cell_detail = {"cell_index": index, "cell_type": cell.cell_type}
        for name in sorted(result.namespace_delta):
            if name.startswith("_"):
                continue  # __continue_requested__ / internals, not bindings
            _record_assertion(
                name, result.namespace_delta[name], versions, ledger, cell_detail
            )

        cont = bool(result.namespace_delta.get("__continue_requested__"))
        # The executed entry records what the cell was observed to DO (when
        # anything was) — the ledger-visible face of the effect observation.
        exec_detail = dict(cell_detail)
        if observed:
            exec_detail["observed_effects"] = list(observed)
        ledger.record(
            "continue_requested" if cont else "executed", detail=exec_detail
        )
        return _CellOutcome(
            rendered="".join(parts), continue_requested=cont,
            ok=True, value=result.value,
        )


@dataclass(frozen=True)
class _CellOutcome:
    """The result of running one cell body (:meth:`_run_resolved_cell`).

    """

    rendered: str
    continue_requested: bool
    #: The refusal text when the cell's run refused (D4): the document stops.
    refusal: str | None = None
    #: The cause when the cell failed (spec E1): the document stops.
    failure: str | None = None
    #: Whether the cell ran and succeeded; then ``value`` is its trailing
    #: expression's value (``CellResult.value``), else None.
    ok: bool = False
    value: Any = None


def _keep_value(values: dict[int, Any], index: int, outcome: _CellOutcome) -> None:
    """Record the cell's value after a run that succeeded; forget it otherwise."""
    if outcome.ok:
        values[index] = outcome.value
    else:
        values.pop(index, None)


#: Cell types that are never the output cell (final review M3): prose, and the
#: annotation cells whose compiled source is lackpy's, not the author's (a
#: ``@write`` cell's trailing ``write_text`` returns a character count).
_NOT_OUTPUT_CELLS = frozenset({"prose", "read", "write", "diff", "continue"})


def _last_code_value(cells: list[Cell], values: dict[int, Any]) -> Any:
    """The value of the last code cell (not prose, not an annotation cell), or None
    when it did not succeed."""
    code = [i for i, c in enumerate(cells) if c.cell_type not in _NOT_OUTPUT_CELLS]
    return values.get(code[-1]) if code else None


def _record_assertion(
    name: str,
    value: Any,
    versions: BindingVersions,
    ledger: Ledger,
    cell_detail: dict[str, Any],
) -> None:
    """Record one binding assertion in the version history (L1.3).

    The single choke point for the version history: every bind — an executed
    cell's namespace delta — passes through here.  A first assertion just
    records version 1; a RE-ASSERTION marks the prior version superseded and
    writes the one ``superseded`` ledger entry, whose ``detail`` carries the
    version transition (mirroring AIDR's ``_aidr_bindings``
    version/``superseded_by`` model) plus the kind of the superseded value.
    """
    new, prior = versions.assert_binding(name, value)
    if prior is not None:
        ledger.record(
            SUPERSEDED,
            name=name,
            detail={
                "from_version": prior.version,
                "to_version": new.version,
                "prior": value_kind(prior.value),
                **cell_detail,
            },
        )


def _annotation_text(entry: LedgerEntry) -> str | None:
    """The [kernel]-channel note text for ONE ledger entry, or ``None``.

    The ledger→annotation projection that closes exp1 (L1 ledger → L2 channel):
    a notable failure / versioning entry becomes one inert channel note on
    the cell it concerns; a plain ``executed`` / ``continue_requested`` entry
    carries no note (the source itself is the record).
    :func:`~.kernel.forgiveness.describe_failure` supplies the failure text —
    so the error string (e.g. the exception type) survives into the canonical
    render — and a supersede gets a one-liner.
    """
    et = entry.entry_type
    if et in (HOLE_OPENED, ERROR_REIFIED, SOURCE_UNAVAILABLE, TIMED_OUT):
        return describe_failure(entry)
    if et == REFUSED:
        return f"refused: {entry.detail.get('error', 'refused')}"
    if et == SUPERSEDED:
        name = entry.name or "binding"
        frm = entry.detail.get("from_version")
        to = entry.detail.get("to_version")
        prior = entry.detail.get("prior", "value")
        return f"superseded: '{name}' v{frm}→v{to} (prior {prior})"
    return None


def _annotations_from_ledger(entries: list[LedgerEntry]) -> dict[int, list[str]]:
    """Project a round's ledger slice to ``cell_index -> [note text, …]``.

    The single source the canonical render draws its ``[kernel]`` notes from —
    NO parallel annotation emitter.  Entries with no ``cell_index`` (or no
    notable text) contribute nothing; the render passes the result straight to
    :func:`~.kernel.formats.render_markdown_from_cells`.
    """
    ann: dict[int, list[str]] = {}
    for entry in entries:
        index = entry.detail.get("cell_index")
        if index is None:
            continue
        note = _annotation_text(entry)
        if note:
            ann.setdefault(index, []).append(note)
    return ann


_UNBOUND_EXEMPT_NAMES = frozenset({
    # Kernel-injected at execution time for @continue cells; never an unknown
    # name even though it is absent from the namespace between cells.
    "__literate_continue__",
})


def _unbound_names(cell: Cell, kernel: LightweightKernel) -> set[str]:
    """The names ``cell`` reads that no earlier cell bound (spec E1).

    The rules are the kernel's own static check (``check_cell``): a name the
    cell itself defines is bound. A syntax error or an unknown cell type is not
    a name problem: it returns no names, and the kernel reports the error.
    """
    try:
        tree = ast.parse(compile_cell(cell))
    except (ValueError, SyntaxError):
        return set()
    defined, referenced = collect_names(tree)
    return referenced - kernel.known_names() - BUILTIN_NAMES - defined - _UNBOUND_EXEMPT_NAMES


def runs_no_code(cell: Cell) -> bool:
    """True for a prose cell with no interpolation: it renders without running code.

    Such a cell takes no checkpoint, and a timeout never belongs to it (Plan 3
    Task 7). The prefix rule (``lackpy.run.sessions``) uses the same test.
    """
    return cell.cell_type == "prose" and all(seg.expr is None for seg in prose_segments(cell))


def _first_line(cell: Cell) -> str:
    """The first non-blank line of ``cell``, as an error names the cell (spec E1)."""
    lines = cell.content.strip().splitlines()
    return lines[0].strip() if lines else cell.cell_type


def _gate_violations(parsed: ParseResult, context: ExecutionContext) -> dict[int, str]:
    """The ceiling gate's verdict for each cell that exceeds the ceiling (L1.5).

    Profile -> ceiling wiring: an explicit ``context.config["grade_ceiling"]``
    wins; otherwise the ceiling defaults to the granted toolset's grade. No
    ceiling and no toolset => no gate. Cells that don't compile/parse are
    skipped, so the kernel reports the precise syntax error rather than a
    misleading refusal.
    """
    raw_ceiling = (context.config or {}).get("grade_ceiling")
    if raw_ceiling is None:
        raw_ceiling = getattr(context.tools, "grade", None)
    gate_violations: dict[int, str] = {}
    # Why the ceiling is there, when the caller says (final review M5): a
    # read-only profile's refusal says so, not only the grade numbers.
    ceiling_reason = (context.config or {}).get("grade_ceiling_reason")
    if raw_ceiling is not None:
        ceiling = as_grade(raw_ceiling)
        for index, cell in enumerate(parsed.cells):
            try:
                src = compile_cell(cell)
                ast.parse(src)
            except (ValueError, SyntaxError):
                continue
            effects = classify_cell(cell.cell_type, src)
            violation = exceeds_ceiling(effects, ceiling)
            if violation:
                if ceiling_reason:
                    violation = f"{violation}: {ceiling_reason}"
                gate_violations[index] = f"effect ceiling exceeded: {violation}"
    return gate_violations


def _timeout_writes(entries: list[TraceEntry]) -> list[str]:
    """The writes a stopped cell's trace shows, as the paths the note names.

    A write with a done entry that succeeded is named by its path. A write with
    a start entry and no done entry (the cell was stopped inside it) is named
    ``"{path} (write started, not confirmed)"`` (Review Focus 5). A write that
    failed is not named: it did not write.
    """
    done_steps = {e.step for e in entries if not e.tool.endswith(START_SUFFIX)}
    names: dict[str, None] = {}
    for e in entries:
        tool = e.tool[:-len(START_SUFFIX)] if e.tool.endswith(START_SUFFIX) else e.tool
        if tool not in ("Path.write_text", "Path.write_bytes"):
            continue
        path = str(e.args[0]) if isinstance(e.args, list) and e.args else "?"
        if e.tool.endswith(START_SUFFIX):
            if e.step not in done_steps:
                names.setdefault(f"{path} (write started, not confirmed)", None)
        else:
            if e.success:
                names.setdefault(path, None)
    return list(names)


def _record_timeout(
    cell: Cell,
    index: int,
    resume: Any,
    kernel: LightweightKernel,
    ledger: Ledger,
) -> str:
    """Record a cell stopped at its time limit, and return its note (spec E1, Plan 3 D7).

    Called in the checkpoint, which holds the state from before the cell. The
    cell's own process is gone, so ``resume.entries`` (its trace entries and
    start entries) is the record of what it did. The entries join the
    session's trace (start entries do not). One ``timed_out`` ledger entry
    records the note: ``timeout: the cell took longer than {t:g} s and was
    stopped``, plus ``; it wrote {paths} before it was stopped`` when the trace
    shows writes. Nothing is bound: the document stops at this cell.
    """
    entries = list(resume.entries or [])
    kernel.session.trace.entries.extend(e for e in entries if not e.tool.endswith(START_SUFFIX))
    paths = _timeout_writes(entries)
    note = f"timeout: the cell took longer than {resume.seconds:g} s and was stopped"
    if paths:
        note += f"; it wrote {', '.join(paths)} before it was stopped"
    ledger.record(TIMED_OUT, name=None, detail={
        "cell_index": index, "cell_type": cell.cell_type, "note": note,
        "error_phase": "timeout", "seconds": resume.seconds,
    })
    return note


_INTERNAL_NAMES = frozenset({
    "__literate_continue__", "__builtins__",
    "__continue_requested__",
})


def _build_kernel(context: ExecutionContext) -> LightweightKernel:
    """Build the kernel for a literate document: one ``CellSession`` (D1).

    The session's ``PathPolicy`` is ``context.config["path_policy"]``; with
    none, ``pathlib`` refuses (fail closed). (Literate stepy Task 4 makes the
    service pass the policy it loads for the workspace.) The display threshold is
    ``context.config["display_threshold"]``.

    ``context.params`` bind no name here (final review I4, design D15): the
    batch ``execute()`` writes them into the document's ``params`` literal
    (:func:`_apply_document_params`). A granted toolset's callables do not join
    the namespace: a literate cell can do nothing a lackpy program cannot.
    """
    config = context.config or {}
    # The caller's rules (``config["extra_rules"]``, rule callables) apply to
    # every cell's cumulative validation and every prose expression (final
    # review I3).
    # In the session worker, ``config["trace_listener"]`` streams each Path
    # call to the parent as it happens (Plan 3 D4, Task 6).
    trace = Trace(listener=config.get("trace_listener"))
    session = CellSession(path_policy=config.get("path_policy"),
                          extra_rules=config.get("extra_rules"), trace=trace)
    return LightweightKernel(
        session,
        display_threshold=config.get("display_threshold", DEFAULT_DISPLAY_THRESHOLD),
    )


#: The cell types whose compiled source is the cell's own text, so a params
#: literal in one of them can be rewritten in place.
_SOURCE_CELLS = frozenset({"code", "hidden", "gather", "scratch"})


def _apply_document_params(parsed: ParseResult, params: dict[str, Any]) -> str | None:
    """Replace the defaults of the document's ``params`` literal with ``params``.

    Final review I4: the literate axis applies passed params as ``run_program``
    does (design D15), with ``RestrictedRunner``'s own AST rewrite and errors
    (``runner._apply_params``). The program it checks is every code cell of the
    document, joined: ``params`` must be bound exactly once, by a top-level
    ``params = {"name": <literal>, ...}`` in one cell. Only that literal's text
    is replaced in its cell, so the rest of the cell is kept. The literal itself
    is written back by ``ast.unparse`` on one line: comments inside a multi-line
    literal are lost (the run is the same).
    Returns the error text, or None; on an error no cell is changed.
    """
    from ...run.runner import _apply_params, _params_literal

    trees: list[tuple[Cell, ast.Module]] = []
    for cell in parsed.cells:
        if cell.cell_type == "prose":
            continue
        try:
            trees.append((cell, ast.parse(compile_cell(cell))))
        except (ValueError, SyntaxError):
            continue  # the kernel reports the error when the cell runs
    program = ast.Module(body=[stmt for _, tree in trees for stmt in tree.body], type_ignores=[])
    literal = _params_literal(program)
    error = _apply_params(program, params)
    if error is not None:
        return error
    assert literal is not None  # _apply_params found it
    holder = next(cell for cell, tree in trees
                  if any(getattr(stmt, "value", None) is literal for stmt in tree.body))
    if holder.cell_type not in _SOURCE_CELLS:
        return f"the params literal must be in a code cell, not a @{holder.cell_type} cell"
    holder.content = _splice(holder.content, literal, ast.unparse(literal))
    return None


def _splice(source: str, node: ast.AST, text: str) -> str:
    """``source`` with the span of ``node`` replaced by ``text``.

    ``ast`` column offsets count UTF-8 bytes, so each is converted on its line.
    """
    lines = source.splitlines(keepends=True)

    def offset(lineno: int, col: int) -> int:
        line = lines[lineno - 1]
        return sum(len(x) for x in lines[:lineno - 1]) + len(line.encode()[:col].decode())

    start = offset(node.lineno, node.col_offset)
    end = offset(node.end_lineno, node.end_col_offset)  # type: ignore[arg-type]
    return source[:start] + text + source[end:]


def validate_document(program: str, extra_rules: list | None = None) -> ValidationResult:
    """Validate a literate document cell by cell, without running it (final review M1).

    The checks a run makes before each cell executes: the document must parse,
    and each code cell must extend the program so far as one lackpy program
    (the caller's ``extra_rules`` too), with annotation cells importing
    ``Path`` as the kernel does. Each prose expression is checked as one line
    after the program. ``@diff`` and ``@continue`` cells hold no program text.
    A cell that fails does not join the program, as at run time. Each
    violation's message starts with the document cell
    (:func:`.kernel.lightweight.cell_label`); its ``line`` is the line in the
    cell, or None. Name resolution (unbound names) is a runtime matter and is not
    checked.
    """
    from ...lang import ValidationResult, Violation
    from .compiler import binds_path, prose_segments, without_path_import
    from .kernel.lightweight import cell_label

    parsed = parse(program)
    if parsed.errors:
        violations = [Violation("literate.parse", None, e) for e in parsed.errors]
        return ValidationResult(valid=False, errors=[v.message for v in violations],
                                violations=violations)
    session = CellSession(path_policy=None, extra_rules=extra_rules)
    violations = []
    for index, cell in enumerate(parsed.cells):
        label = cell_label(cell, index)
        if cell.cell_type == "prose":
            found = [v for seg in prose_segments(cell) if seg.expr is not None
                     for v in session.check_expr(seg.expr)]
        elif cell.cell_type in ("diff", "continue"):
            continue
        else:
            source = compile_cell(cell)
            if cell.cell_type in ("read", "write") and binds_path(session.program):
                source = without_path_import(source)
            if not source.strip():
                continue
            found = session.check_cell(source, label=label)
        violations += [Violation(v.rule_id, v.line, f"{label}: {v.message}", v.hint)
                       for v in found]
    return ValidationResult(valid=not violations, errors=[v.message for v in violations],
                            violations=violations)


# Re-export the multi-round fold API. Imported at the end (after LiterateInterpreter
# and _build_kernel are defined) so session.py's lazy imports resolve cleanly.
from .session import (  # noqa: E402,F401
    COMPUTE_CONTINUE_MARKER,
    CONTINUE_MARKER,
    LiterateSession,
    StepResult,
    StopScanner,
    split_at_continue,
    strip_overlap,
    strip_think,
)
