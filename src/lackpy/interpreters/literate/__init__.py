"""Literate programming interpreter — top-level entry point.

Pipeline: document → parser.parse() → [Cell] → kernel.execute_cell() → stdout

The kernel compiles each cell type to Python via compiler._COMPILERS,
runs static analysis (compile check + AST name resolution), then runs
the cell through the document's CellSession (lackpy.run.cells): each
cell is validated as an extension of the program so far and runs in the
restricted namespace. Prose is rendered from text and expression
segments, each expression evaluated in the session. Captured stdout IS
the rendered document. A refusal ends the document (design D4).

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
from .compiler import compile_cell
from .display import DEFAULT_DISPLAY_THRESHOLD
from .effects import as_grade, classify_cell, exceeds_ceiling, path_writes
from .kernel import CellResult, LightweightKernel
from .kernel.driver import REFUSED
from .kernel.forgiveness import (
    ERROR_REIFIED,
    HOLE_OPENED,
    SOURCE_UNAVAILABLE,
    TIMED_OUT,
    ErrorValue,
    Hole,
    Unavailable,
    describe_failure,
    reified_failures,
)
from .kernel.formats import render_markdown_from_cells
from .kernel.ledger import Ledger, LedgerEntry
from .kernel.reactive import DIRTY, DependencyGraph
from .kernel.static_analysis import BUILTIN_NAMES, collect_bindings, collect_names
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
        evaluate a known value and reify each forgiveness class, and that a
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
        ceiling gate -> run cells (forgiveness pre-pass + execute) -> render.

        Shared by the batch ``execute()`` (a fresh kernel each call) and
        ``LiterateSession.step()`` (one persistent kernel threaded across rounds).
        The kernel owns the namespace, so this must NOT rebuild it -- doing so
        would wipe state threaded from prior rounds.

        ``ledger`` is the L1.0 event ledger this path records into (the batch
        path previously had no ledger wiring at all — only the streaming
        driver did).  The session threads ONE ledger across rounds; the batch
        ``execute()`` creates one per call.  Every cell outcome is recorded:
        ``executed`` / ``continue_requested`` on success, ``hole_opened`` /
        ``error_reified`` when a failure is reified (nothing silent).

        Forgiveness (two-layer contract, conflict #2 option (c)): a cell whose
        names cannot resolve binds typed holes and is skipped; the aggregate
        Either for the run is DERIVED from the ledger at the end — ``success``
        is False iff any failure was reified, while the rendered output,
        namespace bindings, and file writes all stand.

        ``versions`` is the L1.3 binding version history (ground rule 3:
        bindings are immutable versions).  Every name a cell binds — a real
        value, a hole, or an error value — is recorded as a new
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
        # effect right now) and are reified in the execution loop below
        # instead of refusing the whole document up front. The gate's
        # refuse-before-running guarantee is kept per cell: an over-ceiling
        # cell is never executed, so its effect never happens; within-ceiling
        # cells run normally (no abort).
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

        # NOTE (L1.2, contract change): the always-on write journal that
        # rolled a failed document's file writes back is RETIRED on this path
        # together with the abort itself -- under the two-layer forgiveness
        # contract a cell failure reifies as a value + ledger entry and the
        # run completes, so there is no failure point left to roll back from.
        # File writes stand, like every other binding (state kept, failure
        # ledgered).
        output_parts: list[str] = []
        continue_requested = False
        assigned_names: set[str] = set()

        # L1.4: the derived dependency graph for this document (reactive
        # dirty-subgraph re-execution).  Built from the SAME static def/ref
        # sets the forgiveness pre-pass uses (collect_names/collect_bindings) —
        # no IR, a side-structure over the source-string cells.  Populated for
        # every cell below (executed or reified) so the post-loop closure is
        # complete; consumed by the re-exec pass after the linear walk.
        graph = DependencyGraph()

        # Reactive re-exec mode (strict|permissive, default strict) — read
        # from context.config like display_threshold / grade_ceiling, so it
        # threads unchanged through both the batch execute() and the session
        # (which passes its one context every round).  Validated up front: a
        # typo'd mode must fail loudly, not silently mean "strict" — and
        # loudly HERE means the structured pre-execution refusal parse
        # errors use (success=False + error, nothing executed), not an
        # uncaught ValueError out of execute()/step().
        try:
            reactive_mode = _reactive_mode(context)
        except ValueError as exc:
            return InterpreterExecutionResult(
                success=False,
                error=f"Configuration error: {exc}",
                output_format="none",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        # What each executed cell was OBSERVED to do (the Path writes in its
        # slice of the session trace; see effects.path_writes) — the evidence
        # the re-exec pass gates on.  A cell absent from this map never
        # executed this round (reified/skipped), so re-running it is a FIRST
        # execution, not a replay.
        observed_effects: dict[int, tuple[str, ...]] = {}
        # The CellSession id of each cell's run, so the re-exec pass re-runs a
        # cell's validated code (``rerun_cell``) instead of appending it again.
        cell_ids: dict[int, int | None] = {}
        # D4: the text of the refusal that ended this run, if one did.
        refusal: str | None = None
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
        # (after the last cell, after a re-run): the parent then stops the
        # cell's deadline, so host work is not charged to the cell.
        #
        # Task 7: a prose cell takes no checkpoint of its own. The hook runs at
        # the start of each segment: the first cell, and each cell after a
        # non-prose cell. A segment is a run of prose cells and the cell after
        # them, so its checkpoint is also the one before each of its prose
        # cells (a resume from it renders only prose again). A timeout returned
        # there belongs to the segment's first non-prose cell: the prose before
        # it renders as usual.
        checkpoint = (context.config or {}).get("checkpoint")
        cell_done = (context.config or {}).get("cell_done")
        timed_out = False
        pending_timeout: Any = None
        index = 0
        while index < len(parsed.cells):
            cell = parsed.cells[index]
            segment_start = index == 0 or parsed.cells[index - 1].cell_type != "prose"
            resume = (checkpoint(index) if checkpoint is not None and kernel.refusal is None
                      and segment_start else None)
            if resume is not None and resume.kind == "resume":
                # Task 7: continue at ``index`` over the new document's cells,
                # with the params the first run applied.
                reparsed = parse(resume.document or "")
                params_error = None
                if context.params and not reparsed.errors:
                    params_error = _apply_document_params(reparsed, context.params)
                if reparsed.errors or params_error is not None:
                    return InterpreterExecutionResult(
                        success=False,
                        error=(f"Configuration error: {params_error}" if params_error
                               else "Parse errors: " + "; ".join(reparsed.errors)),
                        output_format="none",
                        duration_ms=(time.perf_counter() - start) * 1000,
                    )
                parsed = reparsed
                gate_violations = _gate_violations(parsed, context)
                continue  # the hook returns None for the new cell at ``index``
            refs, binds = _static_ref_bind_sets(cell)
            graph.add_cell(index, refs, binds)
            if resume is not None:  # "timeout": the child was stopped at the limit
                timed_out = True
                pending_timeout = resume
            if pending_timeout is not None and cell.cell_type != "prose":
                values.pop(index, None)
                output_parts.append(_reify_timeout_for_cell(
                    cell, index, pending_timeout, kernel, ledger, versions, assigned_names,
                    observed_effects,
                ))
                pending_timeout = None
                index += 1
                continue

            # Unavailable-source reification (L1.5, binding layer): a cell the
            # ceiling gate refuses binds Unavailable values for the names it
            # would have bound, is ledgered (`source_unavailable`), and is
            # SKIPPED — never executed, so the refused effect never happens.
            # Gate-first: the refusal is about the cell's required effects,
            # independent of whether its names would resolve.
            if index in gate_violations and kernel.refusal is None:
                output_parts.append(
                    _reify_unavailable_for_cell(
                        cell, index, gate_violations[index], kernel, ledger,
                        versions, assigned_names,
                    )
                )
                index += 1
                continue

            outcome = self._run_resolved_cell(
                cell, index, kernel, ledger, versions, assigned_names
            )
            _keep_value(values, index, outcome)
            if outcome.executed:
                observed_effects[index] = outcome.observed_effects
                cell_ids[index] = outcome.cell_id
            output_parts.append(outcome.rendered)
            if outcome.continue_requested:
                continue_requested = True
            if outcome.refusal is not None:
                # D4: a refusal ends the document. Later cells do not run and
                # there is no re-exec pass (the session is closed).
                refusal = outcome.refusal
                break
            index += 1

        # L1.4 re-execution pass (batch/session path only — streaming untouched).
        # A name re-asserted during the linear walk (an L1.3 `superseded`
        # entry) makes its transitive dependents stale; re-run ONLY those, in
        # dependency order, against the now-updated namespace.  Composition is
        # free: a re-exec that now hits an unknown name / error / over-ceiling
        # source re-reifies a Hole / ErrorValue / Unavailable and re-versions
        # through the very same per-cell pipeline the linear walk used.
        if cell_done is not None:
            cell_done()
        if refusal is None:
            refusal, rerun_timed_out = self._reexecute_dirty_subgraph(
                parsed.cells, graph, kernel, ledger, versions, assigned_names,
                reactive_mode, observed_effects, gate_violations, run_mark,
                output_parts, cell_ids, values, checkpoint=checkpoint,
                cell_done=cell_done,
            )
            timed_out = timed_out or rerun_timed_out

        elapsed = (time.perf_counter() - start) * 1000

        raw_ns = kernel.get_namespace(assigned_names)
        variables = {
            k: v for k, v in raw_ns.items()
            if k in assigned_names
            and k not in _INTERNAL_NAMES
            and not callable(v)
        }

        rendered = "".join(output_parts)

        # Aggregate layer (conflict #2 option (c)): the run's Either is a VIEW
        # DERIVED FROM THE LEDGER — Left iff any failure was reified — while
        # the output, bindings, and writes above all stand.  ``completed``
        # distinguishes a finished run (Right or aggregate-Left) from a
        # pre-execution refusal (parse errors — since L1.5 the ceiling gate
        # reifies per cell instead of refusing), which returns earlier and
        # never touches state.
        round_entries = ledger.entries()[run_mark:]
        failures = reified_failures(round_entries)

        # Canonical source-preserving render (conflict #5, Teague-approved): the
        # "document" the round-trip law holds for — authored prose + code SOURCE
        # preserved, with this round's LEDGER-derived forgiveness notes routed
        # through the L2 [kernel] channel (inert on reparse). This — NOT the flat
        # stdout in `output` — is what the session feeds back to a stateless
        # writer, closing the exp1 loop. `output` stays flat stdout for the
        # Left-correction path and existing callers.
        rendered_markdown = render_markdown_from_cells(
            parsed.cells, parsed.frontmatter, _annotations_from_ledger(round_entries)
        )

        errors = [describe_failure(e) for e in failures]
        if refusal is not None:
            errors.insert(0, f"refused: {refusal}")
        return InterpreterExecutionResult(
            success=not failures and refusal is None,
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
        rerun_id: int | None = None,
        rerun: bool = False,
    ) -> _CellOutcome:
        """Run one NON-GATED cell through the full forgiveness pipeline (L1.1–1.3).

        The single per-cell body shared by the linear walk AND the L1.4 re-exec
        pass: forgiveness pre-pass (L1.1 typed holes) → execute → error
        reification (L1.2) → version every binding (L1.3).  Sharing it is what
        makes dirty re-execution COMPOSE with forgiveness for free — a re-run
        that now hits an unknown name or raises reifies + versions exactly as a
        first run does, no special-casing.  (The ceiling gate / L1.5
        unavailable path stays in the callers: the linear walk reifies a gated
        cell instead of calling this, and the re-exec pass withholds it via
        its explicit ``gate_violations`` check — a gated cell never reaches
        this body on either pass.)

        Returns the rendered fragment for this cell, whether it requested a
        ``@continue`` pause, whether it actually executed, and what world
        effects its execution was OBSERVED to perform (the ``Path`` writes in
        its trace slice; see :func:`.effects.path_writes`) — the evidence the reactive re-exec
        pass gates on.  Mutates ``kernel`` / ``ledger`` / ``versions`` /
        ``assigned_names`` in place, like the original inline body.
        """
        # Forgiveness pre-pass (L1.1): a cell whose names cannot resolve binds
        # typed holes, is ledgered, and is SKIPPED (never executed, so a hole
        # can't flow into arithmetic). Not after a refusal (review I1): a
        # closed session refuses every cell, so the kernel's refused result
        # below is the outcome, never a hole.
        if kernel.refusal is None:
            reified = _reify_holes_for_cell(
                cell, index, kernel, ledger, versions, assigned_names
            )
            if reified is not None:
                return _CellOutcome(rendered=reified, continue_requested=False)

        known_before = kernel.known_names()
        # Observe the execution we are doing anyway (D7): the Path writes in
        # this cell's slice of the session trace, so the re-exec pass gates on
        # what the cell DID rather than on what its source looks like. The
        # Path stand-in is a cell's only route to the filesystem.
        trace = kernel.session.trace.entries
        mark = len(trace)
        if rerun:
            result = kernel.rerun_cell(cell, index, rerun_id)
        else:
            result = kernel.execute_cell(cell, index)
        observed = path_writes(trace[mark:])

        if result.error_phase == REFUSED:
            # D4: a refusal is not forgiven. It is never reified as a Hole or
            # an ErrorValue: it is ledgered, noted through the [kernel]
            # channel, and the caller stops the document.
            error = result.error or "refused"
            ledger.record(REFUSED, detail={
                "cell_index": index, "cell_type": cell.cell_type,
                "error": error, "error_phase": REFUSED,
            })
            parts = [result.output] if result.output else []
            parts.append(kernel_note(f"refused: {error}") + "\n")
            return _CellOutcome(
                rendered="".join(parts), continue_requested=False,
                executed=True, observed_effects=observed,
                cell_id=result.cell_id, refusal=error,
            )

        if not result.success:
            # L1.2: the failure is reified — an error value bound to the names
            # the cell declared but did not reach — and the run CONTINUES. No
            # abort, no rollback: partial output and bindings stand.
            parts: list[str] = []
            if result.output:
                parts.append(result.output)
            parts.append(
                _reify_error_for_cell(
                    cell, index, result, kernel, ledger, versions,
                    assigned_names, known_before,
                )
            )
            return _CellOutcome(
                rendered="".join(parts), continue_requested=False,
                executed=True, observed_effects=observed,
                cell_id=result.cell_id,
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
            executed=True, observed_effects=observed,
            cell_id=result.cell_id, ok=True, value=result.value,
        )

    def _reexecute_dirty_subgraph(
        self,
        cells: list[Cell],
        graph: DependencyGraph,
        kernel: LightweightKernel,
        ledger: Ledger,
        versions: BindingVersions,
        assigned_names: set[str],
        reactive_mode: str,
        observed_effects: dict[int, tuple[str, ...]],
        gate_violations: dict[int, str],
        run_mark: int,
        output_parts: list[str],
        cell_ids: dict[int, int | None] | None = None,
        values: dict[int, Any] | None = None,
        checkpoint: Any = None,
        cell_done: Any = None,
    ) -> tuple[str | None, bool]:
        """Re-execute the dependents of this run's heals (L1.4, Teague's C1 decision).

        A name re-asserted during the linear walk shows up as an L1.3
        ``superseded`` ledger entry in this run's slice.  Only a HEAL counts:
        a reified version (Hole, ErrorValue, Unavailable) superseded by a
        value.  A plain value-to-value rebind re-runs nothing, so a document
        gives python3's answer (goal 4).  The healed names seed
        :meth:`DependencyGraph.dirty_closure`, limited to the cells whose
        linear run did not succeed (a cell that ran read real values and
        keeps them), which returns the transitive dependent cells in
        dependency order.  Each dirtied cell is
        LEDGERED (a ``dirty`` entry — nothing silent) and, mode permitting,
        executed again through the shared per-cell pipeline so its bindings
        refresh against the updated namespace (and re-version / re-forgive as
        needed).

        EFFECT-REPLAY (the key risk, per the brief).  Re-running a cell's
        source replays its side effects — a dependent that writes a file
        would write it twice.  The gate is RUNTIME OBSERVATION, not static
        classification: the ``Path`` write entries in each cell's slice of the
        session trace (:func:`.effects.path_writes`, design D7) are in
        ``observed_effects``, what each cell was actually SEEN to do.
        Two modes (``reactive_mode``, default strict):

        * ``strict`` — a dirtied dependent OBSERVED to perform a world effect
          on its first run (``Path.write_text``, ``Path.write_bytes`` or
          ``Path.mkdir``, also one that failed part way) is withheld:
          ``reexecuted=False`` + the reason + the observed events.  A
          dependent whose first run was observed effect-free is re-run —
          including method-call dependents (``name.upper()``) and
          helper-wrapped computations, which a static source gate cannot
          grade.
        * ``permissive`` (opt-in) — ALL dirty dependents re-run, effects and
          all; for interactive/exploratory use where re-firing an effect is
          acceptable.  The re-run's own observed events are ledgered.

        NEVER-EXECUTED CELLS — two absences, two meanings.  A cell with no
        entry in ``observed_effects`` never executed this round, but that
        absence is AMBIGUOUS and the two causes get opposite treatment:

        * **ceiling-refused** (``gate_violations``): the effect ceiling
          refused the cell before it ran — a POLICY refusal, static over the
          cell's source, so it holds identically on re-execution.  The cell
          is withheld in BOTH modes (permissive waives replay caution, not
          policy): its bindings stay ``Unavailable`` and the ``dirty`` entry
          records the withhold with the gate's reason.  Running it here
          would execute the exact effect the gate refused.
        * **hole-skipped** (the forgiveness pre-pass reified it because a
          referenced name was unresolved): re-running when the hole fills is
          the L1.4 point — it is a safe FIRST execution and it runs.

        The disambiguator is the gate's own verdict (``gate_violations``),
        not the absence of observation evidence.

        COVERAGE: the ``Path`` stand-in is a cell's only route to the
        filesystem, so the trace sees every write a cell makes. An effectful
        branch the first run skipped is not seen.  When a strict-mode
        re-run of a first-run-pure cell does fire an effect, the effect is
        NOT prevented — but it is observed and recorded on the ``dirty``
        entry, never silent.

        TRUTHFUL LEDGER.  Every ``dirty`` entry carries the ``mode``.  The
        entry for a re-executed cell is recorded AFTER the re-run and reports
        what actually happened: ``outcome="clean"`` for a clean refresh,
        ``outcome="reified"`` (plus the reified names) when the re-run raised
        / re-holed and the shared pipeline reified the failure, and
        ``observed_effects`` when the re-run performed any.  It never claims
        a clean ``reexecuted=True`` for a re-run that failed.

        THROUGH THE SESSION (literate stepy).  A dependent that ran before
        re-runs its already-validated code with ``CellSession.rerun_cell``
        (``cell_ids`` maps the cell index to its session cell id): the
        program text does not change, so nothing is appended or validated
        again.  A dependent that never joined the program (hole-skipped, or
        rejected by the validator) runs as a new cell.  The rebind that made
        it dirty was itself a new cell.  A refusal during the pass stops it
        and is returned (D4); otherwise ``None``. The second value is whether a
        re-run timed out: in the session worker each re-run forks at the
        checkpoint hook too (Plan 3 Task 6), so a re-run that runs over the
        limit is reified as timed out, and the pass goes on.

        RENDERING.  Re-execution ANNOTATES, never rewrites (the L2 refusal law):
        the original render stands, and each re-run appends one kernel-channel
        note rather than re-emitting the cell's recomputed stdout (so a
        re-executed ``print`` cannot double its output).  The authoritative
        record of what the re-exec did — refreshed bindings, new versions, any
        re-reified hole/error — is the ledger + version history.
        """
        run_slice = ledger.entries()[run_mark:]
        # Teague's C1 decision: only a HEAL re-runs anything. A name heals when
        # a reified version (Hole, ErrorValue, Unavailable) is superseded by a
        # value. A plain value-to-value rebind (`rows = rows[1:]`) re-runs
        # nothing: every cell that read the old value already ran with it, as
        # under python3 (goal 4).
        healed = {
            e.name for e in run_slice
            if e.entry_type == SUPERSEDED and e.name
            and e.detail.get("prior") in _REIFIED_KINDS
            and _version_kind(versions, e.name, e.detail.get("to_version")) == "value"
        }
        if not healed:
            return None, False
        timed_out = False
        cell_ids = cell_ids if cell_ids is not None else {}
        # Only a cell whose linear run did not succeed can have read a reified
        # value, so only such a cell re-runs (a cell that ran saw real values
        # and keeps them, also when a name it read is rebound later).
        succeeded = set(values) if values is not None else set()
        eligible = {i for i in range(len(cells)) if i not in succeeded}

        for index, trigger_names in graph.dirty_closure(healed, eligible=eligible):
            triggered_by = sorted(trigger_names)
            detail = {
                "cell_index": index,
                "cell_type": cells[index].cell_type,
                "triggered_by": triggered_by,
                "mode": reactive_mode,
            }
            if index in gate_violations:
                # Ceiling-refused, NOT hole-skipped: the gate's verdict is
                # static over the cell's source, so it stands on re-exec in
                # every mode.  Executing here would run the refused effect
                # (the linear pass never did).  Withheld and ledgered; the
                # cell's bindings remain the reified Unavailable values.
                detail["reexecuted"] = False
                detail["gate_violation"] = gate_violations[index]
                detail["reason"] = (
                    "over effect ceiling ("
                    + gate_violations[index]
                    + ") — refused on the linear pass and not re-executed:"
                    " the policy refusal stands"
                )
                ledger.record(DIRTY, name=None, detail=detail)
                continue

            first_run = observed_effects.get(index, ())
            if reactive_mode == "strict" and first_run:
                # Withheld: the first run was observed performing a world
                # effect, so a re-run would replay it.  Surfaced with the
                # observed events — never silently skipped.
                detail["reexecuted"] = False
                detail["observed_effects"] = list(first_run)
                detail["reason"] = (
                    "observed to perform a world effect on first run ("
                    + ", ".join(first_run)
                    + ") — re-execution would replay it"
                )
                ledger.record(DIRTY, name=None, detail=detail)
                continue

            # Re-run first, record after: the dirty entry must report what
            # the re-execution actually DID, not a precomputed prediction.
            cell_mark = len(ledger.entries())
            resume = checkpoint(index) if checkpoint is not None else None
            if resume is not None:  # the re-run was stopped at the limit
                timed_out = True
                if values is not None:
                    values.pop(index, None)
                output_parts.append(_reify_timeout_for_cell(
                    cells[index], index, resume, kernel, ledger, versions,
                    assigned_names, observed_effects,
                ))
                detail["reexecuted"] = True
                detail["outcome"] = "reified"
                detail["reified"] = sorted(
                    {e.name for e in reified_failures(ledger.entries()[cell_mark:]) if e.name})
                ledger.record(DIRTY, name=None, detail=detail)
                if cell_done is not None:
                    cell_done()
                continue
            outcome = self._run_resolved_cell(
                cells[index], index, kernel, ledger, versions, assigned_names,
                rerun_id=cell_ids.get(index), rerun=True,
            )
            if cell_done is not None:
                cell_done()
            if values is not None:
                _keep_value(values, index, outcome)
            if outcome.refusal is not None:
                output_parts.append(outcome.rendered)
                return outcome.refusal, timed_out
            if outcome.executed:
                if cell_ids.get(index) is None:
                    cell_ids[index] = outcome.cell_id
                # Keep the observation current (and ledger the re-run's own
                # effects — a permissive replay, or a strict-mode effect the
                # first run's branch never reached).
                observed_effects[index] = outcome.observed_effects
                if outcome.observed_effects:
                    detail["observed_effects"] = list(outcome.observed_effects)
            reified = reified_failures(ledger.entries()[cell_mark:])
            detail["reexecuted"] = True
            detail["outcome"] = "reified" if reified else "clean"
            if reified:
                detail["reified"] = sorted({e.name for e in reified if e.name})
            ledger.record(DIRTY, name=None, detail=detail)
            note = (
                f"re-executed cell {index} (dirty: {', '.join(triggered_by)})"
            )
            if reified:
                note += " — failure reified"
            output_parts.append(kernel_note(note) + "\n")
        return None, timed_out


@dataclass(frozen=True)
class _CellOutcome:
    """The result of running one cell body (:meth:`_run_resolved_cell`).

    ``executed`` is False when the forgiveness pre-pass reified/skipped the
    cell (it never ran, so there is nothing observed); ``observed_effects``
    is the sorted, deduplicated ``Path`` write tools the execution was seen to
    perform (empty = no write observed).
    """

    rendered: str
    continue_requested: bool
    executed: bool = False
    observed_effects: tuple[str, ...] = ()
    #: The CellSession id of the run (``None`` when nothing ran in the session).
    cell_id: int | None = None
    #: The refusal text when the cell's run refused (D4): the document stops.
    refusal: str | None = None
    #: Whether the cell ran and succeeded; then ``value`` is its trailing
    #: expression's value (``CellResult.value``), else None.
    ok: bool = False
    value: Any = None


#: The kinds of a reified (forgiven) binding; superseding one by a value heals it.
_REIFIED_KINDS = frozenset({"hole", "error", "unavailable"})


def _version_kind(versions: BindingVersions, name: str, version: Any) -> str | None:
    """The :func:`value_kind` of version ``version`` of ``name``, or None."""
    found = versions.version_of(name, version) if isinstance(version, int) else None
    return value_kind(found.value) if found is not None else None


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


def _static_ref_bind_sets(cell: Cell) -> tuple[set[str], set[str]]:
    """This cell's ``(referenced_names, bound_names)`` for the L1.4 graph.

    Reuses the SAME static extraction the forgiveness pre-pass uses
    (``collect_names`` for references, ``collect_bindings`` for module-level
    bindings) over the cell's compiled source.  A cell that doesn't compile or
    parse (unknown type / syntax error) contributes empty sets — it cannot
    participate in name-flow dependencies, and its error is reported elsewhere.
    """
    try:
        tree = ast.parse(compile_cell(cell))
    except (ValueError, SyntaxError):
        return set(), set()
    _defined, referenced = collect_names(tree)
    return referenced, collect_bindings(tree)


#: Valid values for ``context.config["reactive_mode"]`` (see
#: :meth:`LiterateInterpreter._reexecute_dirty_subgraph`).
REACTIVE_MODES = ("strict", "permissive")


def _reactive_mode(context: ExecutionContext) -> str:
    """The reactive re-exec mode for this run: ``strict`` | ``permissive``.

    Read from ``context.config["reactive_mode"]`` — the same config surface
    ``display_threshold`` and ``grade_ceiling`` use, so it threads unchanged
    through ``LiterateInterpreter.execute`` and every ``LiterateSession``
    round (the session passes its one context each ``step``).  Defaults to
    ``strict``.  An unrecognized value raises ``ValueError`` rather than
    silently meaning strict — a mode typo must not masquerade as a safety
    choice.  ``_run_document`` catches that raise and returns it as the
    structured pre-execution refusal parse errors use (``success=False`` +
    ``error``), so a config typo surfaces on the normal error surface
    instead of crashing the round.
    """
    mode = (context.config or {}).get("reactive_mode", "strict")
    if mode not in REACTIVE_MODES:
        raise ValueError(
            f"reactive_mode must be one of {REACTIVE_MODES}; got {mode!r}"
        )
    return mode


#: Top-level statement types whose execution is proven complete once the
#: cell's top-level control has moved past their last line — the simple
#: (unconditional) binders ``_assignment_completed`` can reason about.
_SIMPLE_BINDER_STMTS = (
    ast.Assign,
    ast.AnnAssign,
    ast.AugAssign,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Import,
    ast.ImportFrom,
)


def _assignment_completed(
    tree: ast.Module | None, name: str, error_lineno: int | None
) -> bool:
    """Whether ``name``'s intended (re)bind COMPLETED before the cell raised.

    The kernel's identity-based namespace delta cannot see a rebind to the
    identical object (``x = x``, or re-binding an interned equal value) — so
    a failed cell that completed such a rebind before raising would look
    "never rebound" and get wrongly reified.  This helper supplies the
    missing runtime signal from the failure's top-level line
    (:attr:`CellResult.error_lineno`): if the LAST top-level statement
    binding ``name`` is a plain unconditional statement that ends strictly
    BEFORE the line that was executing when the cell raised, the bind ran to
    completion.

    Conservative fallbacks — returns False, so the caller reifies exactly as
    before: no line info, no parseable tree, the name's last binder is (or
    is inside) a compound statement (``if``/``for``/``while``/``try``/
    ``with`` — line order alone cannot tell whether its body ran), or the
    failing line is within the binder statement itself.  KNOWN CORNER
    (low, documented): a bind that completed inside a loop/branch before a
    later same-cell failure is still reified when the identity delta cannot
    see it.
    """
    if tree is None or error_lineno is None:
        return False
    last: ast.stmt | None = None
    for stmt in tree.body:
        binds_name = name in collect_bindings(
            ast.Module(body=[stmt], type_ignores=[])
        )
        if not binds_name:
            continue
        # A later binder supersedes an earlier one: the INTENDED final bind
        # is the last one, and only a simple statement's completion can be
        # inferred from line order.
        last = stmt if isinstance(stmt, _SIMPLE_BINDER_STMTS) else None
    if last is None or last.end_lineno is None:
        return False
    return last.end_lineno < error_lineno


def _record_assertion(
    name: str,
    value: Any,
    versions: BindingVersions,
    ledger: Ledger,
    cell_detail: dict[str, Any],
) -> None:
    """Record one binding assertion in the version history (L1.3).

    The single choke point for the binding layer: every bind — an executed
    cell's namespace delta, a reified hole, an error value — passes through
    here.  A first assertion just records version 1; a RE-ASSERTION marks the
    prior version superseded and writes the one ``superseded`` ledger entry,
    whose ``detail`` carries the version transition (mirroring AIDR's
    ``_aidr_bindings`` version/``superseded_by`` model) plus the kind of the
    superseded value (``value`` / ``hole`` / ``error`` — so a hole filled by
    a real definition is visibly a versioned transition, not a mutation).
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
    a notable forgiveness / versioning / reactive entry becomes one inert
    channel note on the cell it concerns; a plain ``executed`` /
    ``continue_requested`` entry carries no note (the source itself is the
    record).  :func:`~.kernel.forgiveness.describe_failure` supplies the
    hole/error/unavailable text — so the error string (e.g. the exception type)
    survives into the canonical render — and supersede / dirty get one-liners.
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
    if et == DIRTY:
        triggered = ", ".join(entry.detail.get("triggered_by") or [])
        if entry.detail.get("reexecuted"):
            if entry.detail.get("outcome") == "reified":
                return f"re-executed (dirty: {triggered}) — failure reified"
            return f"re-executed (dirty: {triggered})"
        reason = entry.detail.get("reason", "withheld")
        return f"dirty (not re-executed: {reason}; triggered by {triggered})"
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


_HOLE_EXEMPT_NAMES = frozenset({
    # Kernel-injected at execution time for @continue cells; never an unknown
    # name even though it is absent from the namespace between cells.
    "__literate_continue__",
})


def _reify_holes_for_cell(
    cell: Cell,
    index: int,
    kernel: LightweightKernel,
    ledger: Ledger,
    versions: BindingVersions,
    assigned_names: set[str],
) -> str | None:
    """Binding-layer forgiveness pre-pass for one cell (L1.1, typed holes).

    Resolves the cell's referenced names against the kernel namespace with the
    SAME rules as the kernel's own static check (`check_cell`), so the two can
    never disagree.  When a referenced name is unknown (an *origin* hole) or
    is bound to an already-reified forgiving value (the *chain*), the cell
    cannot produce real values, so it is reified instead of executed:

    * each unknown referenced name binds ``Hole(name)`` — the typed hole
      ⟨name: unbound⟩;
    * each name the cell would have bound binds a CHAINED hole recording its
      upstream causes (statically skipped, so a hole never reaches ``+``);
    * one ``hole_opened`` ledger entry per hole — nothing silent;
    * the returned kernel-channel note is appended to the rendered output
      (renders annotate, never rewrite), and is inert on reparse (L2).

    Returns the annotation line when the cell was reified/skipped, or ``None``
    when the cell resolves cleanly (the caller executes it).  Syntax errors
    and unknown cell types are not name problems — they return ``None`` and
    take the error path.
    """
    try:
        src = compile_cell(cell)
    except ValueError:
        return None  # unknown cell type — the kernel reports it precisely
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None  # not a name problem — handled on the error path

    defined, referenced = collect_names(tree)
    known = kernel.known_names()
    undefined = referenced - known - BUILTIN_NAMES - defined - _HOLE_EXEMPT_NAMES
    # A referenced name the cell itself defines is satisfied by that definition
    # (check_cell's own rule: available = known | builtins | defined) — so a
    # cell that REDEFINES a holed name is how a hole gets filled, not blocked.
    blocked = {
        n
        for n in (referenced - defined) & known
        if kernel.is_reified(n)
    }
    if not undefined and not blocked:
        return None

    base_detail = {"cell_index": index, "cell_type": cell.cell_type}
    notes: list[str] = []

    for name in sorted(undefined):
        hole = Hole(name)
        kernel.bind(name, hole)
        _record_assertion(name, hole, versions, ledger, base_detail)
        ledger.record(
            HOLE_OPENED, name=name, detail={"reason": "unbound", **base_detail}
        )
        notes.append(f"'{name}' unbound")

    causes = undefined | blocked
    bindings = collect_bindings(tree)
    for name in sorted(bindings):
        # A name may be among its own causes (`x = x + 1` with x holed);
        # prefer naming the *other* causes, fall back to the self-cause.
        cause = tuple(sorted(causes - {name})) or tuple(sorted(causes))
        hole = Hole(name, blocked_by=cause)
        kernel.bind(name, hole)
        # A chained hole may overwrite a REAL prior value (`x = missing + 1`
        # with x previously bound): that too is a versioned supersession,
        # ledgered by _record_assertion — nothing silent.
        _record_assertion(name, hole, versions, ledger, base_detail)
        ledger.record(
            HOLE_OPENED,
            name=name,
            detail={"reason": "blocked", "blocked_by": list(cause), **base_detail},
        )
        notes.append(f"'{name}' blocked by {', '.join(cause)}")

    if not undefined and not bindings:
        # Nothing binds (e.g. prose interpolating a hole), but the skipped
        # cell must still be ledgered and the round must still report Left.
        ledger.record(
            HOLE_OPENED,
            name=None,
            detail={
                "reason": "blocked",
                "blocked_by": sorted(blocked),
                **base_detail,
            },
        )
        notes.append(f"cell skipped (blocked by {', '.join(sorted(blocked))})")

    assigned_names.update(undefined | bindings)
    return kernel_note("holes: " + "; ".join(notes)) + "\n"


def _reify_error_for_cell(
    cell: Cell,
    index: int,
    result: CellResult,
    kernel: LightweightKernel,
    ledger: Ledger,
    versions: BindingVersions,
    assigned_names: set[str],
    known_before: set[str],
) -> str:
    """Binding-layer error reification for one failed cell (L1.2).

    The failed :class:`CellResult` becomes ledger entries + bound values
    instead of an abort:

    * names the cell declared but did NOT successfully (re)bind THIS PASS
      bind an :class:`ErrorValue` carrying the exception info — one
      ``error_reified`` entry per name.  This includes a failed RE-binding of
      a DOCUMENT-ASSERTED name (one with version history): keeping the prior
      value would be a silent stale success, so the failure SUPERSEDES it
      (value→error / hole→error, one L1.3 ``superseded`` entry via the
      normal versioning choke point);
    * names the cell actually bound *before* the error keep their real values
      (completed work is kept; the kernel executes in place and no longer
      rolls anything back) — recorded in the entry's ``kept`` detail.  For an
      already-known name, "actually bound" is observed by identity against
      the recorded current version (the kernel's own delta rule), backed by
      the failure-line signal (:func:`_assignment_completed`) for the rebind
      the identity delta cannot see — ``x = x``, or re-binding an interned
      equal value, completed before a later statement raised;
    * INJECTED names — params, tools, environment names with NO
      document-asserted version history — are NEVER reified on a failed
      re-bind: the document never bound them, so the failure must not
      destroy the provided value.  They keep their live value (the provided
      one, or the cell's rebind if it completed) and are recorded in the
      entry's ``kept_injected`` detail — nothing silent, but no
      ``superseded`` trace fabricated for a value the document never owned;
    * a cell that binds nothing (a bare statement, a syntax error, an unknown
      cell type) still gets one ``error_reified`` entry (``name=None``) —
      nothing silent.

    Returns the kernel-channel annotation to append to the rendered output.
    """
    error = result.error or "unknown error"
    phase = result.error_phase or "runtime"
    base_detail = {
        "cell_index": index,
        "cell_type": cell.cell_type,
        "error": error,
        "error_phase": phase,
    }

    bindings: set[str] = set()
    tree: ast.Module | None = None
    if phase == "runtime":
        # Static phases (syntax error, unknown type) have no parseable AST /
        # no meaningful bindings; runtime failures tell us which names the
        # cell was going to bind.
        try:
            tree = ast.parse(compile_cell(cell))
            bindings = collect_bindings(tree)
        except (ValueError, SyntaxError):
            tree = None
            bindings = set()

    # Partition the cell's intended bindings against KNOWN_BEFORE (the
    # snapshot taken before this cell ran), not the post-failure namespace:
    # a name that already existed is only "kept" if this cell observably
    # rebound it before failing.  Partitioning on the post-failure namespace
    # was the stale-keep defect — an already-known name always looked
    # "kept" and a failed re-assertion silently retained the prior value.
    known = kernel.known_names()
    reify: list[str] = []
    kept: list[str] = []
    kept_injected: list[str] = []
    for name in sorted(bindings):
        if name not in known_before:
            # Fresh name: kept iff the cell bound it before the failure
            # (it is in the namespace now but was not before).
            if name in known:
                kept.append(name)
            else:
                reify.append(name)
            continue
        current = versions.current(name)
        if current is None:
            # INJECTED name (param / tool / environment): no document-
            # asserted version exists — the document never owned this
            # binding, so a failed re-bind must not clobber the provided
            # value with an ErrorValue.  Keep the live value (the provided
            # one; or the cell's rebind, if it completed — which, like
            # every environment-name change, stays unversioned).
            kept_injected.append(name)
        elif current.value is not kernel.lookup(name):
            # Observed rebound (live value differs, by identity, from the
            # recorded current version) → completed work, kept.
            kept.append(name)
        elif _assignment_completed(tree, name, result.error_lineno):
            # The identity delta cannot see a rebind to the identical
            # object (`x = x`, an interned equal value) — but the failure
            # line shows the binding statement completed before the raise.
            # Rebound → completed work, kept.
            kept.append(name)
        else:
            # The re-bind never completed — reify, superseding the prior
            # value/hole rather than silently keeping it stale.
            reify.append(name)

    cell_detail = {"cell_index": index, "cell_type": cell.cell_type}
    partition_detail: dict[str, Any] = {**base_detail, "kept": kept}
    if kept_injected:
        partition_detail["kept_injected"] = kept_injected
    if reify:
        for name in reify:
            error_value = ErrorValue(name=name, error=error, error_phase=phase)
            kernel.bind(name, error_value)
            _record_assertion(name, error_value, versions, ledger, cell_detail)
            ledger.record(ERROR_REIFIED, name=name, detail=dict(partition_detail))
        assigned_names.update(reify)
    else:
        ledger.record(ERROR_REIFIED, name=None, detail=dict(partition_detail))
    # Partial bindings the cell completed before failing surface as variables
    # — as do kept injected names (their provided values remain live).
    assigned_names.update(kept)
    assigned_names.update(kept_injected)

    # L1.3: those partial bindings are real assertions and must be versioned
    # too — a failed CellResult carries no namespace_delta, so observe them
    # directly.  A kept name absent before this cell ran is a first assertion
    # (or a supersession of a deleted name's history); a kept name whose live
    # value now differs (identity, like all kernel delta tracking) from its
    # recorded current version was RE-ASSERTED before the failure — silent
    # overwrite is exactly what ground rule 3 forbids.  A kept pre-existing
    # name with no version history is an environment name (no version 0 by
    # design) whose change is unobservable here; it stays unversioned.
    for name in kept:
        live = kernel.lookup(name)
        current = versions.current(name)
        if name not in known_before:
            _record_assertion(name, live, versions, ledger, cell_detail)
        elif current is not None and current.value is not live:
            _record_assertion(name, live, versions, ledger, cell_detail)

    note = f"error: {error}"
    if reify:
        note += f" — bound as error value: {', '.join(reify)}"
    return kernel_note(note) + "\n"


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


def _timeout_writes(entries: list[TraceEntry]) -> tuple[list[str], tuple[str, ...]]:
    """The writes a stopped cell's trace shows: ``(paths for the note, write tools)``.

    A write with a done entry that succeeded is named by its path. A write with
    a start entry and no done entry (the cell was stopped inside it) is named
    ``"{path} (write started, not confirmed)"`` (Review Focus 5). A write that
    failed is not named: it did not write. The write tools are the cell's
    observed effects, a started write included.
    """
    done_steps = {e.step for e in entries if not e.tool.endswith(START_SUFFIX)}
    names: dict[str, None] = {}
    tools: set[str] = set()
    for e in entries:
        tool = e.tool[:-len(START_SUFFIX)] if e.tool.endswith(START_SUFFIX) else e.tool
        if tool not in ("Path.write_text", "Path.write_bytes"):
            continue
        path = str(e.args[0]) if isinstance(e.args, list) and e.args else "?"
        if e.tool.endswith(START_SUFFIX):
            if e.step not in done_steps:
                tools.add(tool)
                names.setdefault(f"{path} (write started, not confirmed)", None)
        else:
            tools.add(tool)
            if e.success:
                names.setdefault(path, None)
    return list(names), tuple(sorted(tools))


def _reify_timeout_for_cell(
    cell: Cell,
    index: int,
    resume: Any,
    kernel: LightweightKernel,
    ledger: Ledger,
    versions: BindingVersions,
    assigned_names: set[str],
    observed_effects: dict[int, tuple[str, ...]],
) -> str:
    """Binding-layer reification for a cell stopped at its time limit (Plan 3 D7).

    Called in the checkpoint, which holds the state from before the cell. The
    cell's own process is gone, so ``resume.entries`` (what it streamed to the
    parent: its trace entries and start entries) is the record of what it did.

    * the entries join the session's trace (start entries do not), so the
      document's trace shows what the stopped cell read and wrote;
    * each name the cell would bind binds a ``Hole`` (later cells that read it
      see a hole, as after an error), and is versioned;
    * one ``timed_out`` ledger entry whose ``note`` is the ``[kernel]`` note:
      ``timeout: cell {n} took longer than {t:g} s and was stopped``, plus
      ``; it wrote {paths} before it was stopped`` when the trace shows writes.

    A timeout is not a refusal: the session stays open and the document goes on.
    Returns the kernel-channel annotation to append to the rendered output.
    """
    entries = list(resume.entries or [])
    trace = kernel.session.trace
    trace.entries.extend(e for e in entries if not e.tool.endswith(START_SUFFIX))
    paths, tools = _timeout_writes(entries)
    if tools:
        observed_effects[index] = tools
    note = (f"timeout: cell {index + 1} took longer than {resume.seconds:g} s "
            f"and was stopped")
    if paths:
        note += f"; it wrote {', '.join(paths)} before it was stopped"
    _, binds = _static_ref_bind_sets(cell)
    cell_detail = {"cell_index": index, "cell_type": cell.cell_type}
    for name in sorted(binds):
        hole = Hole(name)
        kernel.bind(name, hole)
        _record_assertion(name, hole, versions, ledger, cell_detail)
    assigned_names.update(binds)
    ledger.record(TIMED_OUT, name=None, detail={
        **cell_detail, "note": note, "error_phase": "timeout",
        "seconds": resume.seconds, "bound_as_holes": sorted(binds),
    })
    return kernel_note(note) + "\n"


def _reify_unavailable_for_cell(
    cell: Cell,
    index: int,
    reason: str,
    kernel: LightweightKernel,
    ledger: Ledger,
    versions: BindingVersions,
    assigned_names: set[str],
) -> str:
    """Binding-layer unavailable-source reification for one gated cell (L1.5).

    The cell's required effect exceeds the ceiling, so the gate refuses to run
    it — its source is UNAVAILABLE right now (a raised ceiling / different
    profile would run it).  Instead of refusing the whole document:

    * each name the cell would have bound binds an :class:`Unavailable`
      carrying the gate's reason — one ``source_unavailable`` ledger entry
      per name (nothing silent);
    * a cell that binds nothing (e.g. a bare ``@write`` directive) still gets
      one ``source_unavailable`` entry (``name=None``);
    * the cell is never executed, so the refused effect never happens
      (refuse-before-running, kept per cell);
    * downstream cells that reference an Unavailable name chain as blocked
      holes via the L1.1 pre-pass (``is_forgiving`` covers Unavailable), and
      a later within-ceiling assertion supersedes it via L1.3 versioning.

    Only called for cells that already compiled+parsed during gate
    classification, so ``compile_cell``/``ast.parse`` cannot fail here.
    Returns the kernel-channel annotation to append to the rendered output.

    NAMING (design conflict #4): this concept never uses the word "pending" —
    the streaming driver's ``pending`` status means "not executed due to
    @continue pause", a different thing on an untouched path.
    """
    bindings = collect_bindings(ast.parse(compile_cell(cell)))
    base_detail = {
        "cell_index": index,
        "cell_type": cell.cell_type,
        "reason": reason,
    }
    cell_detail = {"cell_index": index, "cell_type": cell.cell_type}

    notes: list[str] = []
    if bindings:
        for name in sorted(bindings):
            value = Unavailable(name=name, reason=reason)
            kernel.bind(name, value)
            # A supersession of a real prior value by an Unavailable is a
            # versioned transition like any other, ledgered by
            # _record_assertion — nothing silent.
            _record_assertion(name, value, versions, ledger, cell_detail)
            ledger.record(SOURCE_UNAVAILABLE, name=name, detail=dict(base_detail))
            notes.append(f"'{name}' unavailable")
        assigned_names.update(bindings)
    else:
        # Nothing binds, but the skipped cell must still be ledgered and the
        # round must still report Left.
        ledger.record(SOURCE_UNAVAILABLE, name=None, detail=dict(base_detail))
        notes.append("cell skipped")

    return kernel_note(f"unavailable ({reason}): " + "; ".join(notes)) + "\n"


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
    cell, or None. Name resolution (holes) is a runtime matter and is not
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
