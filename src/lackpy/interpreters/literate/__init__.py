"""Literate programming interpreter — top-level entry point.

Pipeline: document → parser.parse() → [Cell] → kernel.execute_cell() → stdout

The kernel compiles each cell type to Python via compiler._COMPILERS,
runs static analysis (compile check + AST name resolution), then
evaluates the code in a shared namespace dict. Captured stdout IS
the rendered document.

This module owns the batch execution path (LiterateInterpreter.execute).
The streaming path uses kernel.StreamingDriver instead — it adds
recovery, plugin orchestration, and parse-as-you-generate execution.
"""

from __future__ import annotations

import ast
import time
from typing import Any

from ..base import (
    ExecutionContext,
    InterpreterExecutionResult,
    InterpreterValidationResult,
)
from .annotations import kernel_note
from .compiler import compile_cell
from .display import DEFAULT_DISPLAY_THRESHOLD, DISPLAY_HELPER_NAME, make_display
from .effects import (
    LITERATE_TOOL_EFFECTS,
    ToolEffect,
    as_grade,
    classify_effects,
    exceeds_ceiling,
    tool_effect_from_spec,
)
from .kernel import CellResult, LightweightKernel
from .kernel.forgiveness import (
    ERROR_REIFIED,
    HOLE_OPENED,
    SOURCE_UNAVAILABLE,
    ErrorValue,
    Hole,
    Unavailable,
    describe_failure,
    is_forgiving,
    reified_failures,
)
from .kernel.ledger import Ledger
from .kernel.static_analysis import BUILTIN_NAMES, collect_bindings, collect_names
from .kernel.versions import SUPERSEDED, BindingVersions, value_kind
from .parser import Cell, parse
from .prompt import LITERATE_HINT
from .tools import make_tool_namespace


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
        """
        namespace = _build_namespace(context)
        kernel = LightweightKernel(namespace=namespace)
        return await self._run_document(program, context, kernel)

    async def _run_document(
        self,
        program: str,
        context: ExecutionContext,
        kernel: LightweightKernel,
        ledger: Ledger | None = None,
        versions: BindingVersions | None = None,
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

        parsed = parse(program)
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
        raw_ceiling = (context.config or {}).get("grade_ceiling")
        if raw_ceiling is None:
            raw_ceiling = getattr(context.tools, "grade", None)
        gate_violations: dict[int, str] = {}
        if raw_ceiling is not None:
            ceiling = as_grade(raw_ceiling)
            effects_map = _gate_effects_map(context)
            for index, cell in enumerate(parsed.cells):
                try:
                    src = compile_cell(cell)
                    ast.parse(src)
                except (ValueError, SyntaxError):
                    continue
                effects = classify_effects(src, tool_effects=effects_map)
                violation = exceeds_ceiling(effects, ceiling)
                if violation:
                    gate_violations[index] = (
                        f"effect ceiling exceeded: {violation}"
                    )

        # NOTE (L1.2, contract change): the always-on write journal that
        # rolled a failed document's file writes back is RETIRED on this path
        # together with the abort itself -- under the two-layer forgiveness
        # contract a cell failure reifies as a value + ledger entry and the
        # run completes, so there is no failure point left to roll back from.
        # File writes stand, like every other binding (state kept, failure
        # ledgered). journal.FileJournal remains available as a component for
        # a future opt-in strict/transactional mode.
        output_parts: list[str] = []
        continue_requested = False
        assigned_names: set[str] = set()

        for index, cell in enumerate(parsed.cells):
            # Unavailable-source reification (L1.5, binding layer): a cell the
            # ceiling gate refuses binds Unavailable values for the names it
            # would have bound, is ledgered (`source_unavailable`), and is
            # SKIPPED — never executed, so the refused effect never happens.
            # Gate-first: the refusal is about the cell's required effects,
            # independent of whether its names would resolve.
            if index in gate_violations:
                output_parts.append(
                    _reify_unavailable_for_cell(
                        cell, index, gate_violations[index], kernel, ledger,
                        versions, assigned_names,
                    )
                )
                continue

            # Forgiveness pre-pass (L1.1, binding layer): a cell whose names
            # cannot resolve — unknown names, or references to already-bound
            # holes/error values — binds typed holes, is ledgered, and is
            # SKIPPED (never executed, so a hole can't flow into arithmetic).
            reified = _reify_holes_for_cell(
                cell, index, kernel, ledger, versions, assigned_names
            )
            if reified is not None:
                output_parts.append(reified)
                continue

            known_before = kernel.known_names()
            result = kernel.execute_cell(cell, index)

            if not result.success:
                # L1.2 (binding layer): the failure is reified — an error
                # value bound to the names the cell declared but did not
                # reach, one `error_reified` ledger entry per bound name
                # (nothing silent) — and the run CONTINUES. No abort, no
                # rollback: partial output and partial bindings are completed
                # work and stand.
                if result.output:
                    output_parts.append(result.output)
                output_parts.append(
                    _reify_error_for_cell(
                        cell, index, result, kernel, ledger, versions,
                        assigned_names, known_before,
                    )
                )
                continue

            if result.output:
                output_parts.append(result.output)

            assigned_names.update(result.namespace_delta.keys())

            # L1.3 (binding layer): every name this cell bound is a new
            # immutable version; a name with a prior version is a
            # RE-ASSERTION — the prior version is marked superseded and one
            # ``superseded`` ledger entry records the transition (nothing
            # silent).  The kernel's namespace_delta is the runtime record of
            # what the cell actually bound (identity-based, like all kernel
            # delta tracking) — an assertion that rebinds the identical
            # object is unobservable and not versioned.
            cell_detail = {"cell_index": index, "cell_type": cell.cell_type}
            for name in sorted(result.namespace_delta):
                if name.startswith("_"):
                    continue  # __continue_requested__ / internals, not bindings
                _record_assertion(
                    name, result.namespace_delta[name], versions, ledger, cell_detail
                )

            if result.namespace_delta.get("__continue_requested__"):
                continue_requested = True
                ledger.record(
                    "continue_requested",
                    detail={"cell_index": index, "cell_type": cell.cell_type},
                )
            else:
                ledger.record(
                    "executed",
                    detail={"cell_index": index, "cell_type": cell.cell_type},
                )

        elapsed = (time.perf_counter() - start) * 1000

        raw_ns = kernel.get_namespace()
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
        failures = reified_failures(ledger.entries()[run_mark:])
        return InterpreterExecutionResult(
            success=not failures,
            output=rendered,
            error="; ".join(describe_failure(e) for e in failures) or None,
            output_format="markdown",
            duration_ms=elapsed,
            metadata={
                "variables": variables,
                "continue_requested": continue_requested,
                "cell_count": len(parsed.cells),
                "completed": True,
                "ledger": ledger,
                "versions": versions,
                "frontmatter": {
                    "echo": parsed.frontmatter.echo,
                    "output": parsed.frontmatter.output,
                    "interpreter": parsed.frontmatter.interpreter,
                },
            },
        )


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
        if is_forgiving(kernel.lookup(n))
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

    * names the cell declared but did NOT reach bind an :class:`ErrorValue`
      carrying the exception info — one ``error_reified`` entry per name;
    * names the cell bound *before* the error keep their real values
      (completed work is kept; the kernel executes in place and no longer
      rolls anything back) — recorded in the entry's ``kept`` detail;
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
    if phase == "runtime":
        # Static phases (syntax error, unknown type) have no parseable AST /
        # no meaningful bindings; runtime failures tell us which names the
        # cell was going to bind.
        try:
            bindings = collect_bindings(ast.parse(compile_cell(cell)))
        except (ValueError, SyntaxError):
            bindings = set()

    known = kernel.known_names()
    missing = sorted(n for n in bindings if n not in known)
    kept = sorted(n for n in bindings if n in known)

    cell_detail = {"cell_index": index, "cell_type": cell.cell_type}
    if missing:
        for name in missing:
            error_value = ErrorValue(name=name, error=error, error_phase=phase)
            kernel.bind(name, error_value)
            _record_assertion(name, error_value, versions, ledger, cell_detail)
            ledger.record(
                ERROR_REIFIED, name=name, detail={**base_detail, "kept": kept}
            )
        assigned_names.update(missing)
    else:
        ledger.record(ERROR_REIFIED, name=None, detail={**base_detail, "kept": kept})
    # Partial bindings the cell completed before failing surface as variables.
    assigned_names.update(kept)

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
    if missing:
        note += f" — bound as error value: {', '.join(missing)}"
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
    "read_file", "write_file", "apply_diff",
    "search_content", "run_command", "run_tests",
    "__literate_continue__", "__literate_display__", "__builtins__",
    "__continue_requested__",
})


def _gate_effects_map(context: ExecutionContext) -> dict[str, ToolEffect]:
    """Effect grades the ceiling gate scores against, for THIS context.

    Starts from the literate builtins (``LITERATE_TOOL_EFFECTS``) and adds any
    profile/toolbox-injected tools (``context.tools``) graded by their own
    ``ToolSpec``. Without this the gate would only know the builtins, so an
    injected write-capable tool would classify as pure and slip under a
    read-only ceiling -- the gate's validation surface must match the execution
    surface (``_build_namespace`` injects the same ``context.tools`` callables).

    Each injected tool is graded by ``tool_effect_from_spec``: a spec that declares
    ``effect_kind`` + ``path_arg`` is graded precisely (and an injected write tool
    with a literal path arg is journalable); a spec that omits them falls back to
    the grade heuristic (kind from ``grade_w``, no path -> gated but not journaled).
    Builtins win over injected tools of the same name (the literate ``write_file``
    in the namespace is what actually runs), so their precise TOML entry stands.
    """
    effects_map = dict(LITERATE_TOOL_EFFECTS)
    resolved = context.tools
    specs = getattr(resolved, "tools", None) if resolved is not None else None
    for name, spec in (specs or {}).items():
        if name in effects_map:
            continue
        effects_map[name] = tool_effect_from_spec(spec)
    return effects_map


def _build_namespace(context: ExecutionContext) -> dict[str, Any]:
    """Build the execution namespace for a literate document."""
    ns: dict[str, Any] = {}

    ns.update(make_tool_namespace(context.base_dir))

    # Display helper for prose interpolation (see display.py), with the
    # threshold configurable per run. The kernel injects a default-threshold
    # helper when absent; this context-aware one takes precedence.
    threshold = (context.config or {}).get(
        "display_threshold", DEFAULT_DISPLAY_THRESHOLD
    )
    ns[DISPLAY_HELPER_NAME] = make_display(threshold)

    import builtins as _builtins_mod
    ns["__builtins__"] = _builtins_mod

    if context.tools:
        for name, fn in context.tools.callables.items():
            ns[name] = fn

    if context.params:
        ns.update(context.params)

    return ns


# Re-export the multi-round fold API. Imported at the end (after LiterateInterpreter
# and _build_namespace are defined) so session.py's lazy imports resolve cleanly.
from .session import (  # noqa: E402,F401
    CONTINUE_MARKER,
    LiterateSession,
    StepResult,
    StopScanner,
    split_at_continue,
    strip_overlap,
    strip_think,
)
