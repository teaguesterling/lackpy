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
    combine,
    exceeds_ceiling,
    tool_effect_from_spec,
)
from .journal import FileJournal
from .kernel import LightweightKernel
from .kernel.forgiveness import (
    HOLE_OPENED,
    Hole,
    describe_failure,
    is_forgiving,
    reified_failures,
)
from .kernel.ledger import Ledger
from .kernel.static_analysis import BUILTIN_NAMES, collect_bindings, collect_names
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
    ) -> InterpreterExecutionResult:
        """Run a literate document through an ALREADY-BUILT kernel: classify ->
        ceiling gate -> write journal -> run cells (forgiveness pre-pass +
        execute) -> render.

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
        """
        start = time.perf_counter()
        ledger = ledger if ledger is not None else Ledger()
        run_mark = len(ledger)

        parsed = parse(program)
        if parsed.errors:
            return InterpreterExecutionResult(
                success=False,
                error="Parse errors: " + "; ".join(parsed.errors),
                output_format="none",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        # Effects-core-to-the-step: classify the document once (statically, before
        # any cell runs), then drive two mechanisms off the result -- an always-on
        # write journal and a conditional ceiling gate.
        #
        # Cells that don't parse are skipped here so the kernel reports the precise
        # syntax error rather than a misleading refusal. (Cells are compiled here
        # and again by the kernel -- cheap for small docs; a later slice can compile
        # once and feed both. This gates/journals only the batch path; the
        # StreamingDriver path is a follow-up slice.)
        effects_map = _gate_effects_map(context)
        cell_effects = []
        for cell in parsed.cells:
            src = compile_cell(cell)
            try:
                ast.parse(src)
            except SyntaxError:
                continue
            cell_effects.append(classify_effects(src, tool_effects=effects_map))
        doc_effects = combine(cell_effects)

        # Ceiling gate (conditional): refuse a document whose aggregate effects
        # exceed the ceiling, before any cell runs. Profile -> ceiling wiring: an
        # explicit context.config["grade_ceiling"] wins; otherwise the ceiling
        # DEFAULTS to the granted toolset's grade (ResolvedProfile.grade == its
        # tools' grade) -- a document may not exceed the effect grade of the tools
        # its profile granted. No ceiling and no toolset => no gate. (End-to-end
        # enforcement in the agent path additionally needs execution-axis dispatch
        # to run literate under a profile -- a separate slice.)
        raw_ceiling = (context.config or {}).get("grade_ceiling")
        if raw_ceiling is None:
            raw_ceiling = getattr(context.tools, "grade", None)
        if raw_ceiling is not None:
            ceiling = as_grade(raw_ceiling)
            violation = exceeds_ceiling(doc_effects, ceiling)
            if violation:
                return InterpreterExecutionResult(
                    success=False,
                    error=f"effect ceiling exceeded: {violation}",
                    output_format="none",
                    duration_ms=(time.perf_counter() - start) * 1000,
                    metadata={
                        "effects": doc_effects,
                        "ceiling": ceiling,
                        "needs_sandbox": doc_effects.needs_sandbox,
                    },
                )

        # Write journal (always-on): snapshot the document's statically known
        # literal writes (doc_effects.writes) so the whole batch run is atomic for
        # those files -- any cell failure rolls them back, so a failed document
        # leaves the filesystem as it was. Dynamic-path/exec/unanalyzable effects
        # are out of the journal's reach (see journal.py); when a ceiling is set it
        # is what keeps those out of a policy-governed run. Transaction boundary is
        # the whole execute() call; per-@continue commit points are a later slice.
        journal = FileJournal(context.base_dir)
        journal.snapshot(doc_effects.writes)

        output_parts: list[str] = []
        continue_requested = False
        assigned_names: set[str] = set()

        for index, cell in enumerate(parsed.cells):
            # Forgiveness pre-pass (L1.1, binding layer): a cell whose names
            # cannot resolve — unknown names, or references to already-bound
            # holes/error values — binds typed holes, is ledgered, and is
            # SKIPPED (never executed, so a hole can't flow into arithmetic).
            reified = _reify_holes_for_cell(cell, index, kernel, ledger, assigned_names)
            if reified is not None:
                output_parts.append(reified)
                continue

            result = kernel.execute_cell(cell, index)

            if not result.success:
                # Roll the document's known writes back to their pre-run state so a
                # failed document does not leave half-applied files behind.
                journal.rollback()
                ledger.record(
                    "aborted",
                    detail={
                        "cell_index": index,
                        "cell_type": cell.cell_type,
                        "error": result.error,
                        "error_phase": result.error_phase,
                    },
                )
                return InterpreterExecutionResult(
                    success=False,
                    error=result.error or "Unknown error",
                    output_format="text",
                    duration_ms=(time.perf_counter() - start) * 1000,
                )

            if result.output:
                output_parts.append(result.output)

            assigned_names.update(result.namespace_delta.keys())

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

        # Every cell ran or was reified: accept the writes (drop the snapshots).
        journal.commit()

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
        # pre-execution refusal (parse errors, ceiling gate), which returns
        # earlier and never touches state.
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
                "frontmatter": {
                    "echo": parsed.frontmatter.echo,
                    "output": parsed.frontmatter.output,
                    "interpreter": parsed.frontmatter.interpreter,
                },
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
        kernel.bind(name, Hole(name))
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
        kernel.bind(name, Hole(name, blocked_by=cause))
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
