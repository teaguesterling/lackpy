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

from ...lang.grader import Grade
from ..base import (
    ExecutionContext,
    InterpreterExecutionResult,
    InterpreterValidationResult,
)
from .compiler import compile_cell
from .effects import (
    LITERATE_TOOL_EFFECTS,
    ToolEffect,
    as_grade,
    classify_effects,
    combine,
    exceeds_ceiling,
)
from .journal import FileJournal
from .kernel import LightweightKernel
from .parser import parse
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
    ) -> InterpreterExecutionResult:
        """Run a literate document through an ALREADY-BUILT kernel: classify ->
        ceiling gate -> write journal -> run cells -> render.

        Shared by the batch ``execute()`` (a fresh kernel each call) and
        ``LiterateSession.step()`` (one persistent kernel threaded across rounds).
        The kernel owns the namespace, so this must NOT rebuild it -- doing so
        would wipe state threaded from prior rounds.
        """
        start = time.perf_counter()

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
            result = kernel.execute_cell(cell, index)

            if not result.success:
                # Roll the document's known writes back to their pre-run state so a
                # failed document does not leave half-applied files behind.
                journal.rollback()
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

        # Every cell ran: accept the writes (drop the snapshots).
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

        return InterpreterExecutionResult(
            success=True,
            output=rendered,
            output_format="markdown",
            duration_ms=elapsed,
            metadata={
                "variables": variables,
                "continue_requested": continue_requested,
                "cell_count": len(parsed.cells),
                "frontmatter": {
                    "echo": parsed.frontmatter.echo,
                    "output": parsed.frontmatter.output,
                    "interpreter": parsed.frontmatter.interpreter,
                },
            },
        )


_INTERNAL_NAMES = frozenset({
    "read_file", "write_file", "apply_diff",
    "search_content", "run_command", "run_tests",
    "__literate_continue__", "__builtins__",
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

    Injected tools carry no literate-specific path metadata, so ``kind`` is
    derived from the grade (w>=3 write, w==2 exec, else read); only the grade
    matters for the ceiling comparison. Conservative defaults (3/3) for tools
    missing a grade fail closed -- safer to over-refuse than under-refuse.
    """
    effects_map = dict(LITERATE_TOOL_EFFECTS)
    resolved = context.tools
    specs = getattr(resolved, "tools", None) if resolved is not None else None
    for name, spec in (specs or {}).items():
        if name in effects_map:
            continue
        w = getattr(spec, "grade_w", 3)
        d = getattr(spec, "effects_ceiling", 3)
        kind = "write" if w >= 3 else "exec" if w == 2 else "read"
        effects_map[name] = ToolEffect(grade=Grade(w, d), kind=kind)
    return effects_map


def _build_namespace(context: ExecutionContext) -> dict[str, Any]:
    """Build the execution namespace for a literate document."""
    ns: dict[str, Any] = {}

    ns.update(make_tool_namespace(context.base_dir))

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
from .session import LiterateSession, StepResult, strip_think  # noqa: E402,F401
