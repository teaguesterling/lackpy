"""Literate programming interpreter for lackpy.

Executes markdown documents with embedded ```lackpy code blocks.
The execution pipeline: parse markdown → cell sequence → execute
cells one-by-one through LightweightKernel → captured stdout IS
the rendered document.
"""

from __future__ import annotations

import time
from typing import Any

from ..base import (
    ExecutionContext,
    InterpreterExecutionResult,
    InterpreterValidationResult,
)
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
        start = time.perf_counter()

        parsed = parse(program)
        if parsed.errors:
            return InterpreterExecutionResult(
                success=False,
                error="Parse errors: " + "; ".join(parsed.errors),
                output_format="none",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        namespace = _build_namespace(context)
        kernel = LightweightKernel(namespace=namespace)

        output_parts: list[str] = []
        continue_requested = False

        for index, cell in enumerate(parsed.cells):
            result = kernel.execute_cell(cell, index)

            if not result.success:
                return InterpreterExecutionResult(
                    success=False,
                    error=result.error or "Unknown error",
                    output_format="text",
                    duration_ms=(time.perf_counter() - start) * 1000,
                )

            if result.output:
                output_parts.append(result.output)

            if result.namespace_delta.get("__continue_requested__"):
                continue_requested = True

        elapsed = (time.perf_counter() - start) * 1000

        # Filter variables the same way the old implementation did:
        # exclude underscore-prefixed, internal tool names, and callables
        raw_ns = kernel.get_namespace()
        variables = {
            k: v for k, v in raw_ns.items()
            if k not in _INTERNAL_NAMES and not callable(v)
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
})


def _build_namespace(context: ExecutionContext) -> dict[str, Any]:
    """Build the execution namespace for a literate document."""
    ns: dict[str, Any] = {}

    ns.update(make_tool_namespace(context.base_dir))

    import builtins as _builtins_mod
    ns["__builtins__"] = _builtins_mod

    if context.kit:
        for name, fn in context.kit.callables.items():
            ns[name] = fn

    if context.params:
        ns.update(context.params)

    return ns
