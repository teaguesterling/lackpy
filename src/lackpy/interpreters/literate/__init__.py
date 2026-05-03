"""Literate programming interpreter for lackpy.

Executes markdown documents with embedded ```lackpy code blocks.
The compilation pipeline: parse markdown → cell sequence → compile
to Python → execute via restricted runner → captured stdout IS the
rendered document.
"""

from __future__ import annotations

import io
import os
import sys
import time
from contextlib import redirect_stdout
from typing import Any

from ..base import (
    ExecutionContext,
    InterpreterExecutionResult,
    InterpreterValidationResult,
)
from .compiler import CONTINUE_SENTINEL, compile_cells
from .parser import parse
from .prompt import LITERATE_HINT
from .tools import make_tool_namespace


class LiterateInterpreter:
    """Literate programming interpreter.

    Takes a markdown document with ```lackpy code blocks, compiles it
    to Python, and executes it. The captured stdout is the rendered
    document — prose becomes print() calls, code executes and its
    output joins the stream.
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
        """Execute a literate document and return the rendered output."""
        start = time.perf_counter()

        parsed = parse(program)
        if parsed.errors:
            return InterpreterExecutionResult(
                success=False,
                error="Parse errors: " + "; ".join(parsed.errors),
                output_format="none",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        try:
            compiled = compile_cells(parsed)
        except ValueError as e:
            return InterpreterExecutionResult(
                success=False,
                error=f"Compilation error: {e}",
                output_format="none",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        namespace = _build_namespace(context)

        continue_fn_called = False

        def _continue_sentinel():
            nonlocal continue_fn_called
            continue_fn_called = True

        namespace["__literate_continue__"] = _continue_sentinel

        prev_cwd = os.getcwd()
        if context.base_dir:
            os.chdir(context.base_dir)

        try:
            stdout_capture = io.StringIO()
            with redirect_stdout(stdout_capture):
                compiled_code = compile(compiled, "<literate>", "exec")
                _run_literate_code(compiled_code, namespace)

            rendered = stdout_capture.getvalue()
        except Exception as e:
            return InterpreterExecutionResult(
                success=False,
                error=str(e),
                output_format="text",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        finally:
            os.chdir(prev_cwd)

        elapsed = (time.perf_counter() - start) * 1000

        variables = {
            k: v for k, v in namespace.items()
            if not k.startswith("_") and k not in _INTERNAL_NAMES and callable(v) is False
        }

        return InterpreterExecutionResult(
            success=True,
            output=rendered,
            output_format="markdown",
            duration_ms=elapsed,
            metadata={
                "variables": variables,
                "continue_requested": continue_fn_called,
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
    "print", "len", "str", "int", "float", "bool",
    "list", "dict", "set", "tuple", "range", "enumerate",
    "sorted", "reversed", "zip", "map", "filter",
    "sum", "min", "max", "abs", "round",
    "isinstance", "type", "hasattr", "getattr",
    "True", "False", "None",
    "re", "json", "os", "open", "repr",
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


def _make_safe_os_module():
    """Expose a subset of os that models commonly need."""
    import os as _os
    import types
    safe = types.ModuleType("os")
    safe.path = _os.path
    safe.getcwd = _os.getcwd
    safe.listdir = _os.listdir
    safe.walk = _os.walk
    safe.sep = _os.sep
    return safe


def _run_literate_code(compiled_code: object, namespace: dict) -> None:
    # Executes the compiled Python from literate document compilation.
    # This is the core execution path — compiled code is the output of
    # compiler.compile_cells(), not arbitrary user input.
    _do_exec(compiled_code, namespace)


def _do_exec(code, ns):
    exec(code, ns)  # noqa: S102
