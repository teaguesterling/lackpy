"""Restricted Python interpreter plugin.

Wraps the original lackpy execution path — AST validation against the
allowed grammar, then restricted exec via :class:`RestrictedRunner`.
This is the first concrete ``Interpreter`` implementation and the one
lackpy has been built around since the beginning.

The validation and execution logic is unchanged from
:meth:`LackpyService.run_program`; this plugin just adapts that path
into the new plugin protocol so other interpreters can sit alongside it.
"""

from __future__ import annotations

import os
import time
from typing import Any

from ..lang.validator import validate
from ..run.runner import RestrictedRunner
from .base import (
    ExecutionContext,
    Interpreter,
    InterpreterExecutionResult,
    InterpreterValidationResult,
)


class PythonInterpreter:
    """Restricted-Python interpreter.

    The program is a lackpy program — a subset of Python 3 with imports,
    function definitions, and dangerous builtins removed. It is validated
    against the kit's allowed names and then executed with tool callables
    injected into the namespace.

    Output shape: whatever the program's last expression evaluated to.
    The ``output_format`` is ``"python"`` because the value may be any
    Python type. Consumers that want a string representation should
    format it themselves.
    """

    name = "python"
    description = "Restricted Python with tool callables from the kit"

    def __init__(self) -> None:
        self._runner = RestrictedRunner()

    def system_prompt_hint(self) -> str:
        """Interpreter-specialized prompt fragment for the python interpreter.

        Emphasizes orchestration over implementation — the primary failure
        mode observed in evaluation is models writing ``def foo(): ...``
        instead of calling pre-loaded tool functions. Includes
        FORBIDDEN_NAMES so models know what NOT to reference.
        """
        from ..lang.grammar import FORBIDDEN_NAMES
        # Select the most commonly-attempted forbidden names for the prompt.
        # The full list is too long; these are the ones models actually reach for.
        key_forbidden = sorted(
            n for n in FORBIDDEN_NAMES
            if n in ("open", "os", "sys", "pathlib", "subprocess", "shutil",
                     "map", "filter", "reduce", "getattr", "setattr",
                     "type", "super", "input")
        )
        forbidden_str = ", ".join(key_forbidden)
        return (
            "You are a program generator. Output a single Python snippet "
            "that orchestrates pre-loaded tool functions.\n"
            "\n"
            "CRITICAL — ORCHESTRATE, DO NOT IMPLEMENT:\n"
            "  - CALL the pre-loaded tools. Do NOT re-implement them.\n"
            "  - Do NOT use open(). Use read_file() for ALL file reading.\n"
            "  - Do NOT define functions, classes, or use import.\n"
            f"  - FORBIDDEN names (will be rejected): {forbidden_str}\n"
            "\n"
            "Output ONLY the program — no markdown, no code fences, no prose.\n"
            "Assign tool results to variables, then end with a bare expression "
            "holding the final answer."
        )

    def validate(
        self,
        program: str,
        context: ExecutionContext,
    ) -> InterpreterValidationResult:
        """Validate against the kit's allowed names and any extra rules.

        Allowed names are the union of the kit's tool names and any
        parameter names injected into the context. This matches the
        original :meth:`LackpyService.run_program` behavior.
        """
        if context.kit is None:
            return InterpreterValidationResult(
                valid=False,
                errors=["PythonInterpreter requires a resolved kit in the execution context"],
            )
        allowed = set(context.kit.tools.keys())
        if context.params:
            allowed |= set(context.params.keys())
        result = validate(program, allowed_names=allowed, extra_rules=context.extra_rules)
        return InterpreterValidationResult(
            valid=result.valid,
            errors=list(result.errors),
        )

    async def execute(
        self,
        program: str,
        context: ExecutionContext,
    ) -> InterpreterExecutionResult:
        """Execute a validated lackpy program under the restricted runner.

        Runs the program with its working directory set to
        ``context.base_dir`` (restored on exit). Tool callables come from
        the resolved kit; parameter values come from the context.
        """
        start = time.perf_counter()
        validation = self.validate(program, context)
        if not validation.valid:
            return InterpreterExecutionResult(
                success=False,
                error="Validation failed: " + "; ".join(validation.errors),
                output_format="none",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        prev_cwd = os.getcwd()
        try:
            os.chdir(context.base_dir)
            exec_result = self._runner.run(
                program,
                context.kit.callables,
                params=context.params or None,
            )
        finally:
            os.chdir(prev_cwd)

        elapsed = (time.perf_counter() - start) * 1000
        return InterpreterExecutionResult(
            success=exec_result.success,
            output=exec_result.output,
            output_format="python",
            error=exec_result.error,
            duration_ms=elapsed,
            metadata={
                "trace": exec_result.trace,
                "variables": exec_result.variables,
            },
        )
