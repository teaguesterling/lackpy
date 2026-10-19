"""Restricted Python interpreter plugin.

Wraps the lackpy execution path — AST validation against the allowed
grammar, then restricted exec via :class:`RestrictedRunner`, with the
``PathPolicy`` of ``context.base_dir``. This is the same path as
:meth:`LackpyService.run_program`, adapted to the plugin protocol. No tool
callables are bound: the kit was removed in stepy100 Plan 2, Task 7.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

from ..lang.validator import validate
from ..profiles import KIT_REMOVED
from ..run.paths import PathPolicy, load_policy
from ..run.runner import RestrictedRunner
from .base import (
    ExecutionContext,
    Interpreter,
    InterpreterExecutionResult,
    InterpreterValidationResult,
)


class PythonInterpreter:
    """Restricted-Python interpreter.

    The program is a lackpy program: plain Python that the validator has
    checked. It is validated with no allowed names and runs in the runtime
    namespace, with file access through the ``Path`` stand-in under the
    ``PathPolicy`` of ``context.base_dir``. A context whose ``tools`` names
    any tool is an error carrying ``KIT_REMOVED``.

    Output shape: whatever the program's last expression evaluated to.
    The ``output_format`` is ``"python"`` because the value may be any
    Python type. Consumers that want a string representation should
    format it themselves.
    """

    name = "python"
    description = "Plain Python checked by the lackpy validator; files through pathlib.Path"

    def __init__(self, *, path_policy_loader: Callable[[Path], PathPolicy] | None = None) -> None:
        """``path_policy_loader`` builds the policy from ``base_dir`` once per
        run; it defaults to ``load_policy``. Tests pass a temporary home."""
        self._runner = RestrictedRunner()
        self._path_policy_loader = path_policy_loader

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
        """Validate with no allowed names and any extra rules.

        This matches :meth:`LackpyService.run_program`. Params bind no bare
        names (stepy100 Task 5), so their names are not allowed either.
        """
        if context.tools is not None and getattr(context.tools, "tools", None):
            return InterpreterValidationResult(valid=False, errors=[KIT_REMOVED])
        result = validate(program, allowed_names=set(), extra_rules=context.extra_rules)
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
        ``context.base_dir`` (restored on exit), under the ``PathPolicy``
        loaded from ``context.base_dir`` for this run. Parameter values come
        from the context. A malformed settings file gives a failed result
        that names the file.
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

        loader = self._path_policy_loader or load_policy
        try:
            policy = loader(Path(context.base_dir))
        except Exception as e:  # host code; never raises out (review M4)
            from ..service import _policy_error
            return InterpreterExecutionResult(
                success=False,
                error=_policy_error(e),
                output_format="none",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        prev_cwd = os.getcwd()
        try:
            os.chdir(context.base_dir)
            exec_result = self._runner.run(program, params=context.params or None,
                                           path_policy=policy)
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
