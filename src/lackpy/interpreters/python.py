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
        """The intro line of the lackpy prompt for this interpreter.

        ``build_system_prompt`` puts the pre-prompt, the style guide and the
        skeleton after it; they teach what the kit-era hint said in prose
        ("use read_file", "do not define functions or import"), which the
        stepy100 language reverses (stepy100 Task 8, Task 7 review M3).
        """
        from ..infer.prompt import INTRO
        return INTRO

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
