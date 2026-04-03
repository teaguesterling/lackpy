"""Cleanup step: deterministic AST rewrites (strip imports, rewrite open, etc.)."""

from __future__ import annotations

import time

from ..context import ProgramState, StepContext, StepTrace
from ..cleanup import deterministic_cleanup


class CleanupStep:
    """Apply deterministic cleanup transforms to the current program.

    Pushes a new ProgramState with the cleaned program. Does not validate.
    """

    name = "cleanup"

    async def run(self, ctx: StepContext) -> StepContext:
        if ctx.current is None:
            return ctx

        start = time.perf_counter()
        cleaned = deterministic_cleanup(ctx.current.program)
        elapsed = (time.perf_counter() - start) * 1000

        ctx.programs.append(ProgramState(
            program=cleaned,
            intent=ctx.current.intent,
            kit=ctx.current.kit,
            valid=None,
            errors=[],
            trace=StepTrace(
                step_name=self.name,
                provider_name=None,
                model=None,
                system_prompt=None,
                user_prompt=None,
                raw_output=None,
                duration_ms=elapsed,
            ),
        ))
        return ctx
