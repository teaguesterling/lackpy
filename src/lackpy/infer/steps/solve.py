"""Solve step: generate standard Python without lackpy restrictions."""

from __future__ import annotations

import time
from typing import Any

from ..context import ProgramState, StepContext, StepTrace
from ..sanitize import sanitize_output


_SOLVE_SYSTEM_PROMPT = """You write short Python scripts to accomplish tasks using pre-loaded helper functions.

Write standard Python. You may use imports, loops, classes -- whatever you need.
Keep it short and direct. Output ONLY the code -- no markdown, no explanation.

Available helper functions (already imported):
{namespace_desc}
"""


class SolveStep:
    """Generate a standard Python program from intent, unconstrained.

    Unlike GenerateStep, this does NOT apply deterministic_cleanup or
    restrict the prompt to the lackpy subset. The output may contain
    imports, classes, function defs -- anything valid Python.
    """

    name = "solve"

    def __init__(self, provider: Any) -> None:
        self._provider = provider

    async def run(self, ctx: StepContext) -> StepContext:
        start = time.perf_counter()
        namespace_desc = ctx.kit.description
        system_prompt = _SOLVE_SYSTEM_PROMPT.format(namespace_desc=namespace_desc)

        raw = await self._provider.generate(ctx.intent, system_prompt)
        elapsed = (time.perf_counter() - start) * 1000

        if raw is None:
            program = ""
        else:
            program = sanitize_output(raw)

        ctx.programs.append(ProgramState(
            program=program,
            intent=ctx.intent,
            kit=ctx.kit,
            valid=None,
            errors=[],
            trace=StepTrace(
                step_name=self.name,
                provider_name=self._provider.name,
                model=getattr(self._provider, '_model', None),
                system_prompt=system_prompt,
                user_prompt=ctx.intent,
                raw_output=raw,
                duration_ms=elapsed,
            ),
        ))
        return ctx
