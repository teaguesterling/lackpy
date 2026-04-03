"""FewShotCorrect step: re-invoke provider with error feedback."""

from __future__ import annotations

import time
from typing import Any

from ..context import ProgramState, StepContext, StepTrace
from ..cleanup import deterministic_cleanup
from ..hints import enrich_errors
from ..sanitize import sanitize_output


class FewShotCorrectStep:
    """Re-call the provider with the original intent plus enriched error feedback.

    Pushes a new ProgramState with the corrected program.
    """

    name = "few_shot_correct"

    def __init__(self, provider: Any) -> None:
        self._provider = provider

    async def run(self, ctx: StepContext) -> StepContext:
        if ctx.current is None:
            return ctx

        start = time.perf_counter()
        namespace_desc = ctx.kit.description
        enriched = enrich_errors(ctx.current.errors, namespace_desc)

        raw = await self._provider.generate(
            ctx.intent, namespace_desc, error_feedback=enriched,
        )
        elapsed = (time.perf_counter() - start) * 1000

        if raw is None:
            program = ""
        else:
            program = sanitize_output(raw)
            program = deterministic_cleanup(program)

        ctx.programs.append(ProgramState(
            program=program,
            intent=ctx.intent,
            kit=ctx.current.kit,
            valid=None,
            errors=[],
            trace=StepTrace(
                step_name=self.name,
                provider_name=self._provider.name,
                model=getattr(self._provider, '_model', None),
                system_prompt=None,
                user_prompt=ctx.intent,
                raw_output=raw,
                duration_ms=elapsed,
            ),
        ))
        return ctx
