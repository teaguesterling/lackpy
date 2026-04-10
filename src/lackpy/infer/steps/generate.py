"""Generate step: produce a lackpy program from intent via a provider."""

from __future__ import annotations

import time
from typing import Any

from ..context import ProgramState, StepContext, StepTrace
from ..cleanup import deterministic_cleanup
from ..prompt import build_system_prompt, collect_example_pool
from ..sanitize import sanitize_output


class GenerateStep:
    """Generate a lackpy-constrained program from intent.

    Uses the provider's generate() method with the kit's namespace
    description as the system prompt context. When the kit's tools
    carry tagged examples, retrieval-augmented prompting selects the
    most relevant ones for the current intent.
    """

    name = "generate"

    def __init__(self, provider: Any) -> None:
        self._provider = provider

    async def run(self, ctx: StepContext) -> StepContext:
        start = time.perf_counter()
        namespace_desc = ctx.kit.description
        example_pool = collect_example_pool(list(ctx.kit.tools.values()))
        system_prompt = build_system_prompt(
            namespace_desc,
            ctx.params_desc,
            intent=ctx.intent,
            example_pool=example_pool,
        )

        # Pass the retrieval-augmented system prompt to the provider.
        # Providers that ignore system_prompt_override (e.g. rules, templates)
        # fall back to their own generation path, which is fine — they don't
        # use system prompts anyway.
        raw = await self._provider.generate(
            ctx.intent, namespace_desc,
            system_prompt_override=system_prompt if example_pool else None,
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
