"""Generate DSL step: produce a constrained program using interpreter hints.

Like GenerateStep, but passes the interpreter's system_prompt_hint() to
the prompt builder so the LLM generates output in the interpreter's DSL
rather than generic Python.

On retry (when ctx.current has errors from a prior attempt), the error
messages are appended to the prompt as feedback so the model can correct
its output. The feedback format uses plain-text ERROR:/HINT: prefixes,
which work better than XML tags for small models (validated in iterative
eval: XML tags leak into 3b completions, plain text works for 7b+).
"""

from __future__ import annotations

import time
from typing import Any

from ..context import ProgramState, StepContext, StepTrace
from ..prompt import build_system_prompt, collect_example_pool
from ..sanitize import sanitize_output


class GenerateDSLStep:
    """Generate a DSL program guided by an interpreter's prompt hint.

    The interpreter's ``system_prompt_hint()`` replaces the default
    Jupyter-cell framing, constraining the model's output to the DSL.

    When the context contains a prior failed attempt (ctx.current with
    errors), the error messages are included as feedback in the prompt.
    """

    name = "generate-dsl"

    def __init__(self, provider: Any, interpreter: Any) -> None:
        self._provider = provider
        self._interpreter = interpreter

    async def run(self, ctx: StepContext) -> StepContext:
        start = time.perf_counter()
        namespace_desc = ctx.kit.description
        example_pool = collect_example_pool(list(ctx.kit.tools.values()))
        system_prompt = build_system_prompt(
            namespace_desc,
            ctx.params_desc,
            intent=ctx.intent,
            example_pool=example_pool,
            interpreter=self._interpreter,
        )

        error_feedback = None
        if ctx.current and ctx.current.errors and not ctx.current.valid:
            error_feedback = ctx.current.errors

        raw = await self._provider.generate(
            ctx.intent, namespace_desc,
            system_prompt_override=system_prompt,
            interpreter=self._interpreter,
            error_feedback=error_feedback,
        )
        elapsed = (time.perf_counter() - start) * 1000

        program = _clean_dsl_output(raw)

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


def _clean_dsl_output(raw: str | None) -> str:
    """Strip code fences, prose, and trailing noise from DSL output."""
    if raw is None:
        return ""
    program = sanitize_output(raw).strip()
    for fence in ("```css", "```", "`"):
        program = program.strip(fence)
    program = program.strip()
    lines = program.split("\n")
    clean = []
    for line in lines:
        s = line.strip()
        if s and not s.startswith(("Output", "Note", "This", "Here", "The ", "I ")):
            clean.append(s)
        elif clean:
            break
    return "\n".join(clean) if clean else program
