"""Restrict step: rewrite standard Python into the lackpy restricted subset."""

from __future__ import annotations

import time
from typing import Any

from ..context import ProgramState, StepContext, StepTrace
from ..cleanup import deterministic_cleanup
from ..sanitize import sanitize_output
from ...lang.grammar import ALLOWED_BUILTINS


_RESTRICT_SYSTEM_PROMPT = """You rewrite Python code to use ONLY a restricted set of pre-loaded functions.

Rules:
- No imports. All functions are pre-loaded.
- No function definitions, class definitions, while loops, try/except.
- For loops, if/else, list comprehensions, f-strings are allowed.
- Assign tool results to variables and reuse them.

Available functions (use ONLY these):
{namespace_desc}

Allowed builtins: {builtins}

Output ONLY the rewritten code. No explanation, no markdown fences."""

_RESTRICT_USER_PROMPT = """Rewrite this Python code using ONLY the available functions listed above.

Original code:
{program}

Rewritten code:"""


class RestrictStep:
    """Rewrite the current program into the lackpy restricted subset.

    Takes standard Python from a Solve step and rewrites it using
    only the available tool functions and allowed builtins.
    """

    name = "restrict"

    def __init__(self, provider: Any) -> None:
        self._provider = provider

    async def run(self, ctx: StepContext) -> StepContext:
        if ctx.current is None:
            return ctx

        start = time.perf_counter()

        kit = ctx.current.kit or ctx.kit
        namespace_desc = kit.description
        builtins_str = ", ".join(sorted(ALLOWED_BUILTINS))

        system_prompt = _RESTRICT_SYSTEM_PROMPT.format(
            namespace_desc=namespace_desc,
            builtins=builtins_str,
        )
        user_prompt = _RESTRICT_USER_PROMPT.format(program=ctx.current.program)

        raw = await self._provider.generate(user_prompt, system_prompt)
        elapsed = (time.perf_counter() - start) * 1000

        if raw is None:
            program = ""
        else:
            program = sanitize_output(raw)
            program = deterministic_cleanup(program)

        ctx.programs.append(ProgramState(
            program=program,
            intent=ctx.intent,
            kit=kit,
            valid=None,
            errors=[],
            trace=StepTrace(
                step_name=self.name,
                provider_name=self._provider.name,
                model=getattr(self._provider, '_model', None),
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                raw_output=raw,
                duration_ms=elapsed,
            ),
        ))
        return ctx
