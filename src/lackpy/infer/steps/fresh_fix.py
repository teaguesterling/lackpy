"""FreshFix step: call provider with specialized fixer prompt."""

from __future__ import annotations

import time
from typing import Any

from ..context import ProgramState, StepContext, StepTrace
from ..cleanup import deterministic_cleanup
from ..fixer import build_fixer_messages
from ..sanitize import sanitize_output


async def _call_fixer(provider: Any, messages: list[dict]) -> str | None:
    """Try provider._chat() then fallback to provider._create_message()."""
    if hasattr(provider, "_chat"):
        try:
            response = await provider._chat(messages, temperature=0.4)
            content = response.get("message", {}).get("content", "")
            return content if content else None
        except Exception:
            pass

    if hasattr(provider, "_create_message"):
        try:
            system = next((m["content"] for m in messages if m["role"] == "system"), "")
            user_messages = [m for m in messages if m["role"] != "system"]
            response = await provider._create_message(system, user_messages)
            return response if isinstance(response, str) else None
        except Exception:
            pass

    return None


class FreshFixStep:
    """Call provider with a specialized fixer prompt to repair a broken program.

    Pushes a new ProgramState with the fixed program.
    """

    name = "fresh_fix"

    def __init__(self, provider: Any) -> None:
        self._provider = provider

    async def run(self, ctx: StepContext) -> StepContext:
        if ctx.current is None:
            return ctx

        start = time.perf_counter()
        namespace_desc = ctx.kit.description
        errors_text = "\n".join(ctx.current.errors)

        messages = build_fixer_messages(
            intent=ctx.intent,
            broken_program=ctx.current.program,
            errors=errors_text,
            namespace_desc=namespace_desc,
        )

        raw = await _call_fixer(self._provider, messages)
        elapsed = (time.perf_counter() - start) * 1000

        if raw is None:
            program = ""
        else:
            program = sanitize_output(raw)
            program = deterministic_cleanup(program)

        system_prompt = next((m["content"] for m in messages if m["role"] == "system"), None)
        user_prompt = next((m["content"] for m in messages if m["role"] == "user"), None)

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
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                raw_output=raw,
                duration_ms=elapsed,
            ),
        ))
        return ctx
