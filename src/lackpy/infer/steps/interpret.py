"""Interpret step: execute a generated program through an interpreter.

Serves two roles in the pipeline:

1. **As a Check** (inside RetryWithFeedback): runs the program through
   the interpreter, caches the result, returns pass/fail + errors.
   Execution failures feed back into the next generation attempt.

2. **As a Step** (after RetryWithFeedback): emits the cached execution
   result into the context for downstream stages. If no cached result
   exists, executes fresh.

This dual role means execution can live inside the retry loop (for DSLs
where runtime failures are common, like SQL) or outside it (for DSLs
with near-100% execution rates, like CSS selectors).
"""

from __future__ import annotations

import time
from typing import Any

from ..context import ProgramState, StepContext, StepTrace
from ...interpreters.base import ExecutionContext


class InterpretStep:
    """Execute programs through an interpreter — usable as both Step and Check.

    As a Step (``run``): executes the program through the interpreter and
    pushes the output into the context as a new ProgramState.

    As a Check (``check``): executes the program, caches the result, and
    returns (ok, errors). The cached result is emitted when later used
    as a Step, avoiding double execution.
    """

    name = "interpret"

    def __init__(self, interpreter: Any, config: dict | None = None,
                 base_dir: str = ".") -> None:
        self._interpreter = interpreter
        self._config = config or {}
        self._base_dir = base_dir
        self._cached_result = None
        self._cached_program = None

    def check(self, program: str, ctx: StepContext) -> tuple[bool, list[str]]:
        """Check protocol: execute and cache result, return pass/fail."""
        import asyncio
        exec_ctx = ExecutionContext(
            kit=ctx.kit,
            config=self._config,
            base_dir=self._base_dir,
        )
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = pool.submit(
                    asyncio.run,
                    self._interpreter.execute(program, exec_ctx)
                ).result()
        else:
            result = asyncio.run(
                self._interpreter.execute(program, exec_ctx)
            )

        self._cached_result = result
        self._cached_program = program

        if not result.success:
            return False, [result.error or "execution failed"]
        return True, []

    async def run(self, ctx: StepContext) -> StepContext:
        start = time.perf_counter()

        if not ctx.current or not ctx.current.program:
            ctx.programs.append(ProgramState(
                program="",
                intent=ctx.intent,
                kit=ctx.kit,
                valid=False,
                errors=["no program to interpret"],
                trace=StepTrace(
                    step_name=self.name,
                    provider_name=None,
                    model=None,
                    system_prompt=None,
                    user_prompt=None,
                    raw_output=None,
                    duration_ms=(time.perf_counter() - start) * 1000,
                ),
            ))
            return ctx

        program = ctx.current.program

        if self._cached_result and self._cached_program == program:
            result = self._cached_result
            self._cached_result = None
            self._cached_program = None
        else:
            exec_ctx = ExecutionContext(
                kit=ctx.kit,
                config=self._config,
                base_dir=self._base_dir,
            )
            result = await self._interpreter.execute(program, exec_ctx)

        elapsed = (time.perf_counter() - start) * 1000
        output_text = str(result.output) if result.output else ""

        ctx.programs.append(ProgramState(
            program=output_text if result.success else program,
            intent=ctx.intent,
            kit=ctx.kit,
            valid=result.success,
            errors=[result.error] if result.error else [],
            trace=StepTrace(
                step_name=self.name,
                provider_name=f"interpreter:{self._interpreter.name}",
                model=None,
                system_prompt=None,
                user_prompt=program,
                raw_output=output_text[:2000] if output_text else None,
                duration_ms=elapsed,
            ),
        ))
        return ctx
