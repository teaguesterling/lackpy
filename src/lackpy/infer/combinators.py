"""Step protocol and combinators for inference pipeline composition.

Combinators compose steps into pipelines:

- ``Sequence``: run steps in order, threading context
- ``Fallback``: try branches until one succeeds
- ``RetryWithFeedback``: wrap a step with validation checks and retry on
  failure, feeding error messages back into the next attempt

RetryWithFeedback is the key combinator for staged DSL pipelines. Each
stage generates output in a constrained DSL; checks validate syntax and
runtime constraints; failures produce targeted feedback that guides the
next attempt.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

from .context import StepContext


class Step(Protocol):
    """A single fold function in the inference pipeline."""

    name: str

    async def run(self, ctx: StepContext) -> StepContext: ...


class Sequence:
    """Run steps in order, threading context through."""

    def __init__(self, steps: list) -> None:
        self.steps = steps
        self.name = "sequence"

    async def run(self, ctx: StepContext) -> StepContext:
        for step in self.steps:
            ctx = await step.run(ctx)
        return ctx


class Fallback:
    """Try branches in order. Return first where ctx.current.valid is True.

    Each branch starts from the context at Fallback entry. All ProgramState
    entries from all branches accumulate in the flat programs list.
    """

    def __init__(self, branches: list) -> None:
        self.branches = branches
        self.name = "fallback"

    async def run(self, ctx: StepContext) -> StepContext:
        if not self.branches:
            return ctx
        for branch in self.branches:
            ctx = await branch.run(ctx)
            if ctx.current and ctx.current.valid:
                return ctx
        return ctx


class Check(Protocol):
    """A validation check that returns (ok, errors) for a program."""

    def check(self, program: str, ctx: StepContext) -> tuple[bool, list[str]]: ...


class InterpreterCheck:
    """Adapt an interpreter's validate() method to the Check protocol."""

    def __init__(self, interpreter: Any) -> None:
        self._interpreter = interpreter

    def check(self, program: str, ctx: StepContext) -> tuple[bool, list[str]]:
        from ..interpreters.base import ExecutionContext
        result = self._interpreter.validate(program, ExecutionContext())
        return result.valid, result.errors if not result.valid else []


class CallableCheck:
    """Adapt any callable (program, ctx) -> (bool, list[str]) to Check."""

    def __init__(self, fn: Callable, name: str = "custom") -> None:
        self._fn = fn
        self.name = name

    def check(self, program: str, ctx: StepContext) -> tuple[bool, list[str]]:
        return self._fn(program, ctx)


class RetryWithFeedback:
    """Wrap a generation step with validation checks and error-guided retry.

    On each attempt:
    1. Run the wrapped step to produce a program
    2. Run all checks against the output
    3. If all pass, mark valid and return
    4. If any fail, collect error messages and re-run the step with
       feedback injected into the context

    The step must support error feedback — GenerateDSLStep reads
    ``ctx.current.errors`` to build retry prompts when it sees a prior
    failed attempt in the programs list.

    Usage::

        RetryWithFeedback(
            GenerateDSLStep(coder_3b, selector_interp),
            checks=[selector_interp, NonEmptyResultCheck()],
            max_retries=2,
        )
    """

    def __init__(
        self,
        step: Any,
        checks: list | None = None,
        max_retries: int = 2,
    ) -> None:
        self._step = step
        self._checks = _normalize_checks(checks or [])
        self._max_retries = max_retries
        self.name = f"retry({step.name})"

    async def run(self, ctx: StepContext) -> StepContext:
        for attempt in range(1 + self._max_retries):
            ctx = await self._step.run(ctx)

            if not ctx.current or not ctx.current.program:
                if attempt < self._max_retries:
                    continue
                return ctx

            all_ok = True
            all_errors: list[str] = []
            for check in self._checks:
                ok, errors = check.check(ctx.current.program, ctx)
                if not ok:
                    all_ok = False
                    all_errors.extend(errors)

            if all_ok:
                ctx.current.valid = True
                ctx.current.errors = []
                return ctx

            ctx.current.valid = False
            ctx.current.errors = all_errors

            if attempt >= self._max_retries:
                return ctx

        return ctx


def _normalize_checks(checks: list) -> list:
    """Convert mixed check types to a uniform list of Check-protocol objects.

    Dispatch order matters: interpreters (have ``validate``) are tested
    before the generic ``check`` fallback so they always get the
    InterpreterCheck adapter even if they also have a ``check`` method.
    Objects with ``check`` (like InterpretStep) are used as-is.
    """
    normalized = []
    for c in checks:
        if hasattr(c, "validate"):
            normalized.append(InterpreterCheck(c))
        elif hasattr(c, "check"):
            normalized.append(c)
        elif callable(c):
            normalized.append(CallableCheck(c))
        else:
            raise TypeError(f"Cannot use {type(c)} as a check — needs check(), validate(), or be callable")
    return normalized
