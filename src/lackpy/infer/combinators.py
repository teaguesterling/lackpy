"""Step protocol and combinators for inference pipeline composition."""

from __future__ import annotations

from typing import Protocol

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
