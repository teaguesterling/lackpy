"""Validate step: check current program against AST whitelist."""

from __future__ import annotations

from ..context import StepContext
from ...lang.validator import validate


class ValidateStep:
    """Validate the current program against allowed names and rules.

    Mutates ctx.current in place: sets valid and errors. Does not
    push a new ProgramState -- validation is a gate, not a transform.
    """

    name = "validate"

    async def run(self, ctx: StepContext) -> StepContext:
        if ctx.current is None:
            return ctx

        allowed = set(ctx.current.tools.tools.keys()) if ctx.current.tools else set()
        if ctx.tools:
            allowed |= set(ctx.tools.tools.keys())

        result = validate(
            ctx.current.program,
            allowed_names=allowed,
            extra_rules=ctx.extra_rules,
        )
        ctx.current.valid = result.valid
        ctx.current.errors = result.errors
        return ctx
