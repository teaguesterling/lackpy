"""Tests for Step protocol, Sequence, and Fallback combinators."""

import pytest
from lackpy.infer.combinators import Sequence, Fallback
from lackpy.infer.context import StepContext, ProgramState, StepTrace
from lackpy.kit.toolbox import ToolSpec
from lackpy.kit.registry import ResolvedKit
from lackpy.lang.grader import Grade


def _make_kit():
    tools = {"read": ToolSpec(name="read", provider="builtin", description="Read file")}
    return ResolvedKit(
        tools=tools,
        callables={n: lambda *a: None for n in tools},
        grade=Grade(w=1, d=1),
        description="read(path) -> str",
    )


def _make_trace(step_name="test"):
    return StepTrace(
        step_name=step_name, provider_name=None, model=None,
        system_prompt=None, user_prompt=None, raw_output=None, duration_ms=0,
    )


def _make_ctx():
    return StepContext(intent="test", kit=_make_kit())


class _AppendStep:
    def __init__(self, program: str, valid: bool | None = None):
        self.name = f"append_{program}"
        self._program = program
        self._valid = valid

    async def run(self, ctx: StepContext) -> StepContext:
        ctx.programs.append(ProgramState(
            program=self._program, intent=ctx.intent, kit=ctx.kit,
            valid=self._valid, errors=[], trace=_make_trace(self.name),
        ))
        return ctx


class _SetValidStep:
    def __init__(self, valid: bool, errors: list[str] | None = None):
        self.name = "set_valid"
        self._valid = valid
        self._errors = errors or []

    async def run(self, ctx: StepContext) -> StepContext:
        if ctx.current:
            ctx.current.valid = self._valid
            ctx.current.errors = self._errors
        return ctx


class TestSequence:
    @pytest.mark.asyncio
    async def test_runs_steps_in_order(self):
        seq = Sequence([_AppendStep("a"), _AppendStep("b"), _AppendStep("c")])
        ctx = _make_ctx()
        ctx = await seq.run(ctx)
        assert [p.program for p in ctx.programs] == ["a", "b", "c"]
        assert ctx.current.program == "c"

    @pytest.mark.asyncio
    async def test_empty_sequence(self):
        seq = Sequence([])
        ctx = _make_ctx()
        ctx = await seq.run(ctx)
        assert ctx.programs == []


class TestFallback:
    @pytest.mark.asyncio
    async def test_returns_first_valid_branch(self):
        fb = Fallback([
            Sequence([_AppendStep("bad", valid=None), _SetValidStep(False)]),
            Sequence([_AppendStep("good", valid=None), _SetValidStep(True)]),
            Sequence([_AppendStep("never_reached", valid=None), _SetValidStep(True)]),
        ])
        ctx = _make_ctx()
        ctx = await fb.run(ctx)
        assert ctx.current.program == "good"
        assert ctx.current.valid is True
        programs = [p.program for p in ctx.programs]
        assert "never_reached" not in programs

    @pytest.mark.asyncio
    async def test_returns_last_failure_when_all_fail(self):
        fb = Fallback([
            Sequence([_AppendStep("a", valid=None), _SetValidStep(False)]),
            Sequence([_AppendStep("b", valid=None), _SetValidStep(False)]),
        ])
        ctx = _make_ctx()
        ctx = await fb.run(ctx)
        assert ctx.current.valid is False
        assert ctx.current.program == "b"

    @pytest.mark.asyncio
    async def test_all_attempts_in_programs_list(self):
        fb = Fallback([
            Sequence([_AppendStep("a", valid=None), _SetValidStep(False)]),
            Sequence([_AppendStep("b", valid=None), _SetValidStep(True)]),
        ])
        ctx = _make_ctx()
        ctx = await fb.run(ctx)
        programs = [p.program for p in ctx.programs]
        assert programs == ["a", "b"]

    @pytest.mark.asyncio
    async def test_empty_fallback(self):
        fb = Fallback([])
        ctx = _make_ctx()
        ctx = await fb.run(ctx)
        assert ctx.current is None
