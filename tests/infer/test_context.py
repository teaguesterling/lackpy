"""Tests for StepContext and ProgramState."""

import pytest
from lackpy.infer.context import StepContext, ProgramState, StepTrace
from lackpy.kit.registry import ResolvedKit
from lackpy.kit.toolbox import ToolSpec
from lackpy.lang.grader import Grade


def _make_kit(tools=None):
    tools = tools or {"read": ToolSpec(name="read", provider="builtin", description="Read file")}
    return ResolvedKit(
        tools=tools,
        callables={n: lambda *a: None for n in tools},
        grade=Grade(w=1, d=1),
        description="read(path) -> str: Read file",
    )


class TestStepTrace:
    def test_create_trace(self):
        trace = StepTrace(
            step_name="generate", provider_name="ollama", model="qwen2.5:3b",
            system_prompt="You are...", user_prompt="find files",
            raw_output="files = glob('**/*.py')", duration_ms=123.4,
        )
        assert trace.step_name == "generate"
        assert trace.duration_ms == 123.4

    def test_trace_with_no_provider(self):
        trace = StepTrace(
            step_name="cleanup", provider_name=None, model=None,
            system_prompt=None, user_prompt=None, raw_output=None, duration_ms=0.5,
        )
        assert trace.provider_name is None


class TestProgramState:
    def test_create_program_state(self):
        trace = StepTrace(
            step_name="generate", provider_name="ollama", model="qwen2.5:3b",
            system_prompt=None, user_prompt="find files",
            raw_output="glob('**/*.py')", duration_ms=100.0,
        )
        state = ProgramState(
            program="glob('**/*.py')", intent="find files", kit=_make_kit(),
            valid=None, errors=[], trace=trace,
        )
        assert state.program == "glob('**/*.py')"
        assert state.valid is None

    def test_program_state_with_errors(self):
        trace = StepTrace(
            step_name="generate", provider_name="ollama", model="qwen2.5:3b",
            system_prompt=None, user_prompt="find files",
            raw_output="import glob", duration_ms=50.0,
        )
        state = ProgramState(
            program="import glob", intent="find files", kit=_make_kit(),
            valid=False, errors=["Forbidden AST node: Import at line 1"], trace=trace,
        )
        assert not state.valid
        assert len(state.errors) == 1


class TestStepContext:
    def test_create_context(self):
        kit = _make_kit()
        ctx = StepContext(intent="find files", kit=kit)
        assert ctx.intent == "find files"
        assert ctx.programs == []
        assert ctx.current is None

    def test_current_returns_last_program(self):
        kit = _make_kit()
        ctx = StepContext(intent="find files", kit=kit)
        trace = StepTrace(
            step_name="generate", provider_name=None, model=None,
            system_prompt=None, user_prompt=None, raw_output=None, duration_ms=0,
        )
        state1 = ProgramState(
            program="v1", intent="find files", kit=kit,
            valid=False, errors=["err"], trace=trace,
        )
        state2 = ProgramState(
            program="v2", intent="find files", kit=kit,
            valid=True, errors=[], trace=trace,
        )
        ctx.programs.append(state1)
        assert ctx.current.program == "v1"
        ctx.programs.append(state2)
        assert ctx.current.program == "v2"
        assert len(ctx.programs) == 2
