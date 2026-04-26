"""Tests for KibitzerPolicySource."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from lackpy.policy.types import PolicyResult, PolicyContext
from lackpy.policy.sources.kibitzer import KibitzerPolicySource
from lackpy.kit.registry import ResolvedKit
from lackpy.lang.grader import Grade
from lackpy.infer.context import StepContext, ProgramState, StepTrace


class FakeHints:
    def __init__(self, hints=None, doc_context=None):
        self.hints = hints or []
        self.doc_context = doc_context


class FakeKibitzerSession:
    def __init__(self, hints=None, doc_context=None, coaching=None):
        self._hints = FakeHints(hints, doc_context)
        self._coaching = coaching
        self.hint_calls: list[dict] = []

    def get_correction_hints(self, errors=None, model=None, attempt=None):
        self.hint_calls.append({"errors": errors, "model": model, "attempt": attempt})
        return self._hints

    def has_coaching(self):
        return self._coaching is not None

    def apply_coaching(self, desc):
        return desc + self._coaching


@pytest.fixture
def empty_kit():
    return ResolvedKit(
        tools={}, callables={}, grade=Grade(w=0, d=0), description="",
    )


def _make_step_trace():
    return StepTrace(
        step_name="test", provider_name=None, model=None,
        system_prompt=None, user_prompt=None, raw_output=None,
        duration_ms=0.0,
    )


class TestKibitzerPolicySourceBasic:
    def test_name_and_priority(self):
        source = KibitzerPolicySource(FakeKibitzerSession())
        assert source.name == "kibitzer"
        assert source.priority == 50

    def test_no_history_passes_through(self, empty_kit):
        source = KibitzerPolicySource(FakeKibitzerSession())
        current = PolicyResult(
            allowed_tools=frozenset({"read_file"}),
            namespace_desc="tools: read_file",
        )
        context: PolicyContext = {"kit": empty_kit}
        result = source.resolve(current, context)
        assert result.allowed_tools == frozenset({"read_file"})
        assert result.namespace_desc == "tools: read_file"
        assert result.resolved is False

    def test_never_modifies_allowed_tools(self, empty_kit):
        session = FakeKibitzerSession(hints=["use read_file instead"])
        source = KibitzerPolicySource(session)
        history = StepContext(intent="test", kit=empty_kit)
        history.programs.append(ProgramState(
            program="open('f')", intent="test", kit=empty_kit,
            valid=False, errors=["Forbidden name: 'open'"],
            trace=_make_step_trace(),
        ))
        current = PolicyResult(allowed_tools=frozenset({"read_file"}))
        context: PolicyContext = {"kit": empty_kit, "history": history}
        result = source.resolve(current, context)
        assert result.allowed_tools == frozenset({"read_file"})


class TestKibitzerPolicySourceHints:
    def test_adds_hints_on_failure(self, empty_kit):
        session = FakeKibitzerSession(hints=["use read_file instead of open"])
        source = KibitzerPolicySource(session)
        history = StepContext(intent="test", kit=empty_kit)
        history.programs.append(ProgramState(
            program="open('f')", intent="test", kit=empty_kit,
            valid=False, errors=["Forbidden name: 'open'"],
            trace=_make_step_trace(),
        ))
        current = PolicyResult(allowed_tools=frozenset({"read_file"}))
        context: PolicyContext = {"kit": empty_kit, "history": history}
        result = source.resolve(current, context)
        assert "use read_file instead of open" in result.prompt_hints

    def test_adds_doc_context(self, empty_kit):
        session = FakeKibitzerSession(doc_context="Signature: read_file(path: str) -> str")
        source = KibitzerPolicySource(session)
        history = StepContext(intent="test", kit=empty_kit)
        history.programs.append(ProgramState(
            program="open('f')", intent="test", kit=empty_kit,
            valid=False, errors=["Forbidden name: 'open'"],
            trace=_make_step_trace(),
        ))
        current = PolicyResult()
        context: PolicyContext = {"kit": empty_kit, "history": history}
        result = source.resolve(current, context)
        assert "Signature: read_file(path: str) -> str" in result.docs

    def test_no_hints_on_valid_program(self, empty_kit):
        session = FakeKibitzerSession(hints=["should not appear"])
        source = KibitzerPolicySource(session)
        history = StepContext(intent="test", kit=empty_kit)
        history.programs.append(ProgramState(
            program="x = 1", intent="test", kit=empty_kit,
            valid=True, errors=[],
            trace=_make_step_trace(),
        ))
        current = PolicyResult()
        context: PolicyContext = {"kit": empty_kit, "history": history}
        result = source.resolve(current, context)
        assert result.prompt_hints == ()


class TestKibitzerPolicySourceCoaching:
    def test_applies_coaching_to_namespace_desc(self, empty_kit):
        session = FakeKibitzerSession(coaching="\n- Never use open()")
        source = KibitzerPolicySource(session)
        current = PolicyResult(namespace_desc="tools: read_file")
        context: PolicyContext = {"kit": empty_kit}
        result = source.resolve(current, context)
        assert result.namespace_desc == "tools: read_file\n- Never use open()"

    def test_no_coaching_when_no_desc(self, empty_kit):
        session = FakeKibitzerSession(coaching="\n- coaching")
        source = KibitzerPolicySource(session)
        current = PolicyResult(namespace_desc=None)
        context: PolicyContext = {"kit": empty_kit}
        result = source.resolve(current, context)
        assert result.namespace_desc is None
