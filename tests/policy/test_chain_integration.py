"""Integration tests: full PolicyLayer chain with all three sources."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from lackpy.policy.layer import PolicyLayer
from lackpy.policy.types import PolicyResult, PolicyContext
from lackpy.policy.sources.kit import KitPolicySource
from lackpy.policy.sources.kibitzer import KibitzerPolicySource
from lackpy.policy.sources.umwelt import UmweltPolicySource
from lackpy.kit.registry import ResolvedKit, resolve_kit
from lackpy.kit.toolbox import Toolbox, ToolSpec, ArgSpec
from lackpy.kit.providers.builtin import BuiltinProvider
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

    def get_correction_hints(self, errors=None, model=None, attempt=None):
        return self._hints

    def has_coaching(self):
        return self._coaching is not None

    def apply_coaching(self, desc):
        return desc + self._coaching


class FakePolicyEngine:
    def __init__(self, tool_entries):
        self._entries = tool_entries

    def resolve_all(self, type="tool"):
        return self._entries


@pytest.fixture
def toolbox():
    tb = Toolbox()
    tb.register_provider(BuiltinProvider())
    for name, desc, grade in [
        ("read_file", "Read file contents", 1),
        ("find_files", "Find files matching pattern", 1),
        ("edit_file", "Edit a file", 3),
    ]:
        tb.register_tool(ToolSpec(
            name=name, provider="builtin", description=desc,
            args=[ArgSpec(name="path", type="str")],
            returns="str", grade_w=grade, effects_ceiling=grade,
        ))
    return tb


def _make_step_trace():
    return StepTrace(
        step_name="test", provider_name=None, model=None,
        system_prompt=None, user_prompt=None, raw_output=None,
        duration_ms=0.0,
    )


class TestStandaloneConfig:
    """Kit source only — no Kibitzer, no umwelt."""

    def test_kit_only(self, toolbox):
        kit = resolve_kit(["read_file", "find_files"], toolbox)
        layer = PolicyLayer()
        layer.add_source(KitPolicySource(toolbox))

        result = layer.resolve({"kit": kit})
        assert result.allowed_tools == frozenset({"read_file", "find_files"})
        assert result.grade.w == 1
        assert result.namespace_desc is not None
        assert result.prompt_hints == ()
        assert result.resolved is False


class TestKitPlusKibitzer:
    """Kit + Kibitzer coaching."""

    def test_kibitzer_enriches_without_changing_tools(self, toolbox):
        kit = resolve_kit(["read_file", "find_files"], toolbox)
        layer = PolicyLayer()
        layer.add_source(KitPolicySource(toolbox))
        layer.add_source(KibitzerPolicySource(
            FakeKibitzerSession(coaching="\n- Always use find_files before read_file"),
        ))

        result = layer.resolve({"kit": kit})
        assert result.allowed_tools == frozenset({"read_file", "find_files"})
        assert "Always use find_files before read_file" in result.namespace_desc

    def test_kibitzer_adds_hints_on_failure(self, toolbox):
        kit = resolve_kit(["read_file"], toolbox)
        history = StepContext(intent="read a file", kit=kit)
        history.programs.append(ProgramState(
            program="open('test.txt')", intent="read a file", kit=kit,
            valid=False, errors=["Forbidden name: 'open'"],
            trace=_make_step_trace(),
        ))

        layer = PolicyLayer()
        layer.add_source(KitPolicySource(toolbox))
        layer.add_source(KibitzerPolicySource(
            FakeKibitzerSession(hints=["Use read_file() instead of open()"]),
        ))

        result = layer.resolve({"kit": kit, "history": history})
        assert "Use read_file() instead of open()" in result.prompt_hints
        assert result.allowed_tools == frozenset({"read_file"})


class TestFullStack:
    """Kit + Kibitzer + umwelt."""

    def test_umwelt_restricts_after_kibitzer_enriches(self, toolbox):
        kit = resolve_kit(["read_file", "find_files", "edit_file"], toolbox)

        layer = PolicyLayer()
        layer.add_source(KitPolicySource(toolbox))
        layer.add_source(KibitzerPolicySource(
            FakeKibitzerSession(coaching="\n- Be careful with edit_file"),
        ))
        layer.add_source(UmweltPolicySource(FakePolicyEngine([
            {"id": "read_file", "visible": "true"},
            {"id": "find_files", "visible": "true"},
            {"id": "edit_file", "visible": "false"},
        ])))

        result = layer.resolve({"kit": kit})
        assert result.allowed_tools == frozenset({"read_file", "find_files"})
        assert "edit_file" in result.denied_tools
        assert "Be careful with edit_file" in result.namespace_desc
        assert result.grade.w == 3

    def test_umwelt_cannot_add_tools_beyond_kit(self, toolbox):
        kit = resolve_kit(["read_file"], toolbox)

        layer = PolicyLayer()
        layer.add_source(KitPolicySource(toolbox))
        layer.add_source(UmweltPolicySource(FakePolicyEngine([
            {"id": "read_file", "visible": "true"},
            {"id": "edit_file", "visible": "true"},
            {"id": "bash", "visible": "true"},
        ])))

        result = layer.resolve({"kit": kit})
        assert result.allowed_tools == frozenset({"read_file"})


class TestKitPlusUmwelt:
    """Kit + umwelt, no Kibitzer."""

    def test_umwelt_restricts_kit(self, toolbox):
        kit = resolve_kit(["read_file", "edit_file"], toolbox)

        layer = PolicyLayer()
        layer.add_source(KitPolicySource(toolbox))
        layer.add_source(UmweltPolicySource(FakePolicyEngine([
            {"id": "read_file", "visible": "true", "max_level": "1"},
            {"id": "edit_file", "visible": "false"},
        ])))

        result = layer.resolve({"kit": kit})
        assert result.allowed_tools == frozenset({"read_file"})
        assert "edit_file" in result.denied_tools
        assert result.tool_constraints["read_file"].max_level == 1
