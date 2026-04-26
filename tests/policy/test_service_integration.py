"""Integration tests: LackpyService uses PolicyLayer correctly."""

from __future__ import annotations

import pytest
from pathlib import Path

from lackpy.service import LackpyService
from lackpy.kit.toolbox import ToolSpec, ArgSpec
from lackpy.policy.layer import PolicyLayer
from lackpy.policy.sources.kit import KitPolicySource


@pytest.fixture
def service(tmp_path):
    svc = LackpyService(workspace=tmp_path)
    svc.toolbox.register_tool(ToolSpec(
        name="read_file", provider="builtin",
        description="Read file contents",
        args=[ArgSpec(name="path", type="str", description="File path")],
        returns="str", grade_w=1, effects_ceiling=1,
    ))
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello world")
    return svc


class TestServiceHasPolicyLayer:
    def test_service_creates_policy_layer(self, service):
        assert hasattr(service, "_policy")
        assert isinstance(service._policy, PolicyLayer)

    def test_kit_source_always_registered(self, service):
        source_names = [s.name for s in service._policy._sources]
        assert "kit" in source_names

    def test_policy_resolves_with_kit(self, service):
        from lackpy.kit.registry import resolve_kit
        kit = resolve_kit(["read_file"], service.toolbox)
        result = service._policy.resolve({"kit": kit})
        assert "read_file" in result.allowed_tools
        assert result.grade.w == 1


class TestServiceDelegateUsesPolicyLayer:
    def test_validate_still_works(self, service):
        result = service.validate("x = read_file('test.txt')", kit=["read_file"])
        assert result.valid
