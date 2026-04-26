"""Tests for policy core types."""

from __future__ import annotations

import dataclasses
import pytest
from types import MappingProxyType


class TestToolConstraints:
    def test_frozen(self):
        from lackpy.policy.types import ToolConstraints
        tc = ToolConstraints(max_level=2)
        with pytest.raises(dataclasses.FrozenInstanceError):
            tc.max_level = 3

    def test_defaults(self):
        from lackpy.policy.types import ToolConstraints
        tc = ToolConstraints()
        assert tc.max_level is None
        assert tc.allow_patterns == ()
        assert tc.deny_patterns == ()

    def test_with_patterns(self):
        from lackpy.policy.types import ToolConstraints
        tc = ToolConstraints(
            max_level=1,
            allow_patterns=("src/**/*.py",),
            deny_patterns=("*.secret",),
        )
        assert tc.max_level == 1
        assert tc.allow_patterns == ("src/**/*.py",)


class TestPrincipal:
    def test_human_default(self):
        from lackpy.policy.types import Principal
        p = Principal(id="teague")
        assert p.kind == "human"
        assert p.parent is None

    def test_subagent(self):
        from lackpy.policy.types import Principal
        p = Principal(id="worker-1", kind="subagent", parent="orchestrator")
        assert p.kind == "subagent"
        assert p.parent == "orchestrator"

    def test_frozen(self):
        from lackpy.policy.types import Principal
        p = Principal(id="teague")
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.id = "other"


class TestModelSpec:
    def test_defaults(self):
        from lackpy.policy.types import ModelSpec
        m = ModelSpec(name="qwen2.5-coder:1.5b")
        assert m.temperature == 0.0
        assert m.context_window is None
        assert m.tier is None

    def test_full(self):
        from lackpy.policy.types import ModelSpec
        m = ModelSpec(name="claude-haiku", temperature=0.2, context_window=200000, tier="large")
        assert m.tier == "large"


class TestPolicyResult:
    def test_empty_defaults(self):
        from lackpy.policy.types import PolicyResult
        r = PolicyResult()
        assert r.allowed_tools == frozenset()
        assert r.denied_tools == frozenset()
        assert r.tool_constraints == MappingProxyType({})
        assert r.grade is None
        assert r.namespace_desc is None
        assert r.prompt_hints == ()
        assert r.docs == ()
        assert r.resolved is False

    def test_frozen(self):
        from lackpy.policy.types import PolicyResult
        r = PolicyResult()
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.resolved = True

    def test_replace_returns_new_instance(self):
        from lackpy.policy.types import PolicyResult
        r1 = PolicyResult(allowed_tools=frozenset({"read_file"}))
        r2 = r1.replace(allowed_tools=frozenset({"read_file", "edit_file"}))
        assert r1.allowed_tools == frozenset({"read_file"})
        assert r2.allowed_tools == frozenset({"read_file", "edit_file"})
        assert r1 is not r2

    def test_replace_preserves_unmodified_fields(self):
        from lackpy.policy.types import PolicyResult
        r1 = PolicyResult(
            allowed_tools=frozenset({"read_file"}),
            namespace_desc="tools available",
            prompt_hints=("hint1",),
        )
        r2 = r1.replace(resolved=True)
        assert r2.allowed_tools == frozenset({"read_file"})
        assert r2.namespace_desc == "tools available"
        assert r2.prompt_hints == ("hint1",)
        assert r2.resolved is True
