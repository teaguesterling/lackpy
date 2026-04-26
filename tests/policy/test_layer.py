"""Tests for PolicyLayer resolution chain."""

from __future__ import annotations

import pytest

from lackpy.policy.types import PolicyResult, PolicyContext
from lackpy.policy.layer import PolicyLayer, PolicySource
from lackpy.kit.registry import ResolvedKit
from lackpy.lang.grader import Grade


class StubSource:
    """A configurable stub PolicySource for testing."""

    def __init__(self, name: str, priority: int, transform=None, mark_resolved=False):
        self.name = name
        self.priority = priority
        self._transform = transform
        self._mark_resolved = mark_resolved
        self.called_with: list[tuple[PolicyResult, PolicyContext]] = []

    def resolve(self, current: PolicyResult, context: PolicyContext) -> PolicyResult:
        self.called_with.append((current, context))
        if self._transform:
            result = self._transform(current, context)
        else:
            result = current
        if self._mark_resolved:
            result = result.replace(resolved=True)
        return result


@pytest.fixture
def minimal_kit():
    """A minimal ResolvedKit for context construction."""
    return ResolvedKit(
        tools={},
        callables={},
        grade=Grade(w=0, d=0),
        description="",
    )


class TestPolicyLayerOrdering:
    def test_sources_run_lowest_priority_first(self, minimal_kit):
        call_order = []

        def make_transform(label):
            def transform(current, context):
                call_order.append(label)
                return current.replace(
                    prompt_hints=current.prompt_hints + (label,),
                )
            return transform

        layer = PolicyLayer()
        layer.add_source(StubSource("high", priority=100, transform=make_transform("high")))
        layer.add_source(StubSource("low", priority=0, transform=make_transform("low")))
        layer.add_source(StubSource("mid", priority=50, transform=make_transform("mid")))

        result = layer.resolve({"kit": minimal_kit})
        assert call_order == ["low", "mid", "high"]
        assert result.prompt_hints == ("low", "mid", "high")

    def test_resolved_stops_chain(self, minimal_kit):
        call_order = []

        def track(label):
            def transform(current, context):
                call_order.append(label)
                return current.replace(prompt_hints=current.prompt_hints + (label,))
            return transform

        layer = PolicyLayer()
        layer.add_source(StubSource("first", priority=0, transform=track("first")))
        layer.add_source(StubSource("stopper", priority=50, transform=track("stopper"), mark_resolved=True))
        layer.add_source(StubSource("skipped", priority=100, transform=track("skipped")))

        result = layer.resolve({"kit": minimal_kit})
        assert call_order == ["first", "stopper"]
        assert "skipped" not in result.prompt_hints
        assert result.resolved is True


class TestPolicyLayerEmpty:
    def test_no_sources_returns_empty_result(self, minimal_kit):
        layer = PolicyLayer()
        result = layer.resolve({"kit": minimal_kit})
        assert result.allowed_tools == frozenset()
        assert result.resolved is False

    def test_single_source(self, minimal_kit):
        def set_tools(current, context):
            return current.replace(allowed_tools=frozenset({"read_file"}))

        layer = PolicyLayer()
        layer.add_source(StubSource("only", priority=0, transform=set_tools))

        result = layer.resolve({"kit": minimal_kit})
        assert result.allowed_tools == frozenset({"read_file"})


class TestPolicyLayerAddSource:
    def test_add_source_maintains_sort(self, minimal_kit):
        layer = PolicyLayer()
        layer.add_source(StubSource("c", priority=100))
        layer.add_source(StubSource("a", priority=0))
        layer.add_source(StubSource("b", priority=50))

        names = [s.name for s in layer._sources]
        assert names == ["a", "b", "c"]
