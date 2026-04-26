"""PolicyLayer: ordered resolution chain of PolicySources."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .types import PolicyContext, PolicyResult


@runtime_checkable
class PolicySource(Protocol):
    """A source of policy decisions in the resolution chain."""

    name: str
    priority: int

    def resolve(self, current: PolicyResult, context: PolicyContext) -> PolicyResult: ...


class PolicyLayer:
    """Ordered chain of PolicySources that produces a PolicyResult.

    Sources are sorted by priority ascending (lowest first).
    Each source receives the accumulated PolicyResult and the request
    context. If a source returns a result with resolved=True, the
    chain stops.
    """

    def __init__(self) -> None:
        self._sources: list[PolicySource] = []

    def add_source(self, source: PolicySource) -> None:
        self._sources.append(source)
        self._sources.sort(key=lambda s: s.priority)

    def resolve(self, context: PolicyContext) -> PolicyResult:
        result = PolicyResult()
        for source in self._sources:
            result = source.resolve(result, context)
            if result.resolved:
                break
        return result
