"""Core policy types: immutable result, context, and value objects."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Required, TypedDict

from types import MappingProxyType

if TYPE_CHECKING:
    from ..kit.registry import ResolvedKit
    from ..infer.context import StepContext
    from ..run.trace import Trace
    from ..lang.grader import Grade


@dataclass(frozen=True)
class ToolConstraints:
    """Per-tool constraints from policy resolution."""

    max_level: int | None = None
    allow_patterns: tuple[str, ...] = ()
    deny_patterns: tuple[str, ...] = ()


EMPTY_CONSTRAINTS: MappingProxyType[str, ToolConstraints] = MappingProxyType({})


@dataclass(frozen=True)
class Principal:
    """Identity of the requesting entity (S5)."""

    id: str
    kind: str = "human"
    parent: str | None = None


@dataclass(frozen=True)
class ModelSpec:
    """Inferencer properties that affect policy decisions (S4)."""

    name: str
    temperature: float = 0.0
    context_window: int | None = None
    tier: str | None = None


@dataclass(frozen=True)
class PolicyResult:
    """Immutable result of policy resolution.

    Sources produce new instances via replace(). The resolved flag
    controls chain propagation: True stops the chain.
    """

    allowed_tools: frozenset[str] = frozenset()
    denied_tools: frozenset[str] = frozenset()
    tool_constraints: MappingProxyType[str, ToolConstraints] = field(
        default_factory=lambda: EMPTY_CONSTRAINTS
    )
    grade: Any = None  # Grade, but Any to avoid circular import at runtime
    namespace_desc: str | None = None
    prompt_hints: tuple[str, ...] = ()
    docs: tuple[str, ...] = ()
    resolved: bool = False

    def replace(self, **changes: Any) -> PolicyResult:
        return dataclasses.replace(self, **changes)


class PolicyContext(TypedDict, total=False):
    """Shared request context passed through the policy chain.

    VSM-informed: S1 (kit), S2 (session_id), S4 (model),
    S5 (principal), S3* (history, trace).
    """

    kit: Required[Any]  # ResolvedKit, Any to avoid circular import
    principal: Principal
    model: ModelSpec
    session_id: str
    history: Any  # StepContext | None
    trace: Any  # Trace | None
