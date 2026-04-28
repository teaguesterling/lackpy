"""Sandbox backend protocol and shared result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

from .constraints import SandboxConstraint


@dataclass
class ConstraintWarning:
    constraint: SandboxConstraint
    reason: str


@dataclass
class CompilationResult:
    config: Any
    warnings: list[ConstraintWarning] = field(default_factory=list)


@dataclass
class SandboxResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    timed_out: bool
    oom_killed: bool
    resource_stats: dict[str, Any] | None = None


class SandboxBackend(Protocol):
    @property
    def name(self) -> str: ...

    def accept_policy_config(self, config: Any) -> CompilationResult | None: ...

    def compile(
        self,
        constraints: Sequence[SandboxConstraint],
        workspace: Path,
    ) -> CompilationResult: ...

    async def run(
        self,
        config: Any,
        command: list[str],
    ) -> SandboxResult: ...
