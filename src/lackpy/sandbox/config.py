"""Sandbox configuration types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .constraints import (
    MemoryLimit, TimeLimit, PidLimit, NetworkAccess, ReadonlyRoot,
    SandboxConstraint,
)


@dataclass
class SandboxBaseConfig:
    enabled: bool = False
    backend: str = "nsjail"
    strategy: str = "subprocess"
    constraint_warnings: str = "warn"
    timeout: int = 120
    memory_mb: int = 512
    pids_max: int = 16
    network: bool = False
    bridge_enabled: bool = False
    bridge_allowed_providers: tuple[str, ...] = ()

    def to_base_constraints(self) -> tuple[SandboxConstraint, ...]:
        return (
            MemoryLimit(amount=self.memory_mb, unit="MB"),
            TimeLimit(seconds=self.timeout),
            PidLimit(max_pids=self.pids_max),
            NetworkAccess(allow=self.network),
            ReadonlyRoot(enabled=True),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SandboxBaseConfig:
        providers = data.get("bridge_allowed_providers", ())
        if isinstance(providers, list):
            providers = tuple(providers)
        return cls(
            enabled=data.get("enabled", False),
            backend=data.get("backend", "nsjail"),
            strategy=data.get("strategy", "subprocess"),
            constraint_warnings=data.get("constraint_warnings", "warn"),
            timeout=data.get("timeout", 120),
            memory_mb=data.get("memory_mb", 512),
            pids_max=data.get("pids_max", 16),
            network=data.get("network", False),
            bridge_enabled=data.get("bridge_enabled", False),
            bridge_allowed_providers=providers,
        )
