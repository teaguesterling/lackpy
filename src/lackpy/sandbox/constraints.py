"""Typed sandbox constraint hierarchy and merge logic."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SandboxConstraint:
    """Base type for all sandbox constraints."""


@dataclass(frozen=True)
class MemoryLimit(SandboxConstraint):
    amount: int
    unit: str  # "MB", "GB"


@dataclass(frozen=True)
class TimeLimit(SandboxConstraint):
    seconds: int


@dataclass(frozen=True)
class PidLimit(SandboxConstraint):
    max_pids: int


@dataclass(frozen=True)
class CpuLimit(SandboxConstraint):
    ms_per_sec: int


@dataclass(frozen=True)
class NetworkAccess(SandboxConstraint):
    allow: bool


@dataclass(frozen=True)
class NetworkRestriction(SandboxConstraint):
    allowed_destinations: tuple[str, ...] = ()
    allowed_protocols: tuple[str, ...] = ()


@dataclass(frozen=True)
class FilesystemMount(SandboxConstraint):
    path: str
    writable: bool


@dataclass(frozen=True)
class ReadonlyRoot(SandboxConstraint):
    enabled: bool = True


@dataclass(frozen=True)
class SeccompLogging(SandboxConstraint):
    enabled: bool


@dataclass(frozen=True)
class SeccompPolicyConstraint(SandboxConstraint):
    policy_string: str


@dataclass(frozen=True)
class UserMapping(SandboxConstraint):
    inside_id: int
    outside_id: int
    count: int = 1


@dataclass(frozen=True)
class BridgedToolPolicy(SandboxConstraint):
    allowed: bool = False
    allowed_tools: tuple[str, ...] = ()
    allowed_providers: tuple[str, ...] = ()


def _to_mb(amount: int, unit: str) -> int:
    if unit == "GB":
        return amount * 1024
    return amount


def merge_constraints(
    constraints: tuple[SandboxConstraint, ...],
) -> tuple[SandboxConstraint, ...]:
    """Merge multiple constraints using most-restrictive-wins semantics.

    Resource limits: minimum wins. Booleans: restrictive wins.
    Filesystem paths: union, same path conflicting writability -> read-only wins.
    BridgedToolPolicy: intersection of allowed, False wins.
    """
    if not constraints:
        return ()

    by_type: dict[type, list[SandboxConstraint]] = {}
    for c in constraints:
        by_type.setdefault(type(c), []).append(c)

    result: list[SandboxConstraint] = []

    for cls, group in by_type.items():
        if cls is MemoryLimit:
            mb_values = [(_to_mb(c.amount, c.unit), c) for c in group]
            min_mb = min(v for v, _ in mb_values)
            result.append(MemoryLimit(amount=min_mb, unit="MB"))

        elif cls is TimeLimit:
            result.append(TimeLimit(seconds=min(c.seconds for c in group)))

        elif cls is PidLimit:
            result.append(PidLimit(max_pids=min(c.max_pids for c in group)))

        elif cls is CpuLimit:
            result.append(CpuLimit(ms_per_sec=min(c.ms_per_sec for c in group)))

        elif cls is NetworkAccess:
            result.append(NetworkAccess(allow=all(c.allow for c in group)))

        elif cls is ReadonlyRoot:
            result.append(ReadonlyRoot(enabled=any(c.enabled for c in group)))

        elif cls is SeccompLogging:
            result.append(SeccompLogging(enabled=any(c.enabled for c in group)))

        elif cls is FilesystemMount:
            by_path: dict[str, bool] = {}
            for c in group:
                if c.path in by_path:
                    by_path[c.path] = by_path[c.path] and c.writable
                else:
                    by_path[c.path] = c.writable
            for path, writable in sorted(by_path.items()):
                result.append(FilesystemMount(path=path, writable=writable))

        elif cls is BridgedToolPolicy:
            if any(not c.allowed for c in group):
                result.append(BridgedToolPolicy(allowed=False))
            else:
                all_tools = [set(c.allowed_tools) for c in group]
                all_providers = [set(c.allowed_providers) for c in group]
                tools = set.intersection(*all_tools) if all_tools else set()
                providers = set.intersection(*all_providers) if all_providers else set()
                result.append(BridgedToolPolicy(
                    allowed=True,
                    allowed_tools=tuple(sorted(tools)),
                    allowed_providers=tuple(sorted(providers)),
                ))

        elif cls is NetworkRestriction:
            all_dests = [set(c.allowed_destinations) for c in group]
            all_protos = [set(c.allowed_protocols) for c in group]
            dests = set.intersection(*all_dests) if all_dests else set()
            protos = set.intersection(*all_protos) if all_protos else set()
            result.append(NetworkRestriction(
                allowed_destinations=tuple(sorted(dests)),
                allowed_protocols=tuple(sorted(protos)),
            ))

        elif cls is UserMapping:
            if len(group) > 1:
                import warnings
                warnings.warn(
                    f"Multiple UserMapping constraints ({len(group)}); "
                    "using the first one. Conflicting UID maps may produce "
                    "invalid sandbox configs.",
                    stacklevel=2,
                )
            result.append(group[0])

        else:
            result.extend(group)

    return tuple(result)
