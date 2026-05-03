"""Sandbox execution: OS-level containment for interpreter execution."""

from .constraints import (
    SandboxConstraint,
    MemoryLimit,
    TimeLimit,
    PidLimit,
    CpuLimit,
    NetworkAccess,
    NetworkRestriction,
    FilesystemMount,
    ReadonlyRoot,
    SeccompLogging,
    SeccompPolicyConstraint,
    UserMapping,
    BridgedToolPolicy,
    merge_constraints,
)
from .backend import (
    SandboxBackend,
    CompilationResult,
    ConstraintWarning,
    SandboxResult,
)
from .backends.nsjail import NsjailBackend
from .config import SandboxBaseConfig
from .provisioning import ToolProvisionKind, classify_tool, partition_kit
from .bridge import ToolBridgeManager, bridge_client

__all__ = [
    "SandboxConstraint",
    "MemoryLimit",
    "TimeLimit",
    "PidLimit",
    "CpuLimit",
    "NetworkAccess",
    "NetworkRestriction",
    "FilesystemMount",
    "ReadonlyRoot",
    "SeccompLogging",
    "SeccompPolicyConstraint",
    "UserMapping",
    "BridgedToolPolicy",
    "merge_constraints",
    "SandboxBackend",
    "CompilationResult",
    "ConstraintWarning",
    "SandboxResult",
    "NsjailBackend",
    "SandboxBaseConfig",
    "ToolProvisionKind",
    "classify_tool",
    "partition_kit",
    "ToolBridgeManager",
    "bridge_client",
]
