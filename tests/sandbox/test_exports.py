"""Tests for sandbox package public exports."""

from __future__ import annotations


class TestPublicExports:
    def test_constraint_types_importable(self):
        from lackpy.sandbox import (
            SandboxConstraint,
            MemoryLimit, TimeLimit, PidLimit, CpuLimit,
            NetworkAccess, NetworkRestriction, FilesystemMount,
            ReadonlyRoot, SeccompLogging, SeccompPolicyConstraint,
            UserMapping, BridgedToolPolicy,
            merge_constraints,
        )
        assert SandboxConstraint is not None

    def test_backend_types_importable(self):
        from lackpy.sandbox import (
            SandboxBackend, CompilationResult, ConstraintWarning, SandboxResult,
        )
        assert SandboxBackend is not None

    def test_nsjail_backend_importable(self):
        from lackpy.sandbox import NsjailBackend
        assert NsjailBackend is not None

    def test_config_importable(self):
        from lackpy.sandbox import SandboxBaseConfig
        assert SandboxBaseConfig is not None

    def test_provisioning_importable(self):
        from lackpy.sandbox import ToolProvisionKind, classify_tool, partition_kit
        assert ToolProvisionKind is not None

    def test_bridge_importable(self):
        from lackpy.sandbox import ToolBridgeManager, bridge_client
        assert ToolBridgeManager is not None

    def test_interpreter_importable(self):
        from lackpy.interpreters.sandbox import SandboxedInterpreter
        assert SandboxedInterpreter is not None
