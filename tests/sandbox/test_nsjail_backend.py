"""Tests for NsjailBackend constraint compilation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lackpy.sandbox.constraints import (
    MemoryLimit, TimeLimit, PidLimit, CpuLimit,
    NetworkAccess, FilesystemMount, ReadonlyRoot,
    SeccompLogging, SeccompPolicyConstraint, UserMapping,
    NetworkRestriction,
)
from lackpy.sandbox.backend import ConstraintWarning


class TestNsjailBackendCompile:
    def test_name(self):
        from lackpy.sandbox.backends.nsjail import NsjailBackend
        backend = NsjailBackend()
        assert backend.name == "nsjail"

    def test_compile_memory_limit(self):
        from lackpy.sandbox.backends.nsjail import NsjailBackend
        backend = NsjailBackend()
        result = backend.compile(
            (MemoryLimit(amount=256, unit="MB"),),
            Path("/workspace"),
        )
        cfg = result.config
        assert cfg.cgroup_mem_max == 256 * 1024 * 1024

    def test_compile_time_limit(self):
        from lackpy.sandbox.backends.nsjail import NsjailBackend
        backend = NsjailBackend()
        result = backend.compile(
            (TimeLimit(seconds=60),),
            Path("/workspace"),
        )
        assert result.config.time_limit == 60

    def test_compile_pid_limit(self):
        from lackpy.sandbox.backends.nsjail import NsjailBackend
        backend = NsjailBackend()
        result = backend.compile(
            (PidLimit(max_pids=16),),
            Path("/workspace"),
        )
        assert result.config.cgroup_pids_max == 16

    def test_compile_network_disabled(self):
        from lackpy.sandbox.backends.nsjail import NsjailBackend
        backend = NsjailBackend()
        result = backend.compile(
            (NetworkAccess(allow=False),),
            Path("/workspace"),
        )
        assert result.config.clone_newnet is True

    def test_compile_network_enabled(self):
        from lackpy.sandbox.backends.nsjail import NsjailBackend
        backend = NsjailBackend()
        result = backend.compile(
            (NetworkAccess(allow=True),),
            Path("/workspace"),
        )
        assert result.config.clone_newnet is False

    def test_compile_filesystem_mount(self):
        from lackpy.sandbox.backends.nsjail import NsjailBackend
        from nsjail.config import MountPt
        backend = NsjailBackend()
        result = backend.compile(
            (FilesystemMount(path="/data", writable=False),),
            Path("/workspace"),
        )
        mounts = result.config.mount
        data_mounts = [m for m in mounts if m.dst == "/data"]
        assert len(data_mounts) == 1
        assert data_mounts[0].rw is False

    def test_compile_readonly_root(self):
        from lackpy.sandbox.backends.nsjail import NsjailBackend
        backend = NsjailBackend()
        result = backend.compile(
            (ReadonlyRoot(enabled=True),),
            Path("/workspace"),
        )
        root_mounts = [m for m in result.config.mount if m.dst == "/"]
        assert any(not m.rw for m in root_mounts)

    def test_compile_seccomp_logging(self):
        from lackpy.sandbox.backends.nsjail import NsjailBackend
        backend = NsjailBackend()
        result = backend.compile(
            (SeccompLogging(enabled=True),),
            Path("/workspace"),
        )
        assert len(result.config.seccomp_string) > 0

    def test_compile_seccomp_policy(self):
        from lackpy.sandbox.backends.nsjail import NsjailBackend
        backend = NsjailBackend()
        result = backend.compile(
            (SeccompPolicyConstraint(policy_string="ALLOW { read, write }"),),
            Path("/workspace"),
        )
        assert "ALLOW { read, write }" in result.config.seccomp_string

    def test_compile_warns_on_network_restriction(self):
        from lackpy.sandbox.backends.nsjail import NsjailBackend
        backend = NsjailBackend()
        result = backend.compile(
            (NetworkRestriction(allowed_destinations=("10.0.0.0/8",), allowed_protocols=("tcp",)),),
            Path("/workspace"),
        )
        assert len(result.warnings) == 1
        assert "on/off" in result.warnings[0].reason.lower() or "binary" in result.warnings[0].reason.lower()

    def test_compile_multiple_constraints(self):
        from lackpy.sandbox.backends.nsjail import NsjailBackend
        backend = NsjailBackend()
        result = backend.compile(
            (
                MemoryLimit(amount=256, unit="MB"),
                TimeLimit(seconds=60),
                PidLimit(max_pids=8),
                NetworkAccess(allow=False),
                ReadonlyRoot(enabled=True),
            ),
            Path("/workspace"),
        )
        cfg = result.config
        assert cfg.time_limit == 60
        assert cfg.clone_newnet is True
        assert cfg.cgroup_mem_max == 256 * 1024 * 1024
        assert cfg.cgroup_pids_max == 8


class TestNsjailBackendAcceptPolicyConfig:
    def test_accepts_nsjail_config(self):
        from lackpy.sandbox.backends.nsjail import NsjailBackend
        from nsjail.config import NsJailConfig
        backend = NsjailBackend()
        cfg = NsJailConfig(time_limit=30)
        result = backend.accept_policy_config(cfg)
        assert result is not None
        assert result.config is cfg
        assert result.warnings == []

    def test_rejects_non_nsjail_config(self):
        from lackpy.sandbox.backends.nsjail import NsjailBackend
        backend = NsjailBackend()
        result = backend.accept_policy_config({"some": "dict"})
        assert result is None

    def test_rejects_none(self):
        from lackpy.sandbox.backends.nsjail import NsjailBackend
        backend = NsjailBackend()
        result = backend.accept_policy_config(None)
        assert result is None
