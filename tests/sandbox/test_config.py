"""Tests for sandbox configuration."""

from __future__ import annotations

import pytest


class TestSandboxBaseConfig:
    def test_defaults(self):
        from lackpy.sandbox.config import SandboxBaseConfig
        cfg = SandboxBaseConfig()
        assert cfg.enabled is False
        assert cfg.backend == "nsjail"
        assert cfg.strategy == "subprocess"
        assert cfg.constraint_warnings == "warn"
        assert cfg.timeout == 120
        assert cfg.memory_mb == 512
        assert cfg.pids_max == 16
        assert cfg.network is False
        assert cfg.bridge_enabled is False
        assert cfg.bridge_allowed_providers == ()

    def test_custom_values(self):
        from lackpy.sandbox.config import SandboxBaseConfig
        cfg = SandboxBaseConfig(
            enabled=True,
            backend="bwrap",
            strategy="jail_call",
            timeout=60,
            memory_mb=256,
            network=True,
        )
        assert cfg.enabled is True
        assert cfg.backend == "bwrap"
        assert cfg.strategy == "jail_call"
        assert cfg.timeout == 60
        assert cfg.memory_mb == 256
        assert cfg.network is True

    def test_to_base_constraints(self):
        from lackpy.sandbox.config import SandboxBaseConfig
        from lackpy.sandbox.constraints import (
            MemoryLimit, TimeLimit, PidLimit, NetworkAccess, ReadonlyRoot,
        )
        cfg = SandboxBaseConfig(timeout=60, memory_mb=256, pids_max=8, network=False)
        constraints = cfg.to_base_constraints()
        types = {type(c) for c in constraints}
        assert MemoryLimit in types
        assert TimeLimit in types
        assert PidLimit in types
        assert NetworkAccess in types
        assert ReadonlyRoot in types
        mem = next(c for c in constraints if isinstance(c, MemoryLimit))
        assert mem.amount == 256

    def test_from_toml_dict(self):
        from lackpy.sandbox.config import SandboxBaseConfig
        data = {
            "enabled": True,
            "backend": "nsjail",
            "strategy": "subprocess",
            "timeout": 90,
            "memory_mb": 1024,
            "pids_max": 32,
            "network": True,
            "bridge_enabled": True,
            "bridge_allowed_providers": ["mcp"],
        }
        cfg = SandboxBaseConfig.from_dict(data)
        assert cfg.enabled is True
        assert cfg.timeout == 90
        assert cfg.memory_mb == 1024
        assert cfg.pids_max == 32
        assert cfg.network is True
        assert cfg.bridge_enabled is True
        assert cfg.bridge_allowed_providers == ("mcp",)

    def test_from_toml_dict_empty(self):
        from lackpy.sandbox.config import SandboxBaseConfig
        cfg = SandboxBaseConfig.from_dict({})
        assert cfg.enabled is False
        assert cfg.timeout == 120


class TestLackpyConfigSandboxField:
    def test_config_has_sandbox_field(self):
        from lackpy.config import LackpyConfig
        from lackpy.sandbox.config import SandboxBaseConfig
        cfg = LackpyConfig()
        assert isinstance(cfg.sandbox, SandboxBaseConfig)
        assert cfg.sandbox.enabled is False

    def test_load_config_reads_sandbox_section(self, tmp_path):
        from lackpy.config import load_config
        config_dir = tmp_path / ".lackpy"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text(
            '[sandbox]\nenabled = true\ntimeout = 30\nmemory_mb = 128\n'
        )
        cfg = load_config(tmp_path)
        assert cfg.sandbox.enabled is True
        assert cfg.sandbox.timeout == 30
        assert cfg.sandbox.memory_mb == 128
