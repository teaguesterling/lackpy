"""Tests for sandbox backend protocol and result types."""

from __future__ import annotations

import pytest


class TestConstraintWarning:
    def test_fields(self):
        from lackpy.sandbox.backend import ConstraintWarning
        from lackpy.sandbox.constraints import NetworkRestriction
        c = NetworkRestriction(allowed_destinations=("10.0.0.0/8",), allowed_protocols=("tcp",))
        w = ConstraintWarning(constraint=c, reason="nsjail only supports on/off network")
        assert w.constraint is c
        assert "on/off" in w.reason


class TestCompilationResult:
    def test_fields(self):
        from lackpy.sandbox.backend import CompilationResult, ConstraintWarning
        from lackpy.sandbox.constraints import NetworkRestriction
        c = NetworkRestriction(allowed_destinations=(), allowed_protocols=())
        w = ConstraintWarning(constraint=c, reason="unsupported")
        r = CompilationResult(config={"key": "value"}, warnings=[w])
        assert r.config == {"key": "value"}
        assert len(r.warnings) == 1

    def test_empty_warnings(self):
        from lackpy.sandbox.backend import CompilationResult
        r = CompilationResult(config={}, warnings=[])
        assert r.warnings == []


class TestSandboxResult:
    def test_successful_result(self):
        from lackpy.sandbox.backend import SandboxResult
        r = SandboxResult(
            returncode=0, stdout=b"hello", stderr=b"",
            timed_out=False, oom_killed=False,
        )
        assert r.returncode == 0
        assert r.stdout == b"hello"
        assert r.resource_stats is None

    def test_timeout_result(self):
        from lackpy.sandbox.backend import SandboxResult
        r = SandboxResult(
            returncode=137, stdout=b"", stderr=b"killed",
            timed_out=True, oom_killed=False,
        )
        assert r.timed_out is True

    def test_oom_result(self):
        from lackpy.sandbox.backend import SandboxResult
        r = SandboxResult(
            returncode=137, stdout=b"", stderr=b"oom",
            timed_out=False, oom_killed=True,
            resource_stats={"peak_rss_mb": 512},
        )
        assert r.oom_killed is True
        assert r.resource_stats["peak_rss_mb"] == 512


class TestSandboxBackendProtocol:
    def test_protocol_has_expected_methods(self):
        from lackpy.sandbox.backend import SandboxBackend
        assert hasattr(SandboxBackend, "name")
        assert hasattr(SandboxBackend, "compile")
        assert hasattr(SandboxBackend, "run")
        assert hasattr(SandboxBackend, "accept_policy_config")

    def test_custom_backend_satisfies_protocol(self):
        from lackpy.sandbox.backend import (
            SandboxBackend, CompilationResult, SandboxResult,
        )
        from lackpy.sandbox.constraints import SandboxConstraint
        from pathlib import Path
        from typing import Any, Sequence

        class FakeBackend:
            name = "fake"

            def accept_policy_config(self, config: Any) -> CompilationResult | None:
                return None

            def compile(
                self, constraints: Sequence[SandboxConstraint], workspace: Path,
            ) -> CompilationResult:
                return CompilationResult(config={}, warnings=[])

            async def run(self, config: Any, command: list[str]) -> SandboxResult:
                return SandboxResult(
                    returncode=0, stdout=b"", stderr=b"",
                    timed_out=False, oom_killed=False,
                )

        backend: SandboxBackend = FakeBackend()
        assert backend.name == "fake"
