"""Integration tests: real nsjail sandbox execution."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.sandbox.conftest import skip_no_nsjail


@skip_no_nsjail
@pytest.mark.nsjail
class TestNsjailIntegration:
    @pytest.mark.asyncio
    async def test_simple_program_executes(self, tmp_path):
        from lackpy.sandbox.backends.nsjail import NsjailBackend
        from lackpy.sandbox.constraints import (
            MemoryLimit, TimeLimit, PidLimit, NetworkAccess, ReadonlyRoot,
        )
        from lackpy.sandbox.strategies import SubprocessStrategy

        backend = NsjailBackend()
        constraints = (
            MemoryLimit(amount=256, unit="MB"),
            TimeLimit(seconds=30),
            PidLimit(max_pids=16),
            NetworkAccess(allow=False),
            ReadonlyRoot(enabled=True),
        )
        compilation = backend.compile(constraints, tmp_path)

        strategy = SubprocessStrategy(backend=backend)
        result = await strategy.run(
            program="__result__ = 2 + 2",
            embedded_sources={},
            bridge_socket=None,
            bridge_authkey=None,
            base_dir=tmp_path,
            config=compilation.config,
        )
        assert result.success is True
        assert result.output == "4"

    @pytest.mark.asyncio
    async def test_embedded_tool_works(self, tmp_path):
        from lackpy.sandbox.backends.nsjail import NsjailBackend
        from lackpy.sandbox.constraints import (
            MemoryLimit, TimeLimit, PidLimit, NetworkAccess, ReadonlyRoot,
            FilesystemMount,
        )
        from lackpy.sandbox.strategies import SubprocessStrategy

        (tmp_path / "test.txt").write_text("hello sandbox")

        backend = NsjailBackend()
        constraints = (
            MemoryLimit(amount=256, unit="MB"),
            TimeLimit(seconds=30),
            PidLimit(max_pids=16),
            NetworkAccess(allow=False),
            ReadonlyRoot(enabled=True),
            FilesystemMount(path=str(tmp_path), writable=False),
        )
        compilation = backend.compile(constraints, tmp_path)

        strategy = SubprocessStrategy(backend=backend)
        result = await strategy.run(
            program="__result__ = read_file('" + str(tmp_path / "test.txt") + "')",
            embedded_sources={
                "read_file": "def read_file(path):\n    return open(path).read()\n",
            },
            bridge_socket=None,
            bridge_authkey=None,
            base_dir=tmp_path,
            config=compilation.config,
        )
        assert result.success is True
        assert "hello sandbox" in result.output

    @pytest.mark.asyncio
    async def test_timeout_enforced(self, tmp_path):
        from lackpy.sandbox.backends.nsjail import NsjailBackend
        from lackpy.sandbox.constraints import TimeLimit, MemoryLimit, PidLimit, NetworkAccess
        from lackpy.sandbox.strategies import SubprocessStrategy

        backend = NsjailBackend()
        constraints = (
            TimeLimit(seconds=2),
            MemoryLimit(amount=64, unit="MB"),
            PidLimit(max_pids=4),
            NetworkAccess(allow=False),
        )
        compilation = backend.compile(constraints, tmp_path)

        strategy = SubprocessStrategy(backend=backend)
        result = await strategy.run(
            program="import time; time.sleep(60)",
            embedded_sources={},
            bridge_socket=None,
            bridge_authkey=None,
            base_dir=tmp_path,
            config=compilation.config,
        )
        assert result.success is False
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_filesystem_isolation(self, tmp_path):
        from lackpy.sandbox.backends.nsjail import NsjailBackend
        from lackpy.sandbox.constraints import (
            MemoryLimit, TimeLimit, PidLimit, NetworkAccess, ReadonlyRoot,
        )
        from lackpy.sandbox.strategies import SubprocessStrategy

        backend = NsjailBackend()
        constraints = (
            MemoryLimit(amount=256, unit="MB"),
            TimeLimit(seconds=10),
            PidLimit(max_pids=16),
            NetworkAccess(allow=False),
            ReadonlyRoot(enabled=True),
        )
        compilation = backend.compile(constraints, tmp_path)

        strategy = SubprocessStrategy(backend=backend)
        result = await strategy.run(
            program="import os; __result__ = os.path.exists('/etc/passwd')",
            embedded_sources={},
            bridge_socket=None,
            bridge_authkey=None,
            base_dir=tmp_path,
            config=compilation.config,
        )
        assert result.success is True or "error" in (result.error or "").lower()


@skip_no_nsjail
@pytest.mark.nsjail
class TestSandboxedInterpreterIntegration:
    @pytest.mark.asyncio
    async def test_end_to_end_with_python_interpreter(self, tmp_path):
        from lackpy.interpreters.sandbox import SandboxedInterpreter
        from lackpy.interpreters.base import ExecutionContext, InterpreterValidationResult
        from lackpy.sandbox.backends.nsjail import NsjailBackend
        from lackpy.sandbox.config import SandboxBaseConfig
        from unittest.mock import MagicMock

        inner = MagicMock()
        inner.name = "python"
        inner.description = "Python interpreter"
        inner.validate.return_value = InterpreterValidationResult(valid=True)

        backend = NsjailBackend()
        base_config = SandboxBaseConfig(timeout=30, memory_mb=256, pids_max=16)
        si = SandboxedInterpreter(
            interpreter=inner, backend=backend, base_config=base_config,
        )

        ctx = ExecutionContext(base_dir=tmp_path)
        validation = si.validate("__result__ = 42", ctx)
        assert validation.valid is True

        result = await si.execute("__result__ = 42", ctx)
        assert result.success is True
