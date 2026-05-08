"""Tests for execution strategies (subprocess and jail_call)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lackpy.interpreters.base import InterpreterExecutionResult


class TestSubprocessStrategy:
    @pytest.mark.asyncio
    async def test_prepares_request_and_reads_result(self):
        from lackpy.sandbox.strategies import SubprocessStrategy
        from lackpy.sandbox.backend import SandboxResult

        mock_backend = MagicMock()
        mock_result = SandboxResult(
            returncode=0, stdout=b"", stderr=b"",
            timed_out=False, oom_killed=False,
        )
        mock_backend.run = AsyncMock(return_value=mock_result)

        strategy = SubprocessStrategy(backend=mock_backend)

        with tempfile.TemporaryDirectory() as td:
            io_dir = Path(td)
            # Pre-write a result file as if the worker produced it
            result_data = {
                "success": True,
                "output": "hello",
                "output_format": "text",
                "error": None,
                "duration_ms": 10.0,
                "metadata": {},
            }
            (io_dir / "result.json").write_text(json.dumps(result_data))

            exec_result = await strategy.run(
                program="x = 1",
                embedded_sources={},
                bridge_socket=None,
                bridge_authkey=None,
                base_dir=Path("/workspace"),
                config=MagicMock(),
                _io_dir=io_dir,
            )
            assert exec_result.success is True
            assert exec_result.output == "hello"

    @pytest.mark.asyncio
    async def test_timeout_returns_failed_result(self):
        from lackpy.sandbox.strategies import SubprocessStrategy
        from lackpy.sandbox.backend import SandboxResult

        mock_backend = MagicMock()
        mock_result = SandboxResult(
            returncode=137, stdout=b"", stderr=b"timeout",
            timed_out=True, oom_killed=False,
        )
        mock_backend.run = AsyncMock(return_value=mock_result)

        strategy = SubprocessStrategy(backend=mock_backend)
        exec_result = await strategy.run(
            program="while True: pass",
            embedded_sources={},
            bridge_socket=None,
            base_dir=Path("/workspace"),
            config=MagicMock(),
        )
        assert exec_result.success is False
        assert "timed out" in exec_result.error.lower()

    @pytest.mark.asyncio
    async def test_oom_returns_failed_result(self):
        from lackpy.sandbox.strategies import SubprocessStrategy
        from lackpy.sandbox.backend import SandboxResult

        mock_backend = MagicMock()
        mock_result = SandboxResult(
            returncode=137, stdout=b"", stderr=b"oom",
            timed_out=False, oom_killed=True,
        )
        mock_backend.run = AsyncMock(return_value=mock_result)

        strategy = SubprocessStrategy(backend=mock_backend)
        exec_result = await strategy.run(
            program="x = [0] * 10**9",
            embedded_sources={},
            bridge_socket=None,
            base_dir=Path("/workspace"),
            config=MagicMock(),
        )
        assert exec_result.success is False
        assert "memory" in exec_result.error.lower() or "oom" in exec_result.error.lower()
