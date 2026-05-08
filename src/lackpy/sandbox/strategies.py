"""Execution strategies: subprocess worker harness."""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from ..interpreters.base import InterpreterExecutionResult
from .backend import SandboxBackend, SandboxResult
from ._worker import write_request, read_result


class SubprocessStrategy:
    """Launch a worker harness inside the jail via subprocess."""

    def __init__(self, backend: SandboxBackend) -> None:
        self._backend = backend

    async def run(
        self,
        program: str,
        embedded_sources: dict[str, str],
        bridge_socket: Path | None,
        config: Any,
        base_dir: Path = Path("."),
        bridge_authkey: bytes | None = None,
        _io_dir: Path | None = None,
    ) -> InterpreterExecutionResult:
        own_io_dir = _io_dir is None
        io_dir = _io_dir or Path(tempfile.mkdtemp(prefix="lackpy_sandbox_"))

        try:
            request = {
                "program": program,
                "embedded_sources": embedded_sources,
                "bridge_socket": str(bridge_socket) if bridge_socket else None,
                "bridge_authkey": bridge_authkey.hex() if bridge_authkey else None,
                "base_dir": str(base_dir),
            }
            write_request(io_dir, request)

            command = [sys.executable, "-m", "lackpy.sandbox._worker", str(io_dir)]
            sandbox_result: SandboxResult = await self._backend.run(config, command)

            if sandbox_result.timed_out:
                return InterpreterExecutionResult(
                    success=False,
                    error="Sandbox execution timed out",
                    metadata={"sandbox_timed_out": True},
                )

            if sandbox_result.oom_killed:
                return InterpreterExecutionResult(
                    success=False,
                    error="Sandbox execution killed: out of memory",
                    metadata={"sandbox_oom_killed": True},
                )

            if sandbox_result.returncode != 0:
                stderr = sandbox_result.stderr.decode(errors="replace")
                return InterpreterExecutionResult(
                    success=False,
                    error=f"Sandbox process exited with code {sandbox_result.returncode}: {stderr}",
                    metadata={"sandbox_returncode": sandbox_result.returncode},
                )

            try:
                result_data = read_result(io_dir)
            except FileNotFoundError:
                return InterpreterExecutionResult(
                    success=False,
                    error="Sandbox process did not produce a result file",
                )

            return InterpreterExecutionResult(
                success=result_data["success"],
                output=result_data.get("output"),
                output_format=result_data.get("output_format", "none"),
                error=result_data.get("error"),
                duration_ms=result_data.get("duration_ms", 0.0),
                metadata=result_data.get("metadata", {}),
            )
        finally:
            if own_io_dir:
                shutil.rmtree(io_dir, ignore_errors=True)
