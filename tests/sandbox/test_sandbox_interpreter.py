"""Tests for SandboxedInterpreter decorator."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lackpy.interpreters.base import (
    ExecutionContext,
    InterpreterExecutionResult,
    InterpreterValidationResult,
)
from lackpy.sandbox.constraints import MemoryLimit, TimeLimit, SandboxConstraint
from lackpy.sandbox.backend import CompilationResult


class TestSandboxedInterpreterValidation:
    def test_validate_delegates_directly(self):
        from lackpy.interpreters.sandbox import SandboxedInterpreter
        inner = MagicMock()
        inner.name = "python"
        inner.description = "Python interpreter"
        inner.validate.return_value = InterpreterValidationResult(valid=True)
        backend = MagicMock()
        si = SandboxedInterpreter(interpreter=inner, backend=backend)
        ctx = ExecutionContext()
        result = si.validate("x = 1", ctx)
        assert result.valid is True
        inner.validate.assert_called_once_with("x = 1", ctx)

    def test_name_delegates(self):
        from lackpy.interpreters.sandbox import SandboxedInterpreter
        inner = MagicMock()
        inner.name = "python"
        inner.description = "Python interpreter"
        backend = MagicMock()
        si = SandboxedInterpreter(interpreter=inner, backend=backend)
        assert si.name == "python"

    def test_description_delegates(self):
        from lackpy.interpreters.sandbox import SandboxedInterpreter
        inner = MagicMock()
        inner.name = "python"
        inner.description = "Python interpreter"
        backend = MagicMock()
        si = SandboxedInterpreter(interpreter=inner, backend=backend)
        assert si.description == "Python interpreter"


class TestSandboxedInterpreterConfigResolution:
    @pytest.mark.asyncio
    async def test_uses_precompiled_config_when_available(self):
        from lackpy.interpreters.sandbox import SandboxedInterpreter
        from lackpy.sandbox.backend import CompilationResult

        inner = MagicMock()
        inner.name = "python"
        inner.description = "Python interpreter"

        precompiled = MagicMock()
        backend = MagicMock()
        backend.name = "nsjail"
        backend.accept_policy_config.return_value = CompilationResult(
            config=precompiled, warnings=[],
        )
        backend.run = AsyncMock(return_value=MagicMock(
            returncode=0, stdout=b"", stderr=b"",
            timed_out=False, oom_killed=False,
        ))

        si = SandboxedInterpreter(interpreter=inner, backend=backend)
        ctx = ExecutionContext(base_dir=Path("/workspace"))

        with patch("lackpy.interpreters.sandbox.SubprocessStrategy") as mock_strat_cls:
            mock_strat = MagicMock()
            mock_strat.run = AsyncMock(return_value=InterpreterExecutionResult(success=True, output="ok"))
            mock_strat_cls.return_value = mock_strat

            result = await si.execute(
                "x = 1", ctx,
                sandbox_constraints=(),
                backend_configs=MappingProxyType({"nsjail": precompiled}),
            )

        backend.accept_policy_config.assert_called_once_with(precompiled)
        backend.compile.assert_not_called()

    @pytest.mark.asyncio
    async def test_falls_back_to_constraint_compilation(self):
        from lackpy.interpreters.sandbox import SandboxedInterpreter

        inner = MagicMock()
        inner.name = "python"
        inner.description = "Python interpreter"

        compiled_config = MagicMock()
        backend = MagicMock()
        backend.name = "nsjail"
        backend.accept_policy_config.return_value = None
        backend.compile.return_value = CompilationResult(
            config=compiled_config, warnings=[],
        )
        backend.run = AsyncMock()

        constraints = (MemoryLimit(amount=256, unit="MB"),)
        si = SandboxedInterpreter(interpreter=inner, backend=backend)
        ctx = ExecutionContext(base_dir=Path("/workspace"))

        with patch("lackpy.interpreters.sandbox.SubprocessStrategy") as mock_strat_cls:
            mock_strat = MagicMock()
            mock_strat.run = AsyncMock(return_value=InterpreterExecutionResult(success=True))
            mock_strat_cls.return_value = mock_strat

            await si.execute(
                "x = 1", ctx,
                sandbox_constraints=constraints,
                backend_configs=MappingProxyType({}),
            )

        backend.compile.assert_called_once()
        call_constraints = backend.compile.call_args[0][0]
        assert any(isinstance(c, MemoryLimit) for c in call_constraints)

    @pytest.mark.asyncio
    async def test_merges_base_config_constraints(self):
        from lackpy.interpreters.sandbox import SandboxedInterpreter
        from lackpy.sandbox.config import SandboxBaseConfig

        inner = MagicMock()
        inner.name = "python"
        inner.description = "Python interpreter"

        backend = MagicMock()
        backend.name = "nsjail"
        backend.accept_policy_config.return_value = None
        backend.compile.return_value = CompilationResult(config=MagicMock(), warnings=[])
        backend.run = AsyncMock()

        base_config = SandboxBaseConfig(timeout=120, memory_mb=512)
        policy_constraints = (TimeLimit(seconds=60),)
        si = SandboxedInterpreter(
            interpreter=inner, backend=backend, base_config=base_config,
        )
        ctx = ExecutionContext(base_dir=Path("/workspace"))

        with patch("lackpy.interpreters.sandbox.SubprocessStrategy") as mock_strat_cls:
            mock_strat = MagicMock()
            mock_strat.run = AsyncMock(return_value=InterpreterExecutionResult(success=True))
            mock_strat_cls.return_value = mock_strat

            await si.execute(
                "x = 1", ctx,
                sandbox_constraints=policy_constraints,
            )

        compiled_constraints = backend.compile.call_args[0][0]
        types = {type(c) for c in compiled_constraints}
        assert TimeLimit in types
        assert MemoryLimit in types


class TestSandboxedInterpreterSystemPromptHint:
    def test_delegates_system_prompt_hint(self):
        from lackpy.interpreters.sandbox import SandboxedInterpreter
        inner = MagicMock()
        inner.name = "python"
        inner.description = "Python interpreter"
        inner.system_prompt_hint.return_value = "Write Python"
        backend = MagicMock()
        si = SandboxedInterpreter(interpreter=inner, backend=backend)
        assert si.system_prompt_hint() == "Write Python"

    def test_no_system_prompt_hint_returns_empty(self):
        from lackpy.interpreters.sandbox import SandboxedInterpreter
        inner = MagicMock(spec=["name", "description", "validate", "execute"])
        inner.name = "python"
        inner.description = "Python interpreter"
        backend = MagicMock()
        si = SandboxedInterpreter(interpreter=inner, backend=backend)
        assert si.system_prompt_hint() == ""
