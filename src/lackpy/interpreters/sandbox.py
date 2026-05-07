"""SandboxedInterpreter: wraps any Interpreter with sandbox execution."""

from __future__ import annotations

import logging
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .base import (
    ExecutionContext,
    InterpreterExecutionResult,
    InterpreterValidationResult,
)
from ..sandbox.backend import SandboxBackend
from ..sandbox.config import SandboxBaseConfig
from ..sandbox.constraints import SandboxConstraint, merge_constraints
from ..sandbox.strategies import SubprocessStrategy

logger = logging.getLogger(__name__)

EMPTY_MAP: MappingProxyType[str, Any] = MappingProxyType({})


class SandboxedInterpreter:
    """Decorator that wraps any Interpreter with sandbox execution."""

    def __init__(
        self,
        interpreter: Any,
        backend: SandboxBackend,
        strategy: str = "subprocess",
        base_config: SandboxBaseConfig | None = None,
    ) -> None:
        self.interpreter = interpreter
        self._backend = backend
        self._strategy = strategy
        self._base_config = base_config or SandboxBaseConfig()

    @property
    def name(self) -> str:
        return self.interpreter.name

    @property
    def description(self) -> str:
        return self.interpreter.description

    def validate(
        self, program: str, context: ExecutionContext,
    ) -> InterpreterValidationResult:
        return self.interpreter.validate(program, context)

    async def execute(
        self,
        program: str,
        context: ExecutionContext,
        sandbox_constraints: tuple[SandboxConstraint, ...] = (),
        backend_configs: Mapping[str, Any] | None = None,
    ) -> InterpreterExecutionResult:
        backend_configs = backend_configs or EMPTY_MAP

        config = self._resolve_config(sandbox_constraints, backend_configs, context)

        if self._strategy == "subprocess":
            strategy = SubprocessStrategy(backend=self._backend)
            return await strategy.run(
                program=program,
                embedded_sources={},
                bridge_socket=None,
                bridge_authkey=None,
                base_dir=context.base_dir,
                config=config,
            )
        elif self._strategy == "jail_call":
            from ..sandbox.strategies import JailCallStrategy
            strategy_jc = JailCallStrategy()
            return await strategy_jc.run_with_interpreter(
                interpreter=self.interpreter,
                program=program,
                context=context,
                config=config,
            )
        else:
            raise ValueError(f"Unknown sandbox strategy: {self._strategy}")

    def _resolve_config(
        self,
        policy_constraints: tuple[SandboxConstraint, ...],
        backend_configs: Mapping[str, Any],
        context: ExecutionContext,
    ) -> Any:
        precompiled = backend_configs.get(self._backend.name)
        if precompiled is not None:
            result = self._backend.accept_policy_config(precompiled)
            if result is not None:
                for w in result.warnings:
                    logger.warning("Sandbox constraint warning: %s", w.reason)
                return result.config

        base_constraints = self._base_config.to_base_constraints()
        all_constraints = merge_constraints(base_constraints + policy_constraints)
        compilation = self._backend.compile(all_constraints, context.base_dir)

        for w in compilation.warnings:
            logger.warning("Sandbox constraint warning: %s", w.reason)

        return compilation.config

    def system_prompt_hint(self) -> str:
        hint_fn = getattr(self.interpreter, "system_prompt_hint", None)
        if hint_fn is not None:
            return hint_fn()
        return ""
