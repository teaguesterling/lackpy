"""Executor protocol for lackpy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .trace import Trace


@dataclass
class ExecutionResult:
    """Result of executing a lackpy program.

    Attributes:
        success: Whether execution completed without error.
        output: The last expression's value, or None.
        error: Error message if execution failed.
        trace: Execution trace with tool call records.
        variables: Variables assigned during execution (excluding params and internals).
    """

    success: bool
    output: Any = None
    error: str | None = None
    trace: Trace = field(default_factory=Trace)
    variables: dict[str, Any] = field(default_factory=dict)


class Executor(Protocol):
    def run(
        self, program: str, namespace: dict[str, Callable],
        params: dict[str, Any] | None = None,
    ) -> ExecutionResult: ...
