"""Executor protocol for lackpy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .trace import Trace


@dataclass
class ExecutionResult:
    """Result of executing a lackpy program.

    Attributes:
        success: Whether execution completed without error.
        output: The last expression's value, or None.
        stdout: Text captured from ``print()`` calls during execution.
        error: Error message if execution failed.
        trace: Execution trace with tool call records.
        variables: Variables assigned during execution (excluding params and internals).
        error_phase: Where a failure came from; ``"refused"`` for a ``LackpyRefusal``,
            otherwise None.
    """

    success: bool
    output: Any = None
    stdout: str = ""
    error: str | None = None
    trace: Trace = field(default_factory=Trace)
    variables: dict[str, Any] = field(default_factory=dict)
    error_phase: str | None = None
    #: A literate run's session (Plan 3 Task 7, D10): pass it back with a
    #: resubmitted document to resume from the first changed cell.
    session_id: str | None = None

    @property
    def effective_output(self) -> Any:
        """The value to hand back to a caller.

        Prefers the typed last-expression value; falls back to the captured
        stdout (stripped) when a program printed its answer instead of leaving
        a bare final expression. Never coerces a present typed value.
        """
        if self.output is not None:
            return self.output
        return self.stdout.strip() or None


class Executor(Protocol):
    def run(
        self, program: str, *, params: dict[str, Any] | None = None,
        program_file: str | None = None, path_policy: Any = None,
    ) -> ExecutionResult: ...
