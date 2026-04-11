"""Two-stage scoring: structural gate then execution assertion.

Stage 1 (always): sanitize the raw generation via lackpy's production
sanitizer, then ask the intent's structural_gate whether the result is
worth executing.

Stage 2 (on gate pass, added in Task 9): run the program through the
appropriate lackpy interpreter against the toybox and check the
intent's exec_assertion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lackpy.infer.sanitize import sanitize_output

from .intents import GateResult, Intent


@dataclass
class CellScore:
    """Complete score for one matrix cell.

    Attributes:
        raw_generation: Exact model output before sanitization.
        sanitized_program: The program after sanitization — what was
            actually scored.
        gate: The structural-gate result (stage 1).
        executed: Whether the program was actually executed (stage 2).
        exec_output: The interpreter's execution output. None if not
            executed.
        exec_error: Error string if execution failed. None on success.
        assertion_passed: True if exec_assertion(exec_output) was True.
        score: Final integer 0/1/2:
            - 0: gate fail
            - 1: gate pass, execution fail OR assertion fail
            - 2: gate pass, execution success, assertion pass
        duration_ms_execution: Wall-clock for stage 2 (0.0 if not executed).
        interpreter_metadata: Whatever metadata the interpreter attached
            (e.g. trace entries, match count for selectors).
    """

    raw_generation: str
    sanitized_program: str
    gate: GateResult
    executed: bool = False
    exec_output: Any = None
    exec_error: str | None = None
    assertion_passed: bool = False
    score: int = 0
    duration_ms_execution: float = 0.0
    interpreter_metadata: dict = field(default_factory=dict)


def run_gate(intent: Intent, raw_generation: str) -> tuple[str, GateResult]:
    """Sanitize the raw generation and run the intent's structural gate.

    Empty input is short-circuited to a gate failure without consulting
    the intent's gate — this avoids double-reporting the emptiness in
    the intent's own error list.

    Returns:
        (sanitized_program, GateResult). The sanitized program is what
        the gate actually saw. On empty input, the sanitized value is
        the empty string and the GateResult carries a single
        "empty raw generation" error.
    """
    if not raw_generation:
        return "", GateResult(passed=False, errors=["empty raw generation"])
    sanitized = sanitize_output(raw_generation)
    gate = intent.structural_gate(sanitized)
    return sanitized, gate
