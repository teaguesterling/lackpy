"""Intent dataclass shared by all interpreter corpora."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class GateResult:
    """Outcome of a structural gate check.

    Attributes:
        passed: Whether the raw/sanitized program is well-formed enough
            to be worth executing.
        errors: Human-readable reasons the gate failed; empty on pass.
    """

    passed: bool
    errors: list[str] = field(default_factory=list)


@dataclass
class Intent:
    """One row of a corpus: natural-language task + scoring hooks.

    Attributes:
        id: Stable identifier, e.g. "py.core.01" — used as the primary
            key when writing JSONL rows and when resuming a killed run.
        interpreter: Which interpreter this intent targets
            ("python" | "ast-select" | "pss" | "plucker").
        difficulty: "core" or "stretch".
        text: The exact natural-language prompt given to the model as
            the user message.
        return_shape: Short label describing what the execution should
            return (e.g. "list[str]", "int", "dict", "markdown").
            Informational only; the assertion does the real check.
        structural_gate: Callable taking the sanitized program string
            and returning a GateResult. If passed=False, scoring stops.
        exec_assertion: Callable taking the execution result's `output`
            and returning True if the answer is correct.
        notes: Human-readable description of what the intent is
            stressing. Surfaces in the report for failure analysis.
    """

    id: str
    interpreter: str
    difficulty: str
    text: str
    return_shape: str
    structural_gate: Callable[[str], GateResult]
    exec_assertion: Callable[[Any], bool]
    notes: str = ""
