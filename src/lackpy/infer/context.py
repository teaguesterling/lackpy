"""Step context: the accumulator threaded through the inference fold."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepTrace:
    """Operational record of one step execution.

    Captures prompts, model, timing, and raw output for observability.
    """

    step_name: str
    provider_name: str | None
    model: str | None
    system_prompt: str | None
    user_prompt: str | None
    raw_output: str | None
    duration_ms: float


@dataclass
class ProgramState:
    """Semantic result of one generation/transform step.

    Each step in the fold pushes a ProgramState onto the context's
    programs list. The most recent entry is the current state.
    """

    program: str
    intent: str
    kit: Any  # ResolvedKit, but Any to avoid circular imports
    valid: bool | None
    errors: list[str]
    trace: StepTrace


@dataclass
class StepContext:
    """The accumulator threaded through the inference pipeline fold.

    Input fields (intent, kit, params_desc, extra_rules) are set at
    strategy entry and remain immutable across the fold. Steps read
    from `current` and push new ProgramState entries onto `programs`.
    """

    intent: str
    kit: Any  # ResolvedKit
    params_desc: str | None = None
    extra_rules: list | None = None
    programs: list[ProgramState] = field(default_factory=list)
    provider: Any = None

    @property
    def current(self) -> ProgramState | None:
        """The most recent ProgramState, or None if no programs yet."""
        return self.programs[-1] if self.programs else None
