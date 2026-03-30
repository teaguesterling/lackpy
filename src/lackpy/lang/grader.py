"""Grade computation for lackpy programs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Grade:
    w: int  # world coupling (0=pure, 1=pinhole read, 2=scoped exec, 3=scoped write)
    d: int  # effects ceiling

    def __str__(self) -> str:
        return f"Grade(w={self.w}, d={self.d})"


DEFAULT_GRADE_W = 3
DEFAULT_EFFECTS_CEILING = 3


def compute_grade(tools: dict[str, dict]) -> Grade:
    if not tools:
        return Grade(w=0, d=0)
    max_w = 0
    max_d = 0
    for spec in tools.values():
        max_w = max(max_w, spec.get("grade_w", DEFAULT_GRADE_W))
        max_d = max(max_d, spec.get("effects_ceiling", DEFAULT_EFFECTS_CEILING))
    return Grade(w=max_w, d=max_d)
