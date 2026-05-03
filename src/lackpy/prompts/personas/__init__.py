"""Persona templates for system prompt composition.

Each persona is a format string with one slot: {interpreter_hint}.
The interpreter fills that slot with its syntax reference.
"""

from __future__ import annotations

from .analyst import ANALYST_TEMPLATE
from .developer import DEVELOPER_TEMPLATE
from .general import GENERAL_TEMPLATE

PERSONAS: dict[str, str] = {
    "general": GENERAL_TEMPLATE,
    "analyst": ANALYST_TEMPLATE,
    "developer": DEVELOPER_TEMPLATE,
}

DEFAULT_PERSONA = "general"


def list_personas() -> list[str]:
    return sorted(PERSONAS)
