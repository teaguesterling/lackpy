"""Persona-based system prompts for lackpy interpreters.

A persona defines the agent's role and workflow style. An interpreter
provides syntax and tool references via system_prompt_hint(). The
compose() function combines them into a complete system prompt.

    from lackpy.prompts import compose
    from lackpy.interpreters import LiterateInterpreter

    prompt = compose("analyst", LiterateInterpreter())
"""

from __future__ import annotations

from .compose import compose
from .personas import DEFAULT_PERSONA, PERSONAS, list_personas

__all__ = [
    "compose",
    "PERSONAS",
    "DEFAULT_PERSONA",
    "list_personas",
]
