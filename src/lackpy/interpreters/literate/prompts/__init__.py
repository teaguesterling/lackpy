"""System prompts for the literate interpreter.

Each prompt is a complete system message for a specific persona.
The harness selects one via --prompt or programmatically.
"""

from __future__ import annotations

from .general import GENERAL_PROMPT
from .analyst import ANALYST_PROMPT
from .developer import DEVELOPER_PROMPT

PROMPTS: dict[str, str] = {
    "general": GENERAL_PROMPT,
    "analyst": ANALYST_PROMPT,
    "developer": DEVELOPER_PROMPT,
}

DEFAULT_PROMPT = "general"

__all__ = [
    "PROMPTS",
    "DEFAULT_PROMPT",
    "GENERAL_PROMPT",
    "ANALYST_PROMPT",
    "DEVELOPER_PROMPT",
]
