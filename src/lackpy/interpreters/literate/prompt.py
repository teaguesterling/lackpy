"""System prompt hint for the literate interpreter.

The general prompt is the default for system_prompt_hint().
Persona-specific prompts live in the prompts/ subpackage.
"""

from __future__ import annotations

from .prompts import DEFAULT_PROMPT, GENERAL_PROMPT, PROMPTS

LITERATE_SYSTEM_PROMPT = GENERAL_PROMPT

__all__ = ["LITERATE_SYSTEM_PROMPT", "PROMPTS", "DEFAULT_PROMPT"]
