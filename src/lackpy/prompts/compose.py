"""Compose a persona template with an interpreter's syntax hint."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .personas import PERSONAS

if TYPE_CHECKING:
    pass


@runtime_checkable
class HasPromptHint(Protocol):
    def system_prompt_hint(self) -> str: ...


def compose(
    persona: str,
    interpreter: HasPromptHint | str,
) -> str:
    """Build a complete system prompt from a persona and interpreter.

    Args:
        persona: Name of a registered persona ("general", "analyst", etc.)
        interpreter: An interpreter instance with system_prompt_hint(),
            or a raw hint string.

    Returns:
        The composed system prompt ready for the model.
    """
    if persona not in PERSONAS:
        available = ", ".join(sorted(PERSONAS))
        raise ValueError(f"Unknown persona {persona!r}. Available: {available}")

    template = PERSONAS[persona]

    if isinstance(interpreter, str):
        hint = interpreter
    elif isinstance(interpreter, HasPromptHint):
        hint = interpreter.system_prompt_hint()
    else:
        raise TypeError(
            f"interpreter must be a string or have system_prompt_hint(), "
            f"got {type(interpreter).__name__}"
        )

    return template.replace("{interpreter_hint}", hint)
