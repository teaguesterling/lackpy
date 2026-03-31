"""Output sanitization — strip model artifacts from generated code."""

from __future__ import annotations

_PREAMBLE_MARKERS = ["here's", "here is", "the following", "the solution"]


def sanitize_output(raw: str) -> str:
    """Strip model artifacts from a raw generated code string.

    Removes leading preamble lines containing common model hedge phrases,
    and unwraps markdown code fences (``` blocks) if present.

    Args:
        raw: Raw string output from an inference provider.

    Returns:
        Cleaned program source with preamble and code fences removed.
        Returns an empty string if the input is blank.
    """
    text = raw.strip()
    if not text:
        return ""
    lines = text.split("\n")
    while lines and any(marker in lines[0].lower() for marker in _PREAMBLE_MARKERS):
        lines = lines[1:]
    text = "\n".join(lines).strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text
