"""Pattern-specific error hints for inference retry feedback.

All hints use positive language ("use X instead") rather than negative
("don't use Y"). Stress tests show 1.5B models respond better to
positive redirection than negative restriction.
"""

from __future__ import annotations


def enrich_errors(errors: list[str], namespace_desc: str) -> list[str]:
    """Augment validation errors with actionable positive hints.

    Args:
        errors: Raw validation error strings.
        namespace_desc: The namespace description (to check available tools).

    Returns:
        Enriched error list with positive hints appended where patterns match.
    """
    enriched = list(errors)
    hints: list[str] = []

    error_text = " ".join(errors).lower()

    # open() instead of read()
    if "open" in error_text and "read(" in namespace_desc:
        hints.append("Use read(path) to get file contents.")

    # Model wrote a function definition instead of calling a tool
    if "functiondef" in error_text:
        hints.append(
            "Use the tools to find and modify existing code. "
            "Call find_definitions() to locate functions, edit() to modify them."
        )

    # Lambda usage outside key= (if still rejected)
    if "lambda" in error_text:
        hints.append("Use sort_by(items, key_name) for sorting, or key=lambda for sorted().")

    # Import attempts
    if "import" in error_text:
        hints.append("All needed functions are already available in the namespace.")

    # While loop
    if "while" in error_text:
        hints.append("Use for-loops with range() or iterate over a list variable.")

    # Class definition
    if "classdef" in error_text:
        hints.append("Use the available tools and builtins directly.")

    # Try/except
    if "try" in error_text or "excepthandler" in error_text:
        hints.append("Let errors propagate to the caller.")

    if hints:
        enriched.append("--- Suggestions ---")
        enriched.extend(hints)

    return enriched
