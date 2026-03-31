"""Pattern-specific error hints for inference retry feedback.

When validation fails, these hints provide actionable guidance to the model
instead of raw validation errors. The model self-corrects better with
specific instructions like "Use read(path), not open()" than with
"Forbidden name: 'open' at line 3".
"""

from __future__ import annotations


def enrich_errors(errors: list[str], namespace_desc: str) -> list[str]:
    """Augment validation errors with actionable hints.

    Args:
        errors: Raw validation error strings.
        namespace_desc: The namespace description (to check available tools).

    Returns:
        Enriched error list with hints appended where patterns match.
    """
    enriched = list(errors)
    hints: list[str] = []

    error_text = " ".join(errors).lower()

    # open() instead of read()
    if "open" in error_text and "read(" in namespace_desc:
        hints.append("Use read(path) to read files, not open(). open() is not available.")

    # Model wrote a function definition instead of calling a tool
    if "functiondef" in error_text:
        hints.append(
            "Do not write function definitions. Call the tools already available "
            "in the namespace instead."
        )

    # Lambda usage (sorting)
    if "lambda" in error_text:
        hints.append("Use sort_by(items, key) for sorting instead of lambda expressions.")

    # Import attempts
    if "import" in error_text:
        hints.append(
            "Do not use import statements. All needed functions are already "
            "available in the namespace."
        )

    # While loop
    if "while" in error_text:
        hints.append("Use for-loops with range() or iterate over a variable instead of while loops.")

    # Class definition
    if "classdef" in error_text:
        hints.append("Do not define classes. Use the available tools and builtins directly.")

    # Try/except
    if "try" in error_text or "excepthandler" in error_text:
        hints.append("Do not use try/except. Let errors propagate to the caller.")

    if hints:
        enriched.append("--- Hints ---")
        enriched.extend(hints)

    return enriched
