"""Pattern-specific error hints for inference retry feedback.

All hints use positive language ("use X instead") rather than negative
("don't use Y"). Stress tests show 1.5B models respond better to
positive redirection than negative restriction.

The hint for an unapproved word (``open``, ``json.load``, ``os``, ...) is its
entry in ``dictionary.UNAPPROVED``, the one source for those hints.
"""

from __future__ import annotations

import re

from ..lang.dictionary import MODULES, hint_for

# The quoted subject of a validator error: "Forbidden name: 'open' at line 3",
# "Unknown module member: 'json.load' at line 2".
_SUBJECT = re.compile(r"'([A-Za-z_][\w.]*)'")


def enrich_errors(errors: list[str], namespace_desc: str) -> list[str]:
    """Augment validation errors with actionable positive hints.

    Args:
        errors: Raw validation error strings.
        namespace_desc: Unused since the kit was removed (stepy100); kept for
            callers' signatures.

    Returns:
        Enriched error list with positive hints appended where patterns match.
    """
    enriched = list(errors)
    hints: list[str] = []

    def add(hint: str) -> None:
        if hint not in hints:
            hints.append(hint)

    for error in errors:
        for subject in _SUBJECT.findall(error):
            hint = hint_for(subject)
            if hint is not None:
                add(f"For {subject}: {hint}.")
            elif subject == "Path" and error.startswith("Unknown function"):
                add("Import Path first: from pathlib import Path.")

    error_text = " ".join(errors).lower()

    if "lambda" in error_text:
        add("lambda is allowed only as key= for sorted, min or max.")

    if "forbidden ast node: import" in error_text or "importfrom" in error_text:
        modules = ", ".join(sorted(MODULES))
        add(f"Import only these modules: {modules}. Use pathlib.Path for files.")

    if "while" in error_text:
        add("Use for-loops with range() or iterate over a list variable.")

    if "classdef" in error_text:
        add("Use a dict, or a top-level def, in place of a class.")

    if hints:
        enriched.append("--- Suggestions ---")
        enriched.extend(hints)

    return enriched
