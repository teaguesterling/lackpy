"""Failure-mode classification for generation outcomes.

The taxonomy itself (the mode identifiers + ALL_MODES) is the SemVer-stable shared
vocabulary and now lives in ``lackpy.lang.failure_modes`` so the coaching side (kibitzer)
can import it without lackpy's runtime. It is re-exported here for back-compat — existing
``from lackpy.infer.failure_modes import STDLIB_LEAK`` callers are unchanged. This module
keeps ``classify_failure``, the runtime logic that maps a failed generation onto a mode.
"""

from __future__ import annotations

from lackpy.lang.failure_modes import (
    ALL_MODES,
    IMPLEMENT_NOT_ORCHESTRATE,
    JUPYTER_CONFUSION,
    KEY_HALLUCINATION,
    PATH_PREFIX,
    STDLIB_LEAK,
    SYNTAX_ARTIFACT,
    WRONG_OUTPUT,
)

__all__ = [
    "ALL_MODES",
    "IMPLEMENT_NOT_ORCHESTRATE",
    "STDLIB_LEAK",
    "PATH_PREFIX",
    "JUPYTER_CONFUSION",
    "SYNTAX_ARTIFACT",
    "KEY_HALLUCINATION",
    "WRONG_OUTPUT",
    "classify_failure",
]


def classify_failure(
    gate_passed: bool,
    gate_errors: list[str],
    exec_error: str | None,
    sanitized_program: str,
) -> str | None:
    """Classify a failed generation into a failure mode.

    Returns None if the generation succeeded (no failure to classify)
    or if the failure doesn't match any known pattern.

    Args:
        gate_passed: Whether the structural gate passed.
        gate_errors: Error strings from the gate (empty if passed).
        exec_error: Runtime error string (None if execution succeeded).
        sanitized_program: The sanitized program text.
    """
    gate_err_text = " ".join(gate_errors).lower()

    if not gate_passed:
        if "functiondef" in gate_err_text or "classdef" in gate_err_text:
            return IMPLEMENT_NOT_ORCHESTRATE
        if "import" in gate_err_text:
            return IMPLEMENT_NOT_ORCHESTRATE
        if "forbidden name" in gate_err_text and "open" in gate_err_text:
            return STDLIB_LEAK
        if "parse error" in gate_err_text or "invalid syntax" in gate_err_text:
            if "->" in sanitized_program or "→" in sanitized_program:
                return SYNTAX_ARTIFACT
        return None

    # Gate passed — check execution errors
    if not exec_error:
        return None

    exec_lower = exec_error.lower()
    stripped = sanitized_program.strip()

    if stripped in ("ipynb", "py", "sql", "python", "jupyter"):
        return JUPYTER_CONFUSION
    if "not defined" in exec_lower and stripped in ("ipynb", "py", "sql"):
        return JUPYTER_CONFUSION

    if "toybox/" in sanitized_program or "toybox\\" in sanitized_program:
        if "no such file" in exec_lower or "errno 2" in exec_lower:
            return PATH_PREFIX
    if "escapes" in exec_lower and "base_dir" in exec_lower:
        return PATH_PREFIX
    if "no such file" in exec_lower or "errno 2" in exec_lower:
        return PATH_PREFIX

    for bad_key in ("'path'", "'filename'", "'body'", "'name'"):
        if bad_key in exec_error:
            return KEY_HALLUCINATION

    return None
