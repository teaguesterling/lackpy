"""Failure mode taxonomy for generation outcomes.

These identifiers are shared between lackpy and kibitzer. Lackpy
classifies the failure mode after validation/execution; kibitzer
accumulates them in the event log and returns them via
get_failure_patterns() and get_prompt_hints().

The taxonomy is deliberately small. Each category maps to a specific
prompt intervention — if two failure modes need the same fix, they
should be the same category.
"""

from __future__ import annotations

# Model defines functions/classes instead of calling pre-loaded tools.
# Fix: "ORCHESTRATE, DO NOT IMPLEMENT" framing.
IMPLEMENT_NOT_ORCHESTRATE = "implement_not_orchestrate"

# Model uses open(), import os, or other stdlib instead of kit tools.
# Fix: "Do NOT use open(). Use read_file() for ALL file reading."
STDLIB_LEAK = "stdlib_leak"

# Model prefixes paths with directory names (e.g. 'toybox/app.py').
# Fix: "All paths are relative to the workspace root."
PATH_PREFIX = "path_prefix"

# Model outputs bare tokens (ipynb, py, sql) from Jupyter framing.
# Fix: use interpreter-specialized prompt instead of Jupyter template.
JUPYTER_CONFUSION = "jupyter_confusion"

# Model emits non-Python syntax (-> annotations, prose, arrow operators).
# Fix: "Output ONLY the program — no annotations, no prose."
SYNTAX_ARTIFACT = "syntax_artifact"

# Model accesses wrong dict keys (e.g. 'path' instead of 'file').
# Fix: document return schema in namespace_desc.
KEY_HALLUCINATION = "key_hallucination"

# Model generates valid code that executes but produces wrong output.
# No single prompt fix — may need better examples or constraints.
WRONG_OUTPUT = "wrong_output"

# All recognized failure mode strings.
ALL_MODES = frozenset({
    IMPLEMENT_NOT_ORCHESTRATE,
    STDLIB_LEAK,
    PATH_PREFIX,
    JUPYTER_CONFUSION,
    SYNTAX_ARTIFACT,
    KEY_HALLUCINATION,
    WRONG_OUTPUT,
})


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
