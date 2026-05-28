"""Failure-mode taxonomy — the shared vocabulary of generation-outcome defects.

This is the SemVer-stable set of identifiers shared between lackpy (which classifies a
generation's failure mode after validation/execution — see
``lackpy.infer.failure_modes.classify_failure``) and kibitzer (which accumulates them and
maps each to a prompt intervention). It lives in ``lackpy.lang`` so the coaching side can
depend on it without pulling lackpy's runtime — it is pure data, stdlib-only, no execution.

The taxonomy is deliberately small. Each category maps to a specific prompt intervention —
if two failure modes need the same fix, they should be the same category.
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
