"""lackpy-lang — the restricted language (grammar · validator · grader · spec).

The pure language layer: stdlib only, **no execution, no LLM, no runtime**. It defines
what a lackpy program *is* and whether it is safe to run — the SemVer-stable contract the
runtime, the generator, and the interpreters all build on (see RFC 0001).

This package is a **leaf**: `tests/lang/test_no_upward_deps.py` guards that nothing here
imports "upward" into run/kit/policy/infer/service/interpreters. That guard is what makes
`lackpy-lang` an extractable boundary rather than an aspiration.
"""
from .grammar import (
    ALLOWED_BUILTINS,
    ALLOWED_NODES,
    FORBIDDEN_NAMES,
    FORBIDDEN_NODES,
)
from .grader import (
    DEFAULT_EFFECTS_CEILING,
    DEFAULT_GRADE_W,
    Grade,
    compute_grade,
)
from .spec import format_spec, get_spec
from .validator import ValidationResult, validate

__all__ = [
    # grammar — the restriction surface
    "ALLOWED_NODES", "FORBIDDEN_NODES", "FORBIDDEN_NAMES", "ALLOWED_BUILTINS",
    # validation — is this program safe?
    "validate", "ValidationResult",
    # grading — how restricted/effectful is it?
    "Grade", "compute_grade", "DEFAULT_GRADE_W", "DEFAULT_EFFECTS_CEILING",
    # spec — the human/agent-facing language description
    "get_spec", "format_spec",
]
