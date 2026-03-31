"""lackpy: Python that lacks most of Python."""

__version__ = "0.2.0"

from .service import LackpyService
from .lang.validator import validate, ValidationResult
from .lang.grader import Grade, compute_grade
from .lang.grammar import ALLOWED_NODES, FORBIDDEN_NODES, FORBIDDEN_NAMES, ALLOWED_BUILTINS
from .kit.toolbox import Toolbox, ToolSpec, ArgSpec
from .kit.registry import resolve_kit, ResolvedKit
from .run.runner import RestrictedRunner
from .run.base import ExecutionResult
from .run.trace import Trace, TraceEntry

__all__ = [
    "LackpyService",
    "validate", "ValidationResult",
    "Grade", "compute_grade",
    "ALLOWED_NODES", "FORBIDDEN_NODES", "FORBIDDEN_NAMES", "ALLOWED_BUILTINS",
    "Toolbox", "ToolSpec", "ArgSpec",
    "resolve_kit", "ResolvedKit",
    "RestrictedRunner", "ExecutionResult",
    "Trace", "TraceEntry",
]
