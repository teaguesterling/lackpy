"""Lackpy interpreter plugins.

An interpreter executes a program against a resolved kit and returns an
execution result. The restricted-Python interpreter that lackpy was
originally built around is now one plugin among several:

- :class:`~lackpy.interpreters.python.PythonInterpreter` — restricted
  Python over the kit's callables (the original execution path)
- :class:`~lackpy.interpreters.ast_select.AstSelectInterpreter` — a
  single CSS selector evaluated against source code via pluckit,
  rendered as markdown

Interpreters are registered in :data:`INTERPRETERS` and selected by name
via ``--interpreter`` on the CLI or the ``interpreter=`` parameter on
service calls.

The protocol is intentionally small: validate before execution, execute
once asynchronously, return a structured result. Enforcement during
execution is the interpreter's own responsibility — the Python
interpreter uses AST restrictions, the ast-select interpreter relies on
pluckit's read-only AST queries, and so on.
"""

from __future__ import annotations

from .base import (
    ExecutionContext,
    Interpreter,
    InterpreterExecutionResult,
    InterpreterValidationResult,
    run_interpreter,
)
from .registry import INTERPRETERS, get_interpreter, list_interpreters, register_interpreter
from .python import PythonInterpreter
from .ast_select import AstSelectInterpreter

# Register the bundled interpreters at import time. Other interpreters
# (e.g. claude-code, pss, pluckit-chain) register themselves when their
# modules are imported.
register_interpreter(PythonInterpreter)
register_interpreter(AstSelectInterpreter)

__all__ = [
    "ExecutionContext",
    "Interpreter",
    "InterpreterValidationResult",
    "InterpreterExecutionResult",
    "INTERPRETERS",
    "get_interpreter",
    "list_interpreters",
    "register_interpreter",
    "run_interpreter",
    "PythonInterpreter",
    "AstSelectInterpreter",
]
