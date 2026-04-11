"""Lackpy interpreter plugins.

An interpreter executes a program against a resolved kit and returns an
execution result. The restricted-Python interpreter that lackpy was
originally built around is now one plugin among several:

- :class:`~lackpy.interpreters.python.PythonInterpreter` — restricted
  Python over the kit's callables (the original execution path)
- :class:`~lackpy.interpreters.ast_select.AstSelectInterpreter` — a
  single CSS selector evaluated against source code via pluckit,
  rendered as markdown with the selector as the heading identifier
- :class:`~lackpy.interpreters.pss.PssInterpreter` — multi-rule
  selector sheets (selector + declaration blocks) rendered as markdown
  via pluckit's ``AstViewer`` plugin
- :class:`~lackpy.interpreters.plucker.PluckerInterpreter` — fluent
  chain expressions over pluckit's Plucker/Selection classes, with
  the entry point ``source(code)`` and any pluckit method chain
  following it

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
from .pss import PssInterpreter
from .plucker import PluckerInterpreter

# Register the bundled interpreters at import time. Other interpreters
# (e.g. claude-code) register themselves when their modules are imported.
register_interpreter(PythonInterpreter)
register_interpreter(AstSelectInterpreter)
register_interpreter(PssInterpreter)
register_interpreter(PluckerInterpreter)

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
    "PssInterpreter",
    "PluckerInterpreter",
]
