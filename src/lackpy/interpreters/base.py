"""Base protocol and shared types for lackpy interpreter plugins.

An interpreter takes a program string (whose shape depends on the interpreter's
language — restricted Python, a CSS selector, a fluent chain expression, etc.)
and executes it against a resolved kit. Interpreters validate before execution
and enforce restrictions during execution; the exact enforcement mechanism is
the interpreter's own concern.

The protocol is intentionally small:

- ``validate(program, context)`` — check that the program parses and is allowed
- ``execute(program, context)`` — run it and return a structured result

Validation runs before execution and returns a :class:`InterpreterValidationResult`.
Execution runs once, asynchronously, and returns an :class:`InterpreterExecutionResult`.
Callers that want both can call :func:`Interpreter.run` as a convenience.

The protocol has its own validation and result types (prefixed with ``Interpreter``)
to avoid colliding with the existing runtime's :class:`lackpy.run.base.ExecutionResult`,
which is specific to restricted-Python execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class ExecutionContext:
    """Execution context passed to an interpreter.

    Bundles everything an interpreter needs to validate and run a program:
    the resolved kit (tools, callables, allowed names), named parameters,
    the workspace base directory, and an optional ``config`` dict for
    interpreter-specific knobs (e.g. ``mode="full"`` for ast-select).

    Attributes:
        kit: The resolved kit the program runs against. Interpreters that
            don't use kits (e.g. ast-select) may ignore this, though most
            will read the kit's description or tool list for validation.
        params: Named parameter values injected into the execution
            namespace. Interpreter-specific; the restricted-Python
            interpreter uses this for parameter substitution.
        base_dir: Directory the interpreter operates against. For
            ast-select, this is where selectors resolve paths from.
            For the Python interpreter, this is the working directory
            during execution. Defaults to the current directory.
        config: Interpreter-specific configuration. For example, the
            ast-select interpreter reads ``config.get("mode", "full")``
            to choose between full-body and brief output.
        extra_rules: Additional validation rules. Meaning depends on
            the interpreter; restricted-Python uses them for AST checks.
    """

    kit: Any = None
    params: dict[str, Any] = field(default_factory=dict)
    base_dir: Path = field(default_factory=Path.cwd)
    config: dict[str, Any] = field(default_factory=dict)
    extra_rules: list | None = None


@dataclass
class InterpreterValidationResult:
    """Result of validating a program against an interpreter.

    Attributes:
        valid: Whether the program is well-formed and allowed.
        errors: Validation error messages. Empty when ``valid`` is True.
        warnings: Non-fatal observations (e.g. "deprecated selector
            syntax"). Present regardless of whether ``valid`` is True.
    """

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class InterpreterExecutionResult:
    """Result of executing a program through an interpreter.

    The output shape is interpreter-specific: the restricted-Python
    interpreter returns whatever the program's last expression evaluated
    to; ast-select returns a markdown string; the pluckit chain
    interpreter returns whatever the terminal operation produced. The
    ``output_format`` field identifies the shape so consumers can
    dispatch accordingly.

    Attributes:
        success: Whether execution completed without error.
        output: The program's result. Shape depends on ``output_format``.
        output_format: A short identifier for the output shape. Known
            values: ``"markdown"``, ``"text"``, ``"python"``, ``"json"``,
            ``"list"``, ``"dict"``, ``"none"``. Interpreters may define
            their own.
        error: Error message if execution failed. None on success.
        duration_ms: Wall-clock execution time in milliseconds.
        metadata: Interpreter-specific metadata (e.g. match count,
            traces, warnings produced during execution). Consumers
            can inspect this for debugging or display.
    """

    success: bool
    output: Any = None
    output_format: str = "none"
    error: str | None = None
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class Interpreter(Protocol):
    """Protocol for lackpy interpreter plugins.

    Implementations must provide:

    - A ``name`` class attribute — a short identifier used in the
      registry and on the command line (e.g. ``"python"``, ``"ast-select"``).
    - A ``description`` class attribute — a one-line human description.
    - ``validate(program, context)`` — synchronous; returns a
      :class:`InterpreterValidationResult` without running the program.
    - ``execute(program, context)`` — asynchronous; runs the program
      and returns an :class:`InterpreterExecutionResult`.

    Implementations SHOULD:

    - Be safe to instantiate without side effects.
    - Accept programs that fail validation by returning a failed result
      from ``execute``, not raising. (Only unexpected infrastructure
      errors should raise.)
    - Set ``output_format`` on the execution result to match what they
      actually produced.

    Implementations MAY:

    - Provide a ``system_prompt_hint()`` method returning a string
      that describes the interpreter's expected program format, syntax
      reference, and output rules. When present, the inference prompt
      builder incorporates this hint to produce interpreter-specialized
      prompts instead of the generic Jupyter-cell framing. Interpreters
      that omit this method get the default prompt.
    """

    name: str
    description: str

    def validate(
        self,
        program: str,
        context: ExecutionContext,
    ) -> InterpreterValidationResult: ...

    async def execute(
        self,
        program: str,
        context: ExecutionContext,
    ) -> InterpreterExecutionResult: ...


async def run_interpreter(
    interpreter: Interpreter,
    program: str,
    context: ExecutionContext,
) -> InterpreterExecutionResult:
    """Convenience: validate then execute, returning a single result.

    If validation fails, returns a failed execution result with the
    validation errors joined as the error message. No execution is
    attempted.
    """
    validation = interpreter.validate(program, context)
    if not validation.valid:
        return InterpreterExecutionResult(
            success=False,
            error="Validation failed: " + "; ".join(validation.errors),
            output_format="none",
        )
    return await interpreter.execute(program, context)
