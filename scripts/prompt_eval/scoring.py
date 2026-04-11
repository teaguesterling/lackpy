"""Two-stage scoring: structural gate then execution assertion.

Stage 1 (always): sanitize the raw generation via lackpy's production
sanitizer, then ask the intent's structural_gate whether the result is
worth executing.

Stage 2 (on gate pass, added in Task 9): run the program through the
appropriate lackpy interpreter against the toybox and check the
intent's exec_assertion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lackpy.infer.sanitize import sanitize_output

from .intents import GateResult, Intent


@dataclass
class CellScore:
    """Complete score for one matrix cell.

    Attributes:
        raw_generation: Exact model output before sanitization.
        sanitized_program: The program after sanitization — what was
            actually scored.
        gate: The structural-gate result (stage 1).
        executed: Whether the program was actually executed (stage 2).
        exec_output: The interpreter's execution output. None if not
            executed.
        exec_error: Error string if execution failed. None on success.
        assertion_passed: True if exec_assertion(exec_output) was True.
        score: Final integer 0/1/2:
            - 0: gate fail
            - 1: gate pass, execution fail OR assertion fail
            - 2: gate pass, execution success, assertion pass
        duration_ms_execution: Wall-clock for stage 2 (0.0 if not executed).
        interpreter_metadata: Whatever metadata the interpreter attached
            (e.g. trace entries, match count for selectors).
    """

    raw_generation: str
    sanitized_program: str
    gate: GateResult
    executed: bool = False
    exec_output: Any = None
    exec_error: str | None = None
    assertion_passed: bool = False
    score: int = 0
    duration_ms_execution: float = 0.0
    interpreter_metadata: dict = field(default_factory=dict)


def run_gate(intent: Intent, raw_generation: str) -> tuple[str, GateResult]:
    """Sanitize the raw generation and run the intent's structural gate.

    Empty input is short-circuited to a gate failure without consulting
    the intent's gate — this avoids double-reporting the emptiness in
    the intent's own error list.

    Returns:
        (sanitized_program, GateResult). The sanitized program is what
        the gate actually saw. On empty input, the sanitized value is
        the empty string and the GateResult carries a single
        "empty raw generation" error.
    """
    if not raw_generation:
        return "", GateResult(passed=False, errors=["empty raw generation"])
    sanitized = sanitize_output(raw_generation)
    gate = intent.structural_gate(sanitized)
    return sanitized, gate


# ─────────────────────────────────────────────────────────────────────
# Stage 2 — execution scoring
# ─────────────────────────────────────────────────────────────────────

import asyncio
import time
from pathlib import Path

from lackpy.interpreters import (
    AstSelectInterpreter,
    ExecutionContext,
    InterpreterExecutionResult,
    PluckerInterpreter,
    PssInterpreter,
    PythonInterpreter,
    run_interpreter,
)

from .eval_kit import build_eval_kit


_INTERPRETER_FACTORIES = {
    "python": PythonInterpreter,
    "ast-select": AstSelectInterpreter,
    "pss": PssInterpreter,
    "plucker": PluckerInterpreter,
}


def _build_context(interpreter_name: str, toybox_dir: Path) -> ExecutionContext:
    """Build an ExecutionContext appropriate for each interpreter.

    The python interpreter needs the eval kit resolved against the
    toybox base dir; the pluckit-backed interpreters need a `code`
    glob pointing at toybox python files.
    """
    toybox_dir = Path(toybox_dir).resolve()
    if interpreter_name == "python":
        kit = build_eval_kit(toybox_dir)
        return ExecutionContext(kit=kit, base_dir=toybox_dir)
    code_glob = str(toybox_dir / "**" / "*.py")
    return ExecutionContext(base_dir=toybox_dir, config={"code": code_glob})


def run_execution(
    intent: Intent,
    sanitized_program: str,
    toybox_dir: Path,
) -> InterpreterExecutionResult:
    """Execute the sanitized program via the matching lackpy interpreter.

    Returns the full InterpreterExecutionResult. Safe to call from a
    sync context — wraps the interpreter's async execute() in asyncio.run.
    """
    factory = _INTERPRETER_FACTORIES[intent.interpreter]
    interp = factory()
    ctx = _build_context(intent.interpreter, toybox_dir)
    return asyncio.run(run_interpreter(interp, sanitized_program, ctx))


def score_cell(
    intent: Intent,
    raw_generation: str,
    toybox_dir: Path,
) -> CellScore:
    """End-to-end cell scoring: gate → execute → assert → score 0/1/2.

    - Score 0: gate fail (program not worth executing).
    - Score 1: gate pass, but execution raised / returned failure / the
      assertion callable raised / assertion returned False.
    - Score 2: gate pass, execution succeeded, assertion passed.
    """
    sanitized, gate = run_gate(intent, raw_generation)
    cs = CellScore(
        raw_generation=raw_generation,
        sanitized_program=sanitized,
        gate=gate,
    )
    if not gate.passed:
        cs.score = 0
        return cs

    start = time.perf_counter()
    try:
        exec_result = run_execution(intent, sanitized, toybox_dir)
    except Exception as e:
        cs.executed = True
        cs.exec_error = f"{type(e).__name__}: {e}"
        cs.score = 1
        cs.duration_ms_execution = (time.perf_counter() - start) * 1000
        return cs
    cs.duration_ms_execution = (time.perf_counter() - start) * 1000
    cs.executed = True
    cs.exec_output = exec_result.output
    cs.exec_error = exec_result.error
    cs.interpreter_metadata = dict(exec_result.metadata or {})

    if not exec_result.success:
        cs.score = 1
        return cs

    try:
        passed = bool(intent.exec_assertion(exec_result.output))
    except Exception as e:
        cs.assertion_passed = False
        cs.exec_error = f"assertion raised: {type(e).__name__}: {e}"
        cs.score = 1
        return cs

    cs.assertion_passed = passed
    cs.score = 2 if passed else 1
    return cs
