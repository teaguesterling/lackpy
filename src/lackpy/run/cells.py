"""CellSession: the per-document cell runtime for literate lackpy (literate stepy D1, D2, D4).

A session owns one namespace from :func:`build_namespace`, so names, imports and
defs flow from cell to cell. Every cell is checked as part of the cumulative
program before it runs: the source of each earlier cell that ran, then this cell,
validated as one lackpy program (D2). Only the violations on this cell's lines
are reported against it.

A ``LackpyRefusal`` never leaves :meth:`CellSession.run_cell`; it gives
``phase="refused"`` (D4). The validator rejects every handler and ``finally``
that could catch a refusal (``safety.except-base``, ``safety.try-finally``), so
a refusal inside a ``try`` still reaches the session (gate 4).
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from ..lang import validate
from .namespace import build_namespace
from .paths import PathPolicy
from .refusal import LackpyRefusal
from .runner import RUNTIME_FAILURES, _run_validated_code, data_variables, make_print_capture
from .trace import Trace, error_text

Phase = Literal["validate", "runtime", "refused"]


@dataclass
class CellOutcome:
    """The result of one cell.

    Attributes:
        ok: Whether the cell validated and ran without error.
        stdout: Text the cell printed, also when it failed part way.
        error: The error text, or None.
        phase: Where the cell stopped: ``"validate"`` (nothing ran), ``"runtime"``
            (an ``Exception``), ``"refused"`` (a ``LackpyRefusal``), or None.
        delta: The names this cell bound or rebound whose values are data.
    """

    ok: bool
    stdout: str = ""
    error: str | None = None
    phase: Phase | None = None
    delta: dict[str, Any] = field(default_factory=dict)


class CellRejected(ValueError):
    """``eval_expr`` was given a source that is not one valid lackpy expression in the session."""


_LINE = re.compile(r"\bline (\d+)\b")


class CellSession:
    """One literate document's namespace, and its cumulative program."""

    def __init__(self, *, path_policy: PathPolicy | None, trace: Trace | None = None,
                 program_file: str | None = None) -> None:
        self.trace = trace if trace is not None else Trace()
        self._captured, print_fn = make_print_capture()
        self._namespace = build_namespace(program_file=program_file, path_policy=path_policy,
                                          print_fn=print_fn, trace=self.trace)
        self._runtime_names = frozenset(self._namespace)
        # The cells in the cumulative program: those that passed validation and
        # were not refused. A cell that failed at runtime stays in (see _admit).
        self._sources: list[str] = []

    def bound_names(self) -> set[str]:
        """The names the cells have bound so far (the runtime's own names left out)."""
        return {k for k in self._namespace if k not in self._runtime_names}

    def run_cell(self, source: str) -> CellOutcome:
        """Validate ``source`` in the cumulative program, then exec it in the namespace."""
        self._captured.clear()
        try:
            return self._run_cell(source)
        except RUNTIME_FAILURES as e:  # never let a refusal (or a host fault) escape
            phase: Phase = "refused" if isinstance(e, LackpyRefusal) else "runtime"
            return CellOutcome(ok=False, stdout="".join(self._captured),
                               error=error_text(e), phase=phase)

    def eval_expr(self, source: str) -> Any:
        """Validate ``source`` as one expression line after the cumulative program, then evaluate it.

        Returns the value as it is: no data filter (the host-side display formats it).
        Text the expression prints is discarded.

        Raises:
            CellRejected: ``source`` is not one line holding one expression, or it
                has a violation on its own line.
            LackpyRefusal: the evaluation refused. The caller (the kernel) decides
                what a refusal does to the document (D4).
            Exception: whatever the expression raised at runtime.
        """
        if "\n" in source or "\r" in source:
            raise CellRejected("an expression must be one line")
        try:
            tree = ast.parse(source, mode="eval")
        except SyntaxError as e:
            raise CellRejected(f"Parse error: {e.msg} (line 1)") from None
        error = self._violations(source)
        if error is not None:
            raise CellRejected(error)
        code = compile(tree, "<lackpy>", "eval")
        try:
            return eval(code, self._namespace)  # noqa: S307 - validated above, restricted namespace
        finally:
            self._captured.clear()

    # -- internals -------------------------------------------------------------

    def _run_cell(self, source: str) -> CellOutcome:
        # Each cell must parse alone: concatenated, ``else:`` or an indented line
        # could join an earlier cell's block, and exec runs this cell alone.
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return CellOutcome(ok=False, phase="validate",
                               error=f"Parse error: {e.msg} (line {e.lineno})")
        error = self._violations(source)
        if error is not None:
            return CellOutcome(ok=False, phase="validate", error=error)
        code = compile(tree, "<lackpy>", "exec")

        # References, not ids: an id can be reused once its object is freed.
        before = dict(self._namespace)
        try:
            _run_validated_code(code, self._namespace)
        except LackpyRefusal as e:
            return CellOutcome(ok=False, stdout="".join(self._captured),
                               error=error_text(e), phase="refused")
        except Exception as e:
            self._sources.append(source)
            return CellOutcome(ok=False, stdout="".join(self._captured),
                               error=error_text(e), phase="runtime")
        self._sources.append(source)
        delta = data_variables(
            self._namespace,
            lambda k: k not in self._runtime_names
            and (k not in before or before[k] is not self._namespace[k]))
        return CellOutcome(ok=True, stdout="".join(self._captured), delta=delta)

    def _violations(self, source: str) -> str | None:
        """Validate the cumulative program ending in ``source``; return this cell's errors, or None.

        A violation counts when its line is in ``source``, or when it has no line.
        Line numbers in the text are mapped to ``source``'s own lines; a line of an
        earlier cell keeps its cumulative number and says so.
        """
        prefix = "\n".join(self._sources)
        offset = prefix.count("\n") + 1 if self._sources else 0
        program = f"{prefix}\n{source}" if self._sources else source
        verdict = validate(program, allowed_names=set())
        mine = [v for v in verdict.violations if v.line is None or v.line > offset]
        if not mine:
            return None

        def local(m: re.Match[str]) -> str:
            n = int(m.group(1))
            return f"line {n - offset}" if n > offset else f"line {n} of the earlier cells"

        # The format of ``service.validation_error`` (message, then the hint), which
        # run/ cannot import: the service module pulls in inference and config.
        parts = [_LINE.sub(local, f"{v.message} ({v.hint})" if v.hint else v.message) for v in mine]
        return f"Validation failed: {'; '.join(parts)}"
