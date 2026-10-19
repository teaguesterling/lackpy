"""CellSession: the per-document cell runtime for literate lackpy (literate stepy D1, D2, D4).

A session owns one namespace from :func:`build_namespace`, so names, imports and
defs flow from cell to cell. The cells that passed validation form the session's
program, in order. A new cell joins only when the program with the cell added
still validates as one lackpy program (D2, review I1). Each violation is reported
against the new cell, also one that lands on an earlier cell's line. The program
starts empty and only a valid extension joins it, so the program never has a
violation of its own: every violation of the extended program is one the cell
caused.

A refusal ends the session (D4, ruling "after any refusal the CellSession is
closed"). After the first ``LackpyRefusal``, from a cell, a re-run or an
expression, every later call gives ``phase="refused"`` without validating or
executing. No ``LackpyRefusal`` leaves the session. The validator rejects every
handler and ``finally`` that could catch one (``safety.except-base``,
``safety.try-finally``), so a refusal inside a ``try`` still reaches the session
(gate 4).
"""

from __future__ import annotations

import ast
import bisect
import re
from dataclasses import dataclass, field
from types import CodeType
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
    """The result of one cell, or of one re-run.

    Attributes:
        ok: Whether the cell validated and ran without error.
        stdout: Text the cell printed, also when it failed part way.
        error: The error text, or None.
        phase: Where the cell stopped: ``"validate"`` (nothing ran), ``"runtime"``
            (an ``Exception``), ``"refused"`` (a ``LackpyRefusal``, now or earlier), or None.
        delta: The names this cell bound or rebound whose values are data.
        cell_id: The cell's identity, for :meth:`CellSession.rerun_cell`.
    """

    ok: bool
    stdout: str = ""
    error: str | None = None
    phase: Phase | None = None
    delta: dict[str, Any] = field(default_factory=dict)
    cell_id: int | None = None


@dataclass
class ExprOutcome:
    """The result of :meth:`CellSession.eval_expr`. ``value`` is the raw value, not filtered."""

    ok: bool
    value: Any = None
    error: str | None = None
    phase: Phase | None = None


@dataclass
class _Cell:
    code: CodeType | None  # None: the cell failed validation and is not in the program
    rerunnable: bool = False  # its first run was ok


# A validator line reference is ``line N`` outside quoted program text. Rules quote
# program text as '<text>' just before " at line N" (``ast.unparse`` output can hold
# quotes of its own), so that span is matched first and kept as it is (review M4).
_LINE = re.compile(r"'[^\n]*?'(?= at line \d)|\bline (\d+)\b")


class CellSession:
    """One literate document's namespace, and its program."""

    def __init__(self, *, path_policy: PathPolicy | None, trace: Trace | None = None,
                 program_file: str | None = None) -> None:
        self.trace = trace if trace is not None else Trace()
        self._captured, print_fn = make_print_capture()
        self._namespace = build_namespace(program_file=program_file, path_policy=path_policy,
                                          print_fn=print_fn, trace=self.trace)
        self._runtime_names = frozenset(self._namespace)
        self._cells: dict[int, _Cell] = {}
        self._next_id = 1
        # The program: the cells that passed validation, including the ones that
        # then failed at runtime (part of them ran, so the validator must see
        # their bindings). ``_starts[i]`` is the first program line of ``_ids[i]``.
        self._program = ""
        self._lines = 0
        self._starts: list[int] = []
        self._ids: list[int] = []
        self._refusal: str | None = None  # the first refusal's text: the session is closed

    def bound_names(self) -> set[str]:
        """The names the cells have bound so far (the runtime's own names left out)."""
        return {k for k in self._namespace if k not in self._runtime_names}

    def run_cell(self, source: str) -> CellOutcome:
        """Validate ``source`` as an extension of the program, then exec it in the namespace."""
        cell_id = self._next_id
        self._next_id += 1
        if self._refusal is not None:
            return self._closed(cell_id)
        self._captured.clear()
        try:
            return self._run_cell(cell_id, source)
        except RUNTIME_FAILURES as e:  # never let a refusal (or a host fault) escape
            return self._failure(cell_id, e)

    def rerun_cell(self, cell_id: int) -> CellOutcome:
        """Exec the stored code of a cell whose first run was ok, again.

        The code is not validated again and not added to the program again: the
        program text does not change. An unknown cell, or one whose first run
        failed, gives a failed outcome with ``phase="validate"`` (nothing ran).
        """
        if self._refusal is not None:
            return self._closed(cell_id)
        cell = self._cells.get(cell_id)
        if cell is None or not cell.rerunnable or cell.code is None:
            reason = "is not a cell of this session" if cell is None else "did not run ok the first time"
            return CellOutcome(ok=False, phase="validate", cell_id=cell_id,
                               error=f"cell {cell_id} cannot be re-run: it {reason}")
        self._captured.clear()
        try:
            return self._exec(cell_id, cell.code)
        except RUNTIME_FAILURES as e:
            return self._failure(cell_id, e)

    def eval_expr(self, source: str) -> ExprOutcome:
        """Validate ``source`` as one expression line after the program, then evaluate it.

        The value is returned as it is: no data filter (the host-side display formats
        it). Text the expression prints is discarded. Nothing is raised: a rejected
        source gives ``phase="validate"``, an exception ``phase="runtime"``, and a
        refusal ``phase="refused"``, which closes the session (ruling, review M2).
        """
        if self._refusal is not None:
            return ExprOutcome(ok=False, phase="refused", error=self._closed_text())
        try:
            return self._eval_expr(source)
        except LackpyRefusal as e:
            self._refusal = error_text(e)
            return ExprOutcome(ok=False, phase="refused", error=self._refusal)
        except Exception as e:
            return ExprOutcome(ok=False, phase="runtime", error=error_text(e))
        finally:
            self._captured.clear()

    # -- internals -------------------------------------------------------------

    def _run_cell(self, cell_id: int, source: str) -> CellOutcome:
        self._cells[cell_id] = _Cell(code=None)
        # Each cell must parse alone: joined, ``else:`` or an indented line could
        # continue an earlier cell's block, and exec runs this cell alone.
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return CellOutcome(ok=False, phase="validate", cell_id=cell_id,
                               error=f"Parse error: {e.msg} (line {e.lineno})")
        error = self._violations(source)
        if error is not None:
            return CellOutcome(ok=False, phase="validate", error=error, cell_id=cell_id)
        code = compile(tree, "<lackpy>", "exec")
        self._cells[cell_id].code = code
        self._join(cell_id, source)
        outcome = self._exec(cell_id, code)
        self._cells[cell_id].rerunnable = outcome.ok
        return outcome

    def _exec(self, cell_id: int, code: CodeType) -> CellOutcome:
        # References, not ids: an id can be reused once its object is freed.
        before = dict(self._namespace)
        try:
            _run_validated_code(code, self._namespace)
        except RUNTIME_FAILURES as e:
            return self._failure(cell_id, e)
        delta = data_variables(
            self._namespace,
            lambda k: k not in self._runtime_names
            and (k not in before or before[k] is not self._namespace[k]))
        return CellOutcome(ok=True, stdout="".join(self._captured), delta=delta, cell_id=cell_id)

    def _eval_expr(self, source: str) -> ExprOutcome:
        if "\n" in source or "\r" in source:
            return ExprOutcome(ok=False, phase="validate", error="an expression must be one line")
        try:
            tree = ast.parse(source, mode="eval")
        except SyntaxError as e:
            return ExprOutcome(ok=False, phase="validate", error=f"Parse error: {e.msg} (line 1)")
        error = self._violations(source)
        if error is not None:
            return ExprOutcome(ok=False, phase="validate", error=error)
        code = compile(tree, "<lackpy>", "eval")
        # Validated above as part of the program; the namespace is the restricted one.
        return ExprOutcome(ok=True, value=eval(code, self._namespace))  # noqa: S307

    def _failure(self, cell_id: int, e: BaseException) -> CellOutcome:
        text = error_text(e)
        if isinstance(e, LackpyRefusal):
            self._refusal = text
            phase: Phase = "refused"
        else:
            phase = "runtime"
        return CellOutcome(ok=False, stdout="".join(self._captured), error=text,
                           phase=phase, cell_id=cell_id)

    def _closed_text(self) -> str:
        return f"the session ended at an earlier refusal: {self._refusal}"

    def _closed(self, cell_id: int) -> CellOutcome:
        return CellOutcome(ok=False, phase="refused", error=self._closed_text(), cell_id=cell_id)

    def _join(self, cell_id: int, source: str) -> None:
        self._starts.append(self._lines + 1)
        self._ids.append(cell_id)
        self._program = f"{self._program}\n{source}" if self._ids[:-1] else source
        self._lines += source.count("\n") + 1

    def _where(self, line: int) -> tuple[int, int]:
        """The (cell id, cell line) of a program line."""
        i = bisect.bisect_right(self._starts, line) - 1
        return self._ids[i], line - self._starts[i] + 1

    def _violations(self, source: str) -> str | None:
        """Validate the program extended by ``source``; return the error text, or None.

        Every violation counts (the program alone has none; see the module text).
        Line numbers in ``source`` become its own; a line of an earlier cell is
        named by cell id and line, and noted as made invalid by this cell.
        Hints are dictionary text and are left as they are.
        """
        offset = self._lines
        program = f"{self._program}\n{source}" if self._ids else source
        verdict = validate(program, allowed_names=set())
        if not verdict.violations:
            return None

        def local(n: int) -> str:
            if n > offset:
                return f"line {n - offset}"
            cell, line = self._where(n)
            return f"line {line} of cell {cell}"

        def renumber(m: re.Match[str]) -> str:
            return m.group(0) if m.group(1) is None else local(int(m.group(1)))

        parts = []
        for v in verdict.violations:
            # The format of ``service.validation_error`` (message, then the hint), which
            # run/ cannot import: the service module pulls in inference and config.
            text = _LINE.sub(renumber, v.message)
            if v.line is not None and v.line <= offset:
                # The message names the line (renumbered above), or the note does.
                where = "" if f"{local(v.line)}" in text else f"{local(v.line)}, "
                text += f" [{where}an earlier cell's line, made invalid by this cell]"
            parts.append(f"{text} ({v.hint})" if v.hint else text)
        return f"Validation failed: {'; '.join(parts)}"
