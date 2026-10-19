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
from collections.abc import Callable, Iterable
from typing import Any, Literal

from ..lang import validate
from .namespace import build_namespace
from .paths import PathPolicy
from .refusal import LackpyRefusal
from .runner import RUNTIME_FAILURES, _is_data, _run_validated_code, data_variables, make_print_capture
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
        line: For a ``"runtime"`` failure, the cell's top-level line that was
            running when it raised (the literate kernel uses it to tell which
            top-level statements completed); else None.
        rebound: Every name the run bound or rebound (by identity, any value),
            with the name of its value's type. Also for a run that failed part
            way. The values themselves are not given (ruling N1).
    """

    ok: bool
    stdout: str = ""
    error: str | None = None
    phase: Phase | None = None
    delta: dict[str, Any] = field(default_factory=dict)
    cell_id: int | None = None
    line: int | None = None
    rebound: dict[str, str] = field(default_factory=dict)


@dataclass
class ExprOutcome:
    """The result of :meth:`CellSession.eval_expr`.

    ``value`` is data (``runner._is_data``) as it is, or else its ``repr`` text,
    as ``f"{...}"`` would show it under python3: a lazy value is never consumed.
    A ``repr`` text has no `` at 0x...`` address (review M7). ``stdout`` is the
    text the expression printed (review M2).
    """

    ok: bool
    value: Any = None
    error: str | None = None
    phase: Phase | None = None
    stdout: str = ""


@dataclass
class _Cell:
    code: CodeType | None  # None: the cell failed validation and is not in the program


# A validator line reference is ``line N`` outside quoted program text. Rules quote
# program text as '<text>' just before " at line N" (``ast.unparse`` output can hold
# quotes of its own), so that span is matched first and kept as it is (review M4).
_LINE = re.compile(r"'[^\n]*?'(?= at line \d)|\bline (\d+)\b")

# A host address in a default repr (``<function f at 0x7f...>``): dropped so renders
# are deterministic (review M7).
_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")

_TYPE_NAME = type.__dict__["__name__"].__get__


class CellSession:
    """One literate document's namespace, and its program."""

    def __init__(self, *, path_policy: PathPolicy | None, trace: Trace | None = None,
                 program_file: str | None = None) -> None:
        self.trace = trace if trace is not None else Trace()
        self._path_policy = path_policy
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

    def data(self, names: Iterable[str]) -> dict[str, Any]:
        """The values of ``names`` (bound names only) that are data (``runner._is_data``).

        The values are the namespace's own objects, not copies, so an in-place
        change made by a later cell shows in them. A value that is not data
        (a function, a module, a ``Path``) is left out: the host never holds a
        live object (ruling N1). Only the names asked for are walked (review M3).
        """
        ns = self._namespace
        out: dict[str, Any] = {}
        for k in names:
            if k in ns and k not in self._runtime_names and _is_data(ns[k]):
                out[k] = ns[k]
        return out

    @property
    def path_policy(self) -> PathPolicy | None:
        """The ``PathPolicy`` the session's ``Path`` stand-in checks against."""
        return self._path_policy

    @property
    def refusal(self) -> str | None:
        """The first refusal's text, or None. Once set, the session is closed."""
        return self._refusal

    @property
    def closed(self) -> bool:
        """Whether a refusal closed the session (every later call refuses)."""
        return self._refusal is not None

    def closed_text(self) -> str | None:
        """The error every call gives once the session is closed, or None while it is open."""
        return None if self._refusal is None else self._closed_text()

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
        """Exec the stored code of a cell in the program, again.

        A cell is in the program when it passed validation: its first run was ok
        or failed at runtime (the forgiveness case: re-run once a dependency
        changes). The code is not validated again and not added to the program
        again: the program text does not change. An unknown cell, or one that
        failed validation, gives a failed outcome with ``phase="validate"``
        (nothing ran). After a refusal the session is closed, so a refused cell
        gives ``phase="refused"``.
        """
        if self._refusal is not None:
            return self._closed(cell_id)
        cell = self._cells.get(cell_id)
        if cell is None or cell.code is None:
            reason = "is not a cell of this session" if cell is None else "failed validation"
            return CellOutcome(ok=False, phase="validate", cell_id=cell_id,
                               error=f"cell {cell_id} cannot be re-run: it {reason}")
        self._captured.clear()
        try:
            return self._exec(cell_id, cell.code)
        except RUNTIME_FAILURES as e:
            return self._failure(cell_id, e)

    def eval_expr(self, source: str) -> ExprOutcome:
        """Validate ``source`` as one expression line after the program, then evaluate it.

        A data value is returned as it is; any other value as its ``repr`` text, made
        here, so the host never holds a live object (ruling N1). The host-side display
        formats the result. Text the expression prints is in ``stdout``. Nothing is raised: a rejected
        source gives ``phase="validate"``, an exception ``phase="runtime"``, and a
        refusal ``phase="refused"``, which closes the session (ruling, review M2).
        """
        if self._refusal is not None:
            return ExprOutcome(ok=False, phase="refused", error=self._closed_text())
        self._captured.clear()
        try:
            return self._eval_expr(source)
        except LackpyRefusal as e:
            self._refusal = error_text(e)
            return ExprOutcome(ok=False, phase="refused", error=self._refusal,
                               stdout="".join(self._captured))
        except Exception as e:
            return ExprOutcome(ok=False, phase="runtime", error=_runtime_text(e),
                               stdout="".join(self._captured))
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
        return self._exec(cell_id, code)

    def _exec(self, cell_id: int, code: CodeType) -> CellOutcome:
        # References, not ids: an id can be reused once its object is freed.
        before = dict(self._namespace)

        def changed(k: str) -> bool:
            return (k not in self._runtime_names
                    and (k not in before or before[k] is not self._namespace[k]))

        try:
            _run_validated_code(code, self._namespace)
        except RUNTIME_FAILURES as e:
            outcome = self._failure(cell_id, e)
            outcome.rebound = self._rebound(changed)
            return outcome
        delta = data_variables(self._namespace, changed)
        return CellOutcome(ok=True, stdout="".join(self._captured), delta=delta, cell_id=cell_id,
                           rebound=self._rebound(changed))

    def _rebound(self, changed: Callable[[str], bool]) -> dict[str, str]:
        # The type's own __name__ descriptor: no metaclass code runs on the host.
        return {k: _TYPE_NAME(type(v)) for k, v in self._namespace.items() if changed(k)}

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
        value = eval(code, self._namespace)  # noqa: S307
        if not _is_data(value):
            # Ruling N1: no live callable, iterator or stand-in reaches the host, where
            # consuming it would run cell code outside this guard. repr runs here, so a
            # refusal it raises closes the session (eval_expr's handlers).
            value = _ADDRESS.sub("", repr(value))
        return ExprOutcome(ok=True, value=value, stdout="".join(self._captured))

    def _failure(self, cell_id: int, e: BaseException) -> CellOutcome:
        line = None
        if isinstance(e, LackpyRefusal):
            text = error_text(e)
            self._refusal = text
            phase: Phase = "refused"
        else:
            text = _runtime_text(e)
            phase = "runtime"
            line = _cell_line(e)
        return CellOutcome(ok=False, stdout="".join(self._captured), error=text,
                           phase=phase, cell_id=cell_id, line=line)

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


def _runtime_text(e: BaseException) -> str:
    """``Type: message``, as python3 shows an exception: ``str(KeyError('k'))`` alone is ``'k'``."""
    return f"{type(e).__name__}: {error_text(e)}"


def _cell_line(e: BaseException) -> int | None:
    """The line of the shallowest ``<lackpy>`` frame: the cell's top-level statement that raised."""
    tb = e.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == "<lackpy>":
            return tb.tb_lineno
        tb = tb.tb_next
    return None
