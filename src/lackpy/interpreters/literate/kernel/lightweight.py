"""Lightweight kernel: literate cells run through one ``CellSession`` (literate stepy D1-D5).

For each cell, the kernel: (1) looks up the compiler for the cell type in
compiler._COMPILERS, (2) runs static_analysis.check_cell() to catch syntax
errors and undefined names before evaluation, (3) runs the cell through the
document's :class:`~lackpy.run.cells.CellSession`, which validates it as an
extension of the program so far and execs it in the restricted namespace.

There is no other way in. The kernel holds no namespace of its own, execs
nothing and evals nothing: a cell can do nothing a lackpy program cannot.

* ``@diff`` runs on the host side (D6): ``session.edit_text`` reads the file
  through the session's ``Path`` stand-in, the patch is applied here in pure
  Python, and the result is written through the same stand-in.
* Code, ``@hidden``, ``@gather``, ``@read`` and ``@write`` run with
  ``session.run_cell``; a re-run (reactive re-execution) uses
  ``session.rerun_cell`` with the cell's id (:meth:`rerun_cell`).
* Prose (D3) is text and ``{expr}`` segments. A text segment renders as it
  is; each expression is evaluated with ``session.eval_expr`` and formatted
  here, on the host side. Prose never joins the program.
* ``@scratch`` runs its body as a cell; the ``[scratch: ...]`` summary is
  made here from the names the body bound.
* ``@continue`` runs nothing: it only asks for a pause.

A refusal (D4) comes back as ``error_phase="refused"``. The session is then
closed, so every later cell refuses too; the caller stops the document.

Forgiveness values (``Hole``, ``ErrorValue``, ``Unavailable``) are bound by
the runner, not by cell code (:meth:`bind`). They live in a host-side layer
over the session's names and never enter the session's namespace: a cell
that refers to one is skipped by the forgiveness pre-pass (D5), and a cell
that binds the name again replaces it.

Called by: StreamingDriver, LiterateInterpreter.execute()
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from typing import Any

from lackpy.run.cells import CellOutcome, CellSession

from ..compiler import _COMPILERS, prose_segments, without_path_import
from ..display import DEFAULT_DISPLAY_THRESHOLD, make_display
from ..parser import Cell
from ..patch import apply_patch
from .forgiveness import is_forgiving
from .interface import CellResult
from .static_analysis import StaticAnalysisError, check_cell, collect_bindings
from .versions import Opaque

class LightweightKernel:
    def __init__(self, session: CellSession, *,
                 display_threshold: int = DEFAULT_DISPLAY_THRESHOLD) -> None:
        self._session = session
        self._display = make_display(display_threshold)
        # Reified forgiveness values, by name (see the module text). Never
        # passed to the session.
        self._reified: dict[str, Any] = {}
        # The current Opaque marker of each name bound to a value that is not
        # data (review I2). A rebind gets a new marker.
        self._markers: dict[str, Opaque] = {}
        # The session cell ids that are in the program (validated), so a
        # re-run can use ``rerun_cell``.
        self._in_program: set[int] = set()
        # L7 fail-closed backstop: a failed startup self-test disables the
        # kernel (see ..selftest.run_selftest).  While disabled, every
        # execute_cell/inspect call returns a structured refusal instead of
        # running — on EVERY path that holds this kernel, even one that
        # ignores the self-test report.
        self._disabled: str | None = None

    @property
    def session(self) -> CellSession:
        """The document's cell runtime."""
        return self._session

    def disable(self, reason: str) -> None:
        """Disable execution permanently (fail-closed).  ``reason`` is the
        legible explanation every refused call carries."""
        self._disabled = reason

    @property
    def disabled(self) -> str | None:
        """The disable reason, or ``None`` while the kernel accepts work."""
        return self._disabled

    @property
    def refusal(self) -> str | None:
        """The refusal that closed the session (D4), or ``None``."""
        return self._session.refusal

    # -- running cells ---------------------------------------------------------

    def execute_cell(self, cell: Cell, cell_index: int) -> CellResult:
        """Run ``cell`` for the first time: a program cell joins the session's program."""
        return self._execute(cell, cell_index, rerun_id=None)

    def rerun_cell(self, cell: Cell, cell_index: int, cell_id: int | None) -> CellResult:
        """Run ``cell`` again after a dependency changed (reactive re-execution).

        A program cell that is in the session's program (``cell_id`` from its
        first run) runs its validated code again with ``session.rerun_cell``:
        the program text does not change. A cell that never joined the program
        (it was skipped for a hole, or failed validation) runs as a new cell.
        Prose is evaluated again; it was never in the program.
        """
        rerun = cell_id if cell_id in self._in_program else None
        return self._execute(cell, cell_index, rerun_id=rerun)

    def _execute(self, cell: Cell, cell_index: int, rerun_id: int | None) -> CellResult:
        if self._disabled is not None:
            return _failed(cell_index, f"kernel disabled: {self._disabled}", "disabled")
        closed = self._session.closed_text()
        if closed is not None:
            # Review I1: the latch holds at every entry point, also for a cell
            # that would never call the session (prose text, @continue).
            return _failed(cell_index, closed, "refused")
        compiler = _COMPILERS.get(cell.cell_type)
        if compiler is None:
            return _failed(cell_index, f"Unknown cell type: {cell.cell_type}", "static")
        if cell.cell_type == "continue":
            return CellResult(success=True, output="", error=None, error_phase=None,
                              namespace_delta={"__continue_requested__": True},
                              cell_index=cell_index)

        if cell.cell_type == "diff":
            return self._apply_diff(cell, cell_index)

        source = compiler(cell)
        if cell.cell_type in ("read", "write") and "Path" in self._session.bound_names():
            # A second import of Path is a rebinding the validator rejects.
            source = without_path_import(source)
        if cell.cell_type == "prose":
            try:
                check_cell(source, self.known_names())
            except StaticAnalysisError as e:
                return _failed(cell_index, str(e), "static")
            return self._render_prose(cell, cell_index)

        if not source.strip() and cell.cell_type != "scratch":
            return CellResult(success=True, output="", error=None, error_phase=None,
                              namespace_delta={}, cell_index=cell_index)
        try:
            check_cell(source, self.known_names())
        except StaticAnalysisError as e:
            return _failed(cell_index, str(e), "static")
        return self._run_program_cell(cell, cell_index, source, rerun_id)

    def _run_program_cell(self, cell: Cell, cell_index: int, source: str,
                          rerun_id: int | None) -> CellResult:
        names_before = self._session.bound_names()
        if not source.strip():
            outcome = CellOutcome(ok=True)  # an empty @scratch body: nothing to run
        elif rerun_id is not None:
            outcome = self._session.rerun_cell(rerun_id)
        else:
            outcome = self._session.run_cell(source)
        if outcome.cell_id is not None and outcome.phase in (None, "runtime"):
            self._in_program.add(outcome.cell_id)

        # The names this run bound or rebound (review I2): the session reports
        # them by identity, with their type names. A data value goes in the
        # delta as it is; any other value as a fresh Opaque marker, so a def or
        # a Path supersedes, dirties and fills a hole like data does.
        data = self._session.data(outcome.rebound)
        delta: dict[str, Any] = {}
        for name, type_name in outcome.rebound.items():
            if name in data:
                self._markers.pop(name, None)
                delta[name] = data[name]
            else:
                marker = Opaque(type_name)
                self._markers[name] = marker
                delta[name] = marker
            self._reified.pop(name, None)
        if outcome.ok:
            # A cell that ran binds its module-level names again (also to the
            # identical object, which identity cannot see): an earlier reified
            # value for any of them is replaced.
            for name in _static_bindings(source):
                self._reified.pop(name, None)

        if not outcome.ok:
            return CellResult(success=False, output=outcome.stdout or None,
                              error=outcome.error, error_phase=outcome.phase,
                              namespace_delta={}, cell_index=cell_index,
                              error_lineno=outcome.line, cell_id=outcome.cell_id)

        output = outcome.stdout
        if cell.cell_type == "scratch":
            new = sorted(n for n in outcome.rebound if n not in names_before)
            parts = [f"{n}={outcome.rebound[n]}" for n in new]
            output += f"[scratch: {', '.join(parts)}]\n"
        return CellResult(success=True, output=output, error=None, error_phase=None,
                          namespace_delta=delta, cell_index=cell_index,
                          cell_id=outcome.cell_id)

    def _apply_diff(self, cell: Cell, cell_index: int) -> CellResult:
        # D6: host side. The session reads and writes the file through its own
        # Path stand-in (the policy checks both); the patch is pure Python here.
        path = cell.annotation_args.get("path", "")
        outcome = self._session.edit_text(path, lambda text: apply_patch(text, cell.content))
        if not outcome.ok:
            return _failed(cell_index, outcome.error or "the diff failed",
                           outcome.phase or "runtime")
        return CellResult(success=True, output="", error=None, error_phase=None,
                          namespace_delta={}, cell_index=cell_index)

    def _render_prose(self, cell: Cell, cell_index: int) -> CellResult:
        # As python3's print(f"...") does: what an expression prints comes out
        # while the line is built, so before the line itself (review M2).
        printed: list[str] = []
        parts: list[str] = []
        for seg in prose_segments(cell):
            if seg.expr is None:
                parts.append(seg.text or "")
                continue
            out = self._session.eval_expr(seg.expr)
            printed.append(out.stdout)
            if not out.ok:
                return _failed(cell_index, out.error or "evaluation failed",
                               out.phase or "runtime", output="".join(printed) or None)
            parts.append(self._display(out.value) if seg.display else str(out.value))
        return CellResult(success=True, output="".join(printed) + "".join(parts) + "\n",
                          error=None, error_phase=None, namespace_delta={},
                          cell_index=cell_index)

    # -- the host's view of the names --------------------------------------------

    def known_names(self) -> set[str]:
        """The name set a cell's static analysis resolves against — exactly
        the ``known_names`` :meth:`execute_cell` passes to ``check_cell``:
        the session's bound names and the reified ones. Used by the batch
        path's forgiveness pre-pass (L1.1) so its undefined-name detection can
        never disagree with the kernel's own check."""
        return self._session.bound_names() | set(self._reified)

    def lookup(self, name: str, default: Any = None) -> Any:
        """The value bound to ``name``: a reified value, a data value, or the
        name's :class:`Opaque` marker when its value is not data (the host
        never holds a live object)."""
        if name in self._reified:
            return self._reified[name]
        if name in self._markers:
            return self._markers[name]
        return self._session.data([name]).get(name, default)

    def is_reified(self, name: str) -> bool:
        """Whether ``name`` holds a reified forgiveness value (no data walk)."""
        return name in self._reified

    def bind(self, name: str, value: Any) -> None:
        """Bind a reified forgiveness value (``Hole``, ``ErrorValue``, ``Unavailable``).

        The forgiveness reification path (L1.1/L1.2/L1.5): holes and error
        values are bound *by the runner*, not by executing cell code. They stay
        on the host side; the session never sees them. Any other value is
        refused: nothing but a cell binds a name in the session."""
        if not is_forgiving(value):
            raise TypeError(f"the kernel binds only reified values, not {type(value).__name__}")
        self._reified[name] = value

    def inspect(self, expr: str) -> str:
        """Evaluate ``expr`` in the session (``eval_expr``) and return its repr, or the error."""
        if self._disabled is not None:
            return f"kernel disabled: {self._disabled}"
        closed = self._session.closed_text()
        if closed is not None:
            return closed
        out = self._session.eval_expr(expr)
        if not out.ok:
            return out.error or "evaluation failed"
        return repr(out.value)

    def get_scope(self) -> dict[str, str]:
        scope: dict[str, str] = {}
        names = self._session.bound_names() | set(self._reified)
        data = self._session.data(names)
        for k in sorted(names):
            if k.startswith("_"):
                continue
            if k in self._reified:
                v = self._reified[k]
            elif k in data:
                v = data[k]
            elif k in self._markers:
                scope[k] = f"{self._markers[k].type_name}: (not data)"
                continue
            else:
                scope[k] = "object: (not data)"
                continue
            scope[k] = f"{type(v).__name__}: {repr(v)[:80]}"
        return scope

    def get_namespace(self, names: Iterable[str] | None = None) -> dict[str, Any]:
        """The data values and reified values by name (no live objects).

        ``names`` limits the result (and the data walk, review M3) to those names."""
        wanted = self._session.bound_names() if names is None else set(names)
        ns = {k: v for k, v in self._session.data(wanted).items() if not k.startswith("_")}
        ns.update({k: v for k, v in self._reified.items() if names is None or k in wanted})
        return ns


def _static_bindings(source: str) -> set[str]:
    try:
        return collect_bindings(ast.parse(source))
    except SyntaxError:
        return set()


def _failed(cell_index: int, error: str, phase: str, output: str | None = None) -> CellResult:
    return CellResult(success=False, output=output, error=error, error_phase=phase,
                      namespace_delta={}, cell_index=cell_index)
