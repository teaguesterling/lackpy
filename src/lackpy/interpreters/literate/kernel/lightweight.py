"""Lightweight kernel: exec-into-dict with static analysis.

For each cell, the kernel: (1) looks up the compiler for the cell type
in compiler._COMPILERS, (2) runs static_analysis.check_cell() to catch
syntax errors and undefined names before evaluation, (3) evaluates the
compiled Python in a persistent namespace dict with stdout captured.

The namespace persists across all cells — variables defined in any cell
(including @gather and @hidden) are available to all later cells. This
is what makes prose interpolation work: {variable} in prose compiles to
an f-string expression evaluated against the same namespace.

Called by: StreamingDriver, LiterateInterpreter.execute()
"""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

from ..compiler import _COMPILERS
from ..parser import Cell
from .interface import CellResult
from .static_analysis import StaticAnalysisError, check_cell

_CONTINUE_FLAG = "__literate_continue_called__"


def _continue_sentinel() -> None:
    """Placeholder called by @continue cells; kernel detects and sets flag."""
    pass


class LightweightKernel:
    def __init__(self, namespace: dict[str, Any] | None = None) -> None:
        self._namespace: dict[str, Any] = namespace or {}
        self._initial_keys: set[str] = set(self._namespace.keys())
        if "__builtins__" not in self._namespace:
            import builtins
            self._namespace["__builtins__"] = builtins

    def execute_cell(self, cell: Cell, cell_index: int) -> CellResult:
        compiler = _COMPILERS.get(cell.cell_type)
        if compiler is None:
            return CellResult(
                success=False,
                output=None,
                error=f"Unknown cell type: {cell.cell_type}",
                error_phase="static",
                namespace_delta={},
                cell_index=cell_index,
            )

        compiled_source = compiler(cell)
        if not compiled_source.strip():
            return CellResult(
                success=True,
                output="",
                error=None,
                error_phase=None,
                namespace_delta={},
                cell_index=cell_index,
            )

        # For @continue cells, inject the sentinel function so static analysis passes
        is_continue = cell.cell_type == "continue"
        if is_continue:
            self._namespace["__literate_continue__"] = _continue_sentinel

        known_names = set(self._namespace.keys()) - {"__builtins__"}
        try:
            check_cell(compiled_source, known_names)
        except StaticAnalysisError as e:
            if is_continue:
                del self._namespace["__literate_continue__"]
            return CellResult(
                success=False,
                output=None,
                error=str(e),
                error_phase="static",
                namespace_delta={},
                cell_index=cell_index,
            )

        before_snapshot = {
            k: v for k, v in self._namespace.items()
            if k != "__builtins__"
        }

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        try:
            code_obj = compile(compiled_source, "<cell>", "exec")
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                _do_exec(code_obj, self._namespace)
        except BaseException as e:
            if isinstance(e, KeyboardInterrupt):
                raise
            captured = stdout_capture.getvalue()
            err_out = stderr_capture.getvalue()
            if err_out:
                captured = (captured + "\n" + err_out).strip()
            return CellResult(
                success=False,
                output=captured or None,
                error=f"{type(e).__name__}: {e}",
                error_phase="runtime",
                namespace_delta={},
                cell_index=cell_index,
            )
        finally:
            if is_continue and "__literate_continue__" in self._namespace:
                del self._namespace["__literate_continue__"]

        # Identity-based: in-place mutations (e.g. list.append) won't appear
        delta: dict[str, Any] = {}
        for k, v in self._namespace.items():
            if k == "__builtins__" or k.startswith("_"):
                continue
            if k not in before_snapshot or before_snapshot[k] is not v:
                delta[k] = v

        if is_continue:
            delta["__continue_requested__"] = True

        output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()
        if stderr_output:
            output = (output + "\n" + stderr_output).strip() if output else stderr_output

        return CellResult(
            success=True,
            output=output,
            error=None,
            error_phase=None,
            namespace_delta=delta,
            cell_index=cell_index,
        )

    def inspect(self, expr: str) -> str:
        try:
            result = eval(expr, self._namespace)  # noqa: S307
            return repr(result)
        except Exception as e:
            return f"{type(e).__name__}: {e}"

    def get_scope(self) -> dict[str, str]:
        scope: dict[str, str] = {}
        for k, v in self._namespace.items():
            if k.startswith("_") or k == "__builtins__":
                continue
            if callable(v) and k in self._initial_keys:
                continue
            scope[k] = f"{type(v).__name__}: {repr(v)[:80]}"
        return scope

    def restart(self) -> None:
        tools = {k: v for k, v in self._namespace.items() if k in self._initial_keys}
        self._namespace.clear()
        self._namespace.update(tools)
        import builtins
        self._namespace["__builtins__"] = builtins

    def snapshot(self) -> dict[str, Any]:
        """Shallow copy of the current namespace, for restore() after a failed
        step. Captures name REBINDINGS only -- in-place mutations of held objects
        are not recorded (consistent with this kernel's own delta tracking, which
        also can't see in-place changes). A deep copy is deliberately avoided
        (arbitrary/unpicklable live objects, cost)."""
        return dict(self._namespace)

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Restore the namespace to a prior snapshot() -- undoes name rebindings
        made since the snapshot. In-place mutations are not reverted."""
        self._namespace.clear()
        self._namespace.update(snapshot)

    def get_namespace(self) -> dict[str, Any]:
        return {
            k: v for k, v in self._namespace.items()
            if k != "__builtins__" and not k.startswith("_")
        }


def _do_exec(code: object, ns: dict) -> None:
    exec(code, ns)  # noqa: S102
