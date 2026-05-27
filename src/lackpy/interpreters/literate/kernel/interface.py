"""Kernel interface protocol and result types.

KernelInterface defines the contract that LightweightKernel implements.
CellResult carries success/failure, captured output, and the namespace
delta (new or changed variables) for each cell execution.

Called by: StreamingDriver._execute_cells(), LiterateInterpreter.execute()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ...base import IncrementalInterpreter
from ..parser import Cell


@dataclass
class CellResult:
    success: bool
    output: str | None
    error: str | None
    error_phase: str | None  # "static" | "runtime"
    namespace_delta: dict[str, Any]
    cell_index: int


class KernelInterface(IncrementalInterpreter, Protocol):
    """Literate's kernel contract — a structural EXTENSION of the runtime
    :class:`~lackpy.interpreters.base.IncrementalInterpreter` (RFC 0001 Addendum E).

    The runtime base owns the incremental core (``execute_cell`` + ``get_namespace``);
    literate adds the kernel-only affordances (``inspect``/``get_scope``/``restart``) the
    streaming driver and recovery handler use. Declaring the inheritance makes the lift
    explicit rather than coincidental — ``LightweightKernel`` is one concrete
    ``IncrementalInterpreter``, and the one-shot ``Interpreter`` is "run the whole program
    as one cell" (see ``tests/interpreters/test_composite.py``)."""

    def execute_cell(self, cell: Cell, cell_index: int) -> CellResult: ...
    def inspect(self, expr: str) -> str: ...
    def get_scope(self) -> dict[str, str]: ...
    def restart(self) -> None: ...
    def get_namespace(self) -> dict[str, Any]: ...
