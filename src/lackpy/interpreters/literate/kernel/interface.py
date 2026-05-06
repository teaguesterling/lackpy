"""Kernel interface protocol and result types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ..parser import Cell


@dataclass
class CellResult:
    success: bool
    output: str | None
    error: str | None
    error_phase: str | None  # "static" | "runtime"
    namespace_delta: dict[str, Any]
    cell_index: int


class KernelInterface(Protocol):
    def execute_cell(self, cell: Cell, cell_index: int) -> CellResult: ...
    def inspect(self, expr: str) -> str: ...
    def get_scope(self) -> dict[str, str]: ...
    def restart(self) -> None: ...
    def get_namespace(self) -> dict[str, Any]: ...
