"""Execution plugin protocol and advice types.

Plugins observe cell execution and provide advice on errors. The
StreamingDriver calls plugin hooks at each lifecycle point:
  on_cell_start → on_cell_success or on_cell_error → on_recovery_result

On error, plugins return PluginAdvice (hints, doc_context, suggestion)
which the driver merges and passes to the RecoveryHandler. This is how
coaching systems like Kibitzer can influence recovery without owning it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..parser import Cell
from .interface import CellResult


@dataclass
class PluginAdvice:
    hints: list[str] = field(default_factory=list)
    doc_context: list[str] = field(default_factory=list)
    suggestion: str | None = None


class ExecutionPlugin(Protocol):
    def on_cell_start(self, cell: Cell, index: int) -> None: ...
    def on_cell_success(self, cell: Cell, result: CellResult) -> None: ...
    def on_cell_error(self, cell: Cell, error: str, scope: dict) -> PluginAdvice: ...
    def on_recovery_result(self, cell: Cell, success: bool, attempt: int) -> None: ...


def merge_advice(advices: list[PluginAdvice]) -> PluginAdvice:
    if not advices:
        return PluginAdvice()
    hints: list[str] = []
    doc_context: list[str] = []
    suggestion: str | None = None
    for a in advices:
        hints.extend(a.hints)
        doc_context.extend(a.doc_context)
        if suggestion is None and a.suggestion is not None:
            suggestion = a.suggestion
    return PluginAdvice(hints=hints, doc_context=doc_context, suggestion=suggestion)
