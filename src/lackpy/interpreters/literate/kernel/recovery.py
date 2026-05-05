"""Recovery handler protocol and built-in handlers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..parser import Cell
from .plugins import PluginAdvice


@dataclass
class RecoveryAction:
    kind: str  # "fix" | "inspect" | "skip" | "abort"
    cells: list[Cell] | None = None
    expr: str | None = None
    target_index: int | None = None


@dataclass
class RecoveryContext:
    failed_cell: Cell
    error: str
    error_phase: str
    scope: dict[str, str]
    cell_index: int
    prior_output: str
    attempt: int
    plugin_advice: PluginAdvice | None = None


class RecoveryHandler(Protocol):
    max_attempts: int

    def on_cell_error(self, ctx: RecoveryContext) -> RecoveryAction: ...
    def on_inspect_result(self, ctx: RecoveryContext, result: str) -> RecoveryAction: ...


class NoRecoveryHandler:
    max_attempts: int = 0

    def on_cell_error(self, ctx: RecoveryContext) -> RecoveryAction:
        return RecoveryAction(kind="abort")

    def on_inspect_result(self, ctx: RecoveryContext, result: str) -> RecoveryAction:
        return RecoveryAction(kind="abort")
