"""Streaming driver - orchestrates parse, kernel, recovery, plugins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..parser import Cell
from .interface import CellResult, KernelInterface
from .plugins import PluginAdvice, merge_advice
from .recovery import NoRecoveryHandler, RecoveryAction, RecoveryContext, RecoveryHandler
from .streaming_parser import StreamingCellParser


@dataclass
class CellExecutionEvent:
    cell: Cell
    cell_index: int
    result: CellResult | None
    status: str
    recovery_attempts: int = 0
    generation: int = 0


class StreamingDriver:
    def __init__(
        self,
        kernel: KernelInterface,
        recovery: RecoveryHandler | None = None,
        plugins: list[Any] | None = None,
    ) -> None:
        self._kernel = kernel
        self._recovery = recovery or NoRecoveryHandler()
        self._plugins: list[Any] = plugins or []
        self._parser = StreamingCellParser()
        self._log: list[CellExecutionEvent] = []
        self._output_parts: list[str] = []
        self._cell_counter: int = 0
        self._generation: int = 0
        self._interrupted: bool = False
        self._continue_requested: bool = False

    @property
    def execution_log(self) -> list[CellExecutionEvent]:
        return list(self._log)

    @property
    def rendered_output(self) -> str:
        return "".join(self._output_parts)

    @property
    def generation(self) -> int:
        return self._generation

    def interrupt(self) -> None:
        self._interrupted = True

    def resume(self) -> None:
        self._continue_requested = False
        self._interrupted = False
        self._generation += 1
        self._parser.reset()

    async def feed(self, chunk: str) -> list[CellExecutionEvent]:
        if self._interrupted or self._continue_requested:
            return []
        cells = self._parser.feed(chunk)
        return await self._execute_cells(cells)

    async def flush(self) -> list[CellExecutionEvent]:
        if self._interrupted or self._continue_requested:
            return []
        cells = self._parser.flush()
        return await self._execute_cells(cells)

    async def _execute_cells(self, cells: list[Cell]) -> list[CellExecutionEvent]:
        events: list[CellExecutionEvent] = []
        for cell in cells:
            if self._interrupted or self._continue_requested:
                event = CellExecutionEvent(
                    cell=cell,
                    cell_index=self._cell_counter,
                    result=None,
                    status="pending",
                    generation=self._generation,
                )
                self._log.append(event)
                events.append(event)
                self._cell_counter += 1
                continue

            index = self._cell_counter
            self._cell_counter += 1

            self._notify_start(cell, index)
            result = self._kernel.execute_cell(cell, index)

            if result.success:
                if result.namespace_delta.get("__continue_requested__"):
                    event = CellExecutionEvent(
                        cell=cell,
                        cell_index=index,
                        result=result,
                        status="continue_requested",
                        generation=self._generation,
                    )
                    self._log.append(event)
                    events.append(event)
                    self._continue_requested = True
                    continue

                if result.output:
                    self._output_parts.append(result.output)
                self._notify_success(cell, result)
                event = CellExecutionEvent(
                    cell=cell,
                    cell_index=index,
                    result=result,
                    status="executed",
                    generation=self._generation,
                )
                self._log.append(event)
                events.append(event)
            else:
                event = await self._handle_failure(cell, index, result)
                self._log.append(event)
                events.append(event)
                if event.status not in ("recovered", "skipped"):
                    self._interrupted = True

        return events

    async def _handle_failure(
        self, cell: Cell, index: int, result: CellResult
    ) -> CellExecutionEvent:
        advice = self._collect_advice(cell, result.error or "")
        attempt = 0

        while attempt <= self._recovery.max_attempts:
            ctx = RecoveryContext(
                failed_cell=cell,
                error=result.error or "",
                error_phase=result.error_phase or "runtime",
                scope=self._kernel.get_scope(),
                cell_index=index,
                prior_output=self.rendered_output,
                attempt=attempt,
                plugin_advice=advice,
            )

            action = self._recovery.on_cell_error(ctx)

            if action.kind == "abort":
                self._notify_recovery_result(cell, False, attempt)
                return CellExecutionEvent(
                    cell=cell, cell_index=index, result=result,
                    status="aborted", recovery_attempts=attempt,
                    generation=self._generation,
                )

            if action.kind == "skip":
                self._notify_recovery_result(cell, False, attempt)
                return CellExecutionEvent(
                    cell=cell, cell_index=index, result=result,
                    status="skipped", recovery_attempts=attempt,
                    generation=self._generation,
                )

            if action.kind == "inspect":
                while action.kind == "inspect":
                    inspect_result = self._kernel.inspect(action.expr or "None")
                    action = self._recovery.on_inspect_result(ctx, inspect_result)
                if action.kind != "fix":
                    attempt += 1
                    continue

            if action.kind == "fix" and action.cells:
                fix_succeeded = True
                fix_results: list[CellResult] = []
                for fix_cell in action.cells:
                    fix_result = self._kernel.execute_cell(fix_cell, index)
                    fix_results.append(fix_result)
                    if not fix_result.success:
                        fix_succeeded = False
                        result = fix_result
                        break
                    if fix_result.output:
                        self._output_parts.append(fix_result.output)

                if fix_succeeded:
                    self._notify_recovery_result(cell, True, attempt)
                    final_result = fix_results[-1] if fix_results else result
                    return CellExecutionEvent(
                        cell=action.cells[-1], cell_index=index,
                        result=final_result, status="recovered",
                        recovery_attempts=attempt + 1,
                        generation=self._generation,
                    )

            attempt += 1

        self._notify_recovery_result(cell, False, attempt)
        return CellExecutionEvent(
            cell=cell, cell_index=index, result=result,
            status="aborted", recovery_attempts=attempt,
            generation=self._generation,
        )

    def _notify_start(self, cell: Cell, index: int) -> None:
        for plugin in self._plugins:
            try:
                plugin.on_cell_start(cell, index)
            except Exception:
                pass

    def _notify_success(self, cell: Cell, result: CellResult) -> None:
        for plugin in self._plugins:
            try:
                plugin.on_cell_success(cell, result)
            except Exception:
                pass

    def _notify_recovery_result(self, cell: Cell, success: bool, attempt: int) -> None:
        for plugin in self._plugins:
            try:
                plugin.on_recovery_result(cell, success, attempt)
            except Exception:
                pass

    def _collect_advice(self, cell: Cell, error: str) -> PluginAdvice:
        advices: list[PluginAdvice] = []
        scope = self._kernel.get_scope()
        for plugin in self._plugins:
            try:
                advice = plugin.on_cell_error(cell, error, scope)
                if advice:
                    advices.append(advice)
            except Exception:
                pass
        return merge_advice(advices)
