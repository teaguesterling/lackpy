"""Recovery handler protocol and built-in handlers.

Called by StreamingDriver._handle_failure() when a cell fails.
The handler receives RecoveryContext (failed cell, error, scope,
plugin advice) and returns a RecoveryAction:
  - fix: execute replacement cells (handler parses model response)
  - inspect: evaluate an expression, feed result back to handler
  - skip: mark cell skipped, continue with next
  - abort: stop execution

Two built-in handlers:
  NoRecoveryHandler — always aborts. Used by the batch path and tests.
  InferenceRecoveryHandler — calls a model to generate replacement cells.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..parser import Cell, _BODY_ANNOTATION_RE, _ANNOTATION_TYPES
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


import asyncio
from collections.abc import Awaitable, Callable

from .streaming_parser import StreamingCellParser as _RecoveryParser

InferFn = Callable[[str], Awaitable[str]]


class InferenceRecoveryHandler:
    def __init__(self, infer_fn: InferFn, max_attempts: int = 2) -> None:
        self.max_attempts = max_attempts
        self._infer_fn = infer_fn

    def on_cell_error(self, ctx: RecoveryContext) -> RecoveryAction:
        prompt = self._build_prompt(ctx)
        response = self._call_infer(prompt)
        return self._parse_response(response)

    def on_inspect_result(self, ctx: RecoveryContext, result: str) -> RecoveryAction:
        prompt = self._build_inspect_followup(ctx, result)
        response = self._call_infer(prompt)
        return self._parse_response(response)

    def _call_infer(self, prompt: str) -> str:
        try:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(self._infer_fn(prompt))
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self._infer_fn(prompt))
                return future.result()
        except Exception:
            return ""

    def _build_prompt(self, ctx: RecoveryContext) -> str:
        parts = [
            "A cell failed during execution.\n",
            f"Cell type: {ctx.failed_cell.cell_type}",
            f"Cell content:\n{ctx.failed_cell.content}\n",
            f"Error ({ctx.error_phase}): {ctx.error}\n",
            "Variables in scope:",
        ]
        for name, summary in ctx.scope.items():
            parts.append(f"  {name} = {summary}")

        if ctx.plugin_advice:
            if ctx.plugin_advice.hints:
                parts.append("\nCoaching hints:")
                for hint in ctx.plugin_advice.hints:
                    parts.append(f"  - {hint}")
            if ctx.plugin_advice.doc_context:
                parts.append("\nRelevant documentation:")
                for doc in ctx.plugin_advice.doc_context:
                    parts.append(f"  {doc}")

        if self._is_annotation_placement_error(ctx):
            first_line = ctx.failed_cell.content.split("\n", 1)[0].strip()
            parts.append(
                f"\nNote: '{first_line}' is an annotation — it goes on the fence line "
                f"(```lackpy {first_line}), not inside the code body."
            )

        parts.append(
            "\nFix this cell. Return replacement cells as a literate document fragment. "
            "You may add @hidden blocks before the cell to pre-compute values. "
            "Use @scratch if you need to inspect a value first."
        )
        return "\n".join(parts)

    @staticmethod
    def _is_annotation_placement_error(ctx: RecoveryContext) -> bool:
        if "inside code body" in (ctx.error or ""):
            return True
        if ctx.failed_cell.content:
            first_line = ctx.failed_cell.content.split("\n", 1)[0]
            m = _BODY_ANNOTATION_RE.match(first_line)
            if m and m.group(1) in _ANNOTATION_TYPES:
                return True
        return False

    def _build_inspect_followup(self, ctx: RecoveryContext, result: str) -> str:
        return (
            f"Inspection result: {result}\n\n"
            f"Original error: {ctx.error}\n"
            f"Original cell:\n{ctx.failed_cell.content}\n\n"
            "Now provide the fix as a literate document fragment."
        )

    def _parse_response(self, response: str) -> RecoveryAction:
        if not response.strip():
            return RecoveryAction(kind="abort")

        parser = _RecoveryParser()
        cells = parser.feed(response)
        cells.extend(parser.flush())

        if not cells:
            return RecoveryAction(kind="abort")

        if len(cells) == 1 and cells[0].cell_type == "scratch":
            return RecoveryAction(kind="inspect", expr=cells[0].content.strip())

        return RecoveryAction(kind="fix", cells=cells)
