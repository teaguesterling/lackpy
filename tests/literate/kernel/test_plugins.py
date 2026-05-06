"""Tests for ExecutionPlugin protocol and PluginAdvice."""

from lackpy.interpreters.literate.kernel.plugins import (
    ExecutionPlugin,
    PluginAdvice,
    merge_advice,
)
from lackpy.interpreters.literate.kernel.interface import CellResult
from lackpy.interpreters.literate.parser import Cell


class FakePlugin:
    def __init__(self, hints: list[str] | None = None):
        self.started: list[int] = []
        self.succeeded: list[int] = []
        self.errors: list[str] = []
        self._hints = hints or []

    def on_cell_start(self, cell: Cell, index: int) -> None:
        self.started.append(index)

    def on_cell_success(self, cell: Cell, result: CellResult) -> None:
        self.succeeded.append(result.cell_index)

    def on_cell_error(self, cell: Cell, error: str, scope: dict) -> PluginAdvice:
        self.errors.append(error)
        return PluginAdvice(hints=self._hints, doc_context=[], suggestion=None)

    def on_recovery_result(self, cell: Cell, success: bool, attempt: int) -> None:
        pass


class TestPluginAdvice:
    def test_empty_advice(self):
        advice = PluginAdvice(hints=[], doc_context=[], suggestion=None)
        assert advice.hints == []
        assert advice.suggestion is None

    def test_merge_multiple_advices(self):
        a1 = PluginAdvice(hints=["try X"], doc_context=["doc1"], suggestion="do X")
        a2 = PluginAdvice(hints=["try Y"], doc_context=["doc2"], suggestion="do Y")
        merged = merge_advice([a1, a2])
        assert merged.hints == ["try X", "try Y"]
        assert merged.doc_context == ["doc1", "doc2"]
        assert merged.suggestion == "do X"

    def test_merge_empty_list(self):
        merged = merge_advice([])
        assert merged.hints == []
        assert merged.suggestion is None


class TestPluginProtocol:
    def test_plugin_receives_start(self):
        plugin = FakePlugin()
        cell = Cell(cell_type="code", content="x = 1")
        plugin.on_cell_start(cell, 0)
        assert plugin.started == [0]

    def test_plugin_receives_error(self):
        plugin = FakePlugin(hints=["check imports"])
        cell = Cell(cell_type="code", content="x = foo")
        advice = plugin.on_cell_error(cell, "undefined 'foo'", {})
        assert advice.hints == ["check imports"]
        assert plugin.errors == ["undefined 'foo'"]
