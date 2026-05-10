"""Tests for StreamingDriver - the orchestrator."""

import pytest

from lackpy.interpreters.literate.kernel.driver import CellExecutionEvent, StreamingDriver
from lackpy.interpreters.literate.kernel.lightweight import LightweightKernel
from lackpy.interpreters.literate.kernel.recovery import (
    NoRecoveryHandler,
    RecoveryAction,
    RecoveryContext,
)
from lackpy.interpreters.literate.kernel.plugins import PluginAdvice
from lackpy.interpreters.literate.parser import Cell


@pytest.fixture
def kernel():
    return LightweightKernel()


@pytest.fixture
def driver(kernel):
    return StreamingDriver(kernel=kernel, recovery=NoRecoveryHandler())


class TestBasicExecution:
    @pytest.mark.asyncio
    async def test_single_prose_cell(self, driver):
        events = await driver.feed("Hello world\n")
        flush_events = await driver.flush()
        all_events = events + flush_events
        assert len(all_events) == 1
        assert all_events[0].status == "executed"
        assert "Hello world" in driver.rendered_output

    @pytest.mark.asyncio
    async def test_hidden_then_prose(self, driver):
        text = "```lackpy @hidden\nx = 42\n```\n\nValue: {x}\n"
        events = await driver.feed(text)
        flush_events = await driver.flush()
        all_events = events + flush_events
        assert len(all_events) == 2
        assert all_events[0].cell.cell_type == "hidden"
        assert all_events[1].cell.cell_type == "prose"
        assert "42" in driver.rendered_output

    @pytest.mark.asyncio
    async def test_execution_log_tracks_all(self, driver):
        text = "```lackpy @hidden\na = 1\n```\n\n```lackpy\nb = 2\n```\n"
        await driver.feed(text)
        await driver.flush()
        assert len(driver.execution_log) == 2


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_static_error_aborts_with_no_recovery(self, driver):
        text = "```lackpy\nprint(undefined)\n```\n\nNever reached\n"
        events = await driver.feed(text)
        flush_events = await driver.flush()
        all_events = events + flush_events
        failed = [e for e in all_events if e.status not in ("executed", "pending")]
        assert len(failed) >= 1
        assert failed[0].result.error_phase == "static"

    @pytest.mark.asyncio
    async def test_runtime_error_aborts(self, driver):
        text = "```lackpy @hidden\nx = 1\n```\n\n```lackpy\ny = 1/0\n```\n"
        events = await driver.feed(text)
        flush_events = await driver.flush()
        all_events = events + flush_events
        assert all_events[0].status == "executed"
        assert all_events[1].status != "executed"


class TestRecoveryIntegration:
    @pytest.mark.asyncio
    async def test_fix_action_replaces_cell(self, kernel):
        fix_cell = Cell(cell_type="hidden", content="foo = 'fixed'")

        class FixHandler:
            max_attempts = 2

            def on_cell_error(self, ctx: RecoveryContext) -> RecoveryAction:
                return RecoveryAction(kind="fix", cells=[fix_cell])

            def on_inspect_result(self, ctx: RecoveryContext, result: str) -> RecoveryAction:
                return RecoveryAction(kind="abort")

        driver = StreamingDriver(kernel=kernel, recovery=FixHandler())
        text = "```lackpy\nprint(foo)\n```\n"
        events = await driver.feed(text)
        flush_events = await driver.flush()
        all_events = events + flush_events
        recovered = [e for e in all_events if e.status == "recovered"]
        assert len(recovered) == 1

    @pytest.mark.asyncio
    async def test_skip_action_continues(self, kernel):
        class SkipHandler:
            max_attempts = 1

            def on_cell_error(self, ctx: RecoveryContext) -> RecoveryAction:
                return RecoveryAction(kind="skip")

            def on_inspect_result(self, ctx: RecoveryContext, result: str) -> RecoveryAction:
                return RecoveryAction(kind="abort")

        driver = StreamingDriver(kernel=kernel, recovery=SkipHandler())
        text = "```lackpy\nprint(bad)\n```\n\nAfter skip\n"
        events = await driver.feed(text)
        flush_events = await driver.flush()
        all_events = events + flush_events
        skipped = [e for e in all_events if e.status == "skipped"]
        assert len(skipped) == 1
        executed = [e for e in all_events if e.status == "executed"]
        assert len(executed) >= 1

    @pytest.mark.asyncio
    async def test_inspect_action_evaluates_expr(self, kernel):
        kernel.execute_cell(Cell(cell_type="hidden", content="x = 42"), cell_index=0)

        class InspectThenFixHandler:
            max_attempts = 2

            def __init__(self):
                self.inspected = None

            def on_cell_error(self, ctx: RecoveryContext) -> RecoveryAction:
                return RecoveryAction(kind="inspect", expr="type(x).__name__")

            def on_inspect_result(self, ctx: RecoveryContext, result: str) -> RecoveryAction:
                self.inspected = result
                return RecoveryAction(
                    kind="fix",
                    cells=[Cell(cell_type="code", content="y = str(x)")],
                )

        handler = InspectThenFixHandler()
        driver = StreamingDriver(kernel=kernel, recovery=handler)
        text = "```lackpy\ny = str(undefined)\n```\n"
        await driver.feed(text)
        await driver.flush()
        assert handler.inspected == "'int'"


class TestContinueSemantics:
    @pytest.mark.asyncio
    async def test_continue_pauses_execution(self, driver):
        text = "```lackpy @hidden\nx = 1\n```\n\n```lackpy @continue\n```\n\nNot reached yet\n"
        events = await driver.feed(text)
        flush_events = await driver.flush()
        all_events = events + flush_events
        continue_events = [e for e in all_events if e.status == "continue_requested"]
        assert len(continue_events) == 1
        assert "Not reached" not in driver.rendered_output

    @pytest.mark.asyncio
    async def test_generation_increments_on_continue(self, driver):
        text1 = "```lackpy @hidden\nx = 1\n```\n\n```lackpy @continue\n```\n"
        await driver.feed(text1)
        await driver.flush()
        assert driver.generation == 0

        driver.resume()
        text2 = "```lackpy\ny = x + 1\n```\n"
        await driver.feed(text2)
        await driver.flush()
        assert driver.generation == 1


class TestPluginNotifications:
    @pytest.mark.asyncio
    async def test_plugins_notified_on_success(self, kernel):
        class TrackingPlugin:
            def __init__(self):
                self.started = []
                self.succeeded = []

            def on_cell_start(self, cell, index):
                self.started.append(index)

            def on_cell_success(self, cell, result):
                self.succeeded.append(result.cell_index)

            def on_cell_error(self, cell, error, scope):
                return PluginAdvice()

            def on_recovery_result(self, cell, success, attempt):
                pass

        plugin = TrackingPlugin()
        driver = StreamingDriver(kernel=kernel, recovery=NoRecoveryHandler(), plugins=[plugin])
        text = "```lackpy @hidden\nx = 1\n```\n"
        await driver.feed(text)
        await driver.flush()
        assert plugin.started == [0]
        assert plugin.succeeded == [0]


class TestTruncationWarning:
    @pytest.mark.asyncio
    async def test_truncated_cell_produces_warning(self, driver):
        await driver.feed("```lackpy\nx = 1\n")
        await driver.flush()
        assert "[warning:" in driver.rendered_output
        assert "truncated" in driver.rendered_output
