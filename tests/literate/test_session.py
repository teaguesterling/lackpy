"""Tests for the multi-round literate fold (LiterateSession)."""

import pytest

from lackpy.interpreters.base import ExecutionContext
from lackpy.interpreters.literate import LiterateSession, strip_think
from lackpy.lang.grader import Grade


@pytest.fixture
def context(tmp_path):
    return ExecutionContext(base_dir=tmp_path)


class TestLiveThreading:
    @pytest.mark.asyncio
    async def test_function_defined_in_round_one_is_callable_in_round_two(self, context):
        # The DoD: the old fresh-kernel-per-round loop dropped functions entirely
        # (filtered out of `variables`) and round 2 would fail check_cell with an
        # undefined name. A persistent kernel threads the live function object.
        session = LiterateSession(context)
        r1 = await session.step(
            "```lackpy @hidden\ndef double(n):\n    return n * 2\nbase = 10\n```")
        assert r1.ok, r1.errors

        r2 = await session.step("```lackpy\nresult = double(base) + 5\n```\n\nResult: {result}")
        assert r2.ok, r2.errors
        assert "Result: 25" in r2.clean_doc
        assert r2.variables.get("result") == 25

    @pytest.mark.asyncio
    async def test_variable_threads_across_rounds(self, context):
        session = LiterateSession(context)
        assert (await session.step("```lackpy @hidden\nx = 7\n```")).ok
        r2 = await session.step("```lackpy\ny = x * 3\n```\n\ny={y}")
        assert r2.ok and "y=21" in r2.clean_doc

    @pytest.mark.asyncio
    async def test_rendered_accumulates_clean_docs(self, context):
        session = LiterateSession(context)
        await session.step("Round one.\n\n```lackpy @hidden\na = 1\n```")
        await session.step("Round two.")
        assert "Round one." in session.rendered
        assert "Round two." in session.rendered


class TestEitherSemantics:
    @pytest.mark.asyncio
    async def test_failure_returns_left_with_raw_and_errors(self, context):
        session = LiterateSession(context)
        raw = "```lackpy\nx = 1 / 0\n```"
        result = await session.step(raw)
        assert not result.ok
        assert result.raw == raw            # the un-interpreted message, for correction
        assert result.errors and "ZeroDivisionError" in result.errors[0]

    @pytest.mark.asyncio
    async def test_left_does_not_advance_state(self, context):
        session = LiterateSession(context)
        assert (await session.step("```lackpy @hidden\nx = 1\n```")).ok
        # round 2 rebinds x then fails -> Left must restore x to 1
        bad = await session.step("```lackpy\nx = 2\nraise ValueError('boom')\n```")
        assert not bad.ok
        # round 3 (corrected) sees x == 1, not the failed round's x == 2
        r3 = await session.step("```lackpy\ny = x + 10\n```\n\ny={y}")
        assert r3.ok and "y=11" in r3.clean_doc

    @pytest.mark.asyncio
    async def test_failed_round_is_not_added_to_rendered(self, context):
        session = LiterateSession(context)
        await session.step("Good round.")
        await session.step("```lackpy\nboom = 1 / 0\n```")   # Left
        assert "Good round." in session.rendered
        assert "boom" not in session.rendered

    @pytest.mark.asyncio
    async def test_continue_requested_surfaces(self, context):
        session = LiterateSession(context)
        result = await session.step("```lackpy @continue\n```")
        assert result.ok and result.continue_requested


class TestThinkStripping:
    def test_strip_closed_block(self):
        assert strip_think("<think>reasoning here</think>\nHello") == "Hello"

    def test_strip_unclosed_block(self):
        # model hit the token limit mid-reasoning -> strip to end
        assert strip_think("Doc start\n<think>partial reason") == "Doc start"

    def test_only_reasoning_strips_to_empty(self):
        assert strip_think("<think>all thinking, no doc</think>") == ""

    def test_no_think_is_unchanged(self):
        assert strip_think("plain document") == "plain document"

    @pytest.mark.asyncio
    async def test_session_strips_think_before_running(self, context):
        session = LiterateSession(context)
        result = await session.step(
            "<think>I should set x to 5</think>\n```lackpy\nx = 5\n```\n\nValue: {x}")
        assert result.ok
        assert "Value: 5" in result.clean_doc
        assert "should set x" not in result.clean_doc     # reasoning never rendered

    @pytest.mark.asyncio
    async def test_empty_after_strip_is_a_left(self, context):
        session = LiterateSession(context)
        result = await session.step("<think>only thinking</think>")
        assert not result.ok
        assert "empty" in result.errors[0].lower()


class TestGateStillApplies:
    @pytest.mark.asyncio
    async def test_ceiling_gate_refuses_in_a_session(self, tmp_path):
        # The fold runs through _run_document, so the ceiling gate still applies.
        ctx = ExecutionContext(base_dir=tmp_path, config={"grade_ceiling": Grade(1, 1)})
        session = LiterateSession(ctx)
        result = await session.step("```lackpy @write(o.txt)\nhi\n```")
        assert not result.ok
        assert "effect ceiling exceeded" in result.errors[0]
        assert not (tmp_path / "o.txt").exists()
