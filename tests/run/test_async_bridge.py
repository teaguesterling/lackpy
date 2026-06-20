"""The sync↔async execution bridge (RFC 0002 increment 2a).

Exercises the bridge with a fake async tool — no MCP, no subprocess — so the
concurrency correctness (no deadlock, timeout handling, single-flight cwd lock)
is tested deterministically.
"""

import asyncio

from lackpy.kit.toolbox import ArgSpec, ToolSpec
from lackpy.run.bridge import mark_async
from lackpy.service import LackpyService


class _FuncSource:
    """Minimal ToolSource exposing one named callable (test double)."""

    def __init__(self, spec: ToolSpec, fn):
        self._spec = spec
        self._fn = fn

    @property
    def name(self) -> str:
        return "test"

    def available(self) -> bool:
        return True

    def discover(self):
        return [self._spec]

    def resolve(self, spec):
        return self._fn


def _spec(name: str, arg: str, arg_type: str = "int", returns: str = "int") -> ToolSpec:
    return ToolSpec(name=name, provider="test",
                    args=[ArgSpec(name=arg, type=arg_type)], returns=returns)


async def test_async_tool_runs_through_bridge(tmp_path):
    svc = LackpyService(workspace=tmp_path)
    bridge = svc._bridge

    async def _ainc(x):
        await asyncio.sleep(0.01)
        return x + 1

    @mark_async
    def proxy(x):
        return bridge.call_sync(_ainc(x))

    svc.toolbox.add_source(_FuncSource(_spec("ainc", "x"), proxy))
    res = await svc.run_program("y = ainc(1)\ny", profile=["ainc"])

    assert res.success, res.error
    assert res.output == 2
    assert res.trace.entries[0].tool == "ainc"
    assert res.trace.entries[0].success


async def test_async_tool_timeout_yields_failed_result(tmp_path):
    svc = LackpyService(workspace=tmp_path)
    bridge = svc._bridge

    async def _hang(x):
        await asyncio.sleep(10)
        return x

    @mark_async
    def proxy(x):
        return bridge.call_sync(_hang(x), timeout=0.1)

    svc.toolbox.add_source(_FuncSource(_spec("hang", "x"), proxy))
    res = await svc.run_program("y = hang(1)\ny", profile=["hang"])

    assert not res.success
    assert "timed out" in (res.error or "")
    # Failure is recorded in the trace like any other tool error.
    assert res.trace.entries and res.trace.entries[0].success is False


async def test_exec_lock_serializes_concurrent_executions(tmp_path):
    # Proves the single-flight lock: once execution runs off the loop thread, two
    # overlapping delegations must NOT interleave (which would race the global cwd).
    svc = LackpyService(workspace=tmp_path)
    bridge = svc._bridge
    events: list[str] = []

    async def _rec(tag):
        events.append(f"{tag}_in")
        await asyncio.sleep(0.02)
        events.append(f"{tag}_out")
        return tag

    @mark_async
    def proxy(tag):
        return bridge.call_sync(_rec(tag))

    svc.toolbox.add_source(_FuncSource(_spec("rec", "tag", "str", "str"), proxy))

    async def run(tag):
        return await svc.run_program(f"r = rec({tag!r})\nr", profile=["rec"])

    res_a, res_b = await asyncio.gather(run("A"), run("B"))

    assert res_a.success and res_b.success
    assert events in (
        ["A_in", "A_out", "B_in", "B_out"],
        ["B_in", "B_out", "A_in", "A_out"],
    ), f"interleaved -> lock not holding: {events}"


async def test_inline_path_still_used_without_async_tools(tmp_path):
    # No loop-bound tool -> inline (non-threaded) path; bridge loop never set.
    (tmp_path / "f.txt").write_text("data")
    svc = LackpyService(workspace=tmp_path)
    res = await svc.run_program("c = read_file('f.txt')\nc", profile=["read_file"])
    assert res.success and res.output == "data"
    assert svc._bridge.loop is None
