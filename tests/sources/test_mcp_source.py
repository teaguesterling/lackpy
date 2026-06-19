"""McpToolSource: grade mapping, discovery, and end-to-end tool calls."""

import sys
from pathlib import Path

import pytest

from lackpy.sources.mcp import mcp_available

pytestmark = pytest.mark.skipif(not mcp_available(), reason="mcp SDK not installed")

FIXTURE = str(Path(__file__).parent.parent / "fixtures" / "mcp_echo_server.py")


def _server_cfg() -> dict:
    return {"transport": "stdio", "command": sys.executable, "args": [FIXTURE]}


def test_grade_from_annotations_mapping():
    from lackpy.lang.grader import Grade
    from lackpy.sources.mcp.grade import grade_from_annotations

    class _Ann:
        def __init__(self, **k):
            self.readOnlyHint = k.get("readOnlyHint")
            self.openWorldHint = k.get("openWorldHint")
            self.idempotentHint = k.get("idempotentHint")
            self.destructiveHint = k.get("destructiveHint")

    assert grade_from_annotations(None) == Grade(3, 3)
    assert grade_from_annotations(_Ann(readOnlyHint=True, openWorldHint=False)) == Grade(1, 0)
    assert grade_from_annotations(_Ann(readOnlyHint=True, openWorldHint=True)) == Grade(2, 1)
    assert grade_from_annotations(_Ann(readOnlyHint=False, idempotentHint=True, destructiveHint=False)) == Grade(3, 2)
    assert grade_from_annotations(_Ann(readOnlyHint=False, destructiveHint=True)) == Grade(3, 3)
    assert grade_from_annotations(_Ann(readOnlyHint=True)) == Grade(3, 3)  # open-world unknown


def test_source_discovers_specs_with_grade_and_args():
    from lackpy.sources.mcp.client import McpClient, McpServerSpec
    from lackpy.sources.mcp.source import McpToolSource

    client = McpClient()
    try:
        spec = McpServerSpec(server_id="echo", transport="stdio",
                             command=sys.executable, args=[FIXTURE])
        src = McpToolSource(spec, client)
        assert src.available()
        specs = {s.name: s for s in src.discover()}
        assert {"echo", "add"} <= set(specs)
        # echo: readOnly + closed-world -> (1, 0)
        assert (specs["echo"].grade_w, specs["echo"].effects_ceiling) == (1, 0)
        assert specs["echo"].provider == "mcp:echo"
        assert [a.name for a in specs["echo"].args] == ["text"]
        assert specs["echo"].args[0].type == "str"
    finally:
        client.shutdown()


def test_grade_override_wins():
    from lackpy.sources.mcp.client import McpClient, McpServerSpec
    from lackpy.sources.mcp.source import McpToolSource

    client = McpClient()
    try:
        spec = McpServerSpec(server_id="echo", transport="stdio",
                             command=sys.executable, args=[FIXTURE])
        src = McpToolSource(spec, client, grade_overrides={"echo": (3, 3)})
        specs = {s.name: s for s in src.discover()}
        assert (specs["echo"].grade_w, specs["echo"].effects_ceiling) == (3, 3)
    finally:
        client.shutdown()


async def test_run_program_calls_mcp_tool_end_to_end(tmp_path):
    from lackpy.config import LackpyConfig
    from lackpy.service import LackpyService

    cfg = LackpyConfig(mcp_servers={"echo": _server_cfg()})
    svc = LackpyService(workspace=tmp_path, config=cfg)
    try:
        names = {t["name"] for t in svc.toolbox_list()}
        assert {"echo", "add"} <= names

        res = await svc.run_program("r = echo(text='hi')\nr", profile=["echo"])
        assert res.success, res.error
        assert res.output == "hi"
        assert res.trace.entries[0].tool == "echo"

        res2 = await svc.run_program("s = add(a=2, b=3)\ns", profile=["add"])
        assert res2.success, res2.error
        assert res2.output == 5
    finally:
        svc.close()
