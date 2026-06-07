"""Direct McpClient integration test against a real fastmcp stdio server."""

import sys
from pathlib import Path

import pytest

from lackpy.sources.mcp import mcp_available

pytestmark = pytest.mark.skipif(not mcp_available(), reason="mcp SDK not installed")

FIXTURE = str(Path(__file__).parent.parent / "fixtures" / "mcp_echo_server.py")


def _spec():
    from lackpy.sources.mcp.client import McpServerSpec

    return McpServerSpec(
        server_id="echo", transport="stdio", command=sys.executable, args=[FIXTURE]
    )


def test_connect_discover_call_shutdown():
    from lackpy.sources.mcp.client import McpClient

    client = McpClient()
    try:
        tools = client.connect(_spec(), timeout=30)
        names = {t.name for t in tools}
        assert {"echo", "add"} <= names

        # echo carries readOnlyHint + closed-world annotations
        echo = next(t for t in tools if t.name == "echo")
        assert echo.annotations is not None
        assert echo.annotations.readOnlyHint is True

        assert client.call("echo", "echo", {"text": "hi"}) == "hi"
        assert client.call("echo", "add", {"a": 2, "b": 3}) == 5
    finally:
        client.shutdown()
