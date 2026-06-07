"""Minimal fastmcp stdio server used as an MCP-client integration fixture."""

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

mcp = FastMCP("echo-test")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
def echo(text: str) -> str:
    """Echo back the given text."""
    return text


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


if __name__ == "__main__":
    mcp.run()
