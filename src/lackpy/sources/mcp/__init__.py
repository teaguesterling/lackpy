"""MCP-discovered tool source (RFC 0002 increment 2b).

Connects to MCP servers as a *client*, discovers their tools (full specs +
grades from annotations), and exposes them as lackpy tools whose callables proxy
``call_tool`` over a dedicated client event loop. Requires the optional ``mcp``
dependency (``pip install lackpy[mcp]``).
"""

from __future__ import annotations


def mcp_available() -> bool:
    """Whether the optional ``mcp`` client SDK is importable."""
    try:
        import mcp  # noqa: F401
        return True
    except Exception:
        return False
