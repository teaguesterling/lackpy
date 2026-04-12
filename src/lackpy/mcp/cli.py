"""CLI handlers for `lackpyctl mcp {serve,init}`."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def mcp_init(
    workspace: Path,
    name: str = "lackpy",
    force: bool = False,
) -> int:
    """Add a lackpy entry to .mcp.json in the workspace.

    Returns 0 on success, 1 if the entry already exists (without --force).
    """
    mcp_file = workspace / ".mcp.json"
    if mcp_file.exists():
        data = json.loads(mcp_file.read_text())
    else:
        data = {"mcpServers": {}}

    servers = data.setdefault("mcpServers", {})

    if name in servers and not force:
        print(
            f"lackpyctl: '{name}' already exists in {mcp_file}:\n"
            f"  {json.dumps(servers[name], indent=2)}\n"
            f"Use --force to overwrite.",
            file=sys.stderr,
        )
        return 1

    servers[name] = {
        "command": "lackpyctl",
        "args": ["mcp", "serve", "--workspace", str(workspace.resolve())],
    }

    mcp_file.write_text(json.dumps(data, indent=2) + "\n")
    print(f"{'Updated' if force else 'Added'} '{name}' in {mcp_file}")
    return 0


def mcp_serve(workspace: Path) -> int:
    """Start the MCP server on stdio transport."""
    from .server import mcp, set_workspace

    set_workspace(workspace)
    mcp.run(transport="stdio")
    return 0
