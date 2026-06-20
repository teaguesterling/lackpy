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
        try:
            data = json.loads(mcp_file.read_text())
        except json.JSONDecodeError as e:
            print(f"lackpyctl: {mcp_file} contains invalid JSON: {e}", file=sys.stderr)
            return 1
        if not isinstance(data, dict):
            print(f"lackpyctl: {mcp_file} is not a JSON object", file=sys.stderr)
            return 1
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

    existed = name in servers
    servers[name] = {
        "command": "lackpy-mcp",
        "args": ["--workspace", str(workspace.resolve())],
    }

    mcp_file.write_text(json.dumps(data, indent=2) + "\n")
    print(f"{'Updated' if existed else 'Added'} '{name}' in {mcp_file}")
    return 0


def mcp_serve(workspace: Path) -> int:
    """Start the MCP server on stdio transport."""
    from .server import mcp, set_workspace

    set_workspace(workspace)
    mcp.run(transport="stdio")
    return 0


def mcp_main(argv: list[str] | None = None) -> int:
    """Console-script entry: ``lackpy-mcp [--workspace DIR]`` — serve on stdio.

    A stable, dedicated launch surface for external consumers (e.g. a woollama
    ``mcp.json`` backend that spawns lackpy): one command, no subcommand. Equivalent
    to ``lackpyctl mcp serve``. Also reachable as ``lackpy mcp`` for convenience.
    """
    import argparse

    p = argparse.ArgumentParser(
        prog="lackpy-mcp",
        description="Start the lackpy MCP server on stdio transport.")
    p.add_argument("--workspace", type=Path, default=None,
                   help="Workspace directory (default: cwd)")
    args = p.parse_args(argv)
    return mcp_serve(args.workspace or Path.cwd())
