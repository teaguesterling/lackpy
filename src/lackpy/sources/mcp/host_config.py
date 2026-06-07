"""Read external *host* MCP config files into McpServerSpec descriptors.

Normalizes the common ``{"mcpServers": {name: {...}}}`` shape used by Claude
Desktop / Cursor / Claude Code. stdio entries use ``command``/``args``/``env``;
HTTP entries use ``url``/``headers``. Host configs are an explicit opt-in
(``[mcp].host_configs``) — lackpy never auto-discovers them, since they spawn
subprocesses lackpy didn't author.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .client import McpServerSpec


def load_host_servers(path: str) -> list[McpServerSpec]:
    """Parse a host MCP config file into server specs.

    Raises on unreadable/invalid JSON — the caller skips a bad host config so one
    malformed file never breaks startup.
    """
    p = Path(os.path.expandvars(os.path.expanduser(path)))
    data = json.loads(p.read_text(encoding="utf-8"))
    servers = data.get("mcpServers", data) if isinstance(data, dict) else {}
    out: list[McpServerSpec] = []
    for server_id, cfg in servers.items():
        if not isinstance(cfg, dict):
            continue
        if cfg.get("url"):
            out.append(McpServerSpec(
                server_id=server_id, transport="http",
                url=cfg["url"], headers=cfg.get("headers"),
            ))
        elif cfg.get("command"):
            out.append(McpServerSpec(
                server_id=server_id, transport="stdio",
                command=cfg["command"], args=cfg.get("args", []),
                env=cfg.get("env"), cwd=cfg.get("cwd"),
            ))
        # entries with neither url nor command are skipped
    return out
