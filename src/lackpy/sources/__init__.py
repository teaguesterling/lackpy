"""Tool sources: where fully-defined tools come from.

A *source* owns BOTH discovery (producing a complete ``ToolSpec`` — name, args,
returns, docs, grade) AND resolution (producing the callable). This is the seam
that keeps tool names from being hard-coded in lackpy source: every tool a kit can
reference originates from a source. The first concrete source is
``ConfigToolSource`` (config-defined tools); MCP-discovered and virtual/harness
sources implement the same protocol later (see docs/design/tool-sources.md).
"""

from __future__ import annotations

import tomllib
from importlib.resources import files
from typing import Any


def load_default_tool_defs() -> list[dict[str, Any]]:
    """Load the shipped default tool definitions (formerly hard-coded builtins).

    These ship as data (``default_tools.toml``) rather than Python so the default
    toolbox contains zero hard-coded ``ToolSpec`` lists. User ``[[tools]]`` entries
    in ``.lackpy/config.toml`` override these by name.
    """
    raw = (files(__package__) / "default_tools.toml").read_text(encoding="utf-8")
    return tomllib.loads(raw).get("tools", [])
