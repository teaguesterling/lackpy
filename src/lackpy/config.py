"""Configuration loading from .lackpy/config.toml."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]


@dataclass
class LackpyConfig:
    inference_order: list[str] = field(default_factory=lambda: ["templates", "rules"])
    inference_providers: dict[str, dict[str, Any]] = field(default_factory=dict)
    kit_default: str = "debug"
    sandbox_enabled: bool = False
    sandbox_timeout: int = 120
    sandbox_memory_mb: int = 512
    inference_mode: str | None = None
    tool_providers: dict[str, dict[str, Any]] = field(default_factory=dict)
    tools: list[dict[str, Any]] = field(default_factory=list)
    mcp_servers: dict[str, dict[str, Any]] = field(default_factory=dict)
    mcp_host_configs: list[str] = field(default_factory=list)
    virtual_tools: list[dict[str, Any]] = field(default_factory=list)
    config_dir: Path = field(default_factory=lambda: Path(".lackpy"))


def load_config(workspace: Path | None = None) -> LackpyConfig:
    if workspace is None:
        workspace = Path.cwd()
    config_dir = workspace / ".lackpy"
    config_file = config_dir / "config.toml"
    if not config_file.exists():
        return LackpyConfig(config_dir=config_dir)
    with open(config_file, "rb") as f:
        data = tomllib.load(f)
    inference = data.get("inference", {})
    kit = data.get("kit", {})
    sandbox = data.get("sandbox", {})
    tool_providers = data.get("tool_providers", {})
    tools = data.get("tools", [])
    mcp_servers = data.get("mcp_servers", {})
    mcp_host_configs = data.get("mcp", {}).get("host_configs", [])
    virtual_tools = data.get("virtual_tools", [])
    providers: dict[str, dict[str, Any]] = {}
    for name, cfg in inference.get("providers", {}).items():
        providers[name] = cfg
    return LackpyConfig(
        inference_order=inference.get("order", ["templates", "rules"]),
        inference_providers=providers,
        inference_mode=inference.get("mode"),
        kit_default=kit.get("default", "debug"),
        sandbox_enabled=sandbox.get("enabled", False),
        sandbox_timeout=sandbox.get("timeout_seconds", 120),
        sandbox_memory_mb=sandbox.get("memory_mb", 512),
        tool_providers=tool_providers,
        tools=tools,
        mcp_servers=mcp_servers,
        mcp_host_configs=mcp_host_configs,
        virtual_tools=virtual_tools,
        config_dir=config_dir,
    )
