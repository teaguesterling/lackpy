"""FastMCP server: thin adapter over the service layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from ..service import LackpyService

mcp = FastMCP("lackpy")
_service: LackpyService | None = None
_workspace: Path | None = None


def set_workspace(workspace: Path) -> None:
    """Set the workspace directory and reset the service singleton."""
    global _workspace, _service
    _workspace = workspace
    _service = None


def _get_service() -> LackpyService:
    global _service
    if _service is None:
        _service = LackpyService(workspace=_workspace)
    return _service


@mcp.tool()
async def delegate(intent: str, kit: str | list[str] | dict | None = None,
                   params: dict[str, Any] | None = None, sandbox: str | None = None,
                   rules: list[str] | None = None) -> dict:
    """Generate and run a tool-composition program from natural language intent."""
    return await _get_service().delegate(intent, kit=kit, params=params, sandbox=sandbox, rules=rules)


@mcp.tool()
async def generate(intent: str, kit: str | list[str] | dict | None = None,
                   params: dict[str, Any] | None = None, rules: list[str] | None = None) -> dict:
    """Generate a program from intent without executing it."""
    result = await _get_service().generate(intent, kit=kit, params=params, rules=rules)
    return {"program": result.program, "provider": result.provider_name, "generation_time_ms": result.generation_time_ms}


@mcp.tool()
async def run_program(program: str, kit: str | list[str] | dict | None = None,
                      params: dict[str, Any] | None = None, sandbox: str | None = None,
                      rules: list[str] | None = None) -> dict:
    """Validate and run a provided program."""
    result = await _get_service().run_program(program, kit=kit, params=params, sandbox=sandbox, rules=rules)
    return {"success": result.success, "output": result.output, "error": result.error,
            "trace": [{"step": e.step, "tool": e.tool, "args": e.args, "result": e.result,
                        "success": e.success, "error": e.error} for e in result.trace.entries]}


@mcp.tool()
async def create(program: str, kit: str | list[str] | dict | None = None,
                 name: str = "", pattern: str | None = None) -> dict:
    """Validate and save a program as a reusable template."""
    return await _get_service().create(program, kit=kit, name=name, pattern=pattern)


@mcp.tool()
def validate(program: str, kit: str | list[str] | dict | None = None,
             rules: list[str] | None = None) -> dict:
    """Validate a program without running it."""
    result = _get_service().validate(program, kit=kit, rules=rules)
    return {"valid": result.valid, "errors": result.errors, "calls": result.calls, "variables": result.variables}


@mcp.tool()
def kit_info(kit: str | list[str] | dict) -> dict:
    """Get info about a kit: tools, grades, descriptions."""
    return _get_service().kit_info(kit)


@mcp.tool()
def kit_list() -> list[dict]:
    """List available predefined kits."""
    return _get_service().kit_list()


@mcp.tool()
def kit_create(name: str, tools: list[str], description: str | None = None) -> dict:
    """Create a new predefined kit."""
    return _get_service().kit_create(name, tools, description)


@mcp.tool()
def toolbox_list() -> list[dict]:
    """List all registered tools with metadata."""
    return _get_service().toolbox_list()


# --- Configuration & introspection tools ---

@mcp.tool()
def config() -> dict:
    """Return the current workspace configuration: inference order, providers, kit default, sandbox settings."""
    return _get_service().get_config()


@mcp.tool()
def provider_list() -> list[dict]:
    """List configured inference providers with plugin type, model, and availability status."""
    return _get_service().provider_list()


@mcp.tool()
def language_spec() -> dict:
    """Return the lackpy language specification: allowed AST nodes, builtins, and forbidden names."""
    return _get_service().language_spec()
