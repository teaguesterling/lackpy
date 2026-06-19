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
async def delegate(intent: str, profile: str | list[str] | dict | None = None,
                   params: dict[str, Any] | None = None, sandbox: str | None = None,
                   rules: list[str] | None = None,
                   extra_tools: list[str] | None = None) -> dict:
    """Generate and run a tool-composition program from natural language intent."""
    return await _get_service().delegate(intent, profile=profile, params=params, sandbox=sandbox, rules=rules, extra_tools=extra_tools)


@mcp.tool()
async def generate(intent: str, profile: str | list[str] | dict | None = None,
                   params: dict[str, Any] | None = None, rules: list[str] | None = None,
                   extra_tools: list[str] | None = None) -> dict:
    """Generate a program from intent without executing it."""
    result = await _get_service().generate(intent, profile=profile, params=params, rules=rules, extra_tools=extra_tools)
    return {"program": result.program, "provider": result.provider_name, "generation_time_ms": result.generation_time_ms}


@mcp.tool()
async def run_program(program: str, profile: str | list[str] | dict | None = None,
                      params: dict[str, Any] | None = None, sandbox: str | None = None,
                      rules: list[str] | None = None,
                      extra_tools: list[str] | None = None) -> dict:
    """Validate and run a provided program."""
    result = await _get_service().run_program(program, profile=profile, params=params, sandbox=sandbox, rules=rules, extra_tools=extra_tools)
    return {"success": result.success, "output": result.output, "error": result.error,
            "trace": [{"step": e.step, "tool": e.tool, "args": e.args, "result": e.result,
                        "success": e.success, "error": e.error} for e in result.trace.entries]}


@mcp.tool()
async def create(program: str, profile: str | list[str] | dict | None = None,
                 name: str = "", pattern: str | None = None,
                 extra_tools: list[str] | None = None) -> dict:
    """Validate and save a program as a reusable template."""
    return await _get_service().create(program, profile=profile, name=name, pattern=pattern, extra_tools=extra_tools)


@mcp.tool()
def validate(program: str, profile: str | list[str] | dict | None = None,
             rules: list[str] | None = None,
             extra_tools: list[str] | None = None) -> dict:
    """Validate a program without running it."""
    result = _get_service().validate(program, profile=profile, rules=rules, extra_tools=extra_tools)
    return {"valid": result.valid, "errors": result.errors, "calls": result.calls, "variables": result.variables}


@mcp.tool()
def profile_info(profile: str | list[str] | dict, extra_tools: list[str] | None = None) -> dict:
    """Get info about a profile's tools: names, grades, descriptions."""
    return _get_service().profile_info(profile, extra_tools=extra_tools)


@mcp.tool()
def profile_list() -> list[dict]:
    """List available named tool-set / profile files."""
    return _get_service().profile_list()


@mcp.tool()
def profile_create(name: str, tools: list[str], description: str | None = None) -> dict:
    """Create a new named tool-set."""
    return _get_service().profile_create(name, tools, description)


@mcp.tool()
def toolbox_list() -> list[dict]:
    """List all registered tools with metadata."""
    return _get_service().toolbox_list()


@mcp.tool()
def docs_index(profile: str | list[str] | dict | None = None,
               extra_tools: list[str] | None = None) -> dict:
    """Return documentation references for a kit's tools and kit-level docs."""
    return _get_service().docs_index(profile=profile, extra_tools=extra_tools)


@mcp.tool()
def resolve_doc(doc_path: str) -> dict:
    """Read a documentation file by its relative path. Returns content or null."""
    content = _get_service().resolve_doc(doc_path)
    return {"path": doc_path, "content": content}


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
