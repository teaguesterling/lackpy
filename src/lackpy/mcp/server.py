"""FastMCP server: thin adapter over the service layer.

The surface is split (RFC 0002 §11) so an orchestrator sees a small, clear set:

- **Tools** are *actions* — generate/run code, validate, save. Each carries MCP
  annotations (read-only vs. executes-code/destructive) so a caller can gate them.
- **Resources** are read-only *data* — config, language spec, toolbox, providers,
  profiles — addressed by `lackpy://…` URIs instead of cluttering the tool list.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from ..service import LackpyService

mcp = FastMCP("lackpy")
_service: LackpyService | None = None
_workspace: Path | None = None

# Conservative annotation presets. `delegate`/`run_program` execute generated code, so
# their effect depends on the profile's tools — mark non-read-only + potentially
# destructive + open-world. `create`/`profile_create` write a file (non-destructive,
# idempotent by name). The rest are read-only.
_EXECUTES = ToolAnnotations(readOnlyHint=False, destructiveHint=True,
                            idempotentHint=False, openWorldHint=True)
_WRITES_FILE = ToolAnnotations(readOnlyHint=False, destructiveHint=False,
                               idempotentHint=True, openWorldHint=False)
_READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)


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


# --- Tools: actions -----------------------------------------------------------

@mcp.tool(annotations=_EXECUTES)
async def delegate(intent: str, profile: str | list[str] | dict | None = None,
                   params: dict[str, Any] | None = None, rules: list[str] | None = None,
                   extra_tools: list[str] | None = None) -> dict:
    """Generate a restricted program from a natural-language intent and run it. Use for a
    self-contained, multi-step mechanical task (read/transform/compose tools).

    `profile` selects the tool set + model and is one of: a profile NAME (a configured
    `[profiles.<name>]` entry or a `.profile`/`.kit` file), a LIST of tool names, or an
    alias→tool MAP. Omit for the default profile. Returns a lean result (success, output,
    grade, program, tools-called); call `run_program` if you need the full step trace.
    """
    r = await _get_service().delegate(intent, profile=profile, params=params,
                                      rules=rules, extra_tools=extra_tools)
    return {
        "success": r.get("success"),
        "output": r.get("output"),
        "error": r.get("error"),
        "grade": r.get("grade"),
        "program": r.get("program"),
        "generation_tier": r.get("generation_tier"),
        "total_time_ms": r.get("total_time_ms"),
        "tools_called": [s.get("tool") for s in r.get("trace", [])],
    }


@mcp.tool(annotations=_READ_ONLY)
async def generate(intent: str, profile: str | list[str] | dict | None = None,
                   params: dict[str, Any] | None = None, rules: list[str] | None = None,
                   extra_tools: list[str] | None = None) -> dict:
    """Generate a program from intent WITHOUT running it (inspect before executing)."""
    result = await _get_service().generate(intent, profile=profile, params=params,
                                           rules=rules, extra_tools=extra_tools)
    return {"program": result.program, "provider": result.provider_name,
            "generation_time_ms": result.generation_time_ms}


@mcp.tool(annotations=_EXECUTES)
async def run_program(program: str, profile: str | list[str] | dict | None = None,
                      params: dict[str, Any] | None = None, rules: list[str] | None = None,
                      extra_tools: list[str] | None = None) -> dict:
    """Validate and run a program you already have (skips generation)."""
    result = await _get_service().run_program(program, profile=profile, params=params,
                                              rules=rules, extra_tools=extra_tools)
    return {"success": result.success, "output": result.output, "error": result.error,
            "trace": [{"step": e.step, "tool": e.tool, "args": e.args, "result": e.result,
                        "success": e.success, "error": e.error} for e in result.trace.entries]}


@mcp.tool(annotations=_WRITES_FILE)
async def create(program: str, profile: str | list[str] | dict | None = None,
                 name: str = "", pattern: str | None = None,
                 extra_tools: list[str] | None = None) -> dict:
    """Validate a program and save it as a reusable template file."""
    return await _get_service().create(program, profile=profile, name=name,
                                       pattern=pattern, extra_tools=extra_tools)


@mcp.tool(annotations=_READ_ONLY)
def validate(program: str, profile: str | list[str] | dict | None = None,
             rules: list[str] | None = None,
             extra_tools: list[str] | None = None) -> dict:
    """Check a program against a profile's allowed names + rules, without running it."""
    result = _get_service().validate(program, profile=profile, rules=rules, extra_tools=extra_tools)
    return {"valid": result.valid, "errors": result.errors, "calls": result.calls,
            "variables": result.variables}


@mcp.tool(annotations=_WRITES_FILE)
def profile_create(name: str, tools: list[str], description: str | None = None) -> dict:
    """Create a new named tool-set / profile file from a list of tool names."""
    return _get_service().profile_create(name, tools, description)


@mcp.tool(annotations=_READ_ONLY)
def profile_info(profile: str | list[str] | dict, extra_tools: list[str] | None = None) -> dict:
    """Resolve a profile and report its tools, grades, and description."""
    return _get_service().profile_info(profile, extra_tools=extra_tools)


@mcp.tool(annotations=_READ_ONLY)
def docs_index(profile: str | list[str] | dict | None = None,
               extra_tools: list[str] | None = None) -> dict:
    """Return documentation references for a profile's tools and profile-level docs."""
    return _get_service().docs_index(profile=profile, extra_tools=extra_tools)


@mcp.tool(annotations=_READ_ONLY)
def resolve_doc(doc_path: str) -> dict:
    """Read a documentation file by its relative path. Returns content or null."""
    content = _get_service().resolve_doc(doc_path)
    return {"path": doc_path, "content": content}


# --- Resources: read-only data (addressed by lackpy:// URIs) -------------------

@mcp.resource("lackpy://config")
def config() -> dict:
    """The current workspace configuration: inference order, providers, profile default,
    sandbox settings."""
    return _get_service().get_config()


@mcp.resource("lackpy://providers")
def provider_list() -> list[dict]:
    """Configured inference providers with plugin type, model, and availability."""
    return _get_service().provider_list()


@mcp.resource("lackpy://language-spec")
def language_spec() -> dict:
    """The lackpy language specification: allowed AST nodes, builtins, forbidden names."""
    return _get_service().language_spec()


@mcp.resource("lackpy://toolbox")
def toolbox_list() -> list[dict]:
    """Every registered tool with metadata (the full toolbox a profile selects from)."""
    return _get_service().toolbox_list()


@mcp.resource("lackpy://profiles")
def profile_list() -> list[dict]:
    """Available named tool-set / profile files in the workspace."""
    return _get_service().profile_list()
