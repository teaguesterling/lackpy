"""FastMCP server: thin adapter over the service layer.

The surface is split (RFC 0002 §11) so an orchestrator sees a small, clear set:

- **Tools** are *actions* — generate/run code, validate, save. Each carries MCP
  annotations (read-only vs. executes-code/destructive) so a caller can gate them.
- **Resources** are read-only *data* — config, language spec, providers —
  addressed by `lackpy://…` URIs instead of cluttering the tool list.

No kit in the agent's view (stepy100 Plan 2, Task 7): the tool-set tools and
resources (`profile_create`, `profile_info`, `docs_index`, `resolve_doc`,
`lackpy://toolbox`, `lackpy://profiles`) are gone, and `lackpyctl` keeps the
internal toolbox views.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from ..service import LackpyService

mcp = FastMCP("lackpy")
_service: LackpyService | None = None
_workspace: Path | None = None

# Conservative annotation presets. `delegate`/`run_program` execute code that may
# write files (through pathlib.Path, under the workspace's permission rules) — mark
# non-read-only + potentially destructive + open-world. `create` writes a file
# (non-destructive, idempotent by name). The rest are read-only.
_EXECUTES = ToolAnnotations(readOnlyHint=False, destructiveHint=True,
                            idempotentHint=False, openWorldHint=True)
_WRITES_FILE = ToolAnnotations(readOnlyHint=False, destructiveHint=False,
                               idempotentHint=True, openWorldHint=False)
_READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)


def set_workspace(workspace: Path) -> None:
    """Set the workspace directory and reset the service singleton.

    The old service is closed, so its run zygote does not outlive it.
    """
    global _workspace, _service
    _workspace = workspace
    if _service is not None:
        _service.close()
    _service = None


def _get_service() -> LackpyService:
    global _service
    if _service is None:
        _service = LackpyService(workspace=_workspace)
    return _service


def _json_safe(value: Any) -> Any:
    """``value`` as strict JSON data, for a run result at the MCP boundary (review I2, M9).

    A program's output and trace may hold any data value (``runner._is_data``),
    and FastMCP's serializer fails on bytes that are not UTF-8 and writes nan
    and infinity as the non-standard ``NaN`` and ``Infinity``. So:

    - bytes become their ``repr`` text (``b'\\xff'``), as a non-data output
      already does;
    - a float nan or infinity becomes the text ``"nan"``, ``"inf"`` or
      ``"-inf"`` (Python's ``repr``); a finite float stays a number;
    - a tuple or set becomes a list (as FastMCP already does);
    - a dict key that is not a str becomes its ``repr`` text (``(1, 2)`` gives
      ``"(1, 2)"``, ``None`` gives ``"None"``, ``1`` gives ``"1"``).
    """
    kind = type(value)
    if kind is bytes or kind is bytearray:
        return repr(bytes(value))
    if kind is float and not math.isfinite(value):
        return repr(value)
    if kind is dict:
        return {k if type(k) is str else repr(k): _json_safe(v) for k, v in value.items()}
    if kind in (list, tuple, set, frozenset):
        return [_json_safe(v) for v in value]
    return value


# --- Tools: actions -----------------------------------------------------------
#
# No kit in the agent's view (stepy100 Plan 2, Task 7): no tool takes
# `extra_tools`, and `profile` is a profile name only.

@mcp.tool(annotations=_EXECUTES)
async def delegate(intent: str, profile: str | None = None,
                   params: dict[str, Any] | None = None,
                   rules: list[str] | None = None,
                   timeout: float | None = None) -> dict:
    """Generate a program from a natural-language intent and run it. Use for a
    self-contained, multi-step mechanical task. The program is plain Python that
    the lackpy validator checks; it reads and writes files only through
    pathlib.Path, under the workspace's permission rules. `profile` names a
    configured `[profiles.<name>]` entry (model, mode, rules); omit it for the
    default. `timeout` is the run's limit in seconds (default: the sandbox
    timeout, 120 s), up to the profile's max_timeout. Returns a lean result
    (success, output, stdout, program, the traced calls, error_phase:
    "timeout" when the run was stopped at the limit, "refused" when it was
    refused); call `run_program` with the program if you need the full step trace.
    """
    r = await _get_service().delegate(intent, profile=profile, params=params, rules=rules,
                                      timeout=timeout)
    return _json_safe({
        "success": r.get("success"),
        "output": r.get("output"),
        "error": r.get("error"),
        "error_phase": r.get("error_phase"),
        "stdout": r.get("stdout"),
        "program": r.get("program"),
        "generation_tier": r.get("generation_tier"),
        "total_time_ms": r.get("total_time_ms"),
        "tools_called": [s.get("tool") for s in r.get("trace", [])],
    })


@mcp.tool(annotations=_READ_ONLY)
async def generate(intent: str, profile: str | None = None,
                   params: dict[str, Any] | None = None,
                   rules: list[str] | None = None) -> dict:
    """Generate a program from intent WITHOUT running it (inspect before executing)."""
    result = await _get_service().generate(intent, profile=profile, params=params, rules=rules)
    return {"program": result.program, "provider": result.provider_name,
            "generation_time_ms": result.generation_time_ms}


@mcp.tool(annotations=_EXECUTES)
async def run_program(program: str, profile: str | None = None,
                      params: dict[str, Any] | None = None,
                      rules: list[str] | None = None,
                      timeout: float | None = None) -> dict:
    """Validate and run a program you already have (skips generation). The program
    is plain Python that the lackpy validator checks; it reads and writes files
    only through pathlib.Path, under the workspace's permission rules. With a
    profile whose `execution` is "literate", the program is a literate markdown
    document (prose and ```lackpy code cells): `stdout` is the rendered document,
    `output` is the value of the last code cell's trailing expression, and
    `params` replace the defaults of the document's `params = {...}` literal.
    `rules` names extra checks: no_loops, no_nested_calls, max_depth(N),
    max_calls(N); on the literate axis they apply to every cell. `timeout` is
    the run's limit in seconds (default: the sandbox timeout, 120 s); a value
    over the profile's max_timeout is refused before the run. On the literate
    axis the limit is checked against the cap but not yet enforced.
    `error_phase` is "timeout" when the run was stopped at the limit, "refused"
    when it was refused, else null. Bytes, nan and infinity come back as their
    Python repr text."""
    result = await _get_service().run_program(program, profile=profile, params=params,
                                              rules=rules, timeout=timeout)
    return _json_safe({
        "success": result.success, "output": result.output, "error": result.error,
        "error_phase": result.error_phase, "stdout": result.stdout,
        "trace": [{"step": e.step, "tool": e.tool, "args": e.args, "result": e.result,
                   "success": e.success, "error": e.error} for e in result.trace.entries]})


@mcp.tool(annotations=_WRITES_FILE)
async def create(program: str, profile: str | None = None,
                 name: str = "", pattern: str | None = None) -> dict:
    """Validate a program and save it as a reusable template file."""
    return await _get_service().create(program, profile=profile, name=name, pattern=pattern)


@mcp.tool(annotations=_READ_ONLY)
def validate(program: str, profile: str | None = None,
             rules: list[str] | None = None) -> dict:
    """Check a program against the lackpy validator and any extra rules, without running it.
    Each entry of `violations` has the rule id, line, message and the hint that names
    the approved alternative."""
    result = _get_service().validate(program, profile=profile, rules=rules)
    # Review I3 (spec D17): each violation's hint names the approved alternative.
    return {"valid": result.valid, "errors": result.errors, "calls": result.calls,
            "variables": result.variables,
            "violations": [{"rule": v.rule_id, "line": v.line, "message": v.message,
                            "hint": v.hint} for v in result.violations]}


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
