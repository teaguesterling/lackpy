# langchain-lackpy Integration Package Design

## Goal

A thin bridge package that exposes lackpy's safe, graded program generation and execution to the langchain/langgraph ecosystem. Primary audience: langchain users who want a safe alternative to `PythonREPLTool`.

## Architecture

`langchain-lackpy` is a separate package in `packages/langchain-lackpy/` within the lackpy monorepo. It depends on `langchain-core` and `lackpy` — never the full `langchain`. `langgraph` is an optional dependency.

lackpy remains zero-dependency. The langchain package is a thin adapter — if something requires complex logic, that logic belongs in lackpy core, not the bridge.

### Why not build lackpy on langchain?

lackpy and langchain solve different problems at the tool layer. langchain's `BaseTool` is a callable an agent invokes in a ReAct loop. lackpy's `ToolSpec` is metadata consumed by a code generator, then a validator, then a restricted interpreter. The grade system, effects ceiling, provider lineage, grammar restrictions, and AST validation have no langchain equivalent. lackpy's value is that it's a fundamentally different execution model.

---

## Package Structure

```
packages/langchain-lackpy/
  langchain_lackpy/
    __init__.py          # re-exports LackpyToolkit
    toolkit.py           # LackpyToolkit (BaseToolkit subclass, primary entry point)
    _tool_wrapper.py     # Individual BaseTool wrappers for ToolSpec
    _delegate.py         # Delegate BaseTool (as_delegate)
    _node.py             # LangGraph node factory (as_node)
    _schema.py           # ArgSpec → Pydantic args_schema conversion
  tests/
    test_toolkit.py
    test_tool_wrapping.py
    test_delegate.py
    test_node.py
    conftest.py          # shared fixtures (mock service, toolbox, kit)
  pyproject.toml
  README.md
```

### Dependencies

```toml
[project]
name = "langchain-lackpy"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "langchain-core>=0.3,<1.0",
    "lackpy>=0.9",
]

[project.optional-dependencies]
graph = ["langgraph>=0.3"]
dev = ["pytest>=8", "pytest-asyncio>=0.24"]
```

---

## LackpyToolkit — Primary Entry Point

`LackpyToolkit` is a `BaseToolkit` subclass that wraps a lackpy kit. All integration flows through it.

### Construction

```python
from langchain_lackpy import LackpyToolkit
from lackpy import LackpyService

svc = LackpyService(workspace=Path("."))

# Named kit
toolkit = LackpyToolkit(service=svc, kit="filesystem")

# Tool list
toolkit = LackpyToolkit(service=svc, kit=["read_file", "write_file"])

# From config (convenience classmethod)
toolkit = LackpyToolkit.from_config(workspace=Path("."), kit="filesystem")
```

The `kit` parameter accepts the same types as `LackpyService.delegate()`: `str | list[str] | dict | None`. This is passed through to lackpy — the langchain layer does not interpret it.

At construction, the toolkit calls `resolve_kit(kit, service.toolbox)` to obtain a `ResolvedKit` with full `ToolSpec` metadata (provider, grade, effects ceiling, args) and resolved callables. This resolved kit is used for tool wrapping and description generation.

### Three outputs

```python
# 1. Individual langchain tools (for ReAct agents)
tools = toolkit.get_tools()

# 2. A single delegate tool (safe PythonREPLTool replacement)
delegate = toolkit.as_delegate()

# 3. A LangGraph node (requires langgraph optional dep)
node = toolkit.as_node()
```

---

## Individual Tool Wrappers — get_tools()

Each `BaseTool` produced by `get_tools()` wraps a single `ToolSpec` and its resolved callable from the kit.

### ToolSpec → BaseTool mapping

| ToolSpec field | BaseTool field |
|---|---|
| `name` | `name` |
| `description` | `description` (enriched with provider/grade info) |
| `args: list[ArgSpec]` | `args_schema` (Pydantic model built from ArgSpec list) |
| `provider`, `grade_w`, `effects_ceiling` | `metadata` dict |

### args_schema generation

Each `ArgSpec(name, type, description)` becomes a Pydantic `Field`. Type strings map to Python types: `"str"` → `str`, `"int"` → `int`, `"float"` → `float`, `"bool"` → `bool`, `"dict"` → `dict`, `"Any"` → `Any`.

This mapping utility belongs in lackpy core (not the bridge) since it's generally useful for any integration that needs to convert ArgSpec to typed schemas. If lackpy doesn't already have it, add it there.

### Execution

- `_run()` calls the resolved callable directly with the provided args and returns the result as a string.
- `_arun()` wraps `_run()` in an executor (resolved callables are synchronous).
- Tool metadata (provider, grade_w, effects_ceiling) is available on the BaseTool instance for programmatic filtering.

### What this enables

Langchain users can take any lackpy toolbox and get standard langchain tools usable in any ReAct agent, `ToolNode`, or chain. Tools carry their lineage — users can filter by provider or grade before handing them to an agent.

---

## Delegate Tool — as_delegate()

A single `BaseTool` wrapping `LackpyService.delegate()` — the "safe PythonREPLTool replacement" and the primary integration surface.

### Construction

```python
delegate = toolkit.as_delegate(
    name="lackpy_filesystem",     # optional, defaults to "lackpy_delegate"
    description="...",            # optional, auto-generated from kit
)
```

### Input schema

One required field:

```python
class DelegateInput(BaseModel):
    intent: str = Field(
        description="Natural language description of what to accomplish"
    )
```

The kit, tools, and policy are all configured at toolkit construction. The calling agent's only job is describing what it wants.

### Execution

- `_arun()` calls `service.delegate(intent=intent, kit=self._kit_config)`.
- On success: returns `result["output"]` as a string.
- On failure: returns `result["error"]` as a string. This lets the ReAct agent reason about the failure rather than crashing.
- Emits the full delegate result dict (trace, grade, timing, files_read, files_modified) via `run_manager` callback metadata for observability.
- `_run()` raises `NotImplementedError` — lackpy's inference pipeline is async-only.
- Infrastructure failures (service not configured, kit not found) raise `ToolException`.

### Auto-generated description

When no description is provided, one is generated from the resolved kit:

> "Safely compose file operations into a validated, restricted Python program. Available operations: read_file(path), find_files(pattern), write_file(path, content), edit_file(path, old_str, new_str). Programs are graded for safety before execution."

---

## LangGraph Node — as_node()

A method on `LackpyToolkit` that returns an async function shaped for `StateGraph.add_node()`.

### Construction

```python
node = toolkit.as_node(
    intent_key="intent",          # state key to read intent from
    result_key="lackpy_result",   # state key to write result to
)
```

### What it does

1. Reads `state[intent_key]` for the intent string.
2. Calls `service.delegate(intent=..., kit=...)`.
3. Returns `{result_key: result}` as a partial state update.

### Usage

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(MyState)
builder.add_node("lackpy", toolkit.as_node())
builder.add_edge("planner", "lackpy")
builder.add_conditional_edges("lackpy", route_on_success, [END, "retry"])
graph = builder.compile()
```

The `langgraph` import is guarded at the module level. Importing `langchain_lackpy` without langgraph installed works fine; calling `as_node()` without it raises `ImportError` with a clear message.

---

## Tool Lineage

A key design principle: lackpy tools are not just names. Each `ToolSpec` carries provider lineage (builtin, MCP, mock/pluckit, custom), a safety grade (`grade_w`, `effects_ceiling`), and typed argument metadata. The langchain wrappers preserve all of this.

This matters because:
- Different providers execute differently (the lackpy interpreter handles dispatch).
- The grade system needs provider info to assess risk.
- MCP tools, builtin functions, pluckit mock functions, and custom callables have different characteristics.
- Langchain users can filter tools by grade or provider before handing them to an agent.

The bridge never flattens tools to just names — it wraps the full `ToolSpec`.

---

## Policy Integration

The PolicyLayer (v0.9.0) handles access control. When the delegate tool calls `service.delegate()`, the service internally:

1. Resolves the kit → `ResolvedKit` (S1: what's operationally available)
2. Resolves policy → `PolicyResult` (S3: what's allowed given context)
3. Generates, validates, and executes the program

The langchain layer does not participate in policy resolution. It names the kit; the service and PolicyLayer handle the rest. If umwelt restricts tools or kibitzer adds coaching, that happens transparently through the policy chain.

---

## Testing Strategy

### Unit tests (mock service, no real inference)

- **test_toolkit.py** — Construction from service, `from_config()`, `get_tools()` returns correct BaseTool instances with right names/descriptions/metadata, `as_delegate()` returns a BaseTool, `as_node()` returns a callable.
- **test_tool_wrapping.py** — ArgSpec → Pydantic args_schema mapping for each type string, metadata carries provider/grade, `_run()` delegates to resolved callable with correct args.
- **test_delegate.py** — Returns output string on success, returns error string on failure, emits full result via run_manager, raises `NotImplementedError` on sync `_run()`.
- **test_node.py** — Reads intent from correct state key, returns result under correct key, clear error when langgraph missing.

### Integration test (optional, requires real service)

One end-to-end test: construct a real `LackpyService` with builtin tools, wrap as toolkit, call delegate via langchain's `invoke()` path, verify output.

---

## Scope boundaries — what v0.1 does NOT include

- **Langsmith/callback integration** — Defer until real usage patterns emerge.
- **Trace-to-ToolMessage conversion** — Defer; the delegate returns output, not a message stream.
- **Bidirectional tool import** (langchain tools → lackpy toolbox) — Possible future work, not v0.1.
- **Custom agent executor** — lackpy replaces the ReAct loop; a custom executor could formalize this, but `as_delegate()` in a standard agent suffices for now.
