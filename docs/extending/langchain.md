# LangChain Integration

The `langchain-lackpy` package bridges lackpy's safe, graded program execution into the LangChain and LangGraph ecosystem. It provides a thin adapter — all the safety, grading, and policy logic stays in lackpy.

---

## Installation

```bash
pip install langchain-lackpy
```

For LangGraph node support:

```bash
pip install langchain-lackpy[graph]
```

---

## Quick start

Everything flows through `LackpyToolkit`, which wraps a lackpy kit as a LangChain `BaseToolkit`.

```python
from pathlib import Path
from lackpy import LackpyService
from langchain_lackpy import LackpyToolkit

svc = LackpyService(workspace=Path("."))
toolkit = LackpyToolkit(service=svc, kit=["read_file", "find_files"])
```

The `kit` parameter accepts the same types as `LackpyService.delegate()` — a kit name, a list of tool names, or a dict mapping.

From the toolkit you can get three different things:

```python
# 1. Individual LangChain tools
tools = toolkit.get_tools()

# 2. A single delegate tool (safe PythonREPLTool replacement)
delegate = toolkit.as_delegate()

# 3. A LangGraph node
node = toolkit.as_node()
```

---

## Individual tools — `get_tools()`

Each tool in the kit becomes a standard LangChain `BaseTool`. These work in any ReAct agent, `ToolNode`, or chain.

```python
tools = toolkit.get_tools()
# [LackpyToolWrapper(name='read_file'), LackpyToolWrapper(name='find_files')]
```

Each wrapped tool preserves the full `ToolSpec` metadata:

- **name** and **description** map directly
- **args_schema** is a Pydantic model built from the tool's `ArgSpec` list
- **metadata** carries `provider`, `grade_w`, and `effects_ceiling`

This means you can filter tools before handing them to an agent:

```python
safe_tools = [t for t in toolkit.get_tools() if t.metadata["grade_w"] <= 1]
```

### Using with a ReAct agent

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=my_llm,
    tools=toolkit.get_tools(),
    prompt="You have access to file system tools.",
)

result = agent.invoke({"messages": [HumanMessage(content="Find all .py files")]})
```

---

## Delegate tool — `as_delegate()`

The delegate tool wraps `LackpyService.delegate()` — lackpy's full generate-then-execute pipeline — as a single `BaseTool`. This is the **safe alternative to `PythonREPLTool`**.

Instead of the LLM deciding which tools to call one at a time across multiple turns, lackpy generates a complete program that composes multiple tools, validates it for safety, grades it, and executes it in a restricted interpreter — all in one call.

```python
delegate = toolkit.as_delegate()
# BaseTool with name="lackpy_delegate"
```

The agent's only input is a natural language intent:

```python
result = await delegate.ainvoke({"intent": "Read config.yaml and extract the database host"})
# "db.example.com"
```

### Naming for multi-kit setups

If you have multiple kits, give each delegate a distinct name so the agent can choose between them:

```python
file_toolkit = LackpyToolkit(service=svc, kit=["read_file", "find_files"])
data_toolkit = LackpyToolkit(service=svc, kit=["query_db", "format_csv"])

file_delegate = file_toolkit.as_delegate(name="lackpy_files")
data_delegate = data_toolkit.as_delegate(name="lackpy_data")

agent = create_react_agent(model=my_llm, tools=[file_delegate, data_delegate])
```

### Error handling

On success, the delegate returns the program's output as a string. On failure, it returns the error message — this lets the agent reason about the failure rather than crashing. Infrastructure errors (missing kit, unconfigured service) raise `ToolException`.

---

## LangGraph node — `as_node()`

For LangGraph state graphs, the toolkit produces an async node function directly.

```python
from langgraph.graph import StateGraph, START, END

node = toolkit.as_node(intent_key="intent", result_key="lackpy_result")

builder = StateGraph(MyState)
builder.add_node("lackpy", node)
builder.add_edge("planner", "lackpy")
builder.add_conditional_edges("lackpy", route_on_success, [END, "retry"])
graph = builder.compile()
```

The node reads `state[intent_key]`, calls `service.delegate()`, and returns `{result_key: full_result_dict}`. The result dict includes `success`, `output`, `error`, `trace`, `grade`, `files_read`, `files_modified`, and timing information.

---

## Convenience constructor

If you don't need to configure the service separately, `from_config()` discovers `lackpy.toml` / `.lackpy/config.toml` automatically:

```python
toolkit = LackpyToolkit.from_config(workspace=Path("."), kit="filesystem")
```

---

## When to use what

| Scenario | Use |
|----------|-----|
| Agent calls tools one at a time (ReAct) | `get_tools()` |
| Agent delegates a multi-tool task to lackpy | `as_delegate()` |
| Replacing `PythonREPLTool` / `PythonAstREPLTool` | `as_delegate()` |
| lackpy as a node in a LangGraph pipeline | `as_node()` |
| Multiple tool kits with different safety levels | Multiple toolkits, each with `as_delegate(name=...)` |

---

## How it relates to lackpy's policy layer

The langchain adapter does not participate in policy resolution. When the delegate tool calls `service.delegate()`, lackpy internally resolves the kit, runs the PolicyLayer chain (which may include Kibitzer coaching and umwelt restrictions), generates and validates a program, and executes it. All of that is transparent to the langchain layer.

If you need to restrict tools based on the calling agent's identity, configure a `PolicySource` on the lackpy service — see [Writing Policy Sources](policy-sources.md).
