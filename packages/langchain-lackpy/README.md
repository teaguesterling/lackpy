# langchain-lackpy

LangChain integration for [lackpy](https://github.com/teague/lackpy) — safe, graded, restricted Python execution as a LangChain tool.

## Why

LangChain's `PythonREPLTool` lets an LLM run arbitrary Python with no restrictions. lackpy generates restricted Python programs that are validated, graded for safety, and executed in a controlled interpreter. `langchain-lackpy` brings that safety into the LangChain ecosystem.

## Installation

```bash
pip install langchain-lackpy
```

For LangGraph support:

```bash
pip install langchain-lackpy[graph]
```

## Quick Start

```python
from pathlib import Path
from lackpy import LackpyService
from langchain_lackpy import LackpyToolkit

svc = LackpyService(workspace=Path("."))
toolkit = LackpyToolkit(service=svc, kit=["read_file", "find_files"])
```

### Individual tools for ReAct agents

```python
tools = toolkit.get_tools()
# Each tool is a standard BaseTool with provider/grade metadata
```

### Delegate tool (safe PythonREPLTool replacement)

```python
delegate = toolkit.as_delegate()
result = await delegate.ainvoke({"intent": "Find all .py files and read setup.cfg"})
```

### LangGraph node

```python
node = toolkit.as_node()
builder.add_node("lackpy", node)
```

## When to use what

| Scenario | Use |
|----------|-----|
| Agent calls tools one at a time | `get_tools()` |
| Agent delegates a multi-tool task | `as_delegate()` |
| Replacing `PythonREPLTool` | `as_delegate()` |
| Node in a LangGraph pipeline | `as_node()` |

## Documentation

Full guide: [LangChain Integration](https://lackpy.readthedocs.io/extending/langchain/)
