# langchain-lackpy

LangChain integration for [lackpy](https://github.com/teague/lackpy) — safe, graded, restricted Python execution as a LangChain tool.

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

# Individual tools for ReAct agents
tools = toolkit.get_tools()

# Single delegate tool (safe PythonREPLTool replacement)
delegate = toolkit.as_delegate()
```
