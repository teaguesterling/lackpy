# lackpy

**Python that lacks most of Python.** Translate natural language intent into restricted, graded, sandboxed programs that compose tools safely.

[![PyPI](https://img.shields.io/pypi/v/lackpy)](https://pypi.org/project/lackpy/)
[![Tests](https://github.com/teaguesterling/lackpy/actions/workflows/ci.yml/badge.svg)](https://github.com/teaguesterling/lackpy/actions)
[![Docs](https://readthedocs.org/projects/lackpy/badge/?version=latest)](https://lackpy.readthedocs.io)

## What it does

```bash
$ lackpy delegate "read main.py and count its lines" --kit read_file,find_files
```

An LLM agent asks to "read a file and count lines." Most frameworks hand it an unrestricted Python REPL. lackpy generates a restricted program from intent, validates it against an AST grammar, grades it for safety, and runs it inside an OS-level sandbox with only the tools it needs. One MCP call replaces N tool round-trips.

### Three layers of defense

1. **Language restriction** — programs are validated against a strict AST whitelist before running. No imports, no attribute access, no escape hatches.
2. **Policy grading** — every tool has a safety grade and effects ceiling. The PolicyLayer controls which tools are available based on kit, model, and world-model constraints.
3. **OS-level sandbox** — programs run inside nsjail (or bwrap, with more backends planned) with namespace isolation, seccomp filtering, cgroup limits, and no network access by default.

## Install

```bash
pip install lackpy            # core (zero dependencies)
pip install lackpy[ollama]    # + local model inference
pip install lackpy[sandbox]   # + nsjail sandbox containment
pip install lackpy[full]      # all optional features
```

## Quick start

```bash
lackpy init --ollama-url http://localhost:11434
lackpy delegate "find all python files" --kit read_file,find_files
```

```python
from lackpy import LackpyService
import asyncio

async def main():
    svc = LackpyService()
    result = await svc.delegate("read file README.md", kit=["read_file"])
    print(result["output"])

asyncio.run(main())
```

## Key features

**Multiple interpreters** — restricted Python for tool composition, ast-select for structural code queries, plucker for data extraction, PSS for pattern matching. Each interpreter has its own grammar and validation rules.

**PolicyLayer** — pluggable policy sources (kit baseline, kibitzer coaching, umwelt world-model) resolve tool access, constraints, and prompt hints per request. Sandbox constraints flow through the same policy chain.

**Sandbox containment** — the `SandboxedInterpreter` decorator wraps any interpreter with OS-level isolation. Constraints (memory, time, PIDs, network, filesystem) are compiled to backend-specific configs. Tools are provisioned inside the sandbox as embedded source or bridged via IPC.

**MCP server** — expose lackpy as an MCP tool provider so any MCP-compatible agent can delegate multi-tool tasks through it.

**LangChain integration** — the `langchain-lackpy` package provides `LackpyToolkit` for wrapping lackpy tools as LangChain `BaseTool` instances, a delegate tool (safe REPL replacement), and a LangGraph node.

## Architecture

```
Intent
  -> Inference (local 1-3B model via Ollama, or Anthropic)
  -> AST Validation (grammar whitelist)
  -> Policy Grading (tool access, safety grade, constraints)
  -> Sandboxed Interpreter (nsjail/bwrap containment)
  -> Traced Result (output, tool calls, timing)
```

## Documentation

Full docs at [lackpy.readthedocs.io](https://lackpy.readthedocs.io):

- [Getting Started](https://lackpy.readthedocs.io/getting-started/)
- [Tutorial](https://lackpy.readthedocs.io/tutorial/)
- [API Reference](https://lackpy.readthedocs.io/reference/api/)

## Part of the Rigged Suite

lackpy is part of the [Rigged](https://github.com/teaguesterling) developer tool suite — alongside [umwelt](https://github.com/teaguesterling/umwelt) (world-model policy engine) and [kibitzer](https://github.com/teaguesterling/kibitzer) (LLM coaching) — but is independently installable.

## License

MIT
