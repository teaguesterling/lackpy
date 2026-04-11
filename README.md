# lackpy

**Python that lacks most of Python.** A micro-inferencer that translates natural language intent into sandboxed tool-composition programs.

[![PyPI](https://img.shields.io/pypi/v/lackpy)](https://pypi.org/project/lackpy/)
[![Tests](https://github.com/teaguesterling/lackpy/actions/workflows/ci.yml/badge.svg)](https://github.com/teaguesterling/lackpy/actions)
[![Docs](https://readthedocs.org/projects/lackpy/badge/?version=latest)](https://lackpy.readthedocs.io)

## What it does

```bash
$ lackpy delegate "read file main.py and count its lines" --kit read_file,find_files
```

lackpy takes an intent, generates a restricted Python program using a local 1.5B model, validates it against a strict AST whitelist, and runs it with traced tool calls. One MCP call replaces N tool round-trips.

## Install

```bash
pip install lackpy            # core (zero dependencies)
pip install lackpy[ollama]    # + local model inference
pip install lackpy[full]      # + all optional features
```

## Quick start

```bash
lackpy init --ollama-url http://localhost:11434
lackpy delegate "find all python files" --kit read_file,find_files
```

```python
from lackpy import LackpyService, ToolSpec, ArgSpec
import asyncio

async def main():
    svc = LackpyService()
    result = await svc.delegate("read file README.md", kit=["read_file"])
    print(result["output"])

asyncio.run(main())
```

## Documentation

Full docs at [lackpy.readthedocs.io](https://lackpy.readthedocs.io):

- [Getting Started](https://lackpy.readthedocs.io/getting-started/)
- [Tutorial](https://lackpy.readthedocs.io/tutorial/)
- [API Reference](https://lackpy.readthedocs.io/reference/api/)

## Part of the Rigged Suite

lackpy is part of the [Rigged](https://github.com/teaguesterling) developer tool suite but is independently installable.

## License

MIT
