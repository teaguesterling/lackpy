# lackpy

**Python that lacks most of Python.** A micro-inferencer that translates natural language intent into sandboxed tool-composition programs.

[![PyPI](https://img.shields.io/pypi/v/lackpy)](https://pypi.org/project/lackpy/)
[![Tests](https://github.com/teaguesterling/lackpy/actions/workflows/ci.yml/badge.svg)](https://github.com/teaguesterling/lackpy/actions)
[![Docs](https://readthedocs.org/projects/lackpy/badge/?version=latest)](https://lackpy.readthedocs.io)

## What it does

```bash
$ lackpy -c "read file main.py and count its lines" --profile read_file,find_files
```

lackpy takes an intent, generates a restricted Python program using a local model you configure (any Ollama-served coder model), validates it against a strict AST whitelist, and runs it with traced tool calls. One MCP call replaces N tool round-trips.

## Install

```bash
pip install lackpy            # runtime + local/cloud model inference (via woollama)
pip install lackpy[full]      # + all optional features (sandbox, mcp, blq, …)

pip install lackpy-lang       # JUST the language (grammar/validator/grader/spec),
                              # no runtime — for tools that only reason about programs
```

Model inference is built in — calls route through woollama's core (a dependency), so
local Ollama and cloud backends (Anthropic, OpenAI, …) all work without a per-vendor
extra; cloud backends just need the relevant API key in your environment.

`lackpy` and `lackpy-lang` share the `lackpy` namespace ([PEP 420](https://peps.python.org/pep-0420/));
neither ships a top-level `lackpy/__init__.py`, so import from submodules
(`from lackpy.lang import validate`, `from lackpy.service import LackpyService`).

### Development (monorepo)

This repo builds two distributions. For a working editable checkout, install **both** —
`lackpy` pins `lackpy-lang`, which isn't on PyPI until its own tag, so install the local
copy first:

```bash
pip install -e ./packages/lackpy-lang -e ".[dev]"
```

## Quick start

```bash
lackpyctl init --ollama-url http://localhost:11434   # writes .lackpy/config.toml
lackpy -c "find all python files" --profile read_file,find_files
```

`lackpyctl init` wires a local Ollama model into the inference order — pick the model
with `--ollama-model` (the choice is per-machine, not a package default). Without it,
only the deterministic templates/rules tiers run and compositional intents won't generate.

```python
from lackpy.service import LackpyService
from lackpy.kit.toolbox import ToolSpec, ArgSpec
import asyncio

async def main():
    svc = LackpyService()
    result = await svc.delegate("read file README.md", profile=["read_file"])
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
