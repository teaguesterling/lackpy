# lackpy

**Python that lacks most of Python.** lackpy runs plain Python that a validator has checked.

[![PyPI](https://img.shields.io/pypi/v/lackpy)](https://pypi.org/project/lackpy/)
[![Tests](https://github.com/teaguesterling/lackpy/actions/workflows/ci.yml/badge.svg)](https://github.com/teaguesterling/lackpy/actions)
[![Docs](https://readthedocs.org/projects/lackpy/badge/?version=latest)](https://lackpy.readthedocs.io)

## What it does

A lackpy program is a plain Python 3 program. It uses only a small part of the
standard library, and it reads and writes files through `pathlib.Path`:

```python
from pathlib import Path

path = Path("notes.md")
text = path.read_text()
assert text.count("Status: draft") == 1, "the status line occurs once"
path.write_text(text.replace("Status: draft", "Status: final"))
```

Before a program runs, a validator checks it against the
[style guide](docs/reference/style-guide.md). Then lackpy runs it. Each file
access goes through a [path policy](docs/concepts/paths.md) that reads the
`Read(...)` and `Edit(...)` permission rules of Claude Code. A refusal stops the
program with an error. It never changes a result.

The same program runs unchanged under `python3` from the workspace directory,
with the same output. Inside lackpy, it can only do less.

lackpy can also write the program. `delegate` sends a natural-language intent to
a model that you configure, then validates and runs the program it gets back.

## Run a program

`run_program` validates and runs a program that you already have:

```bash
lackpy patch.py                      # a program file is the first argument
lackpy count.py --param name=README.md
```

```python title="Python API: run_program"
import asyncio
from lackpy.service import LackpyService

PROGRAM = '''
from pathlib import Path

params = {"name": "README.md"}
text = Path(params["name"]).read_text()
print(len(text.splitlines()))
'''

async def main():
    svc = LackpyService()
    result = await svc.run_program(PROGRAM, params={"name": "pyproject.toml"})
    print(result.success, result.stdout, result.error)

asyncio.run(main())
```

## Delegate an intent

`delegate` generates a program from an intent, then runs it:

```bash
lackpy -c "count the TODO lines in each .cpp file under src"
```

```python title="Python API: delegate"
import asyncio
from lackpy.service import LackpyService

async def main():
    svc = LackpyService()
    result = await svc.delegate("count the TODO lines in each .cpp file under src")
    print(result["program"])
    print(result["stdout"])

asyncio.run(main())
```

The MCP server (`lackpyctl mcp serve`, with `pip install lackpy[mcp]`) gives the same
two actions as the `run_program` and `delegate` tools.

## Install

```bash
pip install lackpy            # runtime + local/cloud model inference (via woollama)
pip install lackpy[full]      # + all optional features (sandbox, mcp, blq, …)

pip install lackpy-lang       # JUST the language (validator, style guide, spec),
                              # no runtime — for tools that only reason about programs
```

Model calls go through woollama's core, a dependency. Local Ollama and cloud
backends (Anthropic, OpenAI, …) all work with no extra for each vendor. A cloud
backend needs its API key in your environment.

`lackpy` and `lackpy-lang` share the `lackpy` namespace ([PEP 420](https://peps.python.org/pep-0420/)).
Neither ships a top-level `lackpy/__init__.py`, so import from submodules
(`from lackpy.lang import validate`, `from lackpy.service import LackpyService`).

### Development (monorepo)

This repo builds two distributions. For an editable checkout, install **both**.
`lackpy` pins `lackpy-lang`, so install the local copy first:

```bash
pip install -e ./packages/lackpy-lang -e ".[dev]"
```

## Quick start

```bash
lackpyctl init --ollama-model qwen2.5-coder:7b   # writes .lackpy/config.toml
lackpy -c "read the file README.md"
```

`lackpyctl init` adds a local Ollama model to the inference order. The model
choice is per machine, not a package default. Without a model, only the
`templates` and `rules` tiers run, so most intents do not generate.

## Documentation

Full docs at [lackpy.readthedocs.io](https://lackpy.readthedocs.io):

- [Getting Started](docs/getting-started.md)
- [Tutorial](docs/tutorial.md)
- [File access: the path policy](docs/concepts/paths.md)
- [Style guide](docs/reference/style-guide.md)
- [API Reference](docs/reference/api.md)

## Part of the Rigged Suite

lackpy is part of the [Rigged](https://github.com/teaguesterling) developer tool suite but is independently installable.

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE). Commercial licenses are available: see [COMMERCIAL.md](COMMERCIAL.md).
