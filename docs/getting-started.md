# Getting Started

## Installation

=== "Base"

    ```bash
    pip install lackpy
    ```

    The base install validates and runs programs. It includes model inference
    through woollama's core, a dependency. With no model configured, only the
    `templates` and `rules` tiers generate programs.

=== "With Ollama"

    Local inference with [Ollama](https://ollama.com) works with the base
    install. Pull a model that your Ollama host serves well. The choice is per
    machine, not a package default:

    ```bash
    pip install lackpy
    ollama pull qwen2.5-coder:7b
    ```

=== "Full"

    All optional features (sandbox, MCP server, blq, kibitzer):

    ```bash
    pip install "lackpy[full]"
    ```

    A cloud backend needs its API key in your environment:

    ```bash
    export ANTHROPIC_API_KEY=sk-ant-...
    ```

---

## Optional dependencies

| Extra | Required for |
|-------|-------------|
| `mcp` | The MCP server (`lackpyctl mcp serve`) |
| `sandbox` | The nsjail sandbox executor (Python 3.12 and later) |
| `blq`, `fledgling`, `kibitzer` | Integrations with those tools |
| `full` | All of the above |

---

## Initialize a workspace

The workspace is the directory a run starts in. lackpy keeps its configuration
and templates under `.lackpy/` there:

```bash
cd my-project
lackpyctl init --ollama-model qwen2.5-coder:7b
```

This makes:

```
.lackpy/
  config.toml       # inference order, providers, profiles, [paths] rules
  templates/        # saved programs: Lackey files and .tmpl templates
  kits/             # not used: tool kits were removed
```

### Config file

The generated `.lackpy/config.toml` looks like this:

```toml
[inference]
order = ["templates", "rules", "local"]

[inference.providers.local]
plugin = "woollama"
model = "ollama/qwen2.5-coder:7b"
base_url = "http://localhost:11434/v1"
```

Add a `[paths]` table to change which files a program can read and write. See
[File access](concepts/paths.md). With no rules, a program can read and write
inside the workspace only.

See [Concepts: Inference Pipeline](concepts/inference.md) for the inference options.

---

## First use: CLI

Write a program, and save it as `count.py`:

```python
from pathlib import Path

params = {"name": "README.md"}
text = Path(params["name"]).read_text()
print(len(text.splitlines()))
```

Run it. A program file is the first argument of `lackpy`:

```bash
lackpy count.py
lackpy count.py --param name=pyproject.toml
```

The result is JSON. `stdout` holds what the program printed:

```json
{
  "success": true,
  "output": "42",
  "stdout": "42\n",
  "error": null
}
```

Generate and run a program from an intent:

```bash
lackpy -c "read the file README.md"
```

!!! tip "Which tier answered"
    The `generation_tier` field names the tier that wrote the program:
    `templates`, `rules`, or the name of a model provider.

---

## First use: Python API

```python title="Python API: validate and delegate"
import asyncio
from lackpy.service import LackpyService

PROGRAM = 'from pathlib import Path\n\nprint(Path("pyproject.toml").read_text())\n'

async def main():
    svc = LackpyService()

    result = svc.validate(PROGRAM)
    print(result.valid, result.errors)

    result = await svc.run_program(PROGRAM)
    print(result.stdout)

    result = await svc.delegate("read the file pyproject.toml")
    print(result["stdout"])

asyncio.run(main())
```

!!! note "Async API"
    `delegate`, `generate` and `run_program` are `async`. Await them in an async
    function, or wrap them with `asyncio.run()`.

!!! note "No tool kits"
    lackpy no longer has tool kits. A program uses `pathlib.Path` for file
    access. A profile or a `--profile` value that names tools fails with
    `tool kits were removed in lackpy stepy100; use pathlib.Path (docs/concepts/paths.md)`.

---

## Next steps

- [Tutorial](tutorial.md): the three worked examples, step by step
- [File access](concepts/paths.md): the path policy
- [Style guide](reference/style-guide.md): every rule of the language
- [Profiles](concepts/profiles.md): model and mode per task
- [CLI Reference](reference/cli.md)
