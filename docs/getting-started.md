# Getting Started

## Installation

=== "Base (no inference)"

    The core validator, runner, and trace work with no extra dependencies:

    ```bash
    pip install lackpy
    ```

    With the base install you can validate and run programs, manage kits and templates, and use the full Python API — but inference (delegate and `--generate`) will only work via the built-in `templates` and `rules` tiers. Compositional intents won't generate until you configure a local Ollama model with `lackpyctl init --ollama-model`.

=== "With Ollama"

    Local LLM inference using [Ollama](https://ollama.com) works with the base
    install — model calls route through woollama's core (a dependency), no extra
    needed:

    ```bash
    pip install lackpy
    ```

    Then pull whatever model your Ollama host serves best — the choice is per-machine, not a package default:

    ```bash
    ollama pull qwen2.5-coder:1.5b
    ```

=== "Full (Ollama + Anthropic)"

    For all inference providers:

    ```bash
    pip install "lackpy[full]"
    ```

    Set your Anthropic key if using the Anthropic provider:

    ```bash
    export ANTHROPIC_API_KEY=sk-ant-...
    ```

---

## Optional dependencies

| Extra | Package | Required for |
|-------|---------|-------------|
| `ollama` | `ollama` | Local model inference via Ollama |
| `anthropic` | `anthropic` | Cloud inference via Anthropic API |
| `tomli` | `tomli` | TOML config parsing on Python < 3.11 |

Python 3.11+ ships `tomllib` in the standard library; `tomli` is only needed on older versions.

---

## Initialize a workspace

lackpy stores configuration, kits, and templates under `.lackpy/` in your workspace directory. Create this structure with `lackpyctl`:

```bash
cd my-project
lackpyctl init
```

This creates:

```
.lackpy/
  config.toml       # inference order, kit defaults, sandbox settings
  kits/             # .kit files defining tool subsets
  templates/        # .tmpl files for the ratchet pattern
```

The inference order wires in a local Ollama provider so compositional intents work once a model is served. The model choice is **per-machine** — configure it at init time:

```bash
lackpyctl init --ollama-model codellama:7b
```

### Config file

The generated `.lackpy/config.toml` looks like:

```toml
[inference]
order = ["templates", "rules", "local"]

[inference.providers.local]
plugin = "woollama"
model = "ollama/qwen2.5-coder:1.5b"
base_url = "http://localhost:11434/v1"

[kit]
default = "debug"

[sandbox]
enabled = false
timeout_seconds = 120
memory_mb = 512
```

See [Concepts: Inference Pipeline](concepts/inference.md) for all config options.

---

## First use — CLI

Check what tools are available:

```bash
lackpyctl toolbox list
```

Generate and run a program (delegate is the default `lackpy -c` mode):

```bash
lackpy -c "read the file README.md" --kit read_file
```

The output is JSON with the generated program, trace, and result:

```json
{
  "success": true,
  "program": "content = read_file('README.md')\ncontent",
  "grade": {"w": 1, "d": 1},
  "generation_tier": "rules",
  "generation_time_ms": 0.4,
  "execution_time_ms": 1.2,
  "total_time_ms": 1.6,
  "trace": [
    {"step": 0, "tool": "read_file", "args": {"path": "README.md"}, "result": "...", "duration_ms": 1.1, "success": true, "error": null}
  ],
  "output": "# My Project\n...",
  "stdout": "",
  "error": null
}
```

!!! note "Printed output is captured"
    When a generated program `print()`s its answer instead of leaving a bare final expression, the printed text is captured in `stdout` and surfaced as `output` (so `print(...)` no longer yields `output: null`).

!!! tip "Check inference tier"
    The `generation_tier` field tells you which provider handled the request: `templates` (tier 0), `rules` (tier 1), `ollama` (tier 2), or `anthropic` (tier 3).

---

## First use — Python API

```python
import asyncio
from lackpy.service import LackpyService

async def main():
    svc = LackpyService()

    # Validate a program manually
    result = svc.validate(
        'content = read_file("pyproject.toml")\ncontent',
        kit=["read_file"],
    )
    print(result.valid)   # True
    print(result.errors)  # []

    # Generate and run
    result = await svc.delegate(
        intent="read the file pyproject.toml",
        kit=["read_file"],
    )
    print(result["output"])

asyncio.run(main())
```

!!! note "Async API"
    `delegate`, `generate`, and `run_program` are all `async` — they must be awaited inside an async context or wrapped with `asyncio.run()`.

---

## Next steps

- [Tutorial](tutorial.md) — hands-on walkthrough of every feature
- [Concepts: Architecture](concepts/architecture.md) — understand the pipeline
- [Concepts: Language Spec](concepts/language-spec.md) — what Python constructs are allowed
- [Concepts: Kits & Toolbox](concepts/kits.md) — tool organization
- [CLI Reference](reference/cli.md) — complete CLI documentation
