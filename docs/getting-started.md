# Getting Started

## Installation

=== "Base (no inference)"

    The core validator, runner, and trace work with no extra dependencies:

    ```bash
    pip install lackpy
    ```

    With the base install you can validate and run programs, manage kits and templates, and use the full Python API — but inference (`delegate`, `generate`) will only work via the built-in `templates` and `rules` tiers.

=== "With Ollama"

    For local LLM inference using [Ollama](https://ollama.com):

    ```bash
    pip install "lackpy[ollama]"
    ```

    Then pull a model:

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

lackpy stores configuration, kits, and templates under `.lackpy/` in your workspace directory. Create this structure with:

```bash
cd my-project
lackpy init
```

This creates:

```
.lackpy/
  config.toml       # inference order, kit defaults, sandbox settings
  kits/             # .kit files defining tool subsets
  templates/        # .tmpl files for the ratchet pattern
```

To configure a specific Ollama model at init time:

```bash
lackpy init --ollama-model codellama:7b
```

### Config file

The generated `.lackpy/config.toml` looks like:

```toml
[inference]
order = ["templates", "rules", "ollama-local"]

[inference.providers.ollama-local]
plugin = "ollama"
host = "http://localhost:11434"
model = "qwen2.5-coder:1.5b"

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
lackpy toolbox list
```

Generate and run a program:

```bash
lackpy delegate "read the file README.md" --kit read_file
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
  "error": null
}
```

!!! tip "Check inference tier"
    The `generation_tier` field tells you which provider handled the request: `templates` (tier 0), `rules` (tier 1), `ollama` (tier 2), or `anthropic` (tier 3).

---

## First use — Python API

```python
import asyncio
from lackpy import LackpyService

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
