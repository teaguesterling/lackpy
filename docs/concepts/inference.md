# Inference Pipeline

## Tiers

| Tier | Provider | Plugin | Latency | Requires |
|------|----------|--------|---------|----------|
| 0 | `TemplatesProvider` | built-in | ~0 ms | `.lackpy/templates/*.tmpl` files |
| 1 | `RulesProvider` | built-in | ~0 ms | nothing |
| 2 | `WoollamaProvider` | `woollama` | 200–3000 ms | a `"<provider>/<model>"` model in config (local Ollama, or a cloud key) |
| — | `CascadeProvider` | `cascade` | 200–2000 ms | optional; a raw-completion strategy for small Ollama coder models |

The LLM tier is a single `WoollamaProvider`: model calls go through woollama's
model-management core, which routes a `"<provider>/<model>"` string to any
woollama-known backend (Ollama, Anthropic, OpenAI, …). lackpy no longer ships a
provider per vendor. `CascadeProvider` is a separate, optional *strategy* (not a
vendor backend) — see [Tier 2](#tier-2-the-llm).

The dispatcher tries each available provider in priority order. A provider is skipped if `available()` returns `False` (e.g. `woollama.core` is not importable). If a provider returns a syntactically valid program that fails AST validation, the dispatcher feeds the errors back for one retry before moving on.

---

## Tier 0 — Templates

Templates are `.tmpl` files in `.lackpy/templates/`. Each file contains a frontmatter block and a program body:

```
---
name: read-file
pattern: "read the file {path}"
success_count: 12
fail_count: 0
---
content = read_file('{path}')
content
```

The `pattern` field is a mini-template: `{name}` placeholders are converted to named regex groups. The intent is matched case-insensitively. On match, placeholders in the program body are substituted with the captured values.

Templates are checked in sorted filename order. The first match wins.

---

## Tier 1 — Rules

The rules tier uses direct regex matching for common intents. It handles:

- `read (the )? file <path>` → `content = read_file('<path>')\ncontent`
- `find (the )? definition(s)? (of|for) <name>` → `results = find_definitions('<name>')\nresults`
- `find (all)? callers|usages|references (of|for) <name>` → `results = find_callers('<name>')\nresults`
- `(find|list) all <ext> files` → `files = find_files('**/*.<ext>')\nfiles`
- `glob <pattern>` → `files = find_files('<pattern>')\nfiles`

Rules are only applied if the corresponding tool name appears in the namespace description. The rules tier always returns `available() = True`.

---

## Tier 2 — The LLM

The `WoollamaProvider` sends a structured system prompt + user intent to a model. The system prompt describes:

- The available tools and their signatures
- The `ALLOWED_BUILTINS`
- Any pre-set parameter variables
- The constraints (no `import`, `def`, `class`, etc.)

If the first generation fails validation, the errors are appended as a few-shot correction message and the model is called again once.

The raw model call is delegated to woollama's `complete()`, which routes a
`"<provider>/<model>"` string to the right backend — `ollama/…` for a local model,
`anthropic/…` / `openai/…` / etc. for a cloud model (with the relevant API key in
the environment). One provider, any backend; lackpy stops doing per-vendor HTTP.

**Optional — the cascade strategy.** `CascadeProvider` (`plugin = "cascade"`) is a
*different way to generate*, not a different vendor: it uses Ollama's raw
`/api/generate` completion endpoint with pattern-completion prompting, which often
beats chat-template prompting for very small coder models. It tries several models in
speed order and returns the first that validates. Add it to `order` as an extra tier
when you're driving tiny local models.

---

## Dispatch flow

```
for provider in providers:
    if not provider.available():
        continue

    raw = await provider.generate(intent, namespace_desc)
    program = sanitize_output(raw)
    result = validate(program, allowed_names, extra_rules)

    if result.valid:
        return GenerationResult(program, provider.name, elapsed_ms)

    # One retry with error feedback
    raw = await provider.generate(intent, namespace_desc, error_feedback=result.errors)
    program = sanitize_output(raw)
    result = validate(program, allowed_names, extra_rules)

    if result.valid:
        return GenerationResult(program, provider.name, elapsed_ms)

raise RuntimeError("All providers failed")
```

---

## Config example

```toml
[inference]
order = ["templates", "rules", "local", "cloud-fallback"]

[inference.providers.local]
plugin = "woollama"
model = "ollama/qwen2.5-coder:1.5b"
base_url = "http://localhost:11434/v1"
temperature = 0.2

[inference.providers.cloud-fallback]
plugin = "woollama"
model = "anthropic/claude-haiku-4-5"   # needs ANTHROPIC_API_KEY in the environment
```

The `order` list controls priority. Built-in providers (`templates`, `rules`) are always prepended regardless of their position in `order`.

!!! note "Model choice is local config, not a package default"
    Which model you run is a **per-machine / per-deployment decision**, not something
    lackpy bakes in. The package's built-in fallback is the generic, small
    `qwen2.5-coder:1.5b`; set the model your host actually serves best in
    `.lackpy/config.toml` (under `[inference.providers.<name>]`). That file is
    **gitignored** — copy `.lackpy/config.toml.example` to start. Don't commit a
    machine-specific big-model pin into the repo; keep the shipped default generic.

---

## The ratchet

The ratchet pattern is a workflow built on top of the template tier:

1. Delegate an intent (`lackpy -c "..."`) — handled by rules or an LLM on the first call.
2. Verify the result is correct.
3. Save the validated program as a **`.tmpl` template** with an intent `pattern:`.
4. Subsequent delegates with matching intents hit tier 0 — zero latency, guaranteed valid.

Over time, the template library grows and LLM calls become less frequent. The template tier acts as a ratchet: once an intent is captured, it stays captured.

```bash
# Step 1: first run (rules or LLM tier)
lackpy -c "read the file pyproject.toml" --profile read_file
```

```python
# Step 2: capture it as a pattern-matched template (Python API).
# The CLI's `--create` flag saves a run-by-path *Lackey file* instead — that's a
# separate reuse mechanism and is NOT matched against future intents. Only `.tmpl`
# files with a `pattern:` populate the tier-0 templates cache, and today the only
# way to write one (other than authoring it by hand) is svc.create(pattern=...).
import asyncio
from lackpy.service import LackpyService

svc = LackpyService()
asyncio.run(svc.create(
    program="content = read_file('pyproject.toml')\ncontent",
    name="read-pyproject",
    pattern="read the file pyproject.toml",
    profile=["read_file"],
))
```

```bash
# Step 3: future runs hit tier 0
lackpy -c "read the file pyproject.toml" --profile read_file
# generation_tier: "templates"
```

---

## Custom providers

Inference providers implement a simple protocol. See [Extending: Inference Providers](../extending/inference-providers.md) for the full guide.

The minimum interface is:

```python
class MyProvider:
    @property
    def name(self) -> str: ...

    def available(self) -> bool: ...

    async def generate(
        self,
        intent: str,
        namespace_desc: str,
        config: dict | None = None,
        error_feedback: list[str] | None = None,
    ) -> str | None: ...
```

Register the provider on the service's dispatcher by appending it to `svc._inference_providers` before calling `delegate` or `generate`.
