# Inference Pipeline

## Tiers

| Tier | Provider | Plugin | Latency | Requires |
|------|----------|--------|---------|----------|
| 0 | `TemplatesProvider` | built-in | ~0 ms | `.lackpy/templates/*.tmpl` files |
| 1 | `RulesProvider` | built-in | ~0 ms | nothing |
| 2 | `OllamaProvider` | `ollama` | 200–2000 ms | `pip install lackpy[ollama]`, running Ollama |
| 3 | `AnthropicProvider` | `anthropic` | 500–3000 ms | `pip install lackpy[full]`, `ANTHROPIC_API_KEY` |

The dispatcher tries each available provider in priority order. A provider is skipped if `available()` returns `False` (e.g. the `ollama` package is not installed). If a provider returns a syntactically valid program that fails AST validation, the dispatcher feeds the errors back for one retry before moving on.

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
content = read('{path}')
content
```

The `pattern` field is a mini-template: `{name}` placeholders are converted to named regex groups. The intent is matched case-insensitively. On match, placeholders in the program body are substituted with the captured values.

Templates are checked in sorted filename order. The first match wins.

---

## Tier 1 — Rules

The rules tier uses direct regex matching for common intents. It handles:

- `read (the )? file <path>` → `content = read('<path>')\ncontent`
- `find (the )? definition(s)? (of|for) <name>` → `results = find_definitions('<name>')\nresults`
- `find (all)? callers|usages|references (of|for) <name>` → `results = find_callers('<name>')\nresults`
- `(find|list) all <ext> files` → `files = glob('**/*.<ext>')\nfiles`
- `glob <pattern>` → `files = glob('<pattern>')\nfiles`

Rules are only applied if the corresponding tool name appears in the namespace description. The rules tier always returns `available() = True`.

---

## Tier 2 — Ollama

The Ollama provider sends a structured system prompt + user intent to a local model. The system prompt describes:

- The available tools and their signatures
- The `ALLOWED_BUILTINS`
- Any pre-set parameter variables
- The constraints (no `import`, `def`, `class`, etc.)

If the first generation fails validation, the errors are appended to the user message and the model is called again once.

---

## Tier 3 — Anthropic

The Anthropic provider works identically to the Ollama provider but calls the Anthropic Messages API. It is intended as a high-quality fallback for intents that a small local model cannot handle.

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
order = ["templates", "rules", "ollama-local", "anthropic-fallback"]

[inference.providers.ollama-local]
plugin = "ollama"
host = "http://localhost:11434"
model = "qwen2.5-coder:1.5b"
temperature = 0.2
keep_alive = "30m"

[inference.providers.anthropic-fallback]
plugin = "anthropic"
model = "claude-haiku-4-5-20251001"
```

The `order` list controls priority. Built-in providers (`templates`, `rules`) are always prepended regardless of their position in `order`.

---

## The ratchet

The ratchet pattern is a workflow built on top of the template tier:

1. Issue `delegate` — the intent is handled by rules or an LLM on the first call.
2. Verify the result is correct.
3. Issue `create` to save the validated program as a template with an intent pattern.
4. Subsequent `delegate` calls with matching intents hit tier 0 — zero latency, guaranteed valid.

Over time, the template library grows and LLM calls become less frequent. The template tier acts as a ratchet: once an intent is captured, it stays captured.

```bash
# Step 1: first run (rules tier)
lackpy delegate "read the file pyproject.toml" --kit read

# Step 2: save as template
cat > read_pyproject.py << 'EOF'
content = read('pyproject.toml')
content
EOF
lackpy create read_pyproject.py --name read-pyproject --kit read

# Step 3: future runs hit tier 0
lackpy delegate "read the file pyproject.toml" --kit read
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
