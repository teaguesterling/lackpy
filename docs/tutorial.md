# Tutorial

This walkthrough covers every major feature of lackpy. Follow along in order, or jump to any section.

**Prerequisites:** lackpy installed, workspace initialized with `lackpy init`. See [Getting Started](getting-started.md).

---

## 1. Setup

Create a project directory and initialize:

```bash
mkdir my-lackpy-project && cd my-lackpy-project
echo "# Hello from lackpy" > README.md
lackpy init
```

Confirm the setup:

```bash
lackpy status
```

```json
{
  "workspace": "/path/to/my-lackpy-project",
  "config_dir": "/path/to/my-lackpy-project/.lackpy",
  "inference_order": ["templates", "rules"],
  "kit_default": "debug",
  "sandbox_enabled": false,
  "tools": 4
}
```

---

## 2. Pipeline overview

Every `delegate` call passes through this pipeline:

```
                    ┌─────────────────────────────────────────┐
  intent            │  InferenceDispatcher                    │
  ─────────────────►│                                         │
                    │  tier 0: templates  (pattern match)     │
                    │  tier 1: rules      (keyword rules)     │
                    │  tier 2: ollama     (local LLM)         │
                    │  tier 3: anthropic  (cloud LLM)         │
                    └──────────────┬──────────────────────────┘
                                   │ raw program text
                                   ▼
                    ┌──────────────────────────┐
                    │  Validator (AST check)   │
                    │  - allowed node types    │
                    │  - forbidden names       │
                    │  - namespace check       │
                    │  - custom rules          │
                    └──────────────┬───────────┘
                                   │ ValidationResult
                                   ▼
                    ┌──────────────────────────┐
                    │  RestrictedRunner        │
                    │  - traced namespace      │
                    │  - restricted __builtins__│
                    └──────────────┬───────────┘
                                   │ ExecutionResult + Trace
                                   ▼
                              delegate() result
```

The tier system means simple intents are handled deterministically and fast; LLMs are only invoked when no cheaper tier can handle the request.

---

## 3. Validating programs

The validator checks programs before they run. It operates entirely on the AST — no code is executed during validation.

### A valid program

```bash
cat > check.py << 'EOF'
files = glob("**/*.py")
count = len(files)
count
EOF

lackpy validate check.py --kit glob
```

```json
{
  "valid": true,
  "errors": [],
  "calls": ["glob", "len"]
}
```

### An invalid program

The following example shows what the validator rejects. This program uses `import`, which is a `Forbidden AST node`:

```bash
cat > bad.py << 'EOF'
import sys
sys.version
EOF

lackpy validate bad.py --kit glob
```

```json
{
  "valid": false,
  "errors": [
    "Forbidden AST node: Import at line 1",
    "Unknown function: 'sys' at line 2 (not in kit or builtins)"
  ],
  "calls": []
}
```

### What gets rejected and why

| Construct | Reason |
|-----------|--------|
| `import` / `from ... import` | Arbitrary module access |
| `def` / `class` | Code definition introduces opaque scope |
| `while` | Unbounded loops |
| `try` / `except` | Error suppression can hide security violations |
| `lambda` | Anonymous functions bypass namespace checks |
| `open`, `getattr`, `setattr` | Direct resource or reflection access |
| Strings containing `__` | Prevents `getattr(obj, "__class__")` patterns |
| Unknown function names | Every call must be in the kit or allowed builtins |

### Python API

```python
from lackpy import LackpyService

svc = LackpyService()
result = svc.validate(
    'files = glob("**/*.py")\nlen(files)',
    kit=["glob"],
)
print(result.valid)     # True
print(result.errors)    # []
print(result.calls)     # ['glob', 'len']
```

---

## 4. Working with kits

A **kit** is a named subset of tools from the toolbox. Kits control which function names are visible to the validator and runner.

### List available tools

```bash
lackpy toolbox list
```

### Use a comma-separated kit

```bash
lackpy delegate "read the file README.md" --kit read,glob
```

### Create a named kit

```bash
lackpy kit create filesystem --tools read glob write --description "File system tools"
```

This creates `.lackpy/kits/filesystem.kit`:

```
---
name: filesystem
description: File system tools
---
read
glob
write
```

### Use the named kit

```bash
lackpy delegate "find all Python files" --kit filesystem
```

### Kit info

```bash
lackpy kit info filesystem
```

```json
{
  "tools": {
    "read": {"description": "", "grade_w": 3, "effects_ceiling": 3},
    "glob": {"description": "", "grade_w": 3, "effects_ceiling": 3},
    "write": {"description": "", "grade_w": 3, "effects_ceiling": 3}
  },
  "grade": {"w": 3, "d": 3},
  "description": "  read(path) -> Any: \n  glob(pattern) -> Any: \n  write(path, content) -> Any: "
}
```

### Python API — kit forms

```python
svc = LackpyService()

# Named kit (from .lackpy/kits/filesystem.kit)
result = await svc.delegate("find all Python files", kit="filesystem")

# List of tool names
result = await svc.delegate("find all Python files", kit=["glob"])

# Dict with aliases
result = await svc.delegate(
    "find all Python files",
    kit={"find_files": "glob"},  # calls are `find_files(...)` in the program
)
```

---

## 5. Generating programs

`generate` runs the inference pipeline without executing the result:

```bash
lackpy generate "find all Python files" --kit glob
```

```python
files = glob('**/*.py')
files
```

### Inference tiers

=== "Tier 0: Templates"

    Templates are matched first. If a template pattern matches the intent, the stored program is returned — no LLM required.

    ```bash
    lackpy generate "read the file config.toml" --kit read
    # → matched by rules tier: content = read('config.toml')
    ```

=== "Tier 1: Rules"

    The rules tier uses keyword-based matching for common patterns. It handles intents like "read file X", "find definitions of Y", "glob Z".

=== "Tier 2: Ollama"

    If templates and rules fail, Ollama is tried. Requires `pip install lackpy[ollama]` and a running Ollama server.

    ```toml
    # .lackpy/config.toml
    [inference.providers.ollama-local]
    plugin = "ollama"
    model = "qwen2.5-coder:1.5b"
    ```

=== "Tier 3: Anthropic"

    Cloud fallback. Requires `pip install lackpy[full]` and `ANTHROPIC_API_KEY`.

    ```toml
    [inference.providers.anthropic-fallback]
    plugin = "anthropic"
    model = "claude-haiku-4-5-20251001"
    ```

### Python API

```python
result = await svc.generate("find all Python files", kit=["glob"])
print(result.program)          # the generated program
print(result.provider_name)    # which tier produced it
print(result.generation_time_ms)
```

---

## 6. Running programs directly

Use `run` when you already have a program file:

```bash
cat > list_py.py << 'EOF'
files = glob("**/*.py")
files
EOF

lackpy run list_py.py --kit glob
```

```json
{
  "success": true,
  "output": ["src/lackpy/__init__.py", "src/lackpy/cli.py"],
  "error": null
}
```

The program is validated before execution. If validation fails, the runner returns an error without running anything.

### Python API

```python
result = await svc.run_program(
    'files = glob("**/*.py")\nfiles',
    kit=["glob"],
)
print(result.success)
print(result.output)
print(result.trace.entries)   # list of TraceEntry
```

---

## 7. Using parameters

Parameters let you pass values into programs without interpolating them into the intent string:

```python
result = await svc.delegate(
    intent="read the target file",
    kit=["read"],
    params={
        "target_file": {
            "value": "README.md",
            "type": "str",
            "description": "file to read",
        }
    },
)
```

Inside the program, `target_file` is available as a pre-set variable:

```python
# generated program
content = read(target_file)
content
```

!!! warning "Name collisions"
    Parameter names must not collide with tool names or allowed builtins. `LackpyService` raises `ValueError` if they do.

---

## 8. Creating templates (the ratchet)

Once you have a working program, save it as a template to make future matching deterministic:

```bash
cat > read_file.py << 'EOF'
content = read('{path}')
content
EOF

lackpy create read_file.py \
  --name read-file \
  --pattern "read the file {path}" \
  --kit read
```

This creates `.lackpy/templates/read-file.tmpl`:

```
---
name: read-file
pattern: "read the file {path}"
success_count: 0
fail_count: 0
---
content = read('{path}')
content
```

Now `lackpy delegate "read the file README.md"` matches at tier 0, with `{path}` substituted to `README.md`. No LLM is called.

### Python API

```python
result = await svc.create(
    program="content = read('{path}')\ncontent",
    kit=["read"],
    name="read-file",
    pattern="read the file {path}",
)
print(result["path"])  # .lackpy/templates/read-file.tmpl
```

---

## 9. Custom rules

Custom rules let you tighten the validator beyond the built-in checks. They're callables: `ast.Module -> list[str]`.

```python
from lackpy.lang.rules import no_loops, max_calls, max_depth, no_nested_calls

# Use in validate
result = svc.validate(
    program,
    kit=["glob"],
    rules=[no_loops, max_calls(5)],
)

# Use in delegate (enforced on the generated program)
result = await svc.delegate(
    "find all Python files",
    kit=["glob"],
    rules=[no_loops, max_depth(2)],
)
```

### Built-in rules

| Rule | Description |
|------|-------------|
| `no_loops` | Rejects any `for` loop |
| `max_depth(n)` | Limits nesting depth (if/for/with) |
| `max_calls(n)` | Limits total number of function calls |
| `no_nested_calls` | Forbids using a call result directly as an argument |

See [Custom Rules](extending/custom-rules.md) for writing your own.

---

## 10. Reading the trace

Every program run produces a `Trace` with an entry for each tool call:

```python
result = await svc.run_program(
    'files = glob("**/*.py")\ncount = len(files)\ncount',
    kit=["glob"],
)

for entry in result.trace.entries:
    print(
        f"Step {entry.step}: {entry.tool}({entry.args}) "
        f"-> {entry.result} ({entry.duration_ms:.1f}ms)"
    )
```

```
Step 0: glob({'pattern': '**/*.py'}) -> ['src/lackpy/__init__.py', ...] (1.3ms)
```

`len` is a builtin, not a tool, so it doesn't appear in the trace. Only kit tools are traced.

### Trace fields

| Field | Type | Description |
|-------|------|-------------|
| `step` | `int` | Call order (0-indexed) |
| `tool` | `str` | Tool name |
| `args` | `dict` | Arguments by name |
| `result` | `Any` | Return value (truncated if large) |
| `duration_ms` | `float` | Wall-clock call time |
| `success` | `bool` | Whether the call succeeded |
| `error` | `str \| None` | Exception message if failed |

The `Trace` also has `files_read` and `files_modified` lists, populated when tools are configured with effects metadata.

---

## Next steps

- [Concepts: Architecture](concepts/architecture.md) — internals of the pipeline
- [Concepts: Language Spec](concepts/language-spec.md) — full allowed/forbidden reference
- [Concepts: Kits & Toolbox](concepts/kits.md) — provider system in depth
- [Concepts: Inference Pipeline](concepts/inference.md) — tier system and config
- [Extending: Custom Rules](extending/custom-rules.md) — write your own validation rules
- [Extending: Tool Providers](extending/tool-providers.md) — register custom tools
- [Extending: Inference Providers](extending/inference-providers.md) — add LLM backends
