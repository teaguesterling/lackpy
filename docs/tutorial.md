# Tutorial

This walkthrough covers every major feature of lackpy. Follow along in order, or jump to any section.

**Prerequisites:** lackpy installed, workspace initialized with `lackpyctl init`. See [Getting Started](getting-started.md).

---

## 1. Setup

Create a project directory and initialize:

```bash
mkdir my-lackpy-project && cd my-lackpy-project
echo "# Hello from lackpy" > README.md
lackpyctl init
```

Confirm the setup:

```bash
lackpyctl status
```

```json
{
  "workspace": "/path/to/my-lackpy-project",
  "config_dir": "/path/to/my-lackpy-project/.lackpy",
  "inference_order": ["templates", "rules"],
  "profile_default": "debug",
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
files = find_files("**/*.py")
count = len(files)
count
EOF

lackpy check.py --validate --profile find_files
```

```json
{
  "valid": true,
  "errors": [],
  "calls": ["find_files", "len"]
}
```

### An invalid program

The following example shows what the validator rejects. This program uses `import`, which is a `Forbidden AST node`:

```bash
cat > bad.py << 'EOF'
import sys
sys.version
EOF

lackpy bad.py --validate --profile find_files
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
from lackpy.service import LackpyService

svc = LackpyService()
result = svc.validate(
    'files = find_files("**/*.py")\nlen(files)',
    profile=["find_files"],
)
print(result.valid)     # True
print(result.errors)    # []
print(result.calls)     # ['find_files', 'len']
```

---

## 4. Working with kits

A **kit** is a named subset of tools from the toolbox. Kits control which function names are visible to the validator and runner.

### List available tools

```bash
lackpyctl toolbox list
```

### Use a comma-separated kit

```bash
lackpy -c "read the file README.md" --profile read_file,find_files
```

### Create a named kit

```bash
lackpyctl profile create filesystem --tools read glob write --description "File system tools"
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
lackpy -c "find all Python files" --profile filesystem
```

### Kit info

```bash
lackpyctl profile info filesystem
```

```json
{
  "tools": {
    "read_file": {"description": "", "grade_w": 3, "effects_ceiling": 3},
    "find_files": {"description": "", "grade_w": 3, "effects_ceiling": 3},
    "write_file": {"description": "", "grade_w": 3, "effects_ceiling": 3}
  },
  "grade": {"w": 3, "d": 3},
  "description": "  read_file(path) -> Any: \n  find_files(pattern) -> Any: \n  write_file(path, content) -> Any: "
}
```

### Python API — kit forms

```python
svc = LackpyService()

# Named kit (from .lackpy/kits/filesystem.kit)
result = await svc.delegate("find all Python files", profile="filesystem")

# List of tool names
result = await svc.delegate("find all Python files", profile=["find_files"])

# Dict with aliases
result = await svc.delegate(
    "find all Python files",
    profile={"ls": "find_files"},  # calls are `ls(...)` in the program
)
```

---

## 5. Generating programs

The `--generate` flag runs the inference pipeline without executing the result:

```bash
lackpy -c "find all Python files" --generate --profile find_files
```

```python
files = find_files('**/*.py')
files
```

### Inference tiers

=== "Tier 0: Templates"

    Templates are matched first. If a template pattern matches the intent, the stored program is returned — no LLM required.

    ```bash
    lackpy -c "read the file config.toml" --generate --profile read_file
    # → matched by rules tier: content = read_file('config.toml')
    ```

=== "Tier 1: Rules"

    The rules tier uses keyword-based matching for common patterns. It handles intents like "read file X", "find definitions of Y", "glob Z".

=== "Tier 2: Local model"

    If templates and rules fail, the LLM tier is tried. Model calls go through
    woollama's core (a built-in dependency); a local model just needs a running
    Ollama server.

    ```toml
    # .lackpy/config.toml
    [inference.providers.local]
    plugin = "woollama"
    model = "ollama/qwen2.5-coder:1.5b"
    base_url = "http://localhost:11434/v1"
    ```

=== "Tier 2: Cloud fallback"

    Same provider, a cloud model. Just needs the relevant API key in the
    environment (e.g. `ANTHROPIC_API_KEY`).

    ```toml
    [inference.providers.cloud-fallback]
    plugin = "woollama"
    model = "anthropic/claude-haiku-4-5"
    ```

### Python API

```python
result = await svc.generate("find all Python files", profile=["find_files"])
print(result.program)          # the generated program
print(result.provider_name)    # which tier produced it
print(result.generation_time_ms)
```

---

## 6. Running programs directly

When you already have a program file, pass its path as the first positional argument (there is no `run` subcommand). A plain program file needs `--profile` or `--tools` to supply its namespace; a Lackey file carries its own tools and needs neither:

```bash
cat > list_py.py << 'EOF'
files = find_files("**/*.py")
files
EOF

lackpy list_py.py --profile find_files
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
    'files = find_files("**/*.py")\nfiles',
    profile=["find_files"],
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
    profile=["read_file"],
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
content = read_file(target_file)
content
```

!!! warning "Name collisions"
    Parameter names must not collide with tool names or allowed builtins. `LackpyService` raises `ValueError` if they do.

---

## 8. Saving programs for reuse (the ratchet)

lackpy gives you two ways to reuse a validated program. They're distinct mechanisms:

### Lackey files (saved program, run by path)

The `--create` flag generates a program from your intent and saves it as a **Lackey file** — a Python class wrapping the program, plus its tools — under `.lackpy/templates/`:

```bash
lackpy -c "read the file README.md" --create --name ReadFile --profile read_file
```

The output is plain text:

```
Created .lackpy/templates/ReadFile.py
```

A Lackey file carries its own tools and a validated program. You run it by path — this skips inference entirely:

```bash
lackpy .lackpy/templates/ReadFile.py
```

A Lackey file is **not** matched against future intents; it's a saved program you invoke directly.

### Templates (pattern-matched at tier 0)

To make *intent matching* deterministic, save a `.tmpl` template with a `pattern`. This is exposed through the Python API (`svc.create`), which writes a `.tmpl` file with frontmatter:

```python
result = await svc.create(
    program="content = read_file('{path}')\ncontent",
    profile=["read_file"],
    name="read-file",
    pattern="read the file {path}",
)
print(result["path"])  # .lackpy/templates/read-file.tmpl
```

This writes `.lackpy/templates/read-file.tmpl`:

```
---
name: read-file
pattern: "read the file {path}"
success_count: 0
fail_count: 0
---
content = read_file('{path}')
content
```

Now a delegate call (the default `lackpy -c` mode) for `"read the file README.md"` matches the template at tier 0, with `{path}` substituted to `README.md`. No LLM is called.

---

## 9. Custom rules

Custom rules let you tighten the validator beyond the built-in checks. They're callables: `ast.Module -> list[str]`.

```python
from lackpy.lang.rules import no_loops, max_calls, max_depth, no_nested_calls

# Use in validate
result = svc.validate(
    program,
    profile=["find_files"],
    rules=[no_loops, max_calls(5)],
)

# Use in delegate (enforced on the generated program)
result = await svc.delegate(
    "find all Python files",
    profile=["find_files"],
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
    'files = find_files("**/*.py")\ncount = len(files)\ncount',
    profile=["find_files"],
)

for entry in result.trace.entries:
    print(
        f"Step {entry.step}: {entry.tool}({entry.args}) "
        f"-> {entry.result} ({entry.duration_ms:.1f}ms)"
    )
```

```
Step 0: find_files({'pattern': '**/*.py'}) -> ['src/lackpy/__init__.py', ...] (1.3ms)
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
- [Concepts: Kits & Toolbox](concepts/profiles.md) — provider system in depth
- [Concepts: Inference Pipeline](concepts/inference.md) — tier system and config
- [Extending: Custom Rules](extending/custom-rules.md) — write your own validation rules
- [Extending: Tool Providers](extending/tool-providers.md) — register custom tools
- [Extending: Inference Providers](extending/inference-providers.md) — add LLM backends
