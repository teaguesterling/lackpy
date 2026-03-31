# Kits & Toolbox

## Toolbox vs Kits

| Concept | What it is | Scope |
|---------|------------|-------|
| **Toolbox** | The global registry of all available tools and their providers | Service-wide |
| **Kit** | A named subset of toolbox tools for a specific task | Per-request |

The `Toolbox` holds every tool that has been registered across all providers. A `Kit` is the subset of those tools that a particular program may call — it defines the allowed namespace for validation and the callable namespace for execution.

---

## ToolSpec fields

`ToolSpec` is the metadata record for a single tool:

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | The function name used in lackpy programs |
| `provider` | `str` | Provider name that resolves this tool (e.g. `"builtin"`, `"python"`) |
| `provider_config` | `dict` | Provider-specific config (e.g. `module`, `function` for the `python` provider) |
| `description` | `str` | Human-readable description, shown to LLMs in the system prompt |
| `args` | `list[ArgSpec]` | Argument names, types, and descriptions |
| `returns` | `str` | Return type annotation string |
| `grade_w` | `int` | World coupling (0–3) |
| `effects_ceiling` | `int` | Effects ceiling (0–3) |

`ArgSpec` fields: `name`, `type` (string), `description`.

---

## Registering tools

Tools are registered by adding a `ToolSpec` to the `Toolbox` and ensuring a matching provider is also registered:

```python
from lackpy import LackpyService
from lackpy.kit.toolbox import ToolSpec, ArgSpec

svc = LackpyService()

# Register a custom tool backed by a Python function
svc.toolbox.register_tool(ToolSpec(
    name="count_lines",
    provider="python",
    provider_config={
        "module": "my_tools",
        "function": "count_lines",
    },
    description="Count the number of lines in a file",
    args=[ArgSpec(name="path", type="str", description="File path")],
    returns="int",
    grade_w=1,
    effects_ceiling=0,
))
```

The `python` provider is always registered. It resolves tools by importing the named module and looking up the function.

---

## Provider table

| Provider | Name | How it resolves tools |
|----------|------|----------------------|
| `BuiltinProvider` | `"builtin"` | Hardcoded implementations for `read`, `glob`, `write`, `edit` |
| `PythonProvider` | `"python"` | `importlib.import_module(module)` then `getattr(module, function)` |
| Custom | any string | Implement the provider protocol (see [Tool Providers](../extending/tool-providers.md)) |

---

## Kit parameter forms

`resolve_kit()` accepts three kit forms:

| Form | Type | Example | Behaviour |
|------|------|---------|-----------|
| Named kit | `str` | `"filesystem"` | Loads `.lackpy/kits/filesystem.kit` |
| Tool list | `list[str]` | `["read", "glob"]` | Uses tool names directly as aliases |
| Tool mapping | `dict` | `{"find": "glob"}` | Alias → actual tool name |
| Nested dict | `dict` | `{"r": {"tool": "read"}}` | Dict entry with `"tool"` key |

With the tool mapping form, the program sees `find(...)` but the toolbox resolves it to the `glob` implementation.

---

## Kit file format

Named kits are stored as `.kit` files in `.lackpy/kits/`:

```
---
name: filesystem
description: Read, write, and search files
---
read
glob
write
edit
```

- The YAML-like frontmatter between `---` lines is metadata.
- Lines after the closing `---` are tool names, one per line.
- Lines starting with `#` are treated as comments.
- The frontmatter is not validated against a schema; only `name` and `description` are used.

---

## CLI management

```bash
# List all kits in .lackpy/kits/
lackpy kit list

# Show tools and grade for a kit
lackpy kit info filesystem

# Show tools and grade for an ad-hoc list
lackpy kit info read,glob,write

# Create a new kit
lackpy kit create mykit --tools read glob --description "Read-only tools"
```

---

## Grade computation

`compute_grade(tools)` takes a dict of `{name: {"grade_w": int, "effects_ceiling": int}}` and returns the element-wise maximum across all tools:

```python
from lackpy import compute_grade

grade = compute_grade({
    "read":  {"grade_w": 1, "effects_ceiling": 0},
    "write": {"grade_w": 3, "effects_ceiling": 3},
})
# Grade(w=3, d=3)
```

This grade is attached to every `ResolvedKit` and reported in `delegate()` results. The grade is informational — lackpy does not block execution based on grade values, but callers can use it to gate access in security-sensitive contexts.
