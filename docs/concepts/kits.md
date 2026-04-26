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
| `docs` | `str \| None` | Relative path to a markdown documentation file |

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
| `BuiltinProvider` | `"builtin"` | Hardcoded implementations for `read_file`, `find_files`, `write_file`, `edit_file` |
| `PythonProvider` | `"python"` | `importlib.import_module(module)` then `getattr(module, function)` |
| Custom | any string | Implement the provider protocol (see [Tool Providers](../extending/tool-providers.md)) |

---

## Kit parameter forms

`resolve_kit()` accepts these kit forms:

| Form | Type | Example | Behaviour |
|------|------|---------|-----------|
| Named kit | `str` | `"filesystem"` | Loads `.lackpy/kits/filesystem.kit` |
| Empty kit | `str` | `"none"` | No base tools (use with `extra_tools`) |
| Tool list | `list[str]` | `["read_file", "find_files"]` | Uses tool names directly as aliases |
| Tool mapping | `dict` | `{"find": "find_files"}` | Alias → actual tool name |
| Nested dict | `dict` | `{"r": {"tool": "read_file"}}` | Dict entry with `"tool"` key |

With the tool mapping form, the program sees `find(...)` but the toolbox resolves it to the `find_files` implementation.

### Extra tools

All kit forms support an optional `extra_tools` parameter — a list of tool names merged into the resolved kit:

```python
# Named kit + extra tools
kit = resolve_kit("debug", toolbox, extra_tools=["edit_file"])

# Standalone tools (no base kit)
kit = resolve_kit("none", toolbox, extra_tools=["read_file", "find_files"])
```

Duplicates are silently ignored. The kit grade is recomputed after merging.

---

## Kit file format

Named kits are stored as `.kit` files in `.lackpy/kits/`:

```
---
name: filesystem
description: Read, write, and search files
---
read_file
find_files
write_file
edit_file
```

- The YAML-like frontmatter between `---` lines is metadata.
- Lines after the closing `---` are tool names, one per line.
- Lines starting with `#` are treated as comments.
- Supported frontmatter fields: `name`, `description`, `docs`.

### Kit-level documentation

Kits can reference documentation files via the `docs` frontmatter field:

```
---
name: filesystem
description: Read, write, and search files
docs: docs/kits/filesystem.md
---
read_file
find_files
write_file
edit_file
```

The `docs` path is relative to the workspace root. It is not loaded at resolution time — consumers (like Kibitzer) query the reference and read the file on demand.

---

## CLI management

```bash
# List all kits in .lackpy/kits/
lackpy kit list

# Show tools and grade for a kit
lackpy kit info filesystem

# Show tools and grade for an ad-hoc list
lackpy kit info read_file,find_files,write_file

# Create a new kit
lackpy kit create mykit --tools read_file find_files --description "Read-only tools"
```

---

## Grade computation

`compute_grade(tools)` takes a dict of `{name: {"grade_w": int, "effects_ceiling": int}}` and returns the element-wise maximum across all tools:

```python
from lackpy import compute_grade

grade = compute_grade({
    "read_file":  {"grade_w": 1, "effects_ceiling": 0},
    "write_file": {"grade_w": 3, "effects_ceiling": 3},
})
# Grade(w=3, d=3)
```

This grade is attached to every `ResolvedKit` and reported in `delegate()` results. The grade is informational — lackpy does not block execution based on grade values, but callers can use it to gate access in security-sensitive contexts.

---

## Tool documentation

Tools and kits can reference markdown documentation files. These references are stored as relative paths and resolved lazily — nothing is loaded into memory until a consumer explicitly reads the file.

### How it works

1. **ToolSpec** has an optional `docs` field — a path relative to the package/workspace root:
   ```python
   ToolSpec(name="read_file", ..., docs="docs/tools/read_file.md")
   ```

2. **Kit files** have an optional `docs` frontmatter field for kit-level documentation.

3. **At resolution time**, `ResolvedKit` collects all doc references (from both the kit file and individual tools) into a `docs` list.

4. **Consumers query, not load**: the service exposes `docs_index()` (returns the reference map) and `resolve_doc()` (reads a specific file on demand).

### API

```python
# Get the docs index for a kit
index = svc.docs_index(kit="debug", extra_tools=["edit_file"])
# {"tool_docs": {"read_file": "docs/tools/read_file.md", ...}, "kit_docs": [...]}

# Read a specific doc file
content = svc.resolve_doc("docs/tools/read_file.md")
```

### Kibitzer integration

Kibitzer (v0.4+) uses the doc index to surface relevant documentation during error correction. The integration has three parts:

#### 1. Doc registration at init

When `LackpyService` initializes a Kibitzer session, it registers all tool doc references:

```python
# Automatic — happens in LackpyService._init_kibitzer():
docs = svc.docs_index()
kibitzer_session.register_docs(
    docs["tool_docs"],
    docs_root=str(workspace),
    refinement=build_doc_refinement(),
)
```

This gives Kibitzer a searchable index of doc paths without loading any files into memory.

#### 2. Doc refinement callbacks

Lackpy provides a `DocRefinement` with a `select` callback that picks doc sections relevant to each failure mode. When Kibitzer retrieves documentation (via pluckit), the select callback filters to the most useful sections:

| Failure mode | Sections selected |
|---|---|
| `stdlib_leak` | Signature, Parameters, Notes |
| `implement_not_orchestrate` | Signature, Examples |
| `key_hallucination` | Signature, Returns, Notes |
| `path_prefix` | Notes, Parameters |
| `wrong_output` | Signature, Returns, Examples |

This keeps the doc context concise — a 1.5B model gets the signature and one relevant section, not the full doc file.

#### 3. Doc context in the correction chain

When a generation fails, the correction chain calls `get_correction_hints(failure_mode, model, attempt, tool)`. The `tool` parameter is inferred from the error — e.g., a `Forbidden name: 'open'` error maps to `read_file`.

Kibitzer returns a `doc_context` field containing the selected doc sections. The correction chain folds these into the error enrichment:

```
--- From tool documentation ---
Signature: read_file(path: str) -> str
Notes: - Raises FileNotFoundError if the path does not exist. ...
```

This gives the small model targeted documentation about the tool it should have used, rather than generic hints or a raw dump of all Kibitzer patterns.

#### External access

External tools (pluckit, fledgling) can query the MCP `docs_index` and `resolve_doc` endpoints to search tool documentation independently of the correction pipeline.
