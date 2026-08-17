# Profiles & Toolbox

!!! note "Profiles replace kits"
    A **profile** is the per-task configuration bundle that generalizes the former
    "kit": a tool selection **plus** optional inference settings (`model`/`mode`/
    `temperature`) and the language/execution model. A profile that selects only tools
    is the degenerate case — *that* is what a "kit" used to be. The user-facing `kit`
    surface (the `kit=` argument, `--kit`, `kit_*` tools) has been **removed in favor of
    `profile`** (no alias). See [RFC 0002 §10](../design/tool-sources.md#10-migration--sequencing).

## Toolbox vs Profiles

| Concept | What it is | Scope |
|---------|------------|-------|
| **Toolbox** | The global registry of all available tools and their providers | Service-wide |
| **Profile** | A named per-task bundle: a tool selection (+ optional inference / language) | Per-request |

The `Toolbox` holds every tool registered across all sources. A profile's **tool
selection** is the subset a particular program may call — it defines the allowed namespace
for validation and the callable namespace for execution; the profile additionally carries
which model/mode and language/execution to run. Define profiles as `[profiles.<name>]`
tables in `.lackpy/config.toml`, or pass a tool list/name inline.

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
from lackpy.service import LackpyService
from lackpy.tools.toolbox import ToolSpec, ArgSpec

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

### Tools come from sources (no hard-coded names)

Tools are never hard-coded in lackpy by name — they come from **tool sources** that
fully define them. The first source is **config-defined**: declare a tool in
`.lackpy/config.toml` under a top-level `[[tools]]` array and it is loaded at service
init (no Python needed):

```toml
[[tools]]
name = "count_lines"
provider = "python"
module = "my_tools"
function = "count_lines"
description = "Count the number of lines in a file"
returns = "int"
grade_w = 1
effects_ceiling = 0
args = [{ name = "path", type = "str", description = "File path" }]
```

The shipped builtins (`read_file`, `find_files`, `write_file`, `edit_file`) are
themselves config-defined data (`lackpy/sources/default_tools.toml`), auto-loaded by
default; a `[[tools]]` entry with the same name overrides one.

### MCP-discovered tools

With the optional `mcp` extra (`pip install lackpy[mcp]`), lackpy can connect to an
MCP server as a client and expose its tools — full specs (params from the tool's
input schema, docs from its description), a return annotation derived from its
`outputSchema`, and a security grade derived from its MCP annotations
(`readOnlyHint`/`destructiveHint`/…, conservative when absent). Declare servers
under `[mcp_servers]`:

```toml
[mcp_servers.fs]
transport = "stdio"          # or "http" with url = "..."
command = "mcp-server-filesystem"
args = ["--root", "."]

# optional per-tool grade override
[mcp_servers.fs.tools.read_file]
grade = { w = 1, d = 0 }

# optional few-shot examples, retrieved by relevance at generation time
[[mcp_servers.fs.tools.read_file.examples]]
intent = "read the project README"
code = "read_file('README.md')"
```

#### Return shapes

`ToolSpec.returns` is derived from the tool's published `outputSchema`, so
`Toolbox.format_description` renders `name(args) -> list[dict]: …` rather than
`-> Any: …`. This matters more than it sounds: the argument schema always reached
the prompt while the return shape never did, so a generator had to *guess* what a
call handed back — and a wrong guess produces a program that validates, executes
and answers incorrectly, which is the one failure class the AST whitelist cannot
catch.

The mapping unwraps fastmcp's `{"properties": {"result": …}}` envelope and falls
back to `Any` for an absent or unrecognised schema, so nothing regresses.

!!! note "Ceiling worth knowing"

    fastmcp emits `{"type": "object", "additionalProperties": true}` for any
    dict-returning tool without a model annotation — a schema carrying no key
    names. Such a tool improves only from `Any` to `dict`. Knowing a value is a
    dict does not tell a generator *which key holds the answer*, so for those
    tools name the keys in the intent, or annotate the return type server-side.

#### Few-shot examples

`[[mcp_servers.<id>.tools.<name>.examples]]` entries (`intent` + `code`) attach
worked examples to a discovered tool. They join the same retrieval pool as
builtin tools' examples (`collect_example_pool` → `retrieve_examples`) and the
most relevant are injected at generation time.

Discovery gives a tool a name, a description and an argument schema, but nothing
about *usage* — so the idiom has to be guessed. Examples close that gap through
the same channel as the signature rather than as prose in the intent, which
matters because prose guidance is model-dependent: the same hint has been
measured to help one model substantially and hurt another.

Entries are **additive** — description, args and grade all survive. This is the
distinction a top-level `[[tools]]` entry of the same name cannot make, since it
would also replace the resolver and break the call. Malformed entries (missing
`intent` or `code`) are dropped rather than passed through.

MCP I/O runs on a dedicated client loop; an MCP-backed tool call from a (synchronous)
lackpy program is bridged to that loop without blocking generation. A server that
fails to connect is skipped (its tools simply don't appear), never breaking the
others. Multiple servers and external host configs (`[mcp].host_configs`) merge by
precedence — local config/builtins win the bare name, then own `[mcp_servers]`, then
host configs; a shadowed tool is dropped.

### Virtual / harness-provided tools

A host embedding lackpy can pass a `harness_resolver` (`name -> callable | None`) and
declare tools under `[[virtual_tools]]` (full spec, no implementation). These resolve
to the harness's callable at run time. Tools the harness can't currently supply are
hidden from generation; if one is withdrawn between generation and the call, the call
fails cleanly. See [Tool Sources (RFC 0002)](../design/tool-sources.md).

---

## Provider table

| Provider | Name | How it resolves tools |
|----------|------|----------------------|
| `BuiltinProvider` | `"builtin"` | Hardcoded implementations for `read_file`, `find_files`, `write_file`, `edit_file` |
| `PythonProvider` | `"python"` | `importlib.import_module(module)` then `getattr(module, function)` |
| Custom | any string | Implement the provider protocol (see [Tool Providers](../extending/tool-providers.md)) |

---

## Profile / tool-selection forms

`resolve_tools()` accepts these kit forms:

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
# Named profile + extra tools
kit = resolve_tools("debug", toolbox, extra_tools=["edit_file"])

# Standalone tools (no base profile)
kit = resolve_tools("none", toolbox, extra_tools=["read_file", "find_files"])
```

Duplicates are silently ignored. The kit grade is recomputed after merging.

---

## Tool-set file format

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

### Profile-level documentation

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
# List all tool-sets in .lackpy/kits/
lackpyctl profile list

# Show tools and grade for a profile
lackpyctl profile info filesystem

# Show tools and grade for an ad-hoc list
lackpyctl profile info read_file,find_files,write_file

# Create a new tool-set
lackpyctl profile create mykit --tools read_file find_files --description "Read-only tools"
```

---

## Grade computation

`compute_grade(tools)` takes a dict of `{name: {"grade_w": int, "effects_ceiling": int}}` and returns the element-wise maximum across all tools:

```python
from lackpy.lang import compute_grade

grade = compute_grade({
    "read_file":  {"grade_w": 1, "effects_ceiling": 0},
    "write_file": {"grade_w": 3, "effects_ceiling": 3},
})
# Grade(w=3, d=3)
```

This grade is attached to every `ResolvedTools` and reported in `delegate()` results. The grade is informational — lackpy does not block execution based on grade values, but callers can use it to gate access in security-sensitive contexts.

---

## Tool documentation

Tools and kits can reference markdown documentation files. These references are stored as relative paths and resolved lazily — nothing is loaded into memory until a consumer explicitly reads the file.

### How it works

1. **ToolSpec** has an optional `docs` field — a path relative to the package/workspace root:
   ```python
   ToolSpec(name="read_file", ..., docs="docs/tools/read_file.md")
   ```

2. **Kit files** have an optional `docs` frontmatter field for kit-level documentation.

3. **At resolution time**, `ResolvedTools` collects all doc references (from both the kit file and individual tools) into a `docs` list.

4. **Consumers query, not load**: the service exposes `docs_index()` (returns the reference map) and `resolve_doc()` (reads a specific file on demand).

### API

```python
# Get the docs index for a profile
index = svc.docs_index(profile="debug", extra_tools=["edit_file"])
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
