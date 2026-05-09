# Literate Interpreter

The literate interpreter executes markdown documents with embedded
`` ```lackpy `` code blocks. Prose becomes `print()` calls, code executes
inline, and the captured stdout IS the rendered document. It replaces
N tool round-trips with a single document submission — the model writes
a complete response as a literate program, and the interpreter produces
the final rendered output.

This is the complement of MCP prompts: prompts shape the *request*,
literate documents shape the *response*.

```
  Model stream ──→ StreamingCellParser ──→ Cell objects
                                                │
                                                ▼
                            ┌──────── StreamingDriver ────────┐
                            │                                  │
                            │  for each cell:                  │
                            │    static analysis (compile+AST) │
                            │    kernel.execute_cell(cell)     │
                            │      ├─ success → output + state │
                            │      └─ failure → recovery loop  │
                            │                                  │
                            └──────────────────────────────────┘
                                                │
                                                ▼
                                        ExecutionLog
                                       ╱            ╲
                              to_notebook()    render_markdown()
                                   ↓                   ↓
                                .ipynb            clean markdown
```

Cells execute one at a time through a persistent kernel. Variables
defined in any cell are available to all later cells and to prose
interpolation. On failure, a pluggable recovery handler can fix, inspect,
skip, or abort — optionally calling a model to produce replacement cells.

## Quick example

Given this literate document (the model's raw response):

````markdown
---
echo: true
---
# File Summary

```lackpy @hidden
content = read_file("README.md")
lines = content.strip().splitlines()
```

The file has {len(lines)} lines.

```lackpy
first_line = lines[0]
first_line
```

The title is: {first_line}
````

The interpreter produces:

```
# File Summary
The file has 42 lines.
README.md — a toolkit for safe code execution
The title is: README.md — a toolkit for safe code execution
```

Prose passes through with `{expr}` interpolation expanded. Code blocks
execute silently unless they produce print output. Variables defined in any block are
available to all later blocks and to prose interpolation.

## Document format

### Frontmatter

Optional YAML frontmatter sets document-level defaults:

```yaml
---
echo: true        # code visibility (true|false|auto)
output: auto      # result visibility (all|last|hide|auto)
interpreter: python  # backend for code blocks
---
```

All fields are optional. Defaults: `echo=true`, `output=auto`,
`interpreter=python`. Frontmatter is stripped before markdown parsing
since `---` is `<hr>` in CommonMark.

!!! note "Phase 0 limitation"

    Frontmatter fields and per-block options (`echo=false`, `output=all`)
    are parsed and stored but do not yet alter compilation or execution
    behavior. All blocks compile and execute unconditionally. Honoring
    these options is planned for Phase 1.

### Prose

Everything outside fenced code blocks is a prose cell. Prose compiles to
`print()` calls. Curly-brace expressions like `{variable}` are expanded
as f-string interpolation — any Python expression works inside the
braces.

### Code blocks

Fenced with `` ```lackpy ``. Per-block options can override frontmatter:

````markdown
```lackpy echo=false output=all
x = expensive_computation()
```
````

## Block annotations

Annotations appear on the fence line after the language tag:

````markdown
```lackpy @hidden
setup_code_here()
```
````

| Annotation | Purpose | Compiles to |
|---|---|---|
| *(none)* | Normal code | pass through |
| `@hidden` | Silent setup | pass through (no print wrapping) |
| `@gather` | Batch exploration | pass through |
| `@continue` | Pause point | sentinel function call |
| `@read(path)` | Display file | `print(read_file(path))` |
| `@write(path)` | Write file | `write_file(path, content)` |
| `@diff(path)` | Apply unified diff | `apply_diff(path, diff_text)` |
| `@scratch` | Working memory | execute + auto-summarize new variables |

### @hidden

Executes code without producing visible output. Use for setup, variable
assignments, and intermediate computation that the reader doesn't need.

### @gather and @continue

The gather pattern batches information-gathering before narration:

````markdown
```lackpy @gather
files = search_content("TODO", "src/")
```

```lackpy @gather
tests = run_tests("tests/")
```

```lackpy @continue
```

Now I can describe what I found: {files}
````

`@gather` blocks execute but produce no output. `@continue` signals a
pause point — execution stops, partial results return to the caller
(typically an agent harness), and the model writes the next section with
access to everything gathered so far.

### @read, @write, @diff

File operations. The path goes in parentheses on the fence line, or as
the first line of the block body (a fallback for models that format it
that way):

````markdown
```lackpy @read(src/main.py)
```

```lackpy @write(output.py)
def hello():
    print("world")
```

```lackpy @diff(config.py)
--- a/config.py
+++ b/config.py
@@ -1,3 +1,3 @@
-DEBUG = True
+DEBUG = False
```
````

### @scratch

Working memory. Variables defined in the block are available to later
code, but the block's output is auto-summarized rather than shown in
full:

````markdown
```lackpy @scratch
a = complex_computation()
b = another_thing()
```
````

Produces: `[scratch: a=int, b=str]` — a summary of what was defined,
not the full values.

## Execution model

Cells execute one at a time through a **kernel** — a persistent
namespace that accumulates state across cells. Each cell goes through
two phases:

1. **Static analysis** — compile the cell's Python and walk the AST to
   check that all referenced names exist in the current scope or builtins.
   Catches syntax errors and undefined references before execution.

2. **Execution** — run the compiled Python in the kernel's namespace,
   capturing stdout. Variables defined by the cell become available to
   all later cells.

The compiler transforms each cell type into Python before the kernel
executes it:

| Cell type | Python output |
|---|---|
| Prose `"Hello {x}"` | `print('Hello ' + f"""{x}""")` |
| Prose `"Hello world"` | `print('Hello world')` |
| Empty prose | `print()` |
| Code | pass through |
| @hidden | pass through |
| @gather | pass through |
| @continue | continue sentinel |
| @read(path) | `print(read_file('path'))` |
| @write(path) | `write_file('path', 'content')` |
| @diff(path) | `apply_diff('path', 'diff_text')` |
| @scratch | capture `locals()` diff, print summary |

String literals in non-interpolated prose use `repr()` for escaping.
Interpolated prose is split into literal and expression parts using a
brace-matching parser; each expression compiles to its own triple-quoted
f-string (`f"""{expr}"""`), which handles nested quotes, backslashes,
and complex expressions safely.

### Streaming execution

The `StreamingCellParser` detects fence boundaries in partial model
output, yielding Cell objects as they complete. This means cells can
execute as the model generates them — the parser watches for
`` ```lackpy `` open fences and `` ``` `` close fences in the text stream,
yielding each cell as soon as its closing fence arrives.

### Recovery

When a cell fails, the `StreamingDriver` invokes a pluggable
`RecoveryHandler`:

- **`NoRecoveryHandler`** — aborts immediately. Used for batch execution
  and tests.
- **`InferenceRecoveryHandler`** — formats the error, scope, and any
  plugin advice into a prompt, calls a model, and parses the response
  as replacement cells. The model can return `@scratch` to inspect a
  value before fixing.

The recovery flow:

1. Cell fails → driver builds `RecoveryContext` (error, scope, plugin advice)
2. Handler returns an action: `fix`, `inspect`, `skip`, or `abort`
3. `fix` — execute replacement cells. If they fail, increment attempt, loop.
4. `inspect` — evaluate an expression via `kernel.inspect()`, feed result
   back to handler. Handler can inspect multiple times before fixing.
5. `skip` — mark cell skipped, continue with next cell.
6. `abort` — stop execution, return partial results.

Recovery operates on current kernel state (patch-forward). The model
can't rewrite history — it emits correction cells that fix state going
forward.

### Artifact formats

The execution log can be serialized to two formats:

- **`.ipynb` (Jupyter notebook)** — each cell becomes a notebook cell
  with lackpy metadata preserved for round-trip fidelity. This is the
  artifact format. Loading an `.ipynb` back through the kernel with
  `NoRecoveryHandler` is "Restart and Run All."

- **Clean markdown** — the literate document the model *should* have
  written, after recovery fixes. Skipped cells are omitted. Useful for
  feeding corrected output back to the model.

## Execution namespace

The literate interpreter provides full Python builtins (security is
enforced by nsjail at the sandbox level, not by Python restrictions).

### Kit tools

These functions are injected into every execution namespace:

| Function | Signature | Purpose |
|---|---|---|
| `read_file` | `(path) → str` | Read file contents |
| `write_file` | `(path, content) → None` | Write file (creates parents) |
| `apply_diff` | `(path, diff) → str` | Apply unified diff, return result |
| `search_content` | `(pattern, path=".") → str` | Grep-like search |
| `run_command` | `(cmd) → str` | Shell command, stdout+stderr |
| `run_tests` | `(path=".") → str` | Run pytest, return output |

If an `ExecutionContext` includes a kit, its callables are also available.

### Builtins and modules

All Python builtins are available: `open()`, `len()`, `print()`,
`type()`, `isinstance()`, etc. Standard library modules can be imported
normally.

## Architecture

```
src/lackpy/interpreters/literate/
├── __init__.py    LiterateInterpreter — validate(), execute(), registration
├── parser.py      markdown-it-py → Cell sequence (frontmatter, fences, prose)
├── compiler.py    Cell → Python source (brace-matching interpolation)
├── tools.py       read_file, write_file, apply_diff, search, shell, tests
├── prompt.py      system_prompt_hint() for model instruction
└── kernel/
    ├── __init__.py          Public exports
    ├── interface.py         KernelInterface protocol + CellResult
    ├── lightweight.py       LightweightKernel (exec-into-dict)
    ├── static_analysis.py   compile check + AST name resolution
    ├── streaming_parser.py  StreamingCellParser (fence detection on partial input)
    ├── driver.py            StreamingDriver orchestrator
    ├── recovery.py          RecoveryHandler protocol + handlers
    ├── plugins.py           ExecutionPlugin protocol + PluginAdvice
    └── formats.py           to_notebook, from_notebook, render_markdown
```

### Parser (`parser.py`)

Uses [markdown-it-py](https://markdown-it-py.readthedocs.io/) with the
`commonmark` preset for proper fence detection. The parser:

1. Strips YAML frontmatter (before markdown parsing, since `---` is
   `<hr>` in CommonMark)
2. Runs markdown-it to find all fence tokens with their source line maps
3. Extracts prose from raw source text between fence regions
4. Parses fence info strings for annotations and options
5. For `@read`/`@write`/`@diff` without a path in parentheses, falls
   back to the first non-empty line of the block body

Returns a `ParseResult` containing `Frontmatter`, a list of `Cell`
objects, and any parse errors. This is the batch parser used by
`LiterateInterpreter.validate()` and for pre-parsed document execution.
The streaming equivalent is `StreamingCellParser` in the kernel package.

### Compiler (`compiler.py`)

Transforms each `Cell` into Python source. Non-interpolated prose uses
`repr()` for safe string literals. Interpolated prose uses a
brace-matching splitter (`_split_interpolation`) that handles nested
braces, then compiles each expression as a standalone triple-quoted
f-string. This avoids `repr()`-related issues with nested quotes and
backslash escapes inside complex expressions.

The compiler functions are used by both the batch path and the kernel —
`LightweightKernel.execute_cell()` calls the same per-cell-type
compilers internally.

### Kernel (`kernel/`)

The incremental execution engine. Cells execute one at a time through a
persistent namespace, with static analysis before each execution.

**`KernelInterface`** — protocol for cell execution backends. Methods:
`execute_cell()`, `inspect()`, `get_scope()`, `restart()`,
`get_namespace()`.

**`LightweightKernel`** — the default backend. Compiles each cell using
the compiler functions, runs static analysis (compile check + AST name
resolution), then `exec()`s into a shared dict namespace. Returns a
`CellResult` with success/failure, captured stdout, error details, and
the namespace delta (what the cell changed).

**`StreamingCellParser`** — incremental fence detection on partial input.
Fed chunks of model output via `feed()`, yields `Cell` objects as fence
boundaries are detected. Used by `StreamingDriver` for parse-as-you-stream
execution.

**`StreamingDriver`** — the orchestrator. Connects parser → kernel →
recovery → plugins. Handles `@continue` pause/resume with generation
tracking. Provides `execution_log` and `rendered_output` properties.

**`RecoveryHandler`** — protocol for error recovery. Built-in handlers:
`NoRecoveryHandler` (always aborts) and `InferenceRecoveryHandler`
(calls a model to fix failing cells).

**`ExecutionPlugin`** — hook protocol for coaching/tracking systems.
Plugins observe cell lifecycle events (`on_cell_start`, `on_cell_success`,
`on_cell_error`, `on_recovery_result`) and can provide `PluginAdvice`
during error recovery. The system works without any plugins; Kibitzer
subscribes to these hooks when present.

**`formats`** — serialization between execution log, `.ipynb` notebook,
and clean markdown. `to_notebook()` produces a Jupyter-compatible
notebook with lackpy metadata. `from_notebook()` recovers cells for
re-execution. `render_markdown()` produces the corrected literate
document.

### Interpreter (`__init__.py`)

Implements the standard `Interpreter` protocol:

- **`validate(program, context)`** — parses the document and reports
  errors without executing
- **`execute(program, context)`** — parses the document, creates a
  `LightweightKernel` with the tool namespace, and executes cells
  one at a time through the kernel

The literate interpreter uses direct Python execution with full
builtins — security is enforced at the sandbox level (nsjail), not by
Python-level AST restrictions.

The execution result includes metadata: `continue_requested` (whether
`@continue` was hit), `variables` (namespace after execution),
`cell_count`, and `frontmatter` settings.

### System prompt (`prompt.py`)

`system_prompt_hint()` returns a prompt fragment that teaches models to
write literate documents. It covers document structure, annotation
syntax (with correct/incorrect examples), the gather pattern, available
tools and builtins, and key rules.

## Plugin architecture

The execution plugin system provides observation hooks for coaching and
tracking systems. It is designed around three principles:

1. **Plugins observe and advise** — they don't control execution flow
2. **The system works without plugins** — no plugin is required
3. **Plugins manage their own state** — no session objects are passed in

```python
class ExecutionPlugin(Protocol):
    def on_cell_start(self, cell: Cell, index: int) -> None: ...
    def on_cell_success(self, cell: Cell, result: CellResult) -> None: ...
    def on_cell_error(self, cell: Cell, error: str, scope: dict) -> PluginAdvice: ...
    def on_recovery_result(self, cell: Cell, success: bool, attempt: int) -> None: ...
```

When a cell fails, plugins return `PluginAdvice` (hints, documentation
context, and an optional suggestion). If multiple plugins are registered,
their advice is merged — hints concatenated, first non-None suggestion
wins. This advice is folded into the recovery prompt when using
`InferenceRecoveryHandler`.

Plugin methods must not raise — errors are logged and ignored. This
keeps the execution path stable regardless of plugin quality.

## Agent harness

`scripts/literate_agent.py` is a standalone harness for testing with
local Ollama models. It implements the agent loop:

1. Send user prompt to model
2. Receive a literate document as the response
3. Execute via `LiterateInterpreter`
4. If `@continue` was hit, feed partial results + variable summary back
   to the model
5. Repeat until complete or max iterations reached

```bash
python scripts/literate_agent.py "Analyze the files in src/"
python scripts/literate_agent.py --model qwen3.5:35b "Find all TODOs"
python scripts/literate_agent.py -v --base-dir /project "Summarize tests"
```

## Annotated example: codebase health report

This walkthrough shows the full execution flow for a real analysis task.
The document gathers data silently, computes metrics in a hidden block,
then renders a formatted report with variable interpolation.

### The document

````markdown
```lackpy @gather
result = search_content("TODO|FIXME|HACK|noqa|type: ignore", "src/lackpy")
pragmas = [l.strip() for l in result.splitlines() if l.strip()]
```

```lackpy @gather
import_result = run_command("find src/lackpy -name '*.py' -exec grep -l 'import asyncio\\|async def' {} +")
async_files = [l.strip() for l in import_result.splitlines() if l.strip()]
src_result = run_command("find src/lackpy -name '*.py' | wc -l")
loc_result = run_command("find src/lackpy -name '*.py' -exec cat {} + | wc -l")
test_loc_result = run_command("find tests -name '*.py' -exec cat {} + | wc -l")
```

```lackpy @hidden
src_count = int(src_result.strip())
src_loc = int(loc_result.strip())
test_loc = int(test_loc_result.strip())
ratio = test_loc / src_loc if src_loc > 0 else 0
```

# Lackpy Codebase Health Report

```lackpy
print(f"| Metric | Value |")
print(f"|--------|-------|")
print(f"| Source files | {src_count} |")
print(f"| Source LOC | {src_loc:,} |")
print(f"| Test LOC | {test_loc:,} |")
print(f"| Test/Source ratio | {ratio:.2f}x |")
print(f"| Async modules | {len(async_files)} |")
```
````

### Execution trace

The interpreter parses this into 6 cells and executes them in order:

| Step | Cell type | What happens | Output |
|------|-----------|-------------|--------|
| 1 | `@gather` | `search_content(...)` runs, `pragmas` variable set | *(silent)* |
| 2 | `@gather` | `run_command(...)` calls run, 5 variables set | *(silent)* |
| 3 | `@hidden` | Integer parsing, ratio computation | *(silent)* |
| 4 | prose | `"# Lackpy Codebase Health Report"` printed | heading |
| 5 | code | `print(f"| Metric | ...")` renders table | table rows |

Steps 1-3 build up the namespace silently. Steps 4-5 render the
report. All 5 `@gather`/`@hidden` variables are available to the
visible code block and to any prose interpolation.

### Rendered output

```
# Lackpy Codebase Health Report

| Metric | Value |
|--------|-------|
| Source files | 101 |
| Source LOC | 10,442 |
| Test LOC | 9,373 |
| Test/Source ratio | 0.90x |
| Async modules | 30 |
```

Total execution time: ~200ms for 6 cells. The gather blocks do real
I/O (grep, find, wc), the hidden block does arithmetic, and the code
block formats the output. The reader sees only the clean report.

### Key patterns demonstrated

**Gather-then-render.** Data collection is separated from presentation.
`@gather` blocks run silently so the reader never sees raw command
output. The code block formats the data into a readable table.

**Variable persistence.** `src_count` defined in the `@hidden` block
(step 3) is used in the code block (step 5). Variables accumulate
across cells — the namespace is shared.

**Tool composition.** `search_content` and `run_command` are injected
tools from the literate namespace. They run real shell commands and
return string results that the code processes with standard Python.

**Prose as glue.** The heading cell is pure prose — it becomes
`print("# Lackpy Codebase Health Report")`. Prose cells can include
`{variable}` interpolation, but here the heading is static.

## Using the interpreter directly

```python
import asyncio
from lackpy.interpreters import LiterateInterpreter, ExecutionContext, run_interpreter

async def main():
    interp = LiterateInterpreter()
    ctx = ExecutionContext(base_dir="/path/to/project")

    doc = """
# Report

```lackpy @hidden
data = read_file("data.csv").splitlines()
```

Found {len(data)} rows.
"""

    result = await run_interpreter(interp, doc, ctx)
    print(result.output)           # "# Report\nFound 42 rows.\n"
    print(result.output_format)    # "markdown"
    print(result.metadata)         # {"continue_requested": False, "cell_count": 2, ...}

asyncio.run(main())
```

## Current scope

The implementation covers:

- All cell types: prose, code, @hidden, @gather, @continue, @read,
  @write, @diff, @scratch
- Frontmatter parsing
- Full Python builtins in the execution namespace
- Kit tool injection
- The @gather/@continue agent loop pattern
- Incremental cell-by-cell execution via the kernel
- Static analysis (compile check + AST name resolution) before each cell
- Streaming parse-as-you-generate execution
- Pluggable recovery (NoRecoveryHandler, InferenceRecoveryHandler)
- Plugin API for coaching systems (ExecutionPlugin protocol)
- Notebook (.ipynb) artifact export and re-import
- Clean markdown rendering from execution log
- Standalone Ollama harness for testing

Future work:

- Jupyter kernel backend (wrapping `jupyter_client.KernelClient`)
- Full plugin lifecycle API (registration, discovery)
- Kibitzer plugin implementation (subscribes to ExecutionPlugin hooks)
- Document diffs: REPLACE_CELL, INSERT, DELETE
- Branching and merging
- Compaction / history squash
- Non-Python interpreter backends for code blocks
