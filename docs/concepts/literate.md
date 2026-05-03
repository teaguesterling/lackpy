# Literate Interpreter

The literate interpreter executes markdown documents with embedded
`` ```lackpy `` code blocks. Prose becomes `print()` calls, code executes
inline, and the captured stdout IS the rendered document. It replaces
N tool round-trips with a single document submission — the model writes
a complete response as a literate program, and the interpreter produces
the final rendered output in one pass.

This is the complement of MCP prompts: prompts shape the *request*,
literate documents shape the *response*.

```
   ┌─────────────┐     ┌────────┐     ┌──────────┐     ┌─────────┐
   │  markdown    │────▶│ parser │────▶│ compiler │────▶│ execute │
   │  document    │     │        │     │          │     │         │
   └─────────────┘     └────────┘     └──────────┘     └─────────┘
                            │              │                 │
                      Cell sequence   Python source     stdout = rendered
                                                        document
```

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
execute silently unless they produce print output. The entire document
compiles to one Python program — variables defined in any block are
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

## Compilation model

The compiler transforms each cell type into Python:

| Cell type | Python output |
|---|---|
| Prose `"Hello {x}"` | `print(f'Hello {x}')` |
| Prose `"Hello world"` | `print('Hello world')` |
| Empty prose | `print()` |
| Code | pass through |
| @hidden | pass through |
| @gather | pass through |
| @continue | `__literate_continue__()` |
| @read(path) | `print(read_file('path'))` |
| @write(path) | `write_file('path', 'content')` |
| @diff(path) | `apply_diff('path', 'diff_text')` |
| @scratch | capture `locals()` diff, print summary |

String literals use `repr()` for escaping, which eliminates edge cases
with quotes and special characters in file content.

All cells concatenate into a single Python program that executes in one
pass. Variables flow through — anything defined in an earlier block is
available to all later blocks and to prose interpolation.

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
├── compiler.py    Cell sequence → Python source (repr()-based string literals)
├── tools.py       read_file, write_file, apply_diff, search, shell, tests
└── prompt.py      system_prompt_hint() for model instruction
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
objects, and any parse errors.

### Compiler (`compiler.py`)

Transforms each `Cell` into Python source. The key design choice is
using `repr()` for all string literal generation — this handles quotes,
newlines, and special characters without manual escaping. Prose cells
with `{expr}` patterns compile to f-string `print()` calls.

### Interpreter (`__init__.py`)

Implements the standard `Interpreter` protocol:

- **`validate(program, context)`** — parses the document and reports
  errors without executing
- **`execute(program, context)`** — parses, compiles, executes with
  `redirect_stdout`, returns the captured output as markdown

The execution result includes metadata: `continue_requested` (whether
`@continue` was hit), `variables` (namespace after execution),
`cell_count`, and `frontmatter` settings.

### System prompt (`prompt.py`)

`system_prompt_hint()` returns a prompt fragment that teaches models to
write literate documents. It covers document structure, annotation
syntax (with correct/incorrect examples), the gather pattern, available
tools and builtins, and key rules.

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

## Phase 0 scope

The current implementation covers:

- All cell types: prose, code, @hidden, @gather, @continue, @read,
  @write, @diff, @scratch
- Frontmatter parsing
- Full Python builtins in the execution namespace
- Kit tool injection
- The @gather/@continue agent loop pattern
- Standalone Ollama harness for testing

Future phases will add:

- Cell-level execution with namespace caching (incremental re-execution)
- Document diffs: REPLACE_CELL, INSERT, DELETE
- Branching and merging
- Compaction / history squash
- Non-Python interpreter backends for code blocks
