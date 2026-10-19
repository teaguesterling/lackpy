# Literate documents

A literate document is a markdown file with code cells. lackpy runs the cells in
order, and the prose between them can show the values the cells compute. The
printed result is the rendered document.

A document replaces many tool round trips with one submission. The model
writes its whole answer as a document, and lackpy runs it.

Every code cell is a lackpy program. The cells of one document form one program.
lackpy validates each cell, and runs it in the same restricted namespace that
`run_program` uses. A cell reads and writes files only through `pathlib.Path`, under the
[path policy](paths.md). A cell can do nothing that a lackpy program cannot do.

## Terms

This page uses one term for each concept.

| Term | Meaning |
| --- | --- |
| document | One markdown text that lackpy runs. |
| cell | One part of a document: a code cell or a prose cell. |
| code cell | A fenced block. Its body is lackpy code. |
| prose cell | The markdown text between code cells. |
| session | The state of one run: the namespace, the program so far, and the trace. |
| program | The code of all code cells that have run, joined in order. |
| refusal | An error from the path policy. It ends the document. |

## Quick example

This document reads `notes.md` and reports on it. The file `notes.md` has three lines.

````markdown
# Notes summary

```lackpy @hidden
from pathlib import Path

text = Path("notes.md").read_text()
lines = text.strip().splitlines()
```

The file has {len(lines)} lines.

```lackpy
first_line = lines[0]
first_line
```

The title is: {first_line}
````

lackpy prints this text (measured on 2026-10-18). The number of blank lines between the lines can differ:

```text
# Notes summary

The file has 3 lines.

The title is: # Release notes
```

Prose prints as it is, and `{expression}` shows a value. A code cell prints only
what `print` prints. The cell `first_line` prints nothing. Its value is the
output of the run, because it is the last expression of the last code cell
(see [Output](#output)).

## One program

The code of the cells, joined in order, is a lackpy program. It runs
unchanged under `python3`, and gives the same output and the same written
files. The exceptions are `@diff` (see [Differences from python3](#differences-from-python3)) and prose.

lackpy checks this rule cell by cell. A cell is valid when the cells before
it, followed by this cell, validate as one lackpy program. So a name or an
import from an earlier cell is known in a later cell. An import in one cell
is not legal again in another cell.

A cell that is not valid does not run. The error names the rule. A prose
`{expression}` is validated in the same way, as a one-line program in the
session.

This document is refused, because the first cell imports `os`:

````markdown title="Refused: import os"
```lackpy
import os

names = os.listdir(".")
```

There are {len(names)} names.
````

The error is (measured on 2026-10-18):

```text
Validation failed: Forbidden name: 'os' at line 3 (use pathlib.Path); Forbidden AST node: Import at line 1 (use pathlib.Path)
```

The cell does not run. The name `names` stays unbound, so the prose that uses it becomes a hole (see [Holes and errors](#holes-and-errors)).

## Document format

### Frontmatter

A document can start with YAML frontmatter:

```yaml
---
echo: true        # code visibility (true|false|auto)
output: auto      # result visibility (all|last|hide|auto)
interpreter: python  # backend for code blocks
---
```

All fields are optional. The defaults are `echo=true`, `output=auto`,
`interpreter=python`. lackpy strips the frontmatter before it parses the
markdown, because `---` is a rule in CommonMark.

!!! note "Parsed, not applied"

    lackpy parses and stores the frontmatter fields and the per-block options
    (`echo=false`, `output=all`). They do not yet change how a document runs.
    Measured on 2026-10-18: a cell with `echo=false output=all` prints the
    same text as a cell without the options.

### Prose cells

Everything outside a fence is a prose cell. Text in braces, such as
`{variable}`, shows the value of that expression. Write `{{` and `}}` for
literal braces.

An expression can be a name, a call, or a longer expression. It must be a
valid lackpy expression, so `{open("f")}` is an error. An expression can use a
conversion or a format spec, for example `{total:.2f}`.

### Code cells

A code cell is a block fenced with `` ```lackpy ``. A block in a `<compute>` tag
is the same cell. The model prompt uses that form.

## Annotations

An annotation goes on the fence line, after the language tag.

| Annotation | Purpose | What it does |
| --- | --- | --- |
| *(none)* | Normal code | Runs the code. |
| `@hidden` | Silent setup | Runs the code. |
| `@gather` | Batch exploration | Runs the code. |
| `@continue` | Pause point | Marks where a streaming run pauses. |
| `@read(path)` | Show a file | `print(Path(path).read_text())` |
| `@write(path)` | Write a file | `Path(path).write_text(body)`. The body is literal text. |
| `@diff(path)` | Apply a unified diff | lackpy applies the diff to the file. |
| `@scratch` | Working memory | Runs the code, then prints the names that the cell bound. |

The path goes in parentheses on the fence line, or on the first line of the
body.

Each annotation that uses a file goes through `Path`, so the path policy
checks it. `@write` makes missing parent directories.

````markdown
```lackpy @write(out/report.txt)
Status: draft
```

```lackpy @read(out/report.txt)
```

```lackpy @scratch
size = len("abc")
label = "x"
```

Size is {size}.
````

This document prints (measured on 2026-10-18):

```text
Status: draft
[scratch: label=str, size=int]

Size is 3.
```

### The Path import rule

An annotation cell for `@read` or `@write` imports `Path` for the document,
unless an earlier cell already binds `Path`. A second import of `Path` is a
validation error. So an earlier `from pathlib import Path` is correct, and a later one,
after an annotation cell, is wrong.

This document is refused:

````markdown title="Refused: second import of Path"
```lackpy @read(notes.md)
```

```lackpy
from pathlib import Path

size = len(Path("notes.md").read_text())
```
````

The error is (measured on 2026-10-18):

```text
Validation failed: Rebinding of imported name: 'Path' at line 1 (imported at line 1 of cell 1) (annotation cells (@read, @write) import Path for the document; use Path without importing it again)
```

### @diff

`@diff` applies a unified diff to a file. lackpy does the work, not the
program. It reads the file through `Path`, applies the diff in Python, and
writes the file through `Path`. So the path policy checks the read and the
write.

````markdown
```lackpy @diff(config.py)
--- a/config.py
+++ b/config.py
@@ -1,2 +1,2 @@
-DEBUG = True
+DEBUG = False
 X = 1
```
````

Every context line and every `-` line must match the file. If one line does
not match, the cell fails and lackpy writes nothing. The error looks like this (measured on 2026-10-18):

```text
ValueError: the diff does not apply: line 1 is 'DEBUG = False', the diff expects 'DEBUG = True'
```

### @gather and @continue

`@gather` cells run and print nothing. `@continue` marks a pause in a streaming
run: lackpy returns the partial result to the caller, and the model writes the
next part with the values in view. In a one-shot run through `run_program`,
a `@continue` cell does not stop the run. Measured on 2026-10-18: the prose
after it printed.

## Holes and errors

A name that no cell binds yet is a hole. lackpy does not stop. It skips the
cell that needs the name, and it reports the hole.

An ordinary error in a cell binds the names of that cell to an error value.
A later cell that uses an error value is skipped, and the run goes on. This
document has one error (measured on 2026-10-18):

````markdown
```lackpy
scores = {"ann": 3}
best = scores["bob"]
```

Best is {best}.

```lackpy
count = len(scores)
```

Count is {count}.
````

lackpy prints this. The run reports `success=False`.

```text
[kernel] error: KeyError: 'bob' — bound as error value: best [/kernel]
[kernel] holes: cell skipped (blocked by best) [/kernel]

Count is 1.
```

The `[kernel] ... [/kernel]` lines are notes from lackpy. The parser ignores
them if a document with notes is read again.

### Forward references

A document can use a name before the cell that binds it. This document does that
(measured on 2026-10-18):

````markdown
The total is {total}.

```lackpy
total = 3 + 4
```

Done.
````

This is what happens:

1. The first prose cell finds the hole `total`. lackpy skips the cell.
2. The code cell binds `total`.
3. lackpy re-runs the first prose cell, because the name exists now.
4. The run still reports the hole. The result has `success=False`, and the error is `hole: 'total' unbound`.

The text of the model prompt about forward references is under review. This
page states what the code does, and it does not say which wording is correct.

## Refusal

A refusal ends the document. A refusal comes from the path policy, for
example a write to `.claude/settings.json`. It is not an ordinary error.
A document cannot catch it. The validator rejects the handlers that could
catch it: `except Exception`, a bare `except`, and `finally`. A refusal
inside a `try` also ends the document.

This document is refused:

````markdown title="Refused: write to .claude/settings.json"
First.

```lackpy
from pathlib import Path

Path(".claude/settings.json").write_text("{}")
```

Second.

```lackpy
later = 1
```
````

The text `First.` prints. The refused cell prints a note, and no later cell
runs. The result has `success=False`. The error is (measured on 2026-10-18, with
`<workspace>` for the workspace path):

```text
refused: writing <workspace>/.claude/settings.json (it is a permission settings location, which lackpy never writes) is not allowed in lackpy
```

## Output

The value of a run is the value of the trailing expression of the last code
cell. A cell that does not end in an expression has no value, so the value is
`None`. The value is data, or the text of its `repr`. A failed run has no value.
An annotation cell (`@read`, `@write`, `@diff`, `@continue`) is never the
last code cell: a document that ends in `@write` has the value of the code
cell before it.

Measured on 2026-10-18: for a document with `x = 2` in one cell and
`x + 1` in the next, the output is `3`. If a later cell is `y = 5`, the output is `None`.

The rendered document is in the `stdout` field of the result.

## Differences from python3

The code cells of a document run unchanged under `python3`, with two exceptions.

- **@diff.** lackpy applies a diff. The joined program holds a comment for the
  cell, `# @diff(path): applied by lackpy, not by this program`. So a document
  with `@diff` has no `python3` twin.
- **Prose.** Python has no prose. Only the code cells give the same output.

Line endings do not decide the result of a diff. Measured on 2026-10-18: a file with
CRLF line endings accepted the diff, and lackpy wrote it back with LF line
endings. `Path.read_text` turns CRLF into LF, as `python3` does.

## Running a document

A document runs when the profile has `execution = "literate"`. Set it in
`.lackpy/config.toml`:

```toml
[profiles.docs]
execution = "literate"

[profiles.docs-readonly]
execution = "literate"
policy = { read_only = true }
```

The same document runs through each of these routes:

- `run_program` with `profile="docs"`, through the library or through MCP.
- `delegate` with the same profile. lackpy generates the document first.
- `lackpyctl render --execute NOTES.md`. This runs the document under the
  workspace and prints the executed result. Use `--profile NAME` to pick a
  profile and `--base-dir DIR` to pick the workspace.

`lackpyctl render` without `--execute` runs nothing. It only converts the
document to markdown.

```python title="Python API: run a document"
from lackpy.profiles import Profile
from lackpy.service import LackpyService

svc = LackpyService(workspace=".")
result = await svc.run_program(document, profile=Profile(execution="literate"))
print(result.stdout)   # the rendered document
print(result.output)   # the last expression, or None
```

### The read-only profile

A profile with `policy = { read_only = true }` adds a deny on every write to
the path policy. Reads keep the normal rules. The key `read_only` is the only
key that the table accepts. A misspelt key, or a policy that is not a table,
is a profile error, and the run does not start.

Two mechanisms apply, in two layers:

1. A static check looks at each cell before it runs. If it sees a write, it
   marks the names as unavailable and skips the cell. The document goes on.
   Measured on 2026-10-18: a cell with `Path("c.txt").write_text("x")` was
   skipped with `effect ceiling exceeded: world-coupling w=3 exceeds ceiling w=1`, and the
   file was not written. The check also found `q = Path` followed by `q("c.txt").write_text("x")`.
2. The path policy refuses a write that the static check did not find. This
   is the boundary. The static check refuses earlier, and it never allows more
   than the policy allows.

The static check is a planner over the policy. It is not a second policy.

## What was removed

The literate interpreter had a kit of tools and ran cells with host builtins.
Both are gone.

- `read_file`, `write_file`, `apply_diff`, `search_content`, `run_command` and
  `run_tests` do not exist. Use `Path`. lackpy has no subprocess access.
- There is no `import os`, no `open`, and no host module. A cell has the
  restricted builtins and the approved modules, as a program has.
- lackpy does not call `os.chdir`. `Path` resolves against the workspace.
- When kibitzer is installed, `delegate` checks the names that a document calls
  against the kibitzer mode policy. A document calls `Path` methods, so this
  check sees almost nothing (inferred from the code, not measured). It adds
  little, and it is not a boundary. The path policy is the boundary.

## Execution model

A session owns one namespace. Each cell goes through these steps:

1. **Holes.** If the cell uses a name that no cell binds, lackpy opens a hole
   and skips the cell.
2. **Validation.** The program so far, followed by the cell, must validate.
3. **Execution.** The cell runs in the session namespace. The path policy checks
   each file access. lackpy captures the output.

Prose is not exec'd code. lackpy splits a prose cell into text and
expressions, validates each expression, and evaluates it in the session. A
display helper formats the value on the host side.

### Streaming

`StreamingCellParser` finds fence boundaries in partial model output. It
yields a cell when its closing fence arrives. So cells run while the model
writes.

### Recovery

When a cell fails, `StreamingDriver` calls a `RecoveryHandler`:

- `NoRecoveryHandler` aborts. Batch runs and tests use it.
- `InferenceRecoveryHandler` formats the error and the scope into a prompt,
  calls a model, and parses the answer as replacement cells.

The handler returns one action: `fix`, `inspect`, `skip` or `abort`. A refusal is
not a cell failure. The driver stops on a refusal.

Recovery works on the current state. The model cannot rewrite earlier cells.
It emits cells that correct the state from now on.

### Artifact formats

The execution log can be written as a Jupyter notebook (`.ipynb`) or as clean
markdown. This page does not cover notebook execution.

## Architecture

```text
src/lackpy/run/cells.py        CellSession: validate and run one cell at a time
src/lackpy/interpreters/literate/
├── __init__.py    LiterateInterpreter: validate(), execute()
├── parser.py      markdown to cells (frontmatter, fences, prose)
├── compiler.py    cell to Python source; annotations to Path code
├── patch.py       the @diff patcher (pure Python, checks every line)
├── display.py     bounded display of a value in prose
├── effects.py     the static check (planner) over the policy
├── prompt.py      the model prompt for documents
└── kernel/
    ├── lightweight.py       LightweightKernel: runs cells through CellSession
    ├── static_analysis.py   name resolution before a cell runs
    ├── streaming_parser.py  StreamingCellParser
    ├── driver.py            StreamingDriver
    ├── recovery.py          RecoveryHandler and the handlers
    └── formats.py           to_notebook, from_notebook, render_markdown
```

No code from a document runs outside `CellSession`. A test scans the package
for `exec`, `eval` and `compile`, and for imports of host modules.
