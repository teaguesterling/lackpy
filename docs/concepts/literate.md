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
| failure | A cell that raises an exception, times out, reads a name that is not bound, fails validation, or passes the effect ceiling. The document stops there. |
| refusal | An error from the path policy. It ends the document and the session. |

lackpy numbers the cells of a document from 0, and prose cells count. An error
names the cell by its number and its first line, for example
`cell 2 (for i in iter(int, 1):)`.

## Quick example

This document reads `notes.md` and reports on it. The file `notes.md` has three lines.

<!-- docs-run: lackpyctl render --execute -->
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

<!-- docs-run: lackpyctl render --execute -->
````markdown title="Refused: import os"
```lackpy
import os

names = os.listdir(".")
```

There are {len(names)} names.
````

`lackpyctl render --execute` prints this (measured on 2026-11-12, exit code 1).
The first line is the error, on stderr:

```text
lackpyctl render --execute: cell 0 (import os): Validation failed: Forbidden name: 'os' at line 3 (use pathlib.Path); Forbidden AST node: Import at line 1 (use pathlib.Path)
[kernel] error: Validation failed: Forbidden name: 'os' at line 3 (use pathlib.Path); Forbidden AST node: Import at line 1 (use pathlib.Path) [/kernel]
```

The cell does not run, and the document stops there. The prose after it does
not print (see [Failures](#failures)).

A later cell can bind a name again, as a script does (`rows = rows[1:]`).
lackpy does not run the earlier cells again for this: each cell keeps the
value it saw, as under `python3`.

`LackpyService.validate` with a literate profile checks a document cell by
cell in the same way, without running it. It does not check that a name is
bound before a cell reads it. Measured on 2026-11-12: `validate` and
`stepy --check` both accept a document whose first prose cell reads a name
that a later cell binds. The run stops at that prose cell.

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
    same text as a cell without the options. The one option that lackpy
    applies is `rerun` (see [rerun=no](#rerunno)).

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

### The intent block

The first code cell can start with an intent block. The block states the goal
of the document, its parameters and the files it may write. The block in the
first code cell covers the whole document. See [Intent block](intent.md).

### rerun=no

A resume can run a cell again (see [Retry by resubmitting](#retry-by-resubmitting)).
For a cell that writes a file, a second run writes the file again. The fence
option `rerun=no` forbids that:

````markdown
```lackpy rerun=no
from pathlib import Path

Path("log.txt").write_text("written once\n")
```
````

The `<compute>` form takes the same option as an attribute: `<compute rerun="no">`.

- `rerun=yes` is the default. A resume can run the cell again.
- With `rerun=no`, after the cell's code has started in a session, no later
  submission in that session can run it again. This is true if the cell
  succeeded or failed.
- A submission that would run the cell again is refused before anything runs.
  The result has `error_phase="refused"`. The session stays open.
- A new session is a new run. `rerun=no` does not reach across sessions.
- `stepy` and `lackpyctl render --execute` have no resume, so `rerun` changes
  nothing there.
- A value other than `yes` or `no` is a validation error.

lackpy does not say what to do after such a refusal. The prose of the document
says it. [Retry by resubmitting](#retry-by-resubmitting) shows a measured run.

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

<!-- docs-run: lackpyctl render --execute -->
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

<!-- docs-run: lackpyctl render --execute -->
````markdown title="Refused: second import of Path"
```lackpy @read(notes.md)
```

```lackpy
from pathlib import Path

size = len(Path("notes.md").read_text())
```
````

`lackpyctl render --execute` prints this (measured on 2026-11-12, exit code 1).
The `@read` cell runs and prints the file. The error names the earlier cell
by its number and its first line:

```text
lackpyctl render --execute: cell 1 (from pathlib import Path): Validation failed: Rebinding of imported name: 'Path' at line 1 (imported at line 1 of cell 0 (read)) (annotation cells (@read, @write) import Path for the document; use Path without importing it again)
# Release notes
line two
line three

[kernel] error: Validation failed: Rebinding of imported name: 'Path' at line 1 (imported at line 1 of cell 0 (read)) (annotation cells (@read, @write) import Path for the document; use Path without importing it again) [/kernel]
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

## Failures

A document is a procedure, and each cell is a step. A step can start only when
the steps before it succeeded. So a document stops at its first failure:

- The failed cell stops. What it printed before the failure stays in the output.
- No later cell runs, and no later cell prints.
- The result has `success=False`. The error names the cell and the cause.
- A prose expression that raises is a failure at its place in the document.
- A session stays open after a failure, so the caller can correct the
  document and resume (see [Retry by resubmitting](#retry-by-resubmitting)).
  A refusal is different: it ends the session.

This applies to every route that runs a document: `stepy`, `lackpyctl render --execute`,
`run_program`, the MCP tools and `literate_agent`.

This document has one error:

<!-- docs-run: lackpyctl render --execute -->
````markdown
```lackpy
scores = {"ann": 3}
print("scores ready")
best = scores["bob"]
```

Best is {best}.

```lackpy
count = len(scores)
```

Count is {count}.
````

`lackpyctl render --execute` prints this (measured on 2026-11-12, exit code 1):

```text
lackpyctl render --execute: cell 0 (scores = {"ann": 3}): KeyError: 'bob'
scores ready
[kernel] error: KeyError: 'bob' [/kernel]
```

The cell printed `scores ready` before the error, so that line stays. The cell
`count = len(scores)` does not depend on `best`, but it does not run. lackpy does
not decide which later cells are still valid.

The `[kernel] ... [/kernel]` lines are notes from lackpy. The parser ignores
them if a document with notes is read again.

### A name must be bound before a cell reads it

A cell that reads a name that no earlier cell binds is a failure. Put the cell
that binds the name first. This document reads `total` too early:

<!-- docs-run: lackpyctl render --execute -->
````markdown
The total is {total}.

```lackpy
total = 3 + 4
```

Done.
````

`lackpyctl render --execute` prints this (measured on 2026-11-12, exit code 1):

```text
lackpyctl render --execute: cell 0 (The total is {total}.): the name 'total' is not bound. Bind it in an earlier cell
[kernel] the name 'total' is not bound. Bind it in an earlier cell [/kernel]
```

The model prompt for documents states the same rule: a cell comes after the
cells whose names it reads, and a failure stops the document.

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

The rendered document is in the `stdout` field of the result. The `printed`
field holds only what the cells printed, with no prose. `stepy` writes
`printed` to its stdout.

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
- `stepy NOTES.md`. This prints only what the cells print, and it enforces the
  declared writes of the intent block. See [`stepy`](../reference/cli.md#stepy-run-a-script).
  `lackpyctl render --execute` and `stepy` need no `execution = "literate"`
  profile: they run the named profile, or the default profile, as literate.

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

1. A static check looks at each cell before it runs. If it sees a write, the
   cell does not run, and this is a failure: the document stops there.
   The check also finds `q = Path` followed by `q("c.txt").write_text("x")`
   (measured on 2026-10-18).
2. The path policy refuses a write that the static check did not find. This
   is the boundary. The static check refuses earlier, and it never allows more
   than the policy allows.

The static check is a planner over the policy. It is not a second policy.

This document runs under the `docs-readonly` profile above:

<!-- docs-run: lackpyctl render --execute --profile docs-readonly -->
````markdown
Before.

```lackpy
from pathlib import Path

Path("c.txt").write_text("x")
```

After.
````

`lackpyctl render --execute --profile docs-readonly` prints this (measured on
2026-11-12, exit code 1). lackpy does not write `c.txt`:

```text
lackpyctl render --execute: cell 1 (from pathlib import Path): effect ceiling exceeded: world-coupling w=3 exceeds ceiling w=1: the profile is read-only (policy.read_only = true)
Before.
[kernel] effect ceiling exceeded: world-coupling w=3 exceeds ceiling w=1: the profile is read-only (policy.read_only = true) [/kernel]
```

## Time limits and retries

Every cell has a time limit. The page [Time limits](time-limits.md) gives the
full rules. This section shows what a document sees.

A code cell, or a prose cell with an interpolation, that runs longer than the
limit is stopped. This is a timeout. A timeout is a failure, so the document
stops there. A timeout is not a refusal: the session stays open, and the
caller can retry.

This document has a cell that never ends. The first code cell writes a file.
The second code cell loops:

<!-- docs-run: lackpyctl render --execute --timeout 2 -->
````markdown title="A cell that never ends"
# Stuck cell

```lackpy
from pathlib import Path

Path("log.txt").write_text("cell 1 ran\n")
total = 5
```

```lackpy
for i in iter(int, 1):
    total = total + 1
```

The total is {total}.
````

`lackpyctl render --execute --timeout 2` prints this (measured on 2026-11-12,
exit code 1). The first line is the error, on stderr, and the rest is the
rendered document:

```text
lackpyctl render --execute: cell 2 (for i in iter(int, 1):): timeout: the cell took longer than 2 s and was stopped
# Stuck cell
[kernel] timeout: the cell took longer than 2 s and was stopped [/kernel]
```

Four points to read in this output:

- The cell number counts every cell of the document from 0, prose included.
  The title line is cell 0. So the looping cell is cell 2.
- The prose `{total}` after the stopped cell does not print. The document
  stopped at cell 2 (see [Failures](#failures)).
- Cell 1 is not affected. The state of the document is the state from before the
  stopped cell.
- If the stopped cell wrote a file before the stop, the note names it:
  `; it wrote log.txt before it was stopped`. The trace has the writes that
  lackpy confirmed. A write that had started but was not confirmed is named in
  the note. It is not in `trace.files_modified`.

### Retry by resubmitting

A failure ends no session, so the caller retries. The caller sends the full
corrected document. lackpy does not retry for the caller.

A document runs in a session worker. The worker takes one checkpoint (a fork of
its process) before each segment. A segment ends at any cell that can run code:
a code cell, or a prose cell with an interpolation. A prose cell with no
interpolation runs no code, so it takes no checkpoint.

A retry resumes at the cell `k`, where `k` is the smaller of two numbers:

- the first cell whose text changed since the last submission;
- the first cell that failed in the last run (a timeout is a failure).

The cells before `k` do not run again, so their writes do not repeat. The
resume starts at the checkpoint of the segment that holds `k`. The comparison
of cell text is exact. It covers the cell type, the source, the annotations
and the frontmatter.

The document runs fresh in these cases:

- the checkpoint for that point is gone (the session dropped it for memory,
  see [Memory](#memory));
- the submission has other params;
- the session failed;
- it is the first submission of a session.

A short script submitted a document with two code cells to one service, using
`LackpyService.run_program`. The first code cell appends a line to `log.txt`
and sets `total = 5`. The second code cell loops. The script made four calls
with `profile=Profile(execution="literate")`. Calls 2 to 4 passed the `session_id` of call 1:

1. the document with the looping second code cell, with `timeout=2`;
2. the same document with the second code cell rewritten as `total = total * 2`;
3. the document with the first code cell changed (`total = 7`);
4. the document of call 2 again, after `close_session`.

The document has a title, so its code cells are cells 1 and 2. The script then
ran the same four calls again, with `rerun=no` on cell 1. The output below was
measured on 2026-11-12. It is pasted without change, except for two things: the
session ids are `<id>`, and the lines for `printed` are left out (each was empty).

```text
===== no option
--- 1. cell 2 loops, timeout 2 s: success=False error_phase=timeout
    error: cell 2 (for i in iter(int, 1):): timeout: the cell took longer than 2 s and was stopped
    log.txt: ['cell 1 ran']
--- 2. cell 2 rewritten, same session_id: success=True error_phase=None
    log.txt: ['cell 1 ran']
--- 3. cell 1 changed, same session_id: success=True error_phase=None
    log.txt: ['cell 1 ran', 'cell 1 ran']
close_session: True
--- 4. closed id: success=False error_phase=None
    error: unknown or expired session_id '<id>'
    log.txt: ['cell 1 ran', 'cell 1 ran']
===== rerun=no on cell 1
--- 1. cell 2 loops, timeout 2 s: success=False error_phase=timeout
    error: cell 2 (for i in iter(int, 1):): timeout: the cell took longer than 2 s and was stopped
    log.txt: ['cell 1 ran']
--- 2. cell 2 rewritten, same session_id: success=True error_phase=None
    log.txt: ['cell 1 ran']
--- 3. cell 1 changed, same session_id: success=False error_phase=refused
    error: cell 1 (from pathlib import Path) has rerun=no and has already run in this session; start a new session to run it again
    log.txt: ['cell 1 ran']
close_session: True
--- 4. closed id: success=False error_phase=None
    error: unknown or expired session_id '<id>'
    log.txt: ['cell 1 ran']
```

Call 2 resumed at the stopped cell. Cell 1 did not run again, so `log.txt` has
one line. Call 3 changed cell 1, so the document ran fresh, and `log.txt` has
two lines. With `rerun=no`, call 3 is refused before anything runs, and
`log.txt` keeps one line. Call 4 shows the error for a closed session.

These cases are not resumed. They run fresh:

- an append-only resubmission after a run that completed all its cells;
- a resubmission that pauses at `@continue` in `literate_agent`;
- a submission after the session failed.

### Sessions over MCP

`run_program` returns a `session_id` for a document on a literate profile. Pass
it back with the full corrected document to resume. A call without a
`session_id` starts a fresh session. `literate_agent` is a script. It keeps one
worker session for its full-document resubmissions. Its first run and its
`@continue` rounds run in-process, with no time limit.

Call `close_session(session_id)` when you are done. A session also ends after
600 s without a call (lackpy reaps it on the next registry call), or when the
registry evicts it. An unknown or expired `session_id` is an error. It is never
a silent fresh run. A session belongs to one workspace and one profile.

### Memory

Each checkpoint is a process. Python writes reference counts, so a checkpoint
copies any value that a cell iterates over. A session of `run_program` (and so
of MCP) keeps at most 32 checkpoints and at most 256 MiB of measured memory
(Pss) across them, and the service keeps at most 2 sessions. The one session of
`literate_agent` has a budget of 1024 MiB. When a
limit is passed, it drops checkpoints and keeps two: the checkpoint before the
last cell that did not succeed, and the latest one.

Measured without the memory budget: 32 checkpoints over about 100 MB of ints
held 2862 MiB Pss. This is why the budget exists. A document with large values
has fewer resume points. A resume point that was dropped runs fresh.

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

1. **Names.** If the cell reads a name that no earlier cell binds, the cell
   fails.
2. **Validation.** The program so far, followed by the cell, must validate.
3. **Execution.** The cell runs in the session namespace. The path policy checks
   each file access. lackpy captures the output.

A failure at any step stops the document.

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
