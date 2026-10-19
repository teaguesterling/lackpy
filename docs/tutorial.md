# Tutorial

This tutorial runs the three worked examples from the lackpy pre-prompt. The
pre-prompt is the page that teaches a model to write lackpy. Each example
shows one habit of the language. Then the tutorial shows parameters,
generation, reuse and the trace.

**Prerequisites:** lackpy is installed. See [Getting Started](getting-started.md).

A lackpy program is plain Python 3 that a validator has checked. The full rule
set is the [style guide](reference/style-guide.md). File access goes through
`pathlib.Path`, under the [path policy](concepts/paths.md).

---

## 1. Setup

Make a workspace with some files to work on:

```bash
mkdir lackpy-tour && cd lackpy-tour
lackpyctl init
printf 'Status: draft\n' > notes.md
mkdir -p data src
printf '{"status": "ok"}\n{"status": "fail"}\nnot json\n{"status": "ok"}\n' > data/rows.jsonl
printf 'TODO: a\nTODO: b\n' > src/a.cpp
```

The workspace is the directory a run starts in. A relative path in a program
is relative to it.

---

## 2. Patch one occurrence in a file

Save this program as `patch.py`:

```python
from pathlib import Path

path = Path("notes.md")
text = path.read_text()
assert text.count("Status: draft") == 1, "the status line occurs once"
path.write_text(text.replace("Status: draft", "Status: final"))
```

Run it. A program file is the first argument of `lackpy`:

```bash
lackpy patch.py
```

```json
{
  "success": true,
  "output": null,
  "stdout": "",
  "error": null
}
```

The output above is shortened. `notes.md` now holds `Status: final`.

Run it again. The `assert` fails, so the program stops before it writes:

```json
{
  "success": false,
  "error": "the status line occurs once"
}
```

This is the lackpy way to guard an edit. Do not stop with `exit()` or `raise`.
State the condition with `assert cond, "why it must hold"`. Every `assert`
needs a message.

---

## 3. Read JSON lines, skip bad ones, count

```python
import json
from collections import Counter
from pathlib import Path

counts = Counter()
text = Path("data/rows.jsonl").read_text()
for line in text.splitlines():
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    counts[row["status"]] += 1
for status in sorted(counts):
    print(status, counts[status])
```

```text
fail 1
ok 2
```

The `try` block names the error it expects. lackpy refuses
`except Exception:` and a bare `except:`. A `try` block holds at most 3
statements. Its handler ends in `continue`, `return`, `pass`, or assignments
to every name that the body assigns.

lackpy has no `finally:`. Put cleanup code after the `try` block.

The program reads the whole file with `read_text()`, then parses each line
with `json.loads`. lackpy refuses `open()` and `json.load(f)`.

---

## 4. Scan files with a helper function

```python
from pathlib import Path


def todo_count(text):
    return text.count("TODO")


paths = Path("src").glob("*.cpp")
for p in sorted(paths):
    text = p.read_text()
    print(p.name, todo_count(text))
```

```text
a.cpp 2
```

A `def` is at the top level, has at most 12 statements, and comes before its
first use. `glob` and `rglob` list files. If the path policy refuses to read an
entry, the listing omits it.

---

## 5. What the validator refuses

The validator checks a program before any of it runs. Validate a file with
`--validate`:

```python title="Refused: open() is not approved"
with open("notes.md") as f:
    text = f.read()
print(text)
```

```bash
lackpy bad.py --validate
```

```json
{
  "valid": false,
  "errors": [
    "Forbidden name: 'open' at line 1"
  ],
  "calls": ["print", "open"]
}
```

`run_program` validates first. If the program is not valid, nothing runs.

The [style guide](reference/style-guide.md) lists every rule, with an
approved and a not-approved example. The [language spec](concepts/language-spec.md)
gives a summary.

---

## 6. Parameters

A program takes its inputs from one top-level `params` literal. The values in
the literal are the defaults:

```python
from pathlib import Path

params = {"name": "notes.md"}
text = Path(params["name"]).read_text()
print(len(text.splitlines()))
```

Pass a value on the command line with `--param`:

```bash
lackpy count.py --param name=src/a.cpp
```

lackpy puts the passed value into the literal before the program runs. The
program reads `params["name"]`, as it does under `python3`. A param binds no
other name.

Rules for passed params:

- Each key must be a key of the program's `params` literal.
- Each value must have the type of its default. An `int` is accepted for a
  `float` default. A `None` default accepts any type.
- Each value must have a literal form: a string, number, bool, `None`, or a
  list, tuple, set or dict of these.
- The program must assign `params` exactly once.

```python title="Python API: pass params"
result = await svc.run_program(program, params={"name": "src/a.cpp"})
```

---

## 7. Generate a program from an intent

`delegate` turns an intent into a program, then runs it. It is the default
mode of `lackpy -c`:

```bash
lackpy -c "read the file notes.md"
```

`--generate` gives the program without a run:

```bash
lackpy -c "read the file notes.md" --generate
```

```python
from pathlib import Path

text = Path('notes.md').read_text()
print(text, end="")
```

The generator tries tiers in the order of `[inference] order`:

1. `templates`: a saved program whose pattern matches the intent.
2. `rules`: keyword rules for simple intents, such as "read the file X".
3. A model provider, such as the local Ollama provider that `lackpyctl init` adds.

A model receives the pre-prompt, the style guide and a program skeleton. The
generated program goes through the same validator as a program you wrote.

```python title="Python API: delegate"
result = await svc.delegate("read the file notes.md")
print(result["program"])
print(result["stdout"])
```

!!! note "Refused inference modes"
    Mode `staged-dsl` and the `literate` execution axis are refused. They
    wait for a migration to the path policy. Use mode `1-shot` or `spm`.

---

## 8. Save a program for reuse

### Lackey files

`--create` generates a program and saves it as a Lackey file under
`.lackpy/templates/`:

```bash
lackpy -c "read the file notes.md" --create --name ReadNotes
```

```text
Created .lackpy/templates/read_notes.py
```

Run a Lackey file by path. This skips generation:

```bash
lackpy .lackpy/templates/read_notes.py
```

### Templates

A template is a program with an intent pattern. The `templates` tier matches
it before any model runs:

```python title="Python API: save a template"
result = await svc.create(
    program='from pathlib import Path\n\nprint(Path("{path}").read_text())\n',
    name="read-file",
    pattern="read the file {path}",
)
print(result["path"])  # .lackpy/templates/read-file.tmpl
```

---

## 9. Custom rules

A custom rule makes the validator stricter. A rule is a callable that takes an
`ast.Module` and returns a list of error strings:

```python title="Python API: custom rules"
from lackpy.lang.rules import no_loops, max_calls

result = svc.validate(program, rules=[no_loops, max_calls(5)])
```

See [Custom Rules](extending/custom-rules.md).

---

## 10. Read the trace

Each `Path` I/O call adds one trace entry:

```python title="Python API: read the trace"
result = await svc.run_program(program)
for entry in result.trace.entries:
    print(entry.step, entry.tool, entry.args, entry.success)
```

```text
0 Path.read_text ['notes.md'] True
```

| Field | Meaning |
| --- | --- |
| `step` | The call order, from 0. |
| `tool` | `Path.<member>`, for example `Path.read_text`. |
| `args` | The path as the program wrote it, and the pattern for `glob` and `rglob`. |
| `result` | The return value. Listings record no result. |
| `success`, `error` | A refused or failed call has `success=False` and the error text. |

Builtins and module functions, such as `len` or `json.loads`, are not traced.

---

## Next steps

- [File access](concepts/paths.md): the path policy and its rules.
- [Style guide](reference/style-guide.md): every rule of the language.
- [Profiles](concepts/profiles.md): model and mode per task.
- [Inference Pipeline](concepts/inference.md): tiers and providers.
