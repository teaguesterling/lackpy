# lackpy

**Python that lacks most of Python.** lackpy runs plain Python that a validator
has checked. It can also write the program: it turns a natural-language intent
into a program with a model that you configure.

```
intent → program → validator → run under the path policy → result and trace
```

A lackpy program is plain Python 3:

```python
from pathlib import Path

paths = Path("src").glob("*.py")
for p in sorted(paths):
    print(p.name, len(p.read_text().splitlines()))
```

---

## Why lackpy?

- **Plain Python.** A program that lackpy runs gives the same output under
  `python3`, from the workspace directory, with only the standard library.
  Inside lackpy it can only do less. A refusal is an error, never a different
  result.
- **A validator before each run.** lackpy checks every program at the syntax
  tree level before any of it runs. The [style guide](reference/style-guide.md)
  lists each rule. Imports are limited to `pathlib`, `json`, `re`,
  `collections`, `math`, `statistics`, `hashlib` and `datetime`. A `try` must
  name its error. `while`, `class` and `global` are not in the language.
- **File access under a path policy.** A program reads and writes files only
  through `pathlib.Path`. Each access goes to the [path policy](concepts/paths.md),
  which reads the `Read(...)` and `Edit(...)` permission rules of Claude Code.
- **Model output gets the same checks.** A generated program goes through the
  same validator and path policy as a program you wrote. A model that writes
  `import os` or `open(p)` gets a refusal before anything runs.
- **Reuse.** Save a validated program as a Lackey file or a template. A template
  matches a later intent with no model call.

---

## Quick start: CLI

lackpy has two commands: `lackpy` runs and delegates programs, and `lackpyctl`
manages the workspace.

```bash
# Initialize a workspace
lackpyctl init

# Run a program file
lackpy patch.py

# Generate a program from an intent, and run it
lackpy -c "read the file README.md"

# Generate only
lackpy -c "read the file README.md" --generate

# Validate a program file
lackpy patch.py --validate
```

## Quick start: Python API

```python title="Python API: delegate"
import asyncio
from lackpy.service import LackpyService

async def main():
    svc = LackpyService()
    result = await svc.delegate("read the file pyproject.toml")
    print(result["program"])
    print(result["stdout"])
    print(result["trace"])  # one entry for each Path I/O call

asyncio.run(main())
```

---

## Reuse a program

**Lackey files.** `--create` generates a program and saves it as a Lackey file
under `.lackpy/templates/`. Run it by path, with no generation:

```bash
lackpy -c "read the file README.md" --create --name ReadFile
lackpy .lackpy/templates/read_file.py
```

**Templates.** A `.tmpl` template has an intent `pattern`. Save one with
`svc.create(pattern=...)`. The `templates` tier matches later intents to the
pattern before any model runs.

---

## Next steps

- [Getting Started](getting-started.md): install, initialize, first commands
- [Tutorial](tutorial.md): the three worked examples, step by step
- [File access](concepts/paths.md): the path policy
- [Style guide](reference/style-guide.md): every rule of the language
- [Python API Reference](reference/api.md)
- [CLI Reference](reference/cli.md)
