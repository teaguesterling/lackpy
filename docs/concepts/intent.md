# Intent block

An intent block states the interface of a script: what it is for, its
parameters, and the files it may write. Another agent, a person or a CI job
can then run the script without reading its code. `stepy --help` prints the
block, and `stepy` enforces its declared writes.

## Terms

| Term | Meaning |
| --- | --- |
| program | A `.py` file, or the text given to `stepy -c`. |
| document | A [literate document](literate.md), a `.md` file. |
| script | A program or a document. |
| intent block | The `# /// intent` comment block at the top of a script. |
| declared writes | The globs in the `writes` key of the intent block. |

Only `stepy` reads the intent block. `lackpy`, `lackpyctl` and the MCP tools
treat it as a comment.

## The form

The block follows the form of PEP 723: a `# /// intent` line, TOML in comment
lines, and a `# ///` line. This program counts the TODO comments under a
directory:

```python
#!/usr/bin/env stepy
# /// intent
# goal = "Count the TODO comments in the Python files under a directory."
# reads = ["{root}/**/*.py"]
# writes = ["{out}"]
# checks = ["each count is 0 or more"]
# output = "printed"
#
# [params]
# root = { type = "str", doc = "the directory to search" }
# out = { type = "str", doc = "the report file" }
# limit = { type = "int", doc = "the most files to report" }
# ///
from pathlib import Path

params = {"root": "src", "out": "todos.txt", "limit": 10}

counts = {}
for path in sorted(Path(params["root"]).rglob("*.py")):
    counts[str(path)] = path.read_text().count("TODO")

lines = [f"{name}: {n}" for name, n in list(counts.items())[: params["limit"]]]
Path(params["out"]).write_text("\n".join(lines) + "\n")
print(f"{len(counts)} files, {sum(counts.values())} TODO comments")
```

## Where the block goes

- **In a program**, at the top, before any code. Only a shebang line, blank
  lines and comments can come before it.
- **In a document**, at the top of the first code cell. The block covers the
  whole document.
- The `# /// intent` line starts at column 0.
- A script has one block at most.

A block in the wrong place is an error, not a comment that lackpy ignores. A
marker after code, or in a later cell of a document, gives an error that names
the line or the cell.

## The keys

| Key | Type | Meaning | Checked |
| --- | --- | --- | --- |
| `goal` | string | What the script is for. | No. |
| `reads` | list of globs | The files the script reads. | No. |
| `writes` | list of globs | The files the script may write. | Yes, under `stepy` (see [Declared writes](paths.md#declared-writes)). |
| `checks` | list of strings | What a reader can check in the result. | No. |
| `output` | `"none"`, `"printed"` or `"value"` | What the script gives back. The default is `"none"`. | Only the value. |
| `[params]` | table | One entry for each parameter. | Yes, against the `params` literal. |

An unknown key is an error. So a misspelt `write` does not become an empty
`writes`. Measured on 2026-11-12:

```text
stepy: intent block: unknown key 'write'. Allowed keys: checks, goal, output, params, reads, writes.
```

A block with no `writes` key, and a script with no block, may write nothing
under `stepy`.

## Parameters

Each `[params]` entry has exactly two keys: `type` and `doc`. The types are
`str`, `int`, `float`, `bool` and `list[str]`.

The default of each parameter is in the program, in one top-level `params = {...}`
literal. The program reads `params["name"]`, as it does under `python3`. lackpy
checks the two in both directions before anything runs:

- Each key of the literal has a `[params]` entry.
- Each `[params]` entry has a key in the literal.
- Each default has the declared type, exactly. A `float` parameter needs a
  `float` default: write `1.0`, not `1`. A `bool` is not an `int`.

A mismatch exits 2. Measured on 2026-11-12:

```text
stepy: The 'params' literal has key 'limit', which [params] does not describe. Add it to [params].
stepy: The default of param 'ratio' is 1, but [params] declares type float.
```

In a document, lackpy checks `[params]` against the code cells joined in
order. So the `params` literal can be in any code cell (inferred from the
code, not measured).

## `stepy --help`

`stepy SCRIPT --help` prints the block. It shows `writes` as enforced and
`reads` as not checked. Measured on 2026-11-12, for the program above:

```text
Count the TODO comments in the Python files under a directory.

parameters:
  --root str (default: 'src') the directory to search
  --out str (default: 'todos.txt') the report file
  --limit int (default: 10) the most files to report
writes: declared, enforced
  {out}
reads: declared, not checked
  {root}/**/*.py
checks:
  each count is 0 or more

--timeout SECONDS is the time limit for the whole run.
usage: stepy SCRIPT.py [--help] [--check] [--timeout SECONDS] [--profile NAME] [--workspace DIR] [--<param> VALUE ...]
       stepy DOC.md [options]
       stepy -c 'program text' [options]
```

See [`stepy`](../reference/cli.md#stepy-run-a-script) for the parameter options.

## Limits

- lackpy does not derive the intent from the code, and it does not compare
  the two. A `goal` or a `reads` glob that does not agree with the code is not
  found.
- `reads` is declared only. A script can read files that `reads` does not name.
