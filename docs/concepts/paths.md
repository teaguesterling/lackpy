# File access

A lackpy program reads and writes files only through `pathlib.Path`. Inside
lackpy, `Path` is a stand-in class. Each file access goes to the **path
policy** (`PathPolicy`) before any I/O starts. The path policy allows the
access or refuses it.

A refusal stops the program with an error. It never changes a result. A
program that lackpy runs gives the same output under `python3` from the
workspace directory. Inside lackpy, it can only do less.

A [literate document](literate.md) uses the same policy for each of its cells.

The path policy uses the `Read(...)` and `Edit(...)` permission rules of
Claude Code. If Claude Code already guards a project, lackpy reads the same
rules.

## Terms

| Term | Meaning |
| --- | --- |
| workspace | The directory a run starts in. A relative path is relative to it. |
| extra directory | A directory that a settings file adds to the workspace (`additionalDirectories`). |
| rule | One `Read(...)` or `Edit(...)` permission string, with an effect: allow or deny. |
| refusal | An error that stops the program. It names the path and the reason. |

## The default

With no rules, the path policy does this:

- It allows reads and writes inside the workspace and the extra directories.
- It refuses every access outside them.
- It refuses every write to a settings location (see [Settings locations](#settings-locations)).

This program reads one file in the workspace. It needs no rule:

```python
from pathlib import Path

text = Path("notes.md").read_text()
print(len(text.splitlines()))
```

This program reads a file outside the workspace, so lackpy refuses it:

```python title="Refused: outside the workspace"
from pathlib import Path

print(Path("/etc/passwd").read_text())
```

The run fails with this error (measured 2026-10-18, PathPolicy and RestrictedRunner):

```text
reading /etc/passwd (it is outside the workspace and no allow rule matches; to allow it, add Read(//etc/passwd) to permissions.allow in .claude/settings.json) is not allowed in lackpy
```

## Sources

The path policy loads its rules on each run, from these files. It joins the
rules from all of them.

1. The Claude Code managed settings file, then each `*.json` file in the
   `managed-settings.d/` directory beside it, in alphabetical order. Hidden
   files (`.*.json`) are skipped. On Linux,
   the file is `/etc/claude-code/managed-settings.json`. On macOS, it is
   `/Library/Application Support/ClaudeCode/managed-settings.json`.
2. `~/.claude/settings.json`.
3. `<workspace>/.claude/settings.json` and `<workspace>/.claude/settings.local.json`.
4. `<workspace>/.lackpy/config.toml`, table `[paths]`.

In a Claude Code settings file, lackpy reads `permissions.allow`,
`permissions.deny`, `permissions.ask` and `permissions.additionalDirectories`.

The `[paths]` table has three keys, and no other key is accepted:

```toml
[paths]
allow = ["Read(~/datasets/**)"]
deny = ["Read(.env)", "Edit(src/generated/**)"]
additional_directories = ["../shared"]
```

A missing file is skipped. lackpy loads no rules from any other place.

### Managed-only mode

A managed source can set `allowManagedPermissionRulesOnly: true`. lackpy reads
the flag at the top level or inside `permissions`. When it is set, only the
managed sources apply. lackpy then ignores the rules and the extra directories
from every other source, including `[paths]`.

## Rule syntax

A rule is `Read(<pattern>)` or `Edit(<pattern>)`.

- A `Read` rule controls reads. A `Read` deny also blocks writes.
- An `Edit` rule controls writes.
- A `Read` allow grants reads only. An `Edit` allow grants writes only.

lackpy ignores rules for other tools, such as `Bash(...)` or `Write(...)`.

### Anchors

| Pattern | Anchored at |
| --- | --- |
| `//p` | The filesystem root: `//etc/hosts` is `/etc/hosts`. |
| `~/p` | Your home directory. |
| `/p` | The directory of the settings source: the workspace for workspace files, `~/.claude` for `~/.claude/settings.json`, and the managed settings directory for managed files. |
| `p` or `./p` | The workspace. |

### Globs

- `*` matches characters in one path segment.
- `**` as a whole segment matches zero or more segments. So `d/**` also matches `d`.
- A deny also matches every path below a path it matches.
- A bare file name (one segment, such as `.env`) matches at any depth. For a
  deny, that is anywhere on the filesystem. For an allow, it is anywhere below
  the workspace.
- A one-segment directory pattern, such as `src/**`, matches at any depth for a
  deny. For an allow, it matches only `<workspace>/src`.

Matching is case-sensitive.

### Order of checks

1. A write to a settings location refuses. No rule changes this.
2. A deny from any source refuses.
3. An allow that matches grants the access.
4. With no matching rule, the default applies.

### Effects in Claude Code settings

- `allow` gives an allow rule.
- `deny` gives a deny rule.
- `ask` gives a **deny** rule. lackpy cannot ask a person during a run, so it refuses.

A bare `Read` or `Edit` (no parentheses) in `deny` or `ask` matches every path.
So a bare `Read` in `ask` blocks all reads and all writes. A bare `Edit` in
`ask` blocks all writes. A bare allow is ignored.

## Examples

This configuration hides `.env` files and protects generated code:

```toml
[paths]
deny = ["Read(.env)", "Edit(src/generated/**)"]
```

With that configuration, a listing omits `.env`. A deny on `.env` does not make
every `glob` fail, and it does not show that the file exists:

```python
from pathlib import Path

for p in sorted(Path(".").glob("*")):
    print(p.name)
```

A direct read of `.env` refuses:

```python title="Refused: denied by Read(.env)"
from pathlib import Path

print(Path(".env").read_text())
```

```text
reading <workspace>/.env (denied by Read(.env) in .lackpy/config.toml) is not allowed in lackpy
```

This configuration lets programs read a dataset in your home directory:

```toml
[paths]
allow = ["Read(~/datasets/**)"]
```

A program cannot use `Path.home()` (see [Limits](#limits)). Write the path as
text, in the `params` literal, so a caller can change it:

```python
import json
from pathlib import Path

params = {"data": "/home/me/datasets/rows.json"}
rows = json.loads(Path(params["data"]).read_text())
print(len(rows))
```

## Settings locations

A write to a settings location would widen the path policy of every later run.
So lackpy refuses each write to:

- any path with a `.claude` or `.lackpy` component, at any depth;
- `~/.claude.json`;
- the managed settings file and its `managed-settings.d/` directory.

lackpy checks the path as written and its resolved target. The check ignores
case, also on Linux. No rule and no extra directory lifts it.

A workspace inside `~/.claude` is therefore read-only: each write in it refuses.

## Host code files

Some files make a later host command run code. A git config or hook runs when
you run `git`. An MCP client starts the commands in `.mcp.json`. direnv runs
`.envrc`. So lackpy also refuses each write to:

- any path with a `.git` component, at any depth;
- any file or directory named `.mcp.json` or `.envrc`, at any depth.

The check is the same as for settings locations: the path as written and its
resolved target, case ignored, and no rule lifts it. Reads stay allowed. lackpy
does not find a git directory that has another name, for example one made with
`git init --separate-git-dir`.

## Symlinks

- An allow must match both the path as the program wrote it and its resolved target.
- A deny refuses if it matches either one.
- A write refuses when the last component is a symlink.
- A write refuses when a directory component is a symlink to a place that is not writable.

lackpy resolves the literal start of each deny pattern, such as `~/.ssh` in
`Read(~/.ssh/**)`. So a symlinked `~/.ssh` does not avoid its deny.

## Limits

- **A malformed rule fails every run.** The error names the rule and the file.
  A dropped deny would loosen the path policy, so lackpy does not skip it. An
  unknown key in `[paths]` fails the same way.
- **A deny with no literal start is not resolved through symlinks.** Such a
  deny matches the path as written and its resolved target. It does not
  follow a symlinked directory back to the directory it names. For example,
  `Read(**/vault/**)` with `vault` a symlink to `data/` refuses
  `vault/f.txt`, but it allows `data/f.txt`. The target is not denied. A
  one-segment pattern, such as `Read(vault/**)`, floats in the same way. An
  anchored deny, such as `Read(./vault/**)`, resolves `vault`, so it refuses
  both paths (measured 2026-10-18).
- **The check and the I/O are two system calls.** A process that swaps a
  symlink between them is not stopped. lackpy assumes one user per workspace.
- **Listings omit refused entries.** `glob`, `rglob` and `iterdir` leave out
  each entry that the path policy refuses for read. Where a deny applies, a
  program sees fewer entries than `python3` does.
- **A `glob` pattern must stay below its directory.** A pattern that is
  absolute or that contains `..` refuses.
- **`Path.home()` and `Path.cwd()` refuse at run time.** Every class
  attribute of the stand-in refuses. The validator does not reject them, so
  the program fails only when it runs. Pass a path in the `params` literal
  instead.

## Path members

A program can use these `Path` members. The [style guide](../reference/style-guide.md#path-members)
is the generated list.

| Member | Access |
| --- | --- |
| `read_text`, `read_bytes`, `exists`, `is_file`, `is_dir` | read |
| `glob`, `rglob`, `iterdir` | read, and refused entries are omitted |
| `write_text`, `write_bytes` | write |
| `mkdir` | write; with `parents=True`, each new directory is checked |
| `name`, `stem`, `suffix`, `parent`, `parts`, `joinpath`, `with_suffix`, `relative_to`, `/` | none: no I/O |

`open()` and `os` are not approved. Use `Path(p).read_text()` and
`Path(p).write_text(s)`.

## The trace

Each I/O call adds one trace entry. The tool name is `Path.<member>`, and the
arguments hold the path as the program wrote it. This entry is measured, for the
`notes.md` of the [tutorial](../tutorial.md) (`Status: draft` and a newline):

```text
TraceEntry(step=0, tool='Path.read_text', args=['notes.md'], result='Status: draft\n', duration_ms=0.12043211609125137, success=True, error=None)
```

A refused call adds a failed entry, then the refusal stops the program.

## Python API

`LackpyService.run_program` and `delegate` call `load_policy(workspace)` on
each run. A malformed settings file gives a failed result, not an exception.

```python title="Python API: load a path policy"
from pathlib import Path

from lackpy.run.paths import load_policy

policy = load_policy(Path("."))
print(len(policy.rules))
```

See the [Python API reference](../reference/api.md#file-access) for
`load_policy`, `PathPolicy`, `parse_rule` and `make_path_class`.
