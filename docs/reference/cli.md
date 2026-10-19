# CLI Reference

lackpy ships **two** command-line entry points:

| Command | Purpose |
|---------|---------|
| `lackpy` | Run inference: delegate / generate / validate / create from intent, and run program files. |
| `lackpyctl` | Manage the workspace: init, status, toolbox, templates, providers, MCP server. |

```bash
lackpy    [--workspace PATH] -c "<intent>" [flags]      # or: lackpy <file> [flags]
lackpyctl [--workspace PATH] <command> [args]
```

Both accept `--workspace PATH` to set the project root (default: current directory).

> **Note:** `lackpy` is **flag-based**, not subcommand-based. The mode is selected by
> flags on a single intent (`-c`): the default is *delegate* (generate + run);
> `--generate`, `--validate`, and `--create` select the other modes.

A program is plain Python. It reads and writes files only through `pathlib.Path`,
under the workspace's permission rules (see [Paths](../concepts/paths.md)). There
are no tool kits: the `--tools` flag and the kit forms of `--profile` (a tool name
or a comma-separated tool list) were removed in lackpy stepy100, and give an error.

---

# `lackpy` — inference

## delegate (default)

Generate a program from a natural-language intent and run it immediately.

```bash
lackpy -c "<intent>" [--profile NAME] [--param k=v ...] [--mode MODE]
```

**Arguments**

| Argument | Required | Description |
|----------|----------|-------------|
| `-c "<intent>"` | yes | Natural-language description of the task |
| `--profile` | no | A `[profiles.<name>]` table in `.lackpy/config.toml` (model, mode, rules) |
| `--param` | no | Parameter `key=value` (repeatable). See [Params](#params). |
| `--mode` | no | Inference mode: `1-shot`, `spm` (default: from config) |

**Output:** JSON with `success`, `program`, `generation_tier`, timing fields,
`trace`, `files_read`, `files_modified`, `output`, `stdout`, and `error`. (`output`
falls back to captured `stdout` when the generated program `print()`s its answer
instead of leaving a bare final expression.) `files_read` and `files_modified`
list the paths of the program's successful `Path.read_text`/`read_bytes` and
`Path.write_text`/`write_bytes` calls, as the program wrote them.

**Exit code:** 0 on success, 1 on failure.

**Streams:** the JSON envelope is written to **stdout** whatever the outcome —
including when every inference provider fails to produce a valid program. Parse
stdout and branch on the exit code (or on `success`); you do not need to read
stderr to learn why a run failed.

!!! note "Changed"

    The all-providers-failed envelope previously went to stderr while the success
    envelope went to stdout. A caller reading only stdout recorded such a failure
    as empty output with no error, which is indistinguishable from "the model
    produced nothing" — a different and much rarer thing.

**Examples:**

```bash
lackpy -c "count the lines in README.md"
lackpy -c "list the python files under src" --profile fast
```

---

## generate (`--generate`)

Run the inference pipeline and print the generated program **without executing it**.

```bash
lackpy -c "<intent>" --generate [--profile NAME]
```

**Output:** The program text (not JSON).

**Exit code:** 0 on success; 1 if generation fails.

**Streams:** stdout carries the program source and nothing else. On failure the
error envelope goes to **stderr** and stdout stays empty, so
`lackpy -c "…" --generate > out.py` never leaves JSON in `out.py`. This is the
opposite of the delegate path above, and deliberately: there stdout is already
JSON in every outcome, so one stream can carry all of them; here stdout is an
artifact, and mixing a diagnostic into it corrupts the artifact. Empty stdout
with exit 1 is the unambiguous "no program" signal.

**Example:**

```bash
lackpy -c "list the python files under src" --generate
```

---

## validate (`--validate`)

Validate a program against the lackpy validator without running it. Validate either
an inline code string (with `-c`) or a file (as a positional argument).

```bash
lackpy -c "<program source>" --validate [--profile NAME]
lackpy <file> --validate [--profile NAME]
```

**Output:** JSON with `valid` (bool), `errors` (list), `calls` (list).

**Exit code:** 0 if valid, 1 if invalid.

**Examples:**

```bash
lackpy my_program.py --validate
lackpy -c "from pathlib import Path
len(Path('x.py').read_text())" --validate
```

---

## create (`--create`)

Generate a program from an intent and save it as a reusable **Lackey file**
(a Python class wrapping the program) under `.lackpy/templates/`.

```bash
lackpy -c "<intent>" --create --name NAME [--profile NAME]
```

**Arguments**

| Argument | Required | Description |
|----------|----------|-------------|
| `-c "<intent>"` | yes | Intent to generate the saved program from |
| `--create` | yes | Select create mode |
| `--name` | no | Class name for the Lackey file (default: `Generated`) |
| `--profile` | no | A `[profiles.<name>]` table (model, mode, rules) |

**Output:** `Created <path>` (plain text).

**Exit code:** 0 on success.

**Example:**

```bash
lackpy -c "read the file README.md" --create --name ReadReadme
```

---

## Running a program file

Pass a file path as the first positional argument. A Lackey file (a `Lackey` class
with a `run` method) is detected and run. A plain program file runs as it is.

```bash
lackpy prog.py                      # run a plain program file
lackpy prog.py --param n=5          # ...with a param
lackpy my_lackey.py                 # run a Lackey file
```

**Output:** JSON with `success`, `output`, `stdout`, `error`.

**Exit code:** 0 on success, 1 on failure or validation error.

**`__file__`:** a program file runs with `__file__ == "<stdin>"`, the same value as
a program on stdin. `python3 prog.py` gives the script path. So a program that
builds paths from `__file__` gives a different result in lackpy. Use paths relative
to the workspace instead.

You can also pipe a program on stdin:

```bash
echo "from pathlib import Path
sorted(str(p) for p in Path('.').glob('*.py'))" | lackpy
```

## Params

A program declares its params as one top-level `params = {"name": <literal>, ...}`
dict and reads `params["name"]`. Each `--param key=value` replaces one default.

The CLI types the value text like the default:

- When the default is a `str`, the value stays the raw text: `--param name=5`
  gives `"5"`.
- Otherwise the text is read as a Python literal: `--param n=5` gives `5`,
  `--param flag=True` gives `True`, `--param xs=[1,2]` gives `[1, 2]`.
- Text that is not a literal stays a string: `--param p=src/*.py`.

The runner then checks the type against the default (an `int` is accepted for a
`float` default). A Lackey file's typed attributes (`n: int = 3`) are its params,
with the same rules.

```bash
lackpy count.py --param n=5 --param name=report
```

---

# `lackpyctl` — workspace management

## `lackpyctl init`

Initialize a `.lackpy/` workspace in the current directory.

```bash
lackpyctl init [--ollama-url URL] [--ollama-model MODEL]
```

**Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--ollama-url` | `http://localhost:11434` | Ollama server URL written into config |
| `--ollama-model` | `qwen2.5-coder:1.5b` | Ollama model written into config |

**Creates:**

- `.lackpy/config.toml` — inference order (`templates`, `rules`, `local`), the `local`
  woollama tier (`model = "ollama/<model>"`), sandbox settings
- `.lackpy/templates/` — directory for saved Lackey files / `.tmpl` files
- `.lackpy/kits/` — kept for the internal toolbox views; no program run reads it

The generated config wires the Ollama provider into the inference order, so
compositional intents work out of the box once a model is served. The model choice is
**per-machine** — pick whatever your Ollama host serves best with `--ollama-model`.

If `config.toml` already exists, `init` prints a warning and does nothing.

**Example:**

```bash
cd my-project
lackpyctl init --ollama-url http://localhost:11434 --ollama-model qwen2.5-coder:3b
```

---

## `lackpyctl status`

Show the current workspace configuration.

```bash
lackpyctl status
```

**Output:** JSON with `workspace`, `config_dir`, `inference_order`, `profile_default`,
`sandbox_enabled`, `tools`.

---

## `lackpyctl spec`

Print the language grammar as JSON.

```bash
lackpyctl spec
```

**Output:** JSON with `allowed_nodes`, `forbidden_nodes`, `forbidden_names`,
`allowed_builtins`.

---

## `lackpyctl toolbox`

Inspect the internal tool catalog. No program run uses it: since lackpy stepy100,
programs use `pathlib.Path` and the approved modules, not tools.

```bash
lackpyctl toolbox list
lackpyctl toolbox show <name>
```

**Output:** JSON: a list of `{name, provider, description, grade_w, effects_ceiling}`,
or one tool's object.

---

## `lackpyctl template`

Manage template files.

### `lackpyctl template list`

List all `.tmpl` files in `.lackpy/templates/`.

```bash
lackpyctl template list
```

**Output:** JSON array of `{name, path}`.

### `lackpyctl template test`

Test a template against an intent (not yet implemented).

```bash
lackpyctl template test <name>
```

---

## `lackpyctl mcp`

Manage the MCP server.

### `lackpyctl mcp serve`

Start the lackpy MCP server over stdio transport.

```bash
lackpyctl mcp serve
```

!!! tip "Dedicated launch command for external consumers"
    An MCP client that spawns lackpy as a backend (e.g. a `mcp.json`/`.mcp.json`
    `command`) should use the stable, dedicated entry point — equivalent to
    `lackpyctl mcp serve`, but a single command with no subcommand:

    ```bash
    lackpy-mcp [--workspace DIR]     # canonical, decoupled
    lackpy mcp [--workspace DIR]     # convenience alias on the runner CLI
    ```

    This is the form `lackpyctl mcp init` writes into `.mcp.json`.

### `lackpyctl mcp init`

Add lackpy to `.mcp.json` (as a `lackpy-mcp` server entry).

```bash
lackpyctl mcp init [--name NAME] [--force]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--name` | `lackpy` | Server name in `.mcp.json` |
| `--force` | off | Overwrite an existing entry |
