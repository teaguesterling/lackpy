# CLI Reference

lackpy ships **two** command-line entry points:

| Command | Purpose |
|---------|---------|
| `lackpy` | Run inference: delegate / generate / validate / create from intent, and run program files. |
| `lackpyctl` | Manage the workspace: init, status, kits, toolbox, templates, providers, MCP server. |

```
lackpy    [--workspace PATH] -c "<intent>" [flags]      # or: lackpy <file> [flags]
lackpyctl [--workspace PATH] <command> [args]
```

Both accept `--workspace PATH` to set the project root (default: current directory).

> **Note:** `lackpy` is **flag-based**, not subcommand-based. The mode is selected by
> flags on a single intent (`-c`): the default is *delegate* (generate + run);
> `--generate`, `--validate`, and `--create` select the other modes.

---

# `lackpy` — inference

## delegate (default)

Generate a program from a natural-language intent and run it immediately.

```bash
lackpy -c "<intent>" [--kit KIT] [--tools TOOLS] [--param k=v ...] [--mode MODE]
```

**Arguments**

| Argument | Required | Description |
|----------|----------|-------------|
| `-c "<intent>"` | yes | Natural-language description of the task |
| `--kit` | no | Kit name, comma-separated tool list, or `@file` |
| `--tools` | no | Extra tool names (comma-separated) to add on top of the kit |
| `--param` | no | Parameter `key=value` (repeatable) |
| `--mode` | no | Inference mode: `1-shot`, `spm` (default: from config) |

**Output:** JSON with `success`, `program`, `grade`, `generation_tier`, timing fields,
`trace`, `output`, `stdout`, and `error`. (`output` falls back to captured `stdout`
when the generated program `print()`s its answer instead of leaving a bare final
expression.)

**Exit code:** 0 on success, 1 on failure.

**Examples:**

```bash
lackpy -c "read the file README.md" --kit read_file
lackpy -c "find all Python files" --kit read_file,find_files
```

---

## generate (`--generate`)

Run the inference pipeline and print the generated program **without executing it**.

```bash
lackpy -c "<intent>" --generate [--kit KIT]
```

**Output:** The program text (not JSON).

**Exit code:** 0 on success; 1 if generation fails.

**Example:**

```bash
lackpy -c "find all Python files" --generate --kit find_files
```

---

## validate (`--validate`)

Validate a program against the AST whitelist without running it. Validate either an
inline code string (with `-c`) or a file (as a positional argument).

```bash
lackpy -c "<program source>" --validate [--kit KIT]
lackpy <file> --validate [--kit KIT]
```

**Output:** JSON with `valid` (bool), `errors` (list), `calls` (list).

**Exit code:** 0 if valid, 1 if invalid.

**Examples:**

```bash
lackpy my_program.py --validate --kit read_file
lackpy -c "read_file('x.py')" --validate --kit read_file
```

---

## create (`--create`)

Generate a program from an intent and save it as a reusable **Lackey file**
(a Python class wrapping the program) under `.lackpy/templates/`.

```bash
lackpy -c "<intent>" --create --name NAME [--kit KIT] [--tools TOOLS]
```

**Arguments**

| Argument | Required | Description |
|----------|----------|-------------|
| `-c "<intent>"` | yes | Intent to generate the saved program from |
| `--create` | yes | Select create mode |
| `--name` | no | Class name for the Lackey file (default: `Generated`) |
| `--kit` / `--tools` | no | Tools available to the generated program |

**Output:** `Created <path>` (plain text).

**Exit code:** 0 on success.

**Example:**

```bash
lackpy -c "read the file README.md" --create --name ReadReadme --kit read_file
```

---

## Running a program file

Pass a file path as the first positional argument. Lackey files (a `Lackey` class with
a `run` method) are detected and run directly; plain program files need `--kit` or
`--tools` to supply a namespace.

```bash
lackpy <file.py> --kit KIT          # run a plain program file
lackpy <file.py> --tools read_file  # ...with ad-hoc tools
lackpy my_lackey.py                 # run a Lackey file (tools come from the file)
```

**Output:** JSON with `success`, `output`, `error`.

**Exit code:** 0 on success, 1 on failure or validation error.

You can also pipe a program on stdin:

```bash
echo "find_files('*.py')" | lackpy --kit find_files
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

- `.lackpy/config.toml` — inference order (`templates`, `rules`, `ollama-local`), kit
  default, sandbox settings
- `.lackpy/templates/` — directory for saved Lackey files / `.tmpl` files
- `.lackpy/kits/` — directory for `.kit` files

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

**Output:** JSON with `workspace`, `config_dir`, `inference_order`, `kit_default`,
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

## `lackpyctl kit`

Manage kit files.

### `lackpyctl kit list`

List all `.kit` files in `.lackpy/kits/`.

```bash
lackpyctl kit list
```

**Output:** JSON array of `{name, path}`.

### `lackpyctl kit info`

Show the tools and grade for a kit.

```bash
lackpyctl kit info <name> [--tools TOOL ...]
```

| Argument | Description |
|----------|-------------|
| `name` | Kit name, or comma-separated tool list |
| `--tools` | Explicit list of tool names (overrides `name`) |

**Output:** JSON with `tools`, `grade`, `description`.

### `lackpyctl kit create`

Create a new kit file.

```bash
lackpyctl kit create <name> --tools TOOL [TOOL ...] [--description TEXT]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `name` | yes | Kit name (filename: `{name}.kit`) |
| `--tools` | yes | One or more tool names |
| `--description` | no | Human-readable description |

**Example:**

```bash
lackpyctl kit create readonly --tools read_file find_files --description "Read-only filesystem tools"
```

---

## `lackpyctl toolbox`

Inspect the registered tool catalog.

### `lackpyctl toolbox list`

List all registered tools.

```bash
lackpyctl toolbox list
```

**Output:** JSON array of `{name, provider, description, grade_w, effects_ceiling}`.

### `lackpyctl toolbox show`

Show details for a single tool.

```bash
lackpyctl toolbox show <name>
```

**Output:** JSON object for the tool.

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

### `lackpyctl mcp init`

Add lackpy to `.mcp.json`.

```bash
lackpyctl mcp init [--name NAME] [--force]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--name` | `lackpy` | Server name in `.mcp.json` |
| `--force` | off | Overwrite an existing entry |

---

## Kit argument format

Any command that accepts `--kit` supports these forms:

| Form | Example | Resolved as |
|------|---------|-------------|
| Named kit | `--kit filesystem` | Loads `.lackpy/kits/filesystem.kit` |
| Comma-separated | `--kit read_file,find_files,write_file` | Ad-hoc list of tool names |
| Single tool | `--kit read_file` | Single-tool kit |
| Empty kit | `--kit none` | No base tools (use with `--tools`) |

---

## Extra tools (`--tools`)

Any inference invocation that accepts `--kit` also accepts `--tools` to add individual
tools on top of the kit:

```bash
# Add edit_file to a named kit
lackpy -c "fix the typo" --kit debug --tools edit_file

# Multiple extra tools
lackpy -c "reorganize" --kit debug --tools edit_file,write_file

# Standalone — no kit, just tools
lackpy -c "read the README" --tools read_file

# Explicit empty kit + tools
lackpy -c "read the README" --kit none --tools read_file
```

**Behaviour:**
- Extra tools are merged into the resolved kit. Duplicates are ignored.
- The kit grade is recomputed after merging (e.g., adding `write_file` raises the grade).
- `--tools` without `--kit` uses the config default kit as the base. Use `--kit none` for no base tools.
