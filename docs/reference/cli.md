# CLI Reference

```
lackpy [--workspace PATH] <command> [args]
```

All commands accept `--workspace PATH` to set the project root (default: current directory).

---

## `lackpy init`

Initialize a `.lackpy/` workspace in the current directory.

```bash
lackpy init [--ollama-model MODEL]
```

**Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--ollama-model` | `qwen2.5-coder:1.5b` | Default Ollama model to write into config |

**Creates:**

- `.lackpy/config.toml` — inference order, kit defaults, sandbox settings
- `.lackpy/templates/` — directory for `.tmpl` files
- `.lackpy/kits/` — directory for `.kit` files

If `config.toml` already exists, `init` prints a warning and does nothing.

**Example:**

```bash
cd my-project
lackpy init --ollama-model codellama:7b
```

---

## `lackpy delegate`

Generate a program from natural language intent and run it immediately.

```bash
lackpy delegate <intent> [--kit KIT] [--sandbox PROFILE]
```

**Arguments**

| Argument | Required | Description |
|----------|----------|-------------|
| `intent` | yes | Natural language description of the task |
| `--kit` | no | Kit name, comma-separated tool list, or `@file` |
| `--sandbox` | no | Sandbox profile name (v2, not yet active) |

**Output:** JSON with `success`, `program`, `grade`, `generation_tier`, timing fields, `trace`, `output`, and `error`.

**Exit code:** 0 on success, 1 on failure.

**Examples:**

```bash
lackpy delegate "read the file README.md" --kit read
lackpy delegate "find all Python files" --kit read,glob
```

---

## `lackpy generate`

Run the inference pipeline and print the generated program, without executing it.

```bash
lackpy generate <intent> [--kit KIT]
```

**Arguments**

| Argument | Required | Description |
|----------|----------|-------------|
| `intent` | yes | Natural language description |
| `--kit` | no | Kit name or comma-separated tool list |

**Output:** The program text (not JSON).

**Exit code:** Always 0.

**Example:**

```bash
lackpy generate "find all Python files" --kit glob
```

---

## `lackpy run`

Validate and run a program from a file.

```bash
lackpy run <file> [--kit KIT]
```

**Arguments**

| Argument | Required | Description |
|----------|----------|-------------|
| `file` | yes | Path to the program file |
| `--kit` | no | Kit name or comma-separated tool list |

**Output:** JSON with `success`, `output`, `error`.

**Exit code:** 0 on success, 1 on failure or validation error.

**Example:**

```bash
lackpy run my_program.py --kit read,glob
```

---

## `lackpy validate`

Validate a program file without running it.

```bash
lackpy validate <file> [--kit KIT]
```

**Arguments**

| Argument | Required | Description |
|----------|----------|-------------|
| `file` | yes | Path to the program file |
| `--kit` | no | Kit name or comma-separated tool list |

**Output:** JSON with `valid` (bool), `errors` (list), `calls` (list).

**Exit code:** 0 if valid, 1 if invalid.

**Example:**

```bash
lackpy validate my_program.py --kit read
```

---

## `lackpy create`

Validate a program and save it as a template.

```bash
lackpy create <file> --name NAME [--kit KIT] [--pattern PATTERN]
```

**Arguments**

| Argument | Required | Description |
|----------|----------|-------------|
| `file` | yes | Path to the program file |
| `--name` | yes | Template name (used as filename: `{name}.tmpl`) |
| `--kit` | no | Kit name or comma-separated tool list |
| `--pattern` | no | Intent pattern with `{placeholder}` variables |

**Output:** JSON with `success`, `path` (or `errors`).

**Exit code:** 0 on success, 1 on validation failure.

**Example:**

```bash
lackpy create read_file.py --name read-file --pattern "read the file {path}" --kit read
```

---

## `lackpy spec`

Print the language grammar as JSON.

```bash
lackpy spec
```

**Output:** JSON with `allowed_nodes`, `forbidden_nodes`, `forbidden_names`, `allowed_builtins`.

**Exit code:** Always 0.

---

## `lackpy status`

Show the current workspace configuration.

```bash
lackpy status
```

**Output:** JSON with `workspace`, `config_dir`, `inference_order`, `kit_default`, `sandbox_enabled`, `tools`.

**Exit code:** Always 0.

---

## `lackpy kit`

Manage kit files.

### `lackpy kit list`

List all `.kit` files in `.lackpy/kits/`.

```bash
lackpy kit list
```

**Output:** JSON array of `{name, path}`.

### `lackpy kit info`

Show the tools and grade for a kit.

```bash
lackpy kit info <name> [--tools TOOL ...]
```

**Arguments**

| Argument | Description |
|----------|-------------|
| `name` | Kit name, or comma-separated tool list |
| `--tools` | Explicit list of tool names (overrides `name`) |

**Output:** JSON with `tools`, `grade`, `description`.

### `lackpy kit create`

Create a new kit file.

```bash
lackpy kit create <name> --tools TOOL [TOOL ...] [--description TEXT]
```

**Arguments**

| Argument | Required | Description |
|----------|----------|-------------|
| `name` | yes | Kit name (filename: `{name}.kit`) |
| `--tools` | yes | One or more tool names |
| `--description` | no | Human-readable description |

**Output:** JSON with `name`, `path`, `tools`.

**Example:**

```bash
lackpy kit create readonly --tools read glob --description "Read-only filesystem tools"
```

---

## `lackpy toolbox`

Inspect the registered tool catalog.

### `lackpy toolbox list`

List all registered tools.

```bash
lackpy toolbox list
```

**Output:** JSON array of `{name, provider, description, grade_w, effects_ceiling}`.

### `lackpy toolbox show`

Show details for a single tool.

```bash
lackpy toolbox show <name>
```

**Arguments**

| Argument | Description |
|----------|-------------|
| `name` | Tool name |

**Output:** JSON object for the tool.

---

## `lackpy template`

Manage template files.

### `lackpy template list`

List all `.tmpl` files in `.lackpy/templates/`.

```bash
lackpy template list
```

**Output:** JSON array of `{name, path}`.

### `lackpy template test`

Test a template against an intent (not yet implemented).

```bash
lackpy template test <name>
```

---

## Kit argument format

Any command that accepts `--kit` supports three forms:

| Form | Example | Resolved as |
|------|---------|-------------|
| Named kit | `--kit filesystem` | Loads `.lackpy/kits/filesystem.kit` |
| Comma-separated | `--kit read,glob,write` | Ad-hoc list of tool names |
| Single tool | `--kit read` | Single-tool kit |
