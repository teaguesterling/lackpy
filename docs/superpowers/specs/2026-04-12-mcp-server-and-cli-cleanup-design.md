# MCP Server Entry Point + CLI Cleanup

**Date:** 2026-04-12
**Status:** Approved

## Goal

Add a proper MCP server entry point under `lackpyctl mcp`, expose configuration
utility tools so MCP clients can make informed delegate calls, and clean up the
`lackpy` CLI by removing all deprecated subcommands.

## Context

`src/lackpy/mcp/server.py` already exposes the core service methods (delegate,
generate, run_program, validate, kit_info, etc.) as FastMCP tools, but there is
no CLI entry point to launch it. MCP clients also lack tools to discover the
workspace configuration, inference providers, and language spec — information
they need to construct valid delegate calls.

Meanwhile, `lackpy` CLI carries 12 deprecated subcommands that duplicate
`lackpyctl` and clutter the help output.

## Design

### 1. New MCP Tools (`mcp/server.py`)

Three read-only tools added alongside the existing ones:

- **`config()`** — workspace config snapshot: inference order, kit default,
  sandbox settings, config dir path.
- **`provider_list()`** — configured inference providers: name, plugin type,
  model, host, temperature, availability status.
- **`language_spec()`** — allowed AST nodes, allowed builtins, forbidden names.

All delegate to new public methods on `LackpyService`.

### 2. Service Layer Methods (`service.py`)

Three new public methods:

- **`get_config() -> dict`** — serializable config snapshot. Replaces direct
  `_config` access in `ctl.py` status handler.
- **`provider_list() -> list[dict]`** — iterates configured providers, reports
  name/plugin/model/host/temperature/available.
- **`language_spec() -> dict`** — delegates to `lang.spec.get_spec()`.

### 3. MCP CLI Module (`mcp/cli.py`) — new file

Handles `lackpyctl mcp {serve,init}` subcommands.

#### `lackpyctl mcp serve [--workspace PATH]`

- Sets the workspace on the server module (e.g., `server.set_workspace(path)`)
  so `_get_service()` uses it when lazily constructing `LackpyService`.
- Imports the FastMCP `mcp` instance from `mcp/server.py`.
- Calls `mcp.run(transport="stdio")`.

#### `lackpyctl mcp init [--workspace PATH] [--name NAME] [--force]`

- Reads existing `.mcp.json` in the workspace root, or starts with
  `{"mcpServers": {}}` if the file doesn't exist.
- Adds the `lackpy` entry (or `--name` override) under `mcpServers`:

```json
{
  "mcpServers": {
    "lackpy": {
      "command": "lackpyctl",
      "args": ["mcp", "serve", "--workspace", "/absolute/path"]
    }
  }
}
```

- Preserves all other entries in the file.
- If the key already exists and `--force` is not passed: prints the existing
  config and exits non-zero with a hint to use `--force`.
- With `--force`: overwrites the existing entry.

### 4. `lackpyctl` Changes (`ctl.py`)

- Add `mcp` subcommand with `serve` and `init` sub-subcommands.
- Delegate to `mcp.cli` module for handling.
- Switch `status` handler to use `svc.get_config()` instead of `svc._config`.

### 5. `lackpy` CLI Cleanup (`cli.py`)

**Remove entirely:**
- All subcommand definitions from `build_parser()`: `delegate`, `generate`,
  `run`, `create`, `validate`, `init`, `status`, `spec`, `kit`, `toolbox`,
  `template`.
- All subcommand handler code in `main()`.
- `_init_config()` function (already in `ctl.py`).
- `_CTL_COMMANDS` set.

**Keep:**
- File execution: `lackpy script.py --kit ... --param ...`
- Intent mode: `lackpy -c "intent" --kit ... [--generate] [--create] [--validate] [--mode ...]`
- Stdin piping: `echo 'program' | lackpy --kit ...`
- Helper functions: `_file_entrypoint()`, `_parse_kit()`, `_parse_params()`,
  `_run_file()`.

**Add:**
- Epilog on help text:
  ```
  Configuration & management:
    Use lackpyctl for workspace init, kit/toolbox/template management,
    and MCP server. Run 'lackpyctl --help' for details.
  ```

## Files Changed

| File | Action |
|------|--------|
| `src/lackpy/mcp/cli.py` | **New** — MCP subcommand handling + `.mcp.json` merge |
| `src/lackpy/mcp/server.py` | **Modified** — add config, provider_list, language_spec tools |
| `src/lackpy/service.py` | **Modified** — add get_config, provider_list, language_spec methods |
| `src/lackpy/ctl.py` | **Modified** — wire mcp subcommand, use get_config() in status |
| `src/lackpy/cli.py` | **Modified** — remove deprecated subcommands, add lackpyctl epilog |

## Files Not Changed

- `pyproject.toml` — `mcp` extra already has `mcp[cli]`, entry points stay.
- Existing MCP tools in `server.py` — untouched.

## Testing

- `lackpyctl mcp serve` launches and responds to MCP protocol on stdio.
- `lackpyctl mcp init` creates `.mcp.json` when absent, merges when present,
  refuses overwrite without `--force`.
- New MCP tools (`config`, `provider_list`, `language_spec`) return expected
  data shapes.
- `lackpy --help` shows clean output with `lackpyctl` hint, no deprecated
  subcommands.
- `lackpy -c`, `lackpy script.py`, and stdin piping still work.
