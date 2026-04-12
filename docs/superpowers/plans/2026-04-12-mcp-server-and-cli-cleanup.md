# MCP Server Entry Point + CLI Cleanup — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire up a `lackpyctl mcp serve` entry point for the existing FastMCP server, add config/provider/spec utility tools, add `lackpyctl mcp init` for `.mcp.json` generation, and clean up the `lackpy` CLI by removing all deprecated subcommands.

**Architecture:** The existing `mcp/server.py` module (FastMCP with 9 tools) stays as-is. We add 3 new MCP tools that delegate to new `LackpyService` public methods. A new `mcp/cli.py` handles the `lackpyctl mcp {serve,init}` subcommands. `ctl.py` gets a thin `mcp` dispatcher. `cli.py` gets gutted of all subcommands.

**Tech Stack:** Python 3.10+, FastMCP (`mcp[cli]`), argparse, json, pathlib

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/lackpy/lang/spec.py` | Modify | Add `get_spec() -> dict` returning structured spec data |
| `src/lackpy/service.py` | Modify | Add `get_config()`, `provider_list()`, `language_spec()` |
| `src/lackpy/mcp/server.py` | Modify | Add `config()`, `provider_list()`, `language_spec()` MCP tools; add `set_workspace()` |
| `src/lackpy/mcp/cli.py` | Create | `lackpyctl mcp {serve,init}` subcommand logic |
| `src/lackpy/ctl.py` | Modify | Wire `mcp` subcommand, use `get_config()` in status |
| `src/lackpy/cli.py` | Modify | Remove deprecated subcommands, add `lackpyctl` epilog |
| `tests/test_service.py` | Modify | Add tests for new service methods |
| `tests/test_mcp_tools.py` | Create | Tests for new MCP tools |
| `tests/test_mcp_cli.py` | Create | Tests for `mcp init` logic |
| `tests/test_ctl.py` | Modify | Update parser tests for `mcp` subcommand |
| `tests/test_cli.py` | Modify | Replace deprecated subcommand tests with new parser tests |

---

### Task 1: Add `get_spec()` to `lang/spec.py`

The existing `format_spec()` returns a human-readable string. MCP tools need structured data. Also, both `cli.py` and `ctl.py` reference `get_spec()` which doesn't exist yet — this fixes that pre-existing issue.

**Files:**
- Modify: `src/lackpy/lang/spec.py`
- Test: `tests/lang/test_spec.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/lang/test_spec.py`:

```python
"""Tests for the language spec module."""

from lackpy.lang.spec import get_spec


def test_get_spec_returns_dict():
    spec = get_spec()
    assert isinstance(spec, dict)


def test_get_spec_has_required_keys():
    spec = get_spec()
    assert "allowed_nodes" in spec
    assert "forbidden_nodes" in spec
    assert "forbidden_names" in spec
    assert "allowed_builtins" in spec


def test_get_spec_values_are_lists():
    spec = get_spec()
    assert isinstance(spec["allowed_nodes"], list)
    assert isinstance(spec["forbidden_nodes"], list)
    assert isinstance(spec["forbidden_names"], list)
    assert isinstance(spec["allowed_builtins"], list)


def test_get_spec_contains_known_entries():
    spec = get_spec()
    assert "Module" in spec["allowed_nodes"]
    assert "Import" in spec["forbidden_nodes"]
    assert "__import__" in spec["forbidden_names"]
    assert "len" in spec["allowed_builtins"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/lang/test_spec.py -v`
Expected: ImportError — `get_spec` not defined in `spec.py`

- [ ] **Step 3: Implement `get_spec()`**

Add to `src/lackpy/lang/spec.py` after the existing `format_spec()`:

```python
def get_spec() -> dict:
    """Return the language spec as structured data."""
    return {
        "allowed_nodes": sorted(n.__name__ for n in ALLOWED_NODES),
        "forbidden_nodes": sorted(n.__name__ for n in FORBIDDEN_NODES),
        "forbidden_names": sorted(FORBIDDEN_NAMES),
        "allowed_builtins": sorted(ALLOWED_BUILTINS),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/lang/test_spec.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/lackpy/lang/spec.py tests/lang/test_spec.py
git commit -m "feat: add get_spec() returning structured language spec data"
```

---

### Task 2: Add service layer methods

Add `get_config()`, `provider_list()`, and `language_spec()` to `LackpyService`.

**Files:**
- Modify: `src/lackpy/service.py`
- Modify: `tests/test_service.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_service.py`:

```python
class TestGetConfig:
    def test_returns_dict(self, service):
        config = service.get_config()
        assert isinstance(config, dict)

    def test_has_required_keys(self, service):
        config = service.get_config()
        assert "inference_order" in config
        assert "kit_default" in config
        assert "sandbox_enabled" in config
        assert "config_dir" in config

    def test_config_dir_is_string(self, service):
        config = service.get_config()
        assert isinstance(config["config_dir"], str)


class TestProviderList:
    def test_returns_list(self, service):
        providers = service.provider_list()
        assert isinstance(providers, list)

    def test_providers_have_required_keys(self, service):
        providers = service.provider_list()
        # At minimum templates and rules are always present
        assert len(providers) >= 2
        for p in providers:
            assert "name" in p
            assert "plugin" in p
            assert "available" in p

    def test_templates_provider_present(self, service):
        providers = service.provider_list()
        names = [p["name"] for p in providers]
        assert "templates" in names

    def test_rules_provider_present(self, service):
        providers = service.provider_list()
        names = [p["name"] for p in providers]
        assert "rules" in names


class TestLanguageSpec:
    def test_returns_dict(self, service):
        spec = service.language_spec()
        assert isinstance(spec, dict)

    def test_has_spec_keys(self, service):
        spec = service.language_spec()
        assert "allowed_nodes" in spec
        assert "allowed_builtins" in spec
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_service.py::TestGetConfig tests/test_service.py::TestProviderList tests/test_service.py::TestLanguageSpec -v`
Expected: AttributeError — methods don't exist yet

- [ ] **Step 3: Implement the three methods**

Add to `LackpyService` class in `src/lackpy/service.py`, after `toolbox_list()` (around line 618):

```python
    def get_config(self) -> dict[str, Any]:
        """Return a serializable snapshot of the workspace configuration."""
        return {
            "workspace": str(self._workspace),
            "config_dir": str(self._config.config_dir),
            "inference_order": self._config.inference_order,
            "inference_mode": self._config.inference_mode,
            "kit_default": self._config.kit_default,
            "sandbox_enabled": self._config.sandbox_enabled,
            "sandbox_timeout": self._config.sandbox_timeout,
            "sandbox_memory_mb": self._config.sandbox_memory_mb,
            "tools": len(self.toolbox.tools),
        }

    def provider_list(self) -> list[dict[str, Any]]:
        """List configured inference providers with availability status."""
        result = []
        for provider in self._inference_providers:
            entry: dict[str, Any] = {
                "name": provider.name,
                "available": provider.available(),
            }
            # Add config details for providers that have them
            provider_cfg = self._config.inference_providers.get(provider.name, {})
            entry["plugin"] = provider_cfg.get("plugin", provider.name)
            if "model" in provider_cfg:
                entry["model"] = provider_cfg["model"]
            if "host" in provider_cfg:
                entry["host"] = provider_cfg["host"]
            if "temperature" in provider_cfg:
                entry["temperature"] = provider_cfg["temperature"]
            result.append(entry)
        return result

    def language_spec(self) -> dict[str, Any]:
        """Return the lackpy language specification as structured data."""
        from .lang.spec import get_spec
        return get_spec()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_service.py -v`
Expected: All tests PASS (existing + new)

- [ ] **Step 5: Commit**

```bash
git add src/lackpy/service.py tests/test_service.py
git commit -m "feat: add get_config, provider_list, language_spec to LackpyService"
```

---

### Task 3: Add MCP tools and `set_workspace()`

Add the three new tools to `mcp/server.py` and a `set_workspace()` function so the CLI can configure the workspace before the service is lazily created.

**Files:**
- Modify: `src/lackpy/mcp/server.py`
- Create: `tests/test_mcp_tools.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mcp_tools.py`:

```python
"""Tests for MCP server tools (config, provider_list, language_spec)."""

import pytest
from pathlib import Path

from lackpy.mcp import server


@pytest.fixture(autouse=True)
def reset_service():
    """Reset the module-level service singleton between tests."""
    server._service = None
    server._workspace = None
    yield
    server._service = None
    server._workspace = None


@pytest.fixture
def workspace(tmp_path):
    server.set_workspace(tmp_path)
    return tmp_path


def test_set_workspace(tmp_path):
    server.set_workspace(tmp_path)
    assert server._workspace == tmp_path


def test_get_service_uses_workspace(workspace):
    svc = server._get_service()
    assert svc._workspace == workspace


def test_config_tool(workspace):
    result = server.config()
    assert isinstance(result, dict)
    assert "inference_order" in result
    assert "config_dir" in result


def test_provider_list_tool(workspace):
    result = server.provider_list()
    assert isinstance(result, list)
    assert len(result) >= 2  # templates + rules at minimum


def test_language_spec_tool(workspace):
    result = server.language_spec()
    assert isinstance(result, dict)
    assert "allowed_nodes" in result
    assert "allowed_builtins" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_mcp_tools.py -v`
Expected: ImportError/AttributeError — new functions don't exist yet

- [ ] **Step 3: Implement changes to `mcp/server.py`**

Replace `src/lackpy/mcp/server.py` with:

```python
"""FastMCP server: thin adapter over the service layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from ..service import LackpyService

mcp = FastMCP("lackpy")
_service: LackpyService | None = None
_workspace: Path | None = None


def set_workspace(workspace: Path) -> None:
    """Set workspace before first tool call. Used by the CLI entry point."""
    global _workspace, _service
    _workspace = workspace
    _service = None  # reset so next call picks up new workspace


def _get_service() -> LackpyService:
    global _service
    if _service is None:
        _service = LackpyService(workspace=_workspace)
    return _service


# --- Core tools (existing) ---

@mcp.tool()
async def delegate(intent: str, kit: str | list[str] | dict | None = None,
                   params: dict[str, Any] | None = None, sandbox: str | None = None,
                   rules: list[str] | None = None) -> dict:
    """Generate and run a tool-composition program from natural language intent."""
    return await _get_service().delegate(intent, kit=kit, params=params, sandbox=sandbox, rules=rules)


@mcp.tool()
async def generate(intent: str, kit: str | list[str] | dict | None = None,
                   params: dict[str, Any] | None = None, rules: list[str] | None = None) -> dict:
    """Generate a program from intent without executing it."""
    result = await _get_service().generate(intent, kit=kit, params=params, rules=rules)
    return {"program": result.program, "provider": result.provider_name, "generation_time_ms": result.generation_time_ms}


@mcp.tool()
async def run_program(program: str, kit: str | list[str] | dict | None = None,
                      params: dict[str, Any] | None = None, sandbox: str | None = None,
                      rules: list[str] | None = None) -> dict:
    """Validate and run a provided program."""
    result = await _get_service().run_program(program, kit=kit, params=params, sandbox=sandbox, rules=rules)
    return {"success": result.success, "output": result.output, "error": result.error,
            "trace": [{"step": e.step, "tool": e.tool, "args": e.args, "result": e.result,
                        "success": e.success, "error": e.error} for e in result.trace.entries]}


@mcp.tool()
async def create(program: str, kit: str | list[str] | dict | None = None,
                 name: str = "", pattern: str | None = None) -> dict:
    """Validate and save a program as a reusable template."""
    return await _get_service().create(program, kit=kit, name=name, pattern=pattern)


@mcp.tool()
def validate(program: str, kit: str | list[str] | dict | None = None,
             rules: list[str] | None = None) -> dict:
    """Validate a program without running it."""
    result = _get_service().validate(program, kit=kit, rules=rules)
    return {"valid": result.valid, "errors": result.errors, "calls": result.calls, "variables": result.variables}


# --- Kit & toolbox tools (existing) ---

@mcp.tool()
def kit_info(kit: str | list[str] | dict) -> dict:
    """Get info about a kit: tools, grades, descriptions."""
    return _get_service().kit_info(kit)


@mcp.tool()
def kit_list() -> list[dict]:
    """List available predefined kits."""
    return _get_service().kit_list()


@mcp.tool()
def kit_create(name: str, tools: list[str], description: str | None = None) -> dict:
    """Create a new predefined kit."""
    return _get_service().kit_create(name, tools, description)


@mcp.tool()
def toolbox_list() -> list[dict]:
    """List all registered tools with metadata."""
    return _get_service().toolbox_list()


# --- Configuration & introspection tools (new) ---

@mcp.tool()
def config() -> dict:
    """Return the current workspace configuration: inference order, providers, kit default, sandbox settings."""
    return _get_service().get_config()


@mcp.tool()
def provider_list() -> list[dict]:
    """List configured inference providers with plugin type, model, and availability status."""
    return _get_service().provider_list()


@mcp.tool()
def language_spec() -> dict:
    """Return the lackpy language specification: allowed AST nodes, builtins, and forbidden names."""
    return _get_service().language_spec()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_mcp_tools.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/lackpy/mcp/server.py tests/test_mcp_tools.py
git commit -m "feat: add config, provider_list, language_spec MCP tools + set_workspace"
```

---

### Task 4: Create `mcp/cli.py`

New module handling `lackpyctl mcp {serve,init}` — MCP server launch and `.mcp.json` management.

**Files:**
- Create: `src/lackpy/mcp/cli.py`
- Create: `tests/test_mcp_cli.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mcp_cli.py`:

```python
"""Tests for lackpyctl mcp CLI (init subcommand)."""

import json
import pytest
from pathlib import Path

from lackpy.mcp.cli import mcp_init


class TestMcpInit:
    def test_creates_mcp_json_when_absent(self, tmp_path):
        result = mcp_init(workspace=tmp_path)
        assert result == 0
        mcp_file = tmp_path / ".mcp.json"
        assert mcp_file.exists()
        data = json.loads(mcp_file.read_text())
        assert "lackpy" in data["mcpServers"]
        assert data["mcpServers"]["lackpy"]["command"] == "lackpyctl"

    def test_includes_workspace_in_args(self, tmp_path):
        mcp_init(workspace=tmp_path)
        data = json.loads((tmp_path / ".mcp.json").read_text())
        args = data["mcpServers"]["lackpy"]["args"]
        assert "--workspace" in args
        assert str(tmp_path) in args

    def test_preserves_existing_entries(self, tmp_path):
        mcp_file = tmp_path / ".mcp.json"
        mcp_file.write_text(json.dumps({
            "mcpServers": {
                "other-tool": {"command": "other", "args": []}
            }
        }))
        mcp_init(workspace=tmp_path)
        data = json.loads(mcp_file.read_text())
        assert "other-tool" in data["mcpServers"]
        assert "lackpy" in data["mcpServers"]

    def test_refuses_overwrite_without_force(self, tmp_path):
        mcp_file = tmp_path / ".mcp.json"
        mcp_file.write_text(json.dumps({
            "mcpServers": {
                "lackpy": {"command": "old", "args": []}
            }
        }))
        result = mcp_init(workspace=tmp_path, force=False)
        assert result == 1
        # Original entry unchanged
        data = json.loads(mcp_file.read_text())
        assert data["mcpServers"]["lackpy"]["command"] == "old"

    def test_overwrites_with_force(self, tmp_path):
        mcp_file = tmp_path / ".mcp.json"
        mcp_file.write_text(json.dumps({
            "mcpServers": {
                "lackpy": {"command": "old", "args": []}
            }
        }))
        result = mcp_init(workspace=tmp_path, force=True)
        assert result == 0
        data = json.loads(mcp_file.read_text())
        assert data["mcpServers"]["lackpy"]["command"] == "lackpyctl"

    def test_custom_name(self, tmp_path):
        mcp_init(workspace=tmp_path, name="my-lackpy")
        data = json.loads((tmp_path / ".mcp.json").read_text())
        assert "my-lackpy" in data["mcpServers"]
        assert "lackpy" not in data["mcpServers"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_mcp_cli.py -v`
Expected: ImportError — module doesn't exist yet

- [ ] **Step 3: Implement `mcp/cli.py`**

Create `src/lackpy/mcp/cli.py`:

```python
"""CLI handlers for `lackpyctl mcp {serve,init}`."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def mcp_init(
    workspace: Path,
    name: str = "lackpy",
    force: bool = False,
) -> int:
    """Add a lackpy entry to .mcp.json in the workspace.

    Returns 0 on success, 1 if the entry already exists (without --force).
    """
    mcp_file = workspace / ".mcp.json"
    if mcp_file.exists():
        data = json.loads(mcp_file.read_text())
    else:
        data = {"mcpServers": {}}

    servers = data.setdefault("mcpServers", {})

    if name in servers and not force:
        print(
            f"lackpyctl: '{name}' already exists in {mcp_file}:\n"
            f"  {json.dumps(servers[name], indent=2)}\n"
            f"Use --force to overwrite.",
            file=sys.stderr,
        )
        return 1

    servers[name] = {
        "command": "lackpyctl",
        "args": ["mcp", "serve", "--workspace", str(workspace.resolve())],
    }

    mcp_file.write_text(json.dumps(data, indent=2) + "\n")
    print(f"{'Updated' if force else 'Added'} '{name}' in {mcp_file}")
    return 0


def mcp_serve(workspace: Path) -> int:
    """Start the MCP server on stdio transport."""
    from .server import mcp, set_workspace

    set_workspace(workspace)
    mcp.run(transport="stdio")
    return 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_mcp_cli.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/lackpy/mcp/cli.py tests/test_mcp_cli.py
git commit -m "feat: add mcp/cli.py with serve and init subcommand handlers"
```

---

### Task 5: Wire `mcp` subcommand into `ctl.py`

Add the `mcp` subcommand to `lackpyctl` parser and dispatch to `mcp/cli.py`. Also switch `status` to use `get_config()`.

**Files:**
- Modify: `src/lackpy/ctl.py`
- Modify: `tests/test_ctl.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ctl.py`:

```python
def test_parser_mcp_serve():
    parser = build_parser()
    args = parser.parse_args(["mcp", "serve"])
    assert args.command == "mcp"
    assert args.mcp_command == "serve"


def test_parser_mcp_init():
    parser = build_parser()
    args = parser.parse_args(["mcp", "init"])
    assert args.command == "mcp"
    assert args.mcp_command == "init"


def test_parser_mcp_init_force():
    parser = build_parser()
    args = parser.parse_args(["mcp", "init", "--force"])
    assert args.force is True


def test_parser_mcp_init_name():
    parser = build_parser()
    args = parser.parse_args(["mcp", "init", "--name", "my-lackpy"])
    assert args.name == "my-lackpy"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ctl.py::test_parser_mcp_serve tests/test_ctl.py::test_parser_mcp_init -v`
Expected: SystemExit or error — `mcp` subcommand not in parser yet

- [ ] **Step 3: Add `mcp` subcommand to `build_parser()` in `ctl.py`**

In `src/lackpy/ctl.py`, add the `mcp` subcommand in `build_parser()` after the `provider` block (before `return parser`):

```python
    # mcp
    mcp_p = subparsers.add_parser("mcp", help="MCP server management")
    mcp_sub = mcp_p.add_subparsers(dest="mcp_command")

    mcp_sub.add_parser("serve", help="Start the MCP server (stdio transport)")

    mcp_init_p = mcp_sub.add_parser("init", help="Add lackpy to .mcp.json")
    mcp_init_p.add_argument("--name", default="lackpy", help="Server name in .mcp.json (default: lackpy)")
    mcp_init_p.add_argument("--force", action="store_true", default=False, help="Overwrite existing entry")
```

- [ ] **Step 4: Add `mcp` handler to `main()` in `ctl.py`**

Add before the `provider` handler block in `main()`. The `mcp` commands don't need `LackpyService`, so they go before the `svc = LackpyService(...)` line:

```python
    if args.command == "mcp":
        from .mcp.cli import mcp_init, mcp_serve
        if args.mcp_command == "serve":
            return mcp_serve(workspace)
        elif args.mcp_command == "init":
            return mcp_init(
                workspace=workspace,
                name=args.name,
                force=args.force,
            )
        else:
            print("Usage: lackpyctl mcp {serve|init}", file=sys.stderr)
            return 1
```

Insert this after the `spec` handler and before the `from .service import LackpyService` line, since `mcp` commands don't need the full service.

- [ ] **Step 5: Switch `status` handler to use `get_config()`**

Replace the `status` block in `main()`:

```python
    if args.command == "status":
        config = svc.get_config()
        print(json.dumps(config, indent=2))
        return 0
```

- [ ] **Step 6: Run all ctl tests**

Run: `python -m pytest tests/test_ctl.py -v`
Expected: All tests PASS (existing + new)

- [ ] **Step 7: Commit**

```bash
git add src/lackpy/ctl.py tests/test_ctl.py
git commit -m "feat: wire lackpyctl mcp {serve,init} subcommands"
```

---

### Task 6: Clean up `lackpy` CLI

Remove all deprecated subcommands and add `lackpyctl` epilog.

**Files:**
- Modify: `src/lackpy/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Update tests first — replace deprecated tests with new tests**

Replace the contents of `tests/test_cli.py`:

```python
"""Tests for the lackpy CLI."""

from lackpy.cli import build_parser


class TestParserFlags:
    """Test the flag-based interface (the kept functionality)."""

    def test_intent_flag(self):
        parser = build_parser()
        args = parser.parse_args(["-c", "read file main.py"])
        assert args.intent == "read file main.py"

    def test_create_flag(self):
        parser = build_parser()
        args = parser.parse_args(["-c", "find python files", "--create", "--name", "FindPy", "--kit", "read,glob"])
        assert args.intent == "find python files"
        assert args.create is True
        assert args.name == "FindPy"

    def test_generate_flag(self):
        parser = build_parser()
        args = parser.parse_args(["-c", "read file main.py", "--generate", "--kit", "read_file"])
        assert args.intent == "read file main.py"
        assert args.generate is True

    def test_param_flag(self):
        parser = build_parser()
        args = parser.parse_args(["-c", "read file test.txt", "--param", "x=1", "--param", "y=2"])
        assert args.param == ["x=1", "y=2"]

    def test_validate_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--validate", "-c", "x = 1"])
        assert args.validate is True
        assert args.intent == "x = 1"

    def test_mode_flag(self):
        parser = build_parser()
        args = parser.parse_args(["-c", "read file", "--mode", "spm"])
        assert args.mode == "spm"

    def test_workspace_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--workspace", "/tmp/test", "-c", "hello"])
        assert str(args.workspace) == "/tmp/test"


class TestNoDeprecatedSubcommands:
    """Verify all deprecated subcommands are removed."""

    def test_no_delegate_subcommand(self):
        parser = build_parser()
        # parse_args with an unknown subcommand should not set command="delegate"
        args = parser.parse_args([])
        assert not hasattr(args, "command") or args.command is None

    def test_help_mentions_lackpyctl(self, capsys):
        parser = build_parser()
        help_text = parser.format_help()
        assert "lackpyctl" in help_text
```

- [ ] **Step 2: Run old tests to see which fail (baseline)**

Run: `python -m pytest tests/test_cli.py -v`
Expected: Some old tests fail because they test deprecated subcommands. Note which ones.

- [ ] **Step 3: Strip `cli.py` of deprecated subcommands**

Replace `src/lackpy/cli.py` with the cleaned-up version. The key changes:

1. Remove `_CTL_COMMANDS`
2. Remove `_init_config()` 
3. Remove all subparsers from `build_parser()` — no `subparsers` at all
4. Remove all subcommand handlers from `main()` (everything after `args.command` checks)
5. Add epilog to parser

The cleaned `build_parser()`:

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lackpy",
        description="lackpy — run Lackey files and delegate natural-language programs",
        epilog=(
            "Configuration & management:\n"
            "  Use lackpyctl for workspace init, kit/toolbox/template management,\n"
            "  and MCP server. Run 'lackpyctl --help' for details."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--workspace", type=Path, default=None,
        help="Workspace directory (default: cwd)",
    )
    parser.add_argument("-c", dest="intent", default=None, help="One-shot intent")
    parser.add_argument("--create", action="store_true", default=False, help="Save as Lackey file")
    parser.add_argument("--generate", action="store_true", default=False, help="Generate without running")
    parser.add_argument("--name", default=None, help="Class name for --create")
    parser.add_argument("--kit", default=None, help="Kit name, comma-separated list, or @file")
    parser.add_argument("--param", action="append", default=None, help="Parameter: key=value (repeatable)")
    parser.add_argument("--validate", action="store_true", default=False, help="Validate without running")
    parser.add_argument("--mode", default=None, help="Inference mode: 1-shot, spm (default: from config or legacy)")
    return parser
```

The cleaned `main()` — remove everything from `if args.command is None and not sys.stdin.isatty()` onward (the subcommand dispatching), keeping only stdin handling and the "no args" help case:

```python
def main(argv: list[str] | None = None) -> int:
    raw_args = argv if argv is not None else sys.argv[1:]
    first_positional = next(
        (a for a in raw_args if not a.startswith("-")), None
    )
    if first_positional and (first_positional.endswith(".py") or "/" in first_positional or first_positional.startswith(".")):
        return _file_entrypoint(raw_args)

    parser = build_parser()
    args = parser.parse_args(argv)

    workspace = args.workspace or Path.cwd()

    # --validate + -c → validate the code string
    if args.validate and args.intent:
        from .service import LackpyService
        svc = LackpyService(workspace=workspace)
        kit = _parse_kit(args.kit) if args.kit else None
        result = svc.validate(args.intent, kit=kit)
        out: dict[str, Any] = {"valid": result.valid, "errors": result.errors, "calls": list(result.calls)}
        print(json.dumps(out, indent=2))
        return 0 if result.valid else 1

    # Runner-style interface (-c flag)
    if args.intent:
        from .service import LackpyService
        svc = LackpyService(workspace=workspace)
        kit = _parse_kit(args.kit) if args.kit else None
        mode = getattr(args, 'mode', None)

        if args.create:
            gen = asyncio.run(svc.generate(args.intent, kit=kit, mode=mode))
            tools = kit if isinstance(kit, list) else []
            path = asyncio.run(svc.create_lackey(
                program=gen.program, name=args.name or "Generated",
                tools=tools,
                creation_log=[
                    {"role": "user", "content": args.intent},
                    {"role": "assistant", "content": gen.program, "accepted": True},
                ],
            ))
            print(f"Created {path}")
            return 0

        if args.generate:
            try:
                gen = asyncio.run(svc.generate(args.intent, kit=kit, mode=mode))
            except RuntimeError as e:
                print(json.dumps({"success": False, "error": str(e)}, indent=2), file=sys.stderr)
                return 1
            print(gen.program)
            return 0

        # Default: delegate (generate + run)
        try:
            result = asyncio.run(svc.delegate(args.intent, kit=kit, mode=mode))
        except RuntimeError as e:
            print(json.dumps({"success": False, "error": str(e)}, indent=2), file=sys.stderr)
            return 1
        print(json.dumps(result, indent=2, default=str))
        return 0 if result["success"] else 1

    # Stdin: read program and run if stdin is not a tty
    if not sys.stdin.isatty():
        program = sys.stdin.read()
        if program.strip():
            from .service import LackpyService
            svc = LackpyService(workspace=workspace)
            kit = _parse_kit(args.kit) if args.kit else None
            exec_result = asyncio.run(svc.run_program(program, kit=kit))
            out_dict = {"success": exec_result.success, "output": exec_result.output, "error": exec_result.error}
            print(json.dumps(out_dict, indent=2, default=str))
            return 0 if exec_result.success else 1

    parser.print_help()
    return 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cli.py -v`
Expected: All tests PASS

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `python -m pytest tests/ -v --ignore=tests/eval`
Expected: All tests PASS. (Eval tests are slow integration tests; skip them here.)

- [ ] **Step 6: Commit**

```bash
git add src/lackpy/cli.py tests/test_cli.py
git commit -m "refactor: remove deprecated subcommands from lackpy CLI, add lackpyctl epilog"
```

---

### Task 7: Integration verification

Final check that all pieces work together.

**Files:** None (read-only verification)

- [ ] **Step 1: Run full non-eval test suite**

Run: `python -m pytest tests/ -v --ignore=tests/eval`
Expected: All tests PASS

- [ ] **Step 2: Verify `lackpy --help` output**

Run: `python -m lackpy.cli --help`
Expected: Clean output with flags only (no subcommands), epilog mentions `lackpyctl`.

- [ ] **Step 3: Verify `lackpyctl mcp --help`**

Run: `python -m lackpy.ctl mcp --help`
Expected: Shows `serve` and `init` subcommands.

- [ ] **Step 4: Test `lackpyctl mcp init` end-to-end**

```bash
cd /tmp && mkdir -p test-lackpy && cd test-lackpy
python -m lackpy.ctl mcp init
cat .mcp.json
```
Expected: `.mcp.json` contains `lackpy` entry with `lackpyctl` command and workspace path.

- [ ] **Step 5: Commit any fixes, if needed**

Only if previous steps revealed issues.
