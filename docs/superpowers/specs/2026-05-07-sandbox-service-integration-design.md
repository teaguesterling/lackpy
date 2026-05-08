# Sandbox Service Integration Design

## Goal

Wire the existing sandbox module into LackpyService so that policy-driven sandbox constraints are actually consumed at execution time. Clean up incomplete code paths (JailCallStrategy) and add missing test coverage (bridge authkey).

## Background

The sandbox module is complete and tested (97 tests): constraints, NsjailBackend, SubprocessStrategy, ToolBridgeManager, worker harness, and tool provisioning. PolicyResult already carries `sandbox_constraints` and `sandbox_backend_configs` fields. But nothing connects them — `delegate()` and `run_program()` ignore the `sandbox` parameter and always execute via `self._runner.run()`.

Three gaps remain:

1. **Service integration** — route execution through SandboxedInterpreter when sandbox is active
2. **JailCallStrategy removal** — dead code, SubprocessStrategy is the real path
3. **Bridge authkey tests** — authkey generation and passing exist but lack test coverage

## Scope

This spec covers wiring only. No new sandbox capabilities, no new constraint types, no new backends. The sandbox module's internal design is unchanged.

## Design

### 1. Service Integration (per-request, policy-driven)

#### Activation

Sandbox activates per-request when either:
- `policy.sandbox_constraints` is non-empty after policy resolution, OR
- The caller passes an explicit `sandbox=True` parameter

The `sandbox` parameter is a boolean opt-in flag (default `False`). When `True`, the service uses the sandbox backend with policy-provided constraints (if any) plus the base config from `LackpyConfig.sandbox`. When policy constraints are non-empty, sandbox activates regardless of the flag — policy is authoritative.

If neither condition is met, execution follows the existing direct `self._runner.run()` path unchanged.

#### LackpyService changes

**`__init__()` additions:**
- Accept an optional `sandbox_backend: SandboxBackend | None` parameter (default `None`)
- Store as `self._sandbox_backend`
- When `LackpyConfig.sandbox` is configured, construct the backend at init time

**`run_program()` changes (line 315):**
- After validation, check sandbox activation: `sandbox=True` OR policy constraints non-empty
- If active: build `SandboxedInterpreter(backend=self._sandbox_backend)`, call its `execute()` with policy constraints, resolved tools, and base config
- If inactive: existing `self._runner.run()` path (unchanged)

**`delegate()` changes (line 347):**
- After generation and kibitzer validation, check sandbox activation (same conditions)
- Same sandbox-or-direct routing as `run_program()`
- Pass `policy.sandbox_constraints` and `policy.sandbox_backend_configs` to `SandboxedInterpreter.execute()`
- Note: `delegate()` already resolves policy at line 306, so the sandbox fields are available

#### Error handling

- If sandbox is requested but `self._sandbox_backend is None`: raise `RuntimeError` with a message pointing to sandbox configuration
- If tool provisioning finds UNAVAILABLE tools: log warnings but proceed with the available subset

### 2. Tool Provisioning Integration

The current `SandboxedInterpreter.execute()` hardcodes `embedded_sources={}`, `bridge_socket=None`, `bridge_authkey=None`. The provisioning module exists but isn't called. Wire it up.

#### Flow

```
SandboxedInterpreter.execute(program, context, tools, providers, ...)
  → extract BridgedToolPolicy from sandbox_constraints
  → partition_kit(tools, bridge_policy) → embedded, bridged, unavailable
  → for EMBEDDED: provider.get_source(spec) → embedded_sources dict
  → for BRIDGED: start ToolBridgeManager(callables) → socket_path, authkey
  → for UNAVAILABLE: log warning
  → SubprocessStrategy.run(
      program, embedded_sources, bridge_socket, bridge_authkey, config)
  → stop ToolBridgeManager in finally block
```

#### SandboxedInterpreter.execute() signature change

Current:
```python
async def execute(
    self, program, context,
    sandbox_constraints=(), backend_configs=None,
) -> InterpreterExecutionResult
```

New:
```python
async def execute(
    self, program, context,
    sandbox_constraints=(), backend_configs=None,
    tools=None, providers=None,
) -> InterpreterExecutionResult
```

Where `tools` is `dict[str, tuple[spec, provider]]` from the resolved kit, and `providers` is optional (can be derived from tools). When `tools` is None, the current behavior (no embedded sources, no bridge) is preserved for backward compatibility.

### 3. JailCallStrategy Removal

**Delete:**
- `JailCallStrategy` class from `src/lackpy/sandbox/strategies.py`
- `jail_call` branch from `SandboxedInterpreter.execute()`
- `JailCallStrategy` from `src/lackpy/sandbox/__init__.py` exports

**Simplify:**
- Remove `strategy` parameter from `SandboxedInterpreter.__init__()` — subprocess is the only strategy
- Remove `self._strategy` field and the strategy dispatch logic in `execute()`
- Inline `SubprocessStrategy` construction directly

### 4. Bridge Authkey Tests

Three test cases in `tests/sandbox/test_bridge.py`:

1. **Matching authkey**: Start `ToolBridgeManager` with tools, connect `bridge_client` with correct authkey, verify `call()` and `list_tools()` work
2. **Wrong authkey**: Start `ToolBridgeManager`, attempt `bridge_client` with incorrect authkey, verify connection fails (expect `AuthenticationError` or similar)
3. **Authkey in worker request**: Verify `SubprocessStrategy.run()` includes `bridge_authkey` hex in the request payload written to `io_dir/request.json`

## Files Changed

| File | Change |
|------|--------|
| `src/lackpy/service.py` | Add sandbox routing in `delegate()` and `run_program()`, accept `sandbox_backend` in `__init__()` |
| `src/lackpy/interpreters/sandbox.py` | Accept tools/providers in `execute()`, wire provisioning, remove strategy dispatch |
| `src/lackpy/sandbox/strategies.py` | Delete `JailCallStrategy` |
| `src/lackpy/sandbox/__init__.py` | Remove `JailCallStrategy` export |
| `tests/sandbox/test_bridge.py` | Add authkey tests (matching, wrong, request payload) |
| `tests/sandbox/test_sandbox_interpreter.py` | Update tests for removed strategy param |

## Testing

- All existing 401 tests continue to pass (no behavioral change for non-sandbox paths)
- New bridge authkey tests (3 cases)
- Updated SandboxedInterpreter tests (strategy param removal)
- Integration test: mock backend + service, verify sandbox routing activates when policy has constraints

## Out of Scope

- New constraint types or backends
- Policy sources that produce sandbox constraints (those exist separately)
- CLI integration for sandbox configuration
- End-to-end nsjail testing (gated behind `@pytest.mark.nsjail`)
