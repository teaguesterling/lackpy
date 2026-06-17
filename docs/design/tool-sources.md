# RFC 0002 — Tool sources (config, MCP, virtual) & the sync↔async bridge

!!! note "Status"
    Increments 1–4 are **implemented** (config-defined source; async bridge + MCP
    client; multi-source precedence + host-config ingestion; virtual/harness tools).
    Increment 5 (kits→profile) is **design-only** here. Siblings:
    [Direction](direction.md), [Interpreter Types](interpreter-types.md), RFC 0001
    (the `lackpy-lang` leaf split).

## 1. Problem & goals

[direction.md](direction.md) calls for retiring "kit" as a primitive and generalizing
it into a runtime-config system. The **foundational invariant** for that work: *tool
names are never hard-coded in lackpy source.* Every tool — its full `ToolSpec` and its
callable — must originate from a **source** that owns it end to end. Before this RFC,
the invariant was violated by `service._BUILTIN_TOOLS` (a hard-coded spec list) and by
`.kit` files referencing names that only resolve against code-built specs.

Goals:

1. A single `ToolSource` abstraction implemented by config-defined, MCP-discovered, and
   virtual/harness tools alike.
2. Design the deferred sources and cross-cutting concerns: multi-config merge &
   namespacing, grade-from-hints, security.
3. Solve the centerpiece: a **synchronous** lackpy program calling an **async** MCP
   `call_tool` mid-execution — without deadlock.

Non-goals: the kits→profile layer that sits *on top* of sources (sequenced last, §10).

## 2. The ToolSource model

`kit/providers/base.py`'s `ToolProvider` does **resolution only**
(`name`/`available`/`resolve(spec)->callable`); discovery was hard-coded by the caller.
The new surface adds **discovery**:

```python
class ToolSource(Protocol):
    @property
    def name(self) -> str: ...
    def available(self) -> bool: ...
    def discover(self) -> list[ToolSpec]: ...      # NEW — the source owns its names
    def resolve(self, spec: ToolSpec) -> Callable[..., Any]: ...
```

`discover()` is the only addition over `ToolProvider`; `resolve()` keeps its contract.
`Toolbox.add_source(source)` discovers specs, registers them, and records the source as
their resolver (`_spec_owner`), so `Toolbox.resolve(name)` dispatches to the owning
source first and falls back to the legacy provider dispatch for directly-registered
tools. Later-added source wins on name collision (user config beats shipped defaults).

**Structural home:** `src/lackpy/sources/` (top-level runtime package), *not*
`kit/providers/` — "kit" is being retired. `lackpy.sources.*` is runtime and may import
freely from `run/`/`policy/`/`kit/`; it must never be imported by `lackpy.lang`
(`tests/lang/test_no_upward_deps.py` now lists `sources` in `FORBIDDEN`). The MCP client
is **not** a candidate for its own PEP 420 dist like `lackpy-lang` — it depends on the
async runtime/event loop, the opposite of a pure leaf.

### Increment 1 (implemented): config-defined source
`ConfigToolSource` reads fully-defined tool dicts (spec + `module`/`function`) and
resolves callables via the existing `PythonProvider`. The former builtins ship as DATA
in `src/lackpy/sources/default_tools.toml`, auto-loaded at service init — removing the
hard-coded `_BUILTIN_TOOLS` list. Users add/override tools via a top-level `[[tools]]`
array in `.lackpy/config.toml`.

## 3. MCP client design (deferred)

lackpy has only an MCP *server* today (`mcp/server.py`, fastmcp). This introduces the
first MCP *client*.

- **SDK:** the low-level `mcp.ClientSession` (already an optional dep) — explicit
  session/loop control, which fastmcp's client hides. MCP stays an optional extra;
  sources report `available() == False` when `mcp` isn't installed.
- **Transports:** stdio (`command`/`args`/`env`/`cwd`) and streamable-http/SSE
  (`url`/`headers`).
- **Own config** — `[mcp_servers.<id>]` (transport + connection + per-tool overrides).
- **Host configs (multiple)** — `[mcp].host_configs = [...]` paths to host MCP server
  maps (the `mcp__claude_ai_*` / `mcp__plugin_*` shape). Normalized into the same
  internal descriptor, tagged with provenance (own / host:<path>) for precedence (§4)
  and security gating (§8). Virtual/harness-provided tools (§7) are the third origin.
- **Discovery: eager, bounded.** Generation composes against real tool names, so the
  inventory must exist before generation — lazy discovery defeats the purpose. Connect
  each server with a per-server timeout; on failure mark that source unavailable and
  continue (one dead server must not break others). Eager connect also fixes the cwd
  hazard (§5.5) by spawning stdio subprocesses before any execution-time `os.chdir`.
- **Spec build:** MCP `name`→name (after namespacing §4), `description`→description+docs,
  `inputSchema`→`ArgSpec`s (unknown/nested → `type="Any"`, matching
  `resolve_python_type`'s fallback), `annotations`→grade (§6).

## 4. Multi-inventory merge, namespacing, precedence (implemented)

Tension: host convention uses long names (`mcp__claude_ai_Gmail__create_draft`), but
`toolbox._MASKING_NAMES` + the `register_tool` warning exist precisely because long/
confusing names degrade the small (1.5B) generator lackpy targets.

**Implemented — short names, drop-with-log on collision (v1):** each source contributes
tools under their bare name; on collision the higher-precedence source keeps the bare
name and the shadowed tool is **dropped and logged** (`logger.info`). v1 does *not*
keep a `<server>__<tool>` alias — the small models lackpy targets never call qualified
names, so the alias machinery would serve no real path; add it in a later increment
only if a workflow needs a shadowed tool. Masking names stay **warn-only** (auto-qualify
deferred).

**Precedence (deterministic), highest wins the bare name** — *auditable local source
wins*: user config `[[tools]]` (20) > shipped defaults/builtins (10) > own
`[mcp_servers]` (5) > host configs (4, descending per file so the earlier file wins,
floored at 1). Equal precedence → the later-added source wins (this is how a user
`[[tools]]` entry overrides a default of the same name). Directly-`register_tool`'d
tools use `inf` (always authoritative). The resolved name set feeds `KitPolicySource`
and `format_description` unchanged.

## 5. The sync↔async bridge (centerpiece, deferred)

lackpy programs run **synchronously** in `run/runner.py` `RestrictedRunner.run` — the
"lacking" language deliberately has no `await`. MCP `call_tool` is a coroutine, invoked
mid-program with args known only at runtime.

### 5.1 The deadlock the obvious implementation creates
Today `service._execute` calls `self._runner.run(...)` **on the event-loop thread**
(inside async `delegate`). If an MCP callable does
`asyncio.run_coroutine_threadsafe(call_tool(...), loop).result()` against that same
loop, it **deadlocks**: the loop is blocked inside the sync runner, so the submitted
coroutine never runs. Therefore the fix is a **required call-site change**, not merely
"use a thread":

```python
async def _execute(self, program, callables, param_values, kibitzer):
    loop = asyncio.get_running_loop()
    prev = os.getcwd()
    def _body():
        os.chdir(self._workspace)
        try:
            return self._runner.run(program, callables, params=param_values,
                                    kibitzer_session=kibitzer)
        finally:
            os.chdir(prev)
    return await loop.run_in_executor(None, _body)
```

Offloading the sync runner to a worker thread frees the loop to service marshaled
coroutines.

### 5.2 The MCP callable
`source.resolve(spec)` returns a plain sync callable the runner traces like any other;
it marshals to the loop and blocks the *worker* thread:
```python
fut = asyncio.run_coroutine_threadsafe(session.call_tool(mcp_name, kwargs), loop)
result = fut.result(timeout=timeout)   # FuturesTimeout -> cancel + raise TimeoutError
# result.isError -> RuntimeError; else unwrap content blocks -> Python value
```

### 5.3 Session lifecycle
"One persistent session per server" means you **cannot** `async with` per call. Use a
long-lived per-server manager coroutine that enters the transport + `ClientSession`
context once, `initialize()`s, does eager `list_tools()`, then awaits a shutdown event
to keep the context (and stdio subprocess) alive for `call_tool`.

### 5.4 Which loop hosts the sessions
- MCP-server path (fastmcp owns a long-lived loop): capture it; sessions persist.
- CLI path (`asyncio.run` per invocation): lazy session init on the ambient loop,
  one-shot per invocation. Escalation (documented, not built): a dedicated client loop
  on its own thread for cross-invocation persistence.

### 5.5 cwd is process-global
`os.chdir` is process-global, not thread-local — once `_execute` runs in a worker
thread, concurrent ops race on cwd, and stdio servers spawned during the window inherit
the wrong cwd. Mitigations: spawn stdio servers eagerly (§3); single-flight execution
per service instance for now; principled fix is dropping `os.chdir` for an explicit
workspace root threaded into tool callables.

**Implemented tradeoff:** the single-flight guard is a per-service `asyncio.Lock`
held across *all* of `_execute` — including the pure-inline (no-async-tool) path. So
a slow MCP-backed delegation serializes even non-MCP delegations on the same service.
This is deliberate (it reproduces today's cwd-safety) but is a real throughput
serialization in a long-lived MCP-server host; lifting it depends on the `os.chdir`
removal above.

### 5.6 Errors — already solved
`run/trace.py` `make_traced` already catches exceptions into a failed `TraceEntry` and
re-raises into a failed `ExecutionResult`. The MCP callable just translates MCP failures
(timeout, `isError`) into ordinary Python exceptions — no new trace plumbing.

### 5.7 Rejected
Nested `asyncio.run`/new loop per call (kills the persistent session, re-`initialize()`
per call is prohibitive); making the language async (violates lackpy's design).

## 6. Grade from MCP hints (deferred)

Map MCP advisory annotations → `Grade(w, d)` (w = world coupling, d = effects ceiling):

| annotations | (w, d) |
|---|---|
| readOnly + closed-world | (1, 0) |
| readOnly + openWorld | (2, 1) |
| write + idempotent + non-destructive | (3, 2) |
| write + (non-idempotent or destructive) | (3, 3) |
| any relevant hint absent | (3, 3) — conservative |

Invariants (ordering canonical; exact integers the maintainer's call): external coupling
raises **w**; destructiveness/non-idempotence raises **d**; **config override always
wins**. Do not mix partial MCP annotation defaults with the mapping — any missing hint
⇒ conservative. Derived grades land on the `ToolSpec` and feed `compute_grade` →
`ResolvedKit.grade` → `KitPolicySource` → kibitzer with **zero downstream change**.

## 7. Virtual / harness-provided tools (implemented)

Implemented as `sources/virtual.py:VirtualToolSource` + the service's
`harness_resolver` (a `name -> callable | None`) and `_gate_kit`. Declared via
`[[virtual_tools]]` (full spec; no module). Two-layer enforcement: `_gate_kit`
drops currently-unavailable virtual tools from the kit before generation, and the
resolve-time proxy raises if the harness withdrew the tool by call time (→ failed
ExecutionResult). v1 harness callables are sync (wrap via §5 if async later).
Precedence sits below local config/builtins, above MCP.

A virtual tool is **fully declared** (`[virtual_tools.<name>]` with full spec) but its
implementation is supplied by the harness/host at call time and may be absent. The
harness passes a resolver into the service; `VirtualToolSource.available()`/`resolve()`
consult it. Enforce at **both** layers: availability gates visibility at generation
(filtered from `allowed_tools`/`namespace_desc`, so the LLM never composes an absent
tool), and a call-time `RuntimeError` if it was withdrawn after generation (flows
through `make_traced` like any failure).

## 8. Security

Three trust tiers, explicit: **config/python** (arbitrary import/exec; trusted because
auditable local source; highest precedence) > **MCP** (remote, opaque; conservative
grade; provenance-tracked so a future `McpPolicySource` can gate host-config tools
tighter) > **virtual** (trust follows the harness). The grade → `KitPolicySource` →
kibitzer chain is unchanged; new sources plug in by populating grades correctly.

## 9. Open questions

1. Connect-timeout budget for eager discovery across many host servers (parallel connect
   + global deadline?).
2. stdio subprocess cwd when host config and lackpy workspace differ.
3. Runtime tool churn (`list_changed`) — re-discover or fix inventory per lifetime?
   (Recommend fixed for v1.)
4. Canonical Python value for MCP content blocks (text→str, structured→dict, multi→list?).
5. Per-call vs per-session timeout granularity; cancelling a hung call mid-program.

## 10. Migration & sequencing

1. **Config-defined source** — *implemented* (this PR). `_BUILTIN_TOOLS` removed;
   builtins now data.
2. **MCP client + bridge** — *implemented*. 2a: `AsyncBridge` + async `_execute`
   with the single-flight `asyncio.Lock` and lazy threaded path (`run/bridge.py`).
   2b: dedicated-loop `McpClient` (loop-ownership decision = dedicated client
   thread, so eager discovery + session reuse compose across the MCP-server and
   per-command CLI contexts), `McpToolSource`, grade-from-hints; own `[mcp_servers]`.
3. **Multi/host config + namespacing** — *implemented*. Precedence + drop-with-log
   collision merge in `Toolbox` (§4); `[mcp].host_configs` ingestion of external
   `mcpServers` files at descending precedence (§3.4).
4. **Virtual/harness tools** — *implemented* (§7).
5. **kits→profile layer** on top of sources, retiring "kit" ([direction.md](direction.md) §2).
   Designed in [Kits → Profiles](kits-to-profiles.md).

Throughout: keep `tests/lang/test_no_upward_deps.py` green.
