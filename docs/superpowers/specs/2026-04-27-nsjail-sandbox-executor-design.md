# nsjail Sandbox Executor Integration

## Goal

Add OS-level containment to lackpy's interpreter execution via nsjail (and later bwrap, Docker, k8s, Slurm). The current trust boundary is RestrictedRunner's AST-restricted execution with empty `__builtins__` — effective at the language level but not a security boundary if the validator misses something. The sandbox is defense-in-depth: even if generated code evades AST checks, nsjail's namespace isolation, seccomp filtering, and cgroup limits contain the damage.

## Threat Model

**Primary (in scope):** Malicious or buggy generated code that evades the AST validator. The inference model produces syntactically valid code that exploits an unanticipated gap in the grammar/rules system. The sandbox prevents that code from accessing the filesystem (beyond the mounted workspace), network, or host resources.

**Secondary (future, out of scope):** Malicious tool implementations from custom providers or MCP servers. This is harder because tools legitimately need filesystem/network access. Not designed for here, but the architecture should not preclude it.

## Architecture

### Component Overview

```
Service
├── PolicyLayer.resolve() → PolicyResult
│   ├── sandbox_constraints: tuple[SandboxConstraint, ...]
│   └── sandbox_backend_configs: {"nsjail": NsJailConfig}
│
├── SandboxedInterpreter(interpreter, backend, strategy)
│   ├── validate() → delegates directly (no sandbox)
│   └── execute() → routes through sandbox
│       ├── Resolve sandbox config:
│       │   1. Check policy for pre-compiled backend config
│       │   2. Fallback: compile constraints via backend
│       │   3. Merge with base config from lackpy.toml
│       └── Run via strategy:
│           ├── SubprocessStrategy (default)
│           └── JailCallStrategy
│
└── SandboxBackend (protocol)
    └── NsjailBackend (first implementation)
```

### Data Flow

```
delegate()
  → PolicyLayer.resolve()
      → PolicyResult with sandbox_constraints + sandbox_backend_configs
  → SandboxedInterpreter.execute(program, context)
      → resolve config:
          policy pre-compiled config for backend? → use it
          else → backend.compile(constraints, workspace)
          merge with base config from lackpy.toml
      → strategy.run(program, context, sandbox_config)
          subprocess: launch harness in jail, tool calls via embedded/bridged
          jail_call:  serialize interpreter.execute() into sandbox
      → InterpreterExecutionResult (unchanged interface)
```

## Sandbox Constraints

### Rule-Based Constraint Model

Constraints are typed, frozen dataclass objects that subclass `SandboxConstraint`. Each represents a single restriction. This design is backend-agnostic — nsjail, bwrap, Docker, and k8s backends each compile the constraints they understand and report warnings for unsupported ones.

```python
@dataclass(frozen=True)
class SandboxConstraint:
    """Base type for all sandbox constraints."""

@dataclass(frozen=True)
class MemoryLimit(SandboxConstraint):
    amount: int
    unit: str  # "MB", "GB"

@dataclass(frozen=True)
class TimeLimit(SandboxConstraint):
    seconds: int

@dataclass(frozen=True)
class PidLimit(SandboxConstraint):
    max_pids: int

@dataclass(frozen=True)
class CpuLimit(SandboxConstraint):
    ms_per_sec: int  # CPU quota in ms per second

@dataclass(frozen=True)
class NetworkAccess(SandboxConstraint):
    allow: bool

@dataclass(frozen=True)
class NetworkRestriction(SandboxConstraint):
    allowed_destinations: tuple[str, ...]
    allowed_protocols: tuple[str, ...]

@dataclass(frozen=True)
class FilesystemMount(SandboxConstraint):
    path: str
    writable: bool

@dataclass(frozen=True)
class ReadonlyRoot(SandboxConstraint):
    enabled: bool = True

@dataclass(frozen=True)
class SeccompLogging(SandboxConstraint):
    enabled: bool

@dataclass(frozen=True)
class SeccompPolicyConstraint(SandboxConstraint):
    policy_string: str  # Kafel syntax

@dataclass(frozen=True)
class UserMapping(SandboxConstraint):
    inside_id: int
    outside_id: int
    count: int = 1

@dataclass(frozen=True)
class BridgedToolPolicy(SandboxConstraint):
    allowed: bool = False
    allowed_tools: tuple[str, ...] = ()
    allowed_providers: tuple[str, ...] = ()
```

### Constraint Merge Semantics

When multiple policy sources emit constraints of the same type:
- **Resource limits** (MemoryLimit, TimeLimit, PidLimit, CpuLimit): take the minimum (most restrictive).
- **Boolean flags** (NetworkAccess, ReadonlyRoot, SeccompLogging): `False`/restrictive wins.
- **Path tuples** (FilesystemMount): union. If the same path appears with conflicting writability, read-only wins (most restrictive).
- **BridgedToolPolicy**: intersection of allowed tools/providers; `allowed=False` wins.

### Backend Compilation

Each backend compiles constraints it understands and warns on unknown ones:

| Constraint | nsjail | bwrap | Docker | k8s |
|-----------|--------|-------|--------|-----|
| MemoryLimit | cgroup_mem_max | (no native) | --memory | resources.limits |
| TimeLimit | time_limit | (external) | (external) | activeDeadlineSeconds |
| PidLimit | cgroup_pids_max | (no native) | --pids-limit | (no native) |
| CpuLimit | cgroup_cpu_ms_per_sec | (no native) | --cpus | resources.limits |
| NetworkAccess | clone_newnet | --unshare-net | --network=none | networkPolicy |
| NetworkRestriction | warn (on/off only) | warn | iptables | networkPolicy |
| FilesystemMount | MountPt | --bind/--ro-bind | -v | volumes |
| ReadonlyRoot | readonly root preset | --ro-bind / | read_only rootfs | readOnlyRootFilesystem |
| SeccompLogging | seccomp_log | (no native) | seccomp profile | (no native) |
| SeccompPolicyConstraint | seccomp_string | --seccomp | seccomp profile | seccomp profile |
| UserMapping | uidmap | --uid/--gid | --user | securityContext |
| BridgedToolPolicy | (handled by harness) | (handled by harness) | (handled by harness) | (handled by harness) |

## Policy Integration

### PolicyResult Extensions

`PolicyResult` gains two new fields:

```python
@dataclass(frozen=True)
class PolicyResult:
    # ... existing fields ...
    sandbox_constraints: tuple[SandboxConstraint, ...] = ()
    sandbox_backend_configs: MappingProxyType[str, Any] = EMPTY_MAP
```

- `sandbox_constraints`: rule objects from any policy source, compiled by the backend at run time.
- `sandbox_backend_configs`: pre-compiled configs keyed by backend name. If umwelt can natively emit an `NsJailConfig`, it goes here and bypasses constraint compilation entirely.

### Two-Tier Config Resolution

1. **Pre-compiled path**: check `sandbox_backend_configs[backend.name]` → hand to `backend.accept_policy_config()`. If accepted, use directly (merge with base config defaults).
2. **Constraint path**: `backend.compile(sandbox_constraints, workspace)` → build config from primitives.

Pre-compiled configs take precedence. This lets umwelt's native nsjail compiler produce optimized configs while the constraint path serves as a universal fallback.

### Policy Source Behavior

- **KitPolicySource**: does not emit sandbox constraints (kit doesn't know about resources).
- **KibitzerPolicySource**: does not emit sandbox constraints initially. Could add coaching-derived limits later.
- **UmweltPolicySource**: the natural home for sandbox constraints. Emits both constraint rules and optionally pre-compiled backend configs via umwelt's native compiler targets.

## Backend Protocol

```python
@dataclass
class ConstraintWarning:
    constraint: SandboxConstraint
    reason: str

@dataclass
class CompilationResult:
    config: Any  # Backend-specific (NsJailConfig, bwrap args, etc.)
    warnings: list[ConstraintWarning]

@dataclass
class SandboxResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    timed_out: bool
    oom_killed: bool
    resource_stats: dict[str, Any] | None  # Backend-specific metrics

class SandboxBackend(Protocol):
    name: str

    def accept_policy_config(self, config: Any) -> CompilationResult | None:
        """Try to use a pre-compiled config from the policy layer.
        Returns None if the config isn't for this backend or isn't valid."""

    def compile(
        self,
        constraints: Sequence[SandboxConstraint],
        workspace: Path,
    ) -> CompilationResult:
        """Compile constraint rules into backend-specific config.
        Warns on unsupported constraints."""

    async def run(
        self,
        config: Any,
        command: list[str],
    ) -> SandboxResult:
        """Run a command inside the sandbox."""
```

### NsjailBackend (Phase 1 Implementation)

Uses nsjail-python's `Jail` fluent builder to compile constraints:

- `MemoryLimit` → `.memory(amount, unit)`
- `TimeLimit` → `.timeout(seconds)`
- `PidLimit` → `.pids(max_pids)`
- `CpuLimit` → `.cpu(ms_per_sec)`
- `NetworkAccess(allow=False)` → `.no_network()`
- `FilesystemMount` → `.mount(path, readonly=not writable)`
- `ReadonlyRoot` → `.readonly_root()`
- `SeccompLogging` → `.seccomp_log()`
- `SeccompPolicyConstraint` ��� `.seccomp(policy_string)`

Default jail config (before constraints):
- Root filesystem: read-only
- Workspace: mounted read-write
- Python environment: mounted read-only
- Network: disabled
- Processes: capped at 16
- Memory: 512 MB
- Timeout: 120 seconds

`accept_policy_config()` checks if the input is an `NsJailConfig` instance and wraps it in a `CompilationResult`.

### Strict vs. Permissive Warning Handling

Configurable in `lackpy.toml`:

```toml
[sandbox]
constraint_warnings = "warn"  # or "error"
```

- `"warn"` (default): log unsupported constraints, proceed.
- `"error"`: any unsupported constraint aborts with a clear error message.

## Tool Provisioning

### Two Tiers

Tools inside the sandbox are provisioned via two mechanisms:

**Tier 1: Embedded tools (common path)**

Tool implementations are injected directly into the sandbox as source code. They run natively inside the jail with no IPC overhead.

Provisioning methods:
- **Prepended source**: Python function definitions prepended to the program text before interpreter invocation.
- **Injected modules**: Implementation files written to a temp directory mounted into the jail, then imported by the harness.

Applicable when: the tool's implementation is available as source and only needs access to resources mounted into the jail (workspace filesystem, etc.). This covers builtin tools (read_file, write_file, find_files) and pure-computation tools.

The tool provider interface gains a method for source extraction:

```python
class ToolProvider(Protocol):
    def get_source(self, spec: ToolSpec) -> str | None:
        """Return the tool's implementation as Python source, or None
        if the tool requires a live connection and must be bridged."""
```

Providers that return source → embedded. Providers that return `None` → must be bridged.

**Tier 2: Bridged tools (rare path, policy-gated)**

Tool calls are proxied through `multiprocessing.managers` to the parent process, where the real callable runs in the trusted context.

Required when: the tool needs resources outside the jail — MCP connections, database handles, external API clients, parent process state.

Gated by `BridgedToolPolicy` constraint:
- Default: bridge disabled, bridged tools unavailable in sandbox.
- Policy can whitelist specific tools by name or by provider type (e.g., `allowed_providers=("mcp",)`).
- If a tool needs the bridge but policy denies it, the tool is excluded from the sandboxed kit. The harness reports it as a constraint warning.

### Tool Availability Outcomes

For each tool in the resolved kit, the sandbox harness determines:

1. **Embedded**: provider returns source → injected into jail, runs natively.
2. **Bridged**: provider returns None, bridge policy allows it → proxied via managers.
3. **Unavailable**: provider returns None, bridge policy denies it ��� excluded from sandboxed kit, reported as warning.

## Communication: multiprocessing.managers Bridge

### Architecture

```
Parent process (trusted)
├── ToolBridgeManager (BaseManager subclass)
│   ├── Registered callables from resolved kit (bridged tools only)
│   └── Listening on Unix socket in temp directory
│
══════════════ nsjail boundary ══════════════
│
Jail harness
├��─ ToolBridgeManager client (connects to parent socket)
├── Proxy callables (transparent function calls over IPC)
└── Interpreter runs with:
    ├── Embedded tools: native callables
    └── Bridged tools: proxy callables via manager
```

### Protocol

The bridge uses Python's `multiprocessing.managers.BaseManager`:

- Parent creates a `ToolBridgeManager` subclass, registers each bridged tool's callable, starts the server on a Unix socket.
- The socket path is mounted into the jail (read-write).
- The harness inside the jail connects to the manager and retrieves proxy callables.
- Tool calls from the program transparently round-trip through the socket.
- Results are serialized using multiprocessing's native protocol.

The manager is created per-invocation and torn down after the sandbox exits. No persistent server.

### Security Considerations

The multiprocessing.managers bridge uses Python's native serialization. The trust boundary is nsjail itself (namespace isolation, seccomp, cgroups), not the serialization protocol. The bridge is only active when `BridgedToolPolicy` explicitly allows it via the policy layer — it is disabled by default.

### Future: MCP Harness Inside Jail (Phase 2)

For non-Python interpreters that can't call Python functions directly:

- The jail harness starts a lightweight MCP server inside the sandbox.
- The MCP server exposes both embedded tools and bridged tool proxies as MCP tools.
- The non-Python interpreter connects as an MCP client and calls tools via the MCP protocol.
- Tool calls to bridged tools flow: interpreter → MCP → harness → managers → parent → real callable.

This is designed for but not built in Phase 1. The multiprocessing.managers bridge is the foundation — MCP inside the jail is a translation layer on top.

## Execution Strategies

### SubprocessStrategy (Default)

1. Prepare inputs:
   - Program text, interpreter name, params as JSON
   - Embedded tool sources (prepended or as module files)
   - Manager socket path (if any bridged tools)
2. Build jail config from compiled constraints + base config
3. Launch: `python -m lackpy.sandbox._worker <io_dir>`
4. Jail mounts: workspace (rw), python env (ro), io_dir (rw), manager socket (rw if bridge active)
5. Worker harness inside the jail:
   - Reads inputs from `<io_dir>/request.json`
   - Loads embedded tool sources
   - Connects to manager for bridged tools (if socket path provided)
   - Constructs tool namespace (embedded + bridged callables)
   - Gets interpreter from registry
   - Calls `interpreter.execute(program, context)`
   - Writes `InterpreterExecutionResult` as JSON to `<io_dir>/result.json`
6. `SandboxedInterpreter` reads and deserializes the result

### JailCallStrategy

1. Uses nsjail-python's `jail_call()` to directly invoke `interpreter.execute(program, context)` inside the sandbox via serialization.
2. Only works when the interpreter instance and its ExecutionContext (including tool callables) are serializable.
3. Embedded tools that are simple functions may be serializable; bridged proxy callables are not.
4. If serialization fails, raises a clear error suggesting subprocess mode.
5. No harness, no serialization protocol, no manager bridge — everything crosses via nsjail-python's internal serialization mechanism.

### Strategy Selection

Configured in `lackpy.toml`, defaults to `subprocess`:

```toml
[sandbox]
strategy = "subprocess"  # or "jail_call"
```

`jail_call` is an optimization for simple cases. `subprocess` is the universal strategy that always works.

## SandboxedInterpreter

### Interface

```python
class SandboxedInterpreter:
    """Decorator that wraps any Interpreter with sandbox execution."""

    def __init__(
        self,
        interpreter: Interpreter,
        backend: SandboxBackend,
        strategy: str = "subprocess",
        base_config: SandboxBaseConfig | None = None,
    ) -> None: ...

    @property
    def name(self) -> str:
        return self.interpreter.name

    @property
    def description(self) -> str:
        return self.interpreter.description

    def validate(
        self, program: str, context: ExecutionContext,
    ) -> InterpreterValidationResult:
        """Delegates directly — validation is pure AST analysis, no sandbox needed."""
        return self.interpreter.validate(program, context)

    async def execute(
        self,
        program: str,
        context: ExecutionContext,
        sandbox_constraints: tuple[SandboxConstraint, ...] = (),
        backend_configs: Mapping[str, Any] | None = None,
    ) -> InterpreterExecutionResult:
        """Runs program inside the sandbox."""
        ...

    def system_prompt_hint(self) -> str:
        return self.interpreter.system_prompt_hint()
```

### Service Wiring

The service wraps interpreters at construction time based on config:

```python
interpreter = get_interpreter(interpreter_name)
if self._config.sandbox_enabled:
    backend = NsjailBackend()
    interpreter = SandboxedInterpreter(
        interpreter=interpreter,
        backend=backend,
        strategy=self._config.sandbox_strategy,
        base_config=self._config.sandbox_base_config,
    )
```

The service passes `PolicyResult.sandbox_constraints` and `PolicyResult.sandbox_backend_configs` to the interpreter's `execute()` call. The `SandboxedInterpreter` handles config resolution internally.

## Configuration

### lackpy.toml

```toml
[sandbox]
enabled = false           # Master switch
backend = "nsjail"        # Backend name
strategy = "subprocess"   # Execution strategy
constraint_warnings = "warn"  # "warn" or "error"

# Base resource limits (overridden by policy)
timeout = 120
memory_mb = 512
pids_max = 16
network = false

# Bridge configuration
bridge_enabled = false
bridge_allowed_providers = []  # e.g., ["mcp"]
```

These values serve as defaults. Policy-provided constraints override them (most restrictive wins).

## File Structure

```
src/lackpy/
├─��� sandbox/
│   ├── __init__.py              # Public exports
│   ├── constraints.py           # SandboxConstraint hierarchy
│   ├── backend.py               # SandboxBackend protocol, CompilationResult, SandboxResult
│   ├── backends/
│   │   ├── __init__.py
│   │   └── nsjail.py            # NsjailBackend implementation
│   ├── bridge.py                # ToolBridgeManager, bridge setup/teardown
│   ├── provisioning.py          # Tool provisioning logic (embedded vs bridged)
│   ├── config.py                # SandboxBaseConfig, config loading from lackpy.toml
│   └── _worker.py               # Subprocess worker harness (runs inside jail)
├── interpreters/
│   ├── sandbox.py               # SandboxedInterpreter decorator
│   └── ...                      # Existing interpreters (unchanged)
├── policy/
│   └── types.py                 # Extended: sandbox_constraints, sandbox_backend_configs
└── ...
```

## Testing Strategy

**Unit tests** (no nsjail binary needed):
- Constraint merging logic
- NsjailBackend constraint compilation (verify Jail builder calls)
- Tool provisioning decisions (embedded vs bridged vs unavailable)
- SandboxedInterpreter config resolution (pre-compiled vs compiled vs base)
- Worker harness serialization/deserialization
- Policy integration (sandbox constraints on PolicyResult)
- Bridge manager setup/teardown lifecycle
- Constraint warning handling (strict vs permissive)

**Integration tests** (require nsjail binary):
- End-to-end: generate + validate + sandboxed execution with real nsjail
- Subprocess strategy with embedded tools
- Subprocess strategy with bridged tools
- jail_call strategy with simple interpreter
- Resource limit enforcement (timeout, memory OOM)
- Filesystem isolation (cannot read outside mounted paths)
- Network isolation

**Skip markers**: integration tests gated by `@pytest.mark.nsjail` so CI can run without nsjail installed.

## Scope

### Phase 1 (This Spec)

- `SandboxConstraint` hierarchy and merge logic
- `SandboxBackend` protocol
- `NsjailBackend` implementation
- `SandboxedInterpreter` decorator
- Both execution strategies: subprocess and jail_call
- Tool provisioning: embedded and bridged tiers
- `ToolBridgeManager` for bridged tools via multiprocessing.managers
- `ToolProvider.get_source()` for embedded tool extraction
- `BridgedToolPolicy` constraint and bridge policy gating
- Policy integration: `sandbox_constraints` and `sandbox_backend_configs` on PolicyResult
- Two-tier config resolution (pre-compiled vs constraint compilation)
- Config in `lackpy.toml`
- Service wiring
- Worker harness for subprocess strategy

### Phase 2 (Designed, Not Built)

- MCP harness inside jail for non-Python interpreters
- bwrap backend
- Per-interpreter strategy configuration

### Future

- Docker, Kubernetes, Slurm backends
- Tool-level sandbox isolation (threat model B)
- Automatic strategy selection based on kit contents
