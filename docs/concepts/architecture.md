# Architecture

## Tool sources

At service init the `Toolbox` is populated by **tool sources** — there are no
hard-coded tool names in lackpy. Each source *discovers* full `ToolSpec`s and
*resolves* their callables: config-defined tools (`[[tools]]`, plus the shipped
`default_tools.toml` builtins), MCP-discovered tools (`[mcp_servers]` + ingested
host configs), and virtual/harness tools (`[[virtual_tools]]`). Sources merge by
precedence (local config/builtins > own MCP > host); a shadowed name is dropped.
See [Kits & Toolbox](kits.md) and the [Tool Sources RFC](../design/tool-sources.md).

## Pipeline

A `delegate()` call traverses four stages in sequence (kit resolution draws on the
source-populated toolbox above):

```
  ┌────────────────────────────────────────────────────────────────┐
  │  LackpyService                                                 │
  │                                                                │
  │  1. Kit resolution                                             │
  │     kit name/list/dict ──► ResolvedKit (tools + callables)    │
  │                                                                │
  │  2. Inference                                                  │
  │     intent + namespace_desc                                    │
  │     ──► InferenceDispatcher                                    │
  │          tier 0: TemplatesProvider (regex pattern match)       │
  │          tier 1: RulesProvider    (keyword rules)              │
  │          tier 2: OllamaProvider   (local LLM, optional)        │
  │          tier 3: AnthropicProvider (cloud LLM, optional)       │
  │     ◄── GenerationResult (program + provider_name + time_ms)  │
  │                                                                │
  │  3. Validation                                                 │
  │     program + allowed_names + extra_rules                      │
  │     ──► validate() (AST walk)                                  │
  │     ◄── ValidationResult (valid, errors, calls, variables)     │
  │                                                                │
  │  4. Execution                                                  │
  │     program + resolved.callables + param_values                │
  │     ──► RestrictedRunner.run()                                 │
  │     ◄── ExecutionResult (success, output, trace, variables)    │
  └────────────────────────────────────────────────────────────────┘
```

Validation is also performed inside `InferenceDispatcher` after each provider attempt. If a provider returns an invalid program, the dispatcher feeds the errors back to the provider for a retry before moving to the next tier.

---

## Modules

| Module | Responsibility | Key dependencies |
|--------|---------------|-----------------|
| `lackpy.service` | Unified service layer; wires all components together | All modules |
| `lackpy.config` | Load `config.toml` from `.lackpy/` | stdlib `tomllib` / `tomli` |
| `lackpy.lang.grammar` | `ALLOWED_NODES`, `FORBIDDEN_NODES`, `FORBIDDEN_NAMES`, `ALLOWED_BUILTINS` | `ast` |
| `lackpy.lang.validator` | AST walk + rule application → `ValidationResult` | `lang.grammar` |
| `lackpy.lang.grader` | `Grade(w, d)` computation from tool specs | none |
| `lackpy.lang.rules` | Built-in custom rule callables | `ast` |
| `lackpy.lang.spec` | Machine-readable grammar spec (used by `lackpy spec`) | `lang.grammar` |
| `lackpy.kit.toolbox` | `Toolbox` — tool registry; `add_source()` merge with precedence + resolution | none |
| `lackpy.kit.registry` | `resolve_kit()` — name/list/dict → `ResolvedKit` | `kit.toolbox`, `lang.grader` |
| `lackpy.kit.providers.python` | Resolve a `ToolSpec` to an importable function | `importlib` |
| `lackpy.sources.base` | `ToolSource` protocol — discover (own the names) + resolve | `kit.toolbox` |
| `lackpy.sources.config` | `ConfigToolSource` — tools fully defined in config; `default_tools.toml` ships the builtins as data | `kit.providers.python` |
| `lackpy.sources.virtual` | `VirtualToolSource` — harness-provided tools (`[[virtual_tools]]` + resolver) | `kit.toolbox` |
| `lackpy.sources.mcp.*` | MCP client (dedicated loop), `McpToolSource`, grade-from-hints, host-config ingestion | `mcp` (optional), `run.bridge` |
| `lackpy.run.trace` | `Trace`, `TraceEntry`, `make_traced()` | `inspect`, `time` |
| `lackpy.run.base` | `ExecutionResult`, `Executor` protocol | `run.trace` |
| `lackpy.run.bridge` | `AsyncBridge` — marshal a coroutine from a worker thread onto a loop | `asyncio` |
| `lackpy.run.runner` | `RestrictedRunner` — compile + exec with traced namespace | `lang.grammar`, `run.trace` |
| `lackpy.infer.dispatch` | `InferenceDispatcher` — priority-ordered provider loop | `lang.validator`, `infer.sanitize` |
| `lackpy.infer.prompt` | `build_system_prompt()`, `format_params_description()` | `lang.grammar` |
| `lackpy.infer.sanitize` | Strip model artifacts (markdown fences, preambles) | none |
| `lackpy.infer.providers.*` | `TemplatesProvider`, `RulesProvider`, `OllamaProvider`, `AnthropicProvider` | `infer.prompt` |
| `lackpy.cli` | `argparse`-based CLI; calls `LackpyService` | `service` |
| `lackpy.mcp` | MCP server exposing the service as tools | `service` |

---

## Service layer role

`LackpyService` is the single entry point. Both the CLI (`lackpy.cli`) and the MCP server (`lackpy.mcp`) call the service methods rather than accessing lower-level modules directly. This means:

- The MCP server and CLI always have identical behaviour.
- Third-party code using the Python API benefits from the same validation, tracing, and grade computation as the built-in interfaces.
- Configuration is loaded once at `LackpyService.__init__` and propagated automatically.

---

## Security model

lackpy uses three layers of defence in depth:

### Layer 1 — AST validation (primary)

Every program is parsed and walked before it is compiled or executed. `ALLOWED_NODES` is a whitelist: if a node type is not in the set, the program is rejected. This means:

- `import` / `from ... import` → structurally impossible
- `def` / `class` / `lambda` → structurally impossible
- `while` / `try` / `except` → structurally impossible

The validator also checks:

- All function calls are to names in the kit or `ALLOWED_BUILTINS`
- No name in `FORBIDDEN_NAMES` appears anywhere
- No string constant contains `__` (prevents dunder access via `getattr`)
- `for` loops must iterate over a function call or a variable (not a literal)

### Layer 2 — Restricted execution namespace (secondary)

`RestrictedRunner` executes programs with `__builtins__` set to `{}` (the empty dict). The only names available at runtime are:

- Kit tools (wrapped in tracing callables)
- `ALLOWED_BUILTINS` (direct references from the `builtins` module)
- Parameter values

Even if a program somehow bypassed the AST check, it could not call `eval`, `exec`, `compile`, or any other dangerous builtin — they are simply not in scope.

### Layer 3 — nsjail (v2, planned)

A future `sandbox` tier will use nsjail for process-level isolation with configurable memory and time limits. The `sandbox_enabled` config flag and `sandbox` parameter on `delegate`/`run_program` are already wired; the nsjail integration is slated for v2.

---

## Grade system

Every kit has a `Grade(w, d)` computed from its tools:

| Field | Meaning | Scale |
|-------|---------|-------|
| `w` | World coupling | 0 = pure, 1 = pinhole read, 2 = scoped exec, 3 = scoped write |
| `d` | Effects ceiling | 0–3, higher = more side effects |

`compute_grade()` takes the element-wise maximum across all tools in the kit. The grade is reported in every `delegate()` result and `kit_info()` response so callers can decide whether a given kit is acceptable for their context.

Tool authors set `grade_w` and `effects_ceiling` on their `ToolSpec`. The built-in tools default to `grade_w=3, effects_ceiling=3` (conservative).
