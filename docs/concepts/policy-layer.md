# Policy Layer

## What it is

The **PolicyLayer** consolidates tool-access policy into a single resolution chain. Instead of scattered `if kibitzer` / `if umwelt` conditionals, the service builds a context, calls `policy.resolve()`, and gets back an immutable result describing what tools are allowed, what constraints apply, and what hints to inject into prompts.

The design supports **progressive enrichment**: standalone lackpy uses kits alone; adding Kibitzer adds coaching and hints; adding umwelt adds world-model policy. Each integration layer enriches the result without replacing earlier layers.

---

## PolicyResult

The output of policy resolution. Frozen — sources produce new instances via `replace()`.

| Field | Type | Description |
|-------|------|-------------|
| `allowed_tools` | `frozenset[str]` | Tools the program may call |
| `denied_tools` | `frozenset[str]` | Tools explicitly blocked by policy |
| `tool_constraints` | `MappingProxyType[str, ToolConstraints]` | Per-tool restrictions (patterns, level limits) |
| `grade` | `Grade \| None` | Aggregate security grade from the kit |
| `namespace_desc` | `str \| None` | Tool descriptions for the inference prompt |
| `prompt_hints` | `tuple[str, ...]` | Correction hints injected on retry |
| `docs` | `tuple[str, ...]` | Doc context strings for the current request |
| `resolved` | `bool` | If True, no further sources are consulted |

---

## PolicyContext

A `TypedDict` describing the request. Only `kit` is required — all other fields are optional and sources degrade gracefully when they're absent.

| Field | Type | VSM Level | Description |
|-------|------|-----------|-------------|
| `kit` | `ResolvedKit` | S1 (operations) | Tools available for this request |
| `principal` | `Principal` | S5 (identity) | Who is requesting — human, agent, or subagent |
| `model` | `ModelSpec` | S4 (inferencer) | Model name, temperature, tier |
| `session_id` | `str` | S2 (coordination) | Lets sources correlate internal state |
| `history` | `StepContext` | S3* (audit) | Inference fold accumulator — prior attempts, errors |
| `trace` | `Trace` | S3* (audit) | Runtime execution trace — tool calls, timing |

The PolicyLayer itself is S3 (control). S0 (environment/world) is source-internal — umwelt brings its world model, others bring nothing.

---

## How resolution works

Sources are registered with a priority. The chain runs lowest-priority-first, threading an immutable `PolicyResult` through each source:

```
1. KitPolicySource (0)       → establishes allowed_tools, grade, namespace_desc
2. KibitzerPolicySource (50)  → adds hints, docs, coaching
3. UmweltPolicySource (100)   → restricts tools, adds per-tool constraints
```

Each source receives the current result and the context, and returns a new `PolicyResult`. If a source sets `resolved=True`, the chain stops early.

```python
from lackpy.policy import PolicyLayer, PolicyResult

layer = PolicyLayer()
layer.add_source(kit_source)      # priority 0
layer.add_source(kibitzer_source)  # priority 50

result = layer.resolve({"kit": resolved_kit})
# result.allowed_tools → frozenset({"read_file", "find_files"})
# result.prompt_hints → ("Use read_file() instead of open()",)
```

---

## Built-in sources

### KitPolicySource (priority 0)

Always present. Translates a `ResolvedKit` into the baseline `PolicyResult` — sets `allowed_tools` from the kit's tool names, `grade` from the kit's aggregate grade, and `namespace_desc` from the toolbox's formatted descriptions.

Never sets `resolved=True`.

### KibitzerPolicySource (priority 50)

Optional. Reads the session history (`StepContext`) to detect failure patterns and adds prompt hints, doc context, and coaching to the namespace description. **Never modifies `allowed_tools` or `denied_tools`** — Kibitzer is a coaching layer, not a policy authority.

### UmweltPolicySource (priority 100)

Optional. Queries umwelt's `PolicyEngine` for capability-taxon entries. Can **restrict** the kit's tool set (deny visible tools, add per-tool constraints) but **cannot grant** tools the kit doesn't have. Kit resolution (S1) is the ground truth for what's available.

---

## Configurations

| Setup | Sources | Behavior |
|-------|---------|----------|
| Standalone | Kit | Tools allowed, grade computed, descriptions formatted |
| + Kibitzer | Kit + Kibitzer | Above + coaching, prompt hints, doc context on failures |
| + umwelt | Kit + Kibitzer + Umwelt | Above + world-model tool restrictions, per-tool constraints |
| Kit + umwelt | Kit + Umwelt | Policy restrictions without coaching |

---

## Kit resolution vs policy resolution

These are separate concerns that happen at different points:

1. **Kit resolution** (`resolve_kit()`) turns a kit name, tool list, or dict into a `ResolvedKit` with tools, callables, and grade. This is S1 — what's operationally available.

2. **Policy resolution** (`PolicyLayer.resolve()`) takes the resolved kit and determines what constraints apply. This is S3 — what's allowed given the current context.

The service calls `resolve_kit()` first, then passes the result into the policy layer via `PolicyContext["kit"]`.
