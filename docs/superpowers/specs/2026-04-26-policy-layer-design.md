# PolicyLayer Design: lackpy-umwelt Integration

**Date:** 2026-04-26
**Status:** Draft

## Problem

lackpy's policy logic is scattered across the service layer: kit resolution in `kit/registry.py`, tool descriptions in `kit/toolbox.py`, Kibitzer coaching via conditional `if self._kibitzer` checks threaded through `service.py`, and umwelt integration not yet wired. Adding umwelt means adding a third source of tool policy with no clean place to put it.

The integration surface is also asymmetric. lackpy must work standalone (kits only), with Kibitzer (kits + coaching), or with umwelt (kits + world-model policy + coaching). Each combination currently requires different code paths in the service layer.

## Design Principles

1. **Progressive enrichment** -- standalone lackpy uses kits; adding Kibitzer adds coaching/hints; adding umwelt adds world-model policy. Each layer enriches rather than replaces.
2. **Kit resolution is separate from policy resolution** -- `resolve_kit()` turns a name into tools/callables (S1 lookup). PolicyLayer determines what constraints apply to those tools (S3 control). These happen at different points in the API.
3. **umwelt is optional** -- lackpy never imports umwelt at module level. The UmweltPolicySource is registered at runtime when an engine is available.
4. **Immutable propagation** -- PolicyResult is frozen. Each source returns a new instance. No mutation, no side effects in the chain.

## Architecture: VSM-Informed PolicyContext

The PolicyContext is modeled on Beer's Viable System Model to ensure the right information is available at each level:

| VSM Level | What | Representation |
|-----------|------|----------------|
| S5 | Principal (identity) | `Principal` object -- who is requesting, human vs agent vs subagent |
| S4 | Inferencer (model) | `ModelSpec` object -- name, temperature, context window, tier |
| S3 | Control | The `PolicyLayer` itself -- not data, the mechanism |
| S3* | Audit trail | `StepContext` (inference history) + `Trace` (runtime execution) |
| S2 | Coordination | `session_id` -- sources correlate their own internal state |
| S1 | Operations | `ResolvedKit` -- the tools available for this request |
| S0 | Environment | Source-internal -- umwelt brings its world model, others bring nothing |

Key insight: S0 (world/environment) is too rich and varied to standardize in the shared context. Each PolicySource brings its own world state internally. The shared context carries only what multiple sources need to make decisions.

The S3* audit trail uses lackpy's existing types rather than inventing new ones. `StepContext` already carries intent (the original prompt), the sequence of `ProgramState` entries (each with errors, validity, step traces with model/prompts), and the kit. `Trace` carries the tool-call execution history. Together they are the conversation log for a lackpy session.

### Future: conversation as taxon

The structured conversation history (system messages, user intent, assistant responses, tool calls, errors) maps naturally to an umwelt taxon vocabulary. Selectors like `error[mode="stdlib_leak"] > tool-call[name="open"]` could match against conversation history as a DOM. This is out of scope but validates the choice of using structured, typed history objects rather than flat strings.

## Core Types

### PolicyContext

```python
class PolicyContext(TypedDict, total=False):
    kit: Required[ResolvedKit]           # S1 -- operations
    principal: Principal                  # S5 -- identity
    model: ModelSpec                      # S4 -- inferencer
    session_id: str                      # S2 -- coordination
    history: StepContext | None           # S3* -- inference fold accumulator
    trace: Trace | None                  # S3* -- runtime execution trace
```

Only `kit` is required. All other fields are optional -- standalone lackpy may not have a principal or model spec. Sources check for the keys they need and degrade gracefully.

### Principal and ModelSpec

```python
@dataclass(frozen=True)
class Principal:
    id: str
    kind: str = "human"              # "human", "agent", "subagent"
    parent: str | None = None        # delegating principal, if subagent

@dataclass(frozen=True)
class ModelSpec:
    name: str
    temperature: float = 0.0
    context_window: int | None = None
    tier: str | None = None          # capability tier for policy decisions
```

`Principal` is deliberately thin. The richer identity model (permissions, trust levels, delegation chains) lives upstream in umwelt. `kind` and `parent` are sufficient for lackpy's policy decisions -- a subagent delegated by another agent might get a tighter tool set than a human-initiated request.

`ModelSpec` carries properties that affect policy -- Kibitzer sizes hints differently for small vs large models, umwelt could have model-specific constraints.

### PolicyResult

```python
@dataclass(frozen=True)
class PolicyResult:
    allowed_tools: frozenset[str] = frozenset()
    denied_tools: frozenset[str] = frozenset()
    tool_constraints: MappingProxyType[str, ToolConstraints] = EMPTY_CONSTRAINTS
    grade: Grade | None = None
    namespace_desc: str | None = None
    prompt_hints: tuple[str, ...] = ()
    docs: tuple[str, ...] = ()
    resolved: bool = False

    def replace(self, **changes: Any) -> PolicyResult:
        return dataclasses.replace(self, **changes)
```

Immutable. Sources call `current.replace(allowed_tools=..., ...)` to produce a new result. The `resolved` field controls chain propagation -- `True` stops the chain, `False` (the default) lets the next source run.

### ToolConstraints

```python
EMPTY_CONSTRAINTS: MappingProxyType[str, ToolConstraints] = MappingProxyType({})

@dataclass(frozen=True)
class ToolConstraints:
    max_level: int | None = None
    allow_patterns: tuple[str, ...] = ()
    deny_patterns: tuple[str, ...] = ()
```

Per-tool constraints, keyed by tool name in `PolicyResult.tool_constraints`. Populated primarily by UmweltPolicySource from capability-taxon entries.

### PolicySource protocol

```python
class PolicySource(Protocol):
    name: str
    priority: int

    def resolve(self, current: PolicyResult, context: PolicyContext) -> PolicyResult: ...
```

Each source receives the current accumulated result and the request context. Returns a new PolicyResult. The `resolved` field on the returned result controls whether the chain continues.

### PolicyLayer

```python
class PolicyLayer:
    def __init__(self) -> None:
        self._sources: list[PolicySource] = []

    def add_source(self, source: PolicySource) -> None:
        self._sources.append(source)
        self._sources.sort(key=lambda s: s.priority)

    def resolve(self, context: PolicyContext) -> PolicyResult:
        result = PolicyResult()  # empty seed
        for source in self._sources:
            result = source.resolve(result, context)
            if result.resolved:
                break
        return result
```

Sources are sorted by priority ascending (lowest first). The chain runs until a source sets `resolved=True` or all sources have run.

## PolicySource Implementations

### KitPolicySource (priority 0, always present)

The baseline. Translates a ResolvedKit into the initial PolicyResult.

```python
class KitPolicySource:
    name = "kit"
    priority = 0

    def __init__(self, toolbox: Toolbox) -> None:
        self._toolbox = toolbox

    def resolve(self, current: PolicyResult, context: PolicyContext) -> PolicyResult:
        kit = context["kit"]
        return PolicyResult(
            allowed_tools=frozenset(kit.tools.keys()),
            grade=kit.grade,
            namespace_desc=self._toolbox.format_description(kit.tools),
            resolved=False,
        )
```

Never sets `resolved=True` -- by itself it has no reason to stop other sources from enriching. Does not inspect `current` since it's always first in the chain.

### KibitzerPolicySource (priority 50, optional)

Adds coaching, prompt hints, and doc context. Reads the session history (S3*) to tailor hints to what's happened.

```python
class KibitzerPolicySource:
    name = "kibitzer"
    priority = 50

    def __init__(self, session: KibitzerSession) -> None:
        self._session = session

    def resolve(self, current: PolicyResult, context: PolicyContext) -> PolicyResult:
        hints = []
        docs = list(current.docs)

        history = context.get("history")
        if history and history.current:
            prog = history.current
            if not prog.valid and prog.errors:
                correction = self._session.get_correction_hints(
                    errors=prog.errors,
                    model=context.get("model"),
                    attempt=len(history.programs),
                )
                if correction.hints:
                    hints.extend(correction.hints)
                if correction.doc_context:
                    docs.append(correction.doc_context)

        ns_desc = current.namespace_desc
        if ns_desc and self._session.has_coaching():
            ns_desc = self._session.apply_coaching(ns_desc)

        return current.replace(
            namespace_desc=ns_desc,
            prompt_hints=current.prompt_hints + tuple(hints),
            docs=tuple(docs),
            resolved=False,
        )
```

Kibitzer never touches `allowed_tools` or `denied_tools`. It is not a policy authority -- it is a coaching layer that reads what's allowed and adds guidance.

### UmweltPolicySource (priority 100, optional)

Wraps a `PolicyEngine` session. Queries the capability taxon for tool-level policy. Gets the final say on tool access.

```python
class UmweltPolicySource:
    name = "umwelt"
    priority = 100

    def __init__(self, engine: PolicyEngine) -> None:
        self._engine = engine

    def resolve(self, current: PolicyResult, context: PolicyContext) -> PolicyResult:
        tool_entries = self._engine.resolve_all(type="tool")

        allowed = set()
        denied = set()
        constraints = {}

        for entry in tool_entries:
            name = entry["id"]
            if entry.get("visible") == "false":
                denied.add(name)
            else:
                allowed.add(name)
            if entry.get("max_level") or entry.get("allow_patterns"):
                constraints[name] = ToolConstraints(
                    max_level=_parse_int(entry.get("max_level")),
                    allow_patterns=tuple(entry.get("allow_patterns", ())),
                    deny_patterns=tuple(entry.get("deny_patterns", ())),
                )

        # Intersect -- umwelt restricts but cannot grant tools the kit lacks
        effective_allowed = current.allowed_tools & frozenset(allowed)
        effective_denied = current.denied_tools | frozenset(denied)

        return current.replace(
            allowed_tools=effective_allowed,
            denied_tools=effective_denied,
            tool_constraints=MappingProxyType(constraints),
            resolved=False,
        )
```

Key constraint: umwelt **restricts** the kit's tool set, it does not expand it. If the kit provides `[read_file, edit_file, bash]` and umwelt says `bash { visible: false; }`, the result is `[read_file, edit_file]`. But umwelt cannot add tools the kit didn't include -- kit resolution (S1) is the ground truth for what's available.

Note: at a separate layer (outside the policy engine), umwelt could define new tools and kits for lackpy. That's a world-taxon to capability-taxon bridge feeding into kit resolution, not policy resolution. Out of scope here.

### Resolution order

Sources run lowest-priority-first:

```
1. KitPolicySource (0)       --> establishes allowed_tools, grade, namespace_desc
2. KibitzerPolicySource (50)  --> adds hints, docs, coaching
3. UmweltPolicySource (100)   --> restricts tools, adds constraints
```

umwelt gets the final say on tool policy. If Kibitzer added hints for a tool that umwelt subsequently removes, those hints are harmless noise that the prompt builder filters out. On the next correction round, Kibitzer sees the post-umwelt tool set in the updated PolicyResult and adjusts its recommendations accordingly.

## Service Layer Wiring

### Current state

Kibitzer integration is scattered across `service.py`:
- `_init_kibitzer()` registers tools
- `_register_kibitzer_docs()` registers docs
- `_apply_kibitzer_hints()` enriches namespace descriptions
- Conditional `if self._kibitzer` checks in `delegate()`, `generate()`, and the correction chain

### Proposed state

```python
class LackpyService:
    def __init__(self, config=None, ...):
        self._toolbox = Toolbox()
        self._policy = PolicyLayer()

        # Always present
        self._policy.add_source(KitPolicySource(self._toolbox))

        # Optional integrations registered at init
        if kibitzer_session:
            self._policy.add_source(KibitzerPolicySource(kibitzer_session))
        if umwelt_engine:
            self._policy.add_source(UmweltPolicySource(umwelt_engine))

    def delegate(self, intent, kit_name, ...):
        kit = resolve_kit(kit_name, self._toolbox, extra_tools=extra_tools)

        context: PolicyContext = {
            "kit": kit,
            "principal": principal,
            "model": ModelSpec(name=model_name, temperature=temperature),
            "session_id": session_id,
        }

        policy = self._policy.resolve(context)

        # policy.allowed_tools   --> validation namespace
        # policy.namespace_desc  --> system prompt
        # policy.prompt_hints    --> injected into prompt
        # policy.tool_constraints --> runtime enforcement
        ...
```

The scattered conditional checks disappear. The service builds a context, calls `resolve()`, and uses the PolicyResult. Adding a new integration is `self._policy.add_source(...)`.

### Correction chain simplification

On retry, the service updates `context["history"]` with the latest `StepContext` and calls `self._policy.resolve(context)` again. Kibitzer sees the failure in the history and returns appropriate hints. No special-case `_apply_kibitzer_hints()` method needed.

## Module Layout

```
src/lackpy/policy/
    __init__.py          # public API: PolicyLayer, PolicyResult, PolicyContext, PolicySource
    types.py             # PolicyResult, PolicyContext, ToolConstraints, Principal, ModelSpec
    layer.py             # PolicyLayer, PolicySource protocol
    sources/
        __init__.py
        kit.py           # KitPolicySource
        kibitzer.py      # KibitzerPolicySource
        umwelt.py        # UmweltPolicySource
```

`sources/kibitzer.py` and `sources/umwelt.py` import their dependencies conditionally or accept them as constructor arguments, avoiding hard imports of optional packages.

## Configurations

| Setup | Sources registered | Behavior |
|-------|-------------------|----------|
| Standalone | KitPolicySource | Kit tools allowed, grade computed, descriptions formatted |
| + Kibitzer | Kit + Kibitzer | Above + coaching, prompt hints, doc context on failures |
| + umwelt | Kit + Kibitzer + Umwelt | Above + world-model tool restrictions, per-tool constraints |
| umwelt only | Kit + Umwelt | Kit baseline + policy restrictions, no coaching |

## Open Questions

1. **PolicyResult field filtering** -- when umwelt removes a tool, should the PolicyLayer automatically strip prompt_hints and docs that reference that tool? Or leave it to the prompt builder?
2. **Source hot-reload** -- should `PolicyLayer.add_source()` be callable after init (e.g., Kibitzer connects mid-session)? Current design supports this since sources are just a sorted list.
3. **Tracing/observability** -- should the PolicyLayer record which source set which field? Useful for debugging but adds complexity. Could be a debug mode that wraps each source call.
