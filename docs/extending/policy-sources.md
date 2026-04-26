# Writing Policy Sources

A **PolicySource** participates in the policy resolution chain. By writing a custom source you can inject constraints, hints, or metadata from any external system — audit logs, feature flags, role-based access control — without modifying lackpy's service layer.

---

## Protocol

A PolicySource is any object with `name`, `priority`, and a `resolve` method:

```python
from lackpy.policy import PolicyResult, PolicyContext


class MySource:
    name = "my_source"
    priority = 75  # between Kibitzer (50) and umwelt (100)

    def resolve(self, current: PolicyResult, context: PolicyContext) -> PolicyResult:
        # Read current state, return a new PolicyResult
        return current.replace(...)
```

There is no abstract base class — Python duck typing is used. The `PolicySource` protocol is `runtime_checkable` if you want to verify conformance with `isinstance()`.

---

## Complete example — role-based tool restrictions

This source restricts tools based on the principal's role, looked up from an external permissions service.

```python
from lackpy.policy import PolicyResult, PolicyContext


class RoleBasedSource:
    """Restricts tools based on principal role from a permissions service."""

    name = "rbac"
    priority = 80

    def __init__(self, permissions_client):
        self._client = permissions_client

    def resolve(self, current: PolicyResult, context: PolicyContext) -> PolicyResult:
        principal = context.get("principal")
        if principal is None:
            return current

        allowed = self._client.get_allowed_tools(
            principal_id=principal.id,
            principal_kind=principal.kind,
        )
        if allowed is None:
            return current

        # Intersect — can only restrict, never grant
        restricted = current.allowed_tools & frozenset(allowed)
        denied = current.allowed_tools - restricted

        return current.replace(
            allowed_tools=restricted,
            denied_tools=current.denied_tools | denied,
        )
```

### Registration

```python
from lackpy import LackpyService
from lackpy.policy.types import Principal

svc = LackpyService()

# Register the custom source
svc._policy.add_source(RoleBasedSource(permissions_client))

# Now resolve with a principal
from lackpy.kit.registry import resolve_kit
kit = resolve_kit(["read_file", "edit_file", "write_file"], svc.toolbox)

result = svc._policy.resolve({
    "kit": kit,
    "principal": Principal(id="intern-42", kind="human"),
})
# result.allowed_tools might be frozenset({"read_file"})
```

---

## Design rules

### Restrict, don't grant

Sources should **intersect** with the current `allowed_tools`, never add tools that weren't already there. Kit resolution (S1) is the ground truth for what's operationally available. Policy sources control *access to* those tools, not *existence of* them.

### Immutable results

`PolicyResult` is frozen. Always use `current.replace(...)` to produce a new instance. Never try to mutate fields in place.

### Use `resolved` sparingly

Setting `resolved=True` stops the chain — no subsequent sources run. This is appropriate when your source has complete authority (e.g., an emergency kill switch that denies all tools). For most sources, leave `resolved=False` so other sources can enrich the result.

### Degrade gracefully

Context fields other than `kit` are optional. Check with `context.get("principal")` rather than `context["principal"]`. A source that requires a model spec should silently pass through when none is provided.

---

## Priority guidelines

| Range | Intended use | Examples |
|-------|-------------|----------|
| 0–10 | Baseline sources | KitPolicySource |
| 20–40 | Enrichment / analytics | Logging, metrics collection |
| 50–60 | Coaching / hints | KibitzerPolicySource |
| 70–90 | Access control | RBAC, feature flags, rate limiting |
| 100+ | Enforcement / override | UmweltPolicySource, emergency kill switch |

Sources at the same priority run in registration order.

---

## Testing

Test sources in isolation by constructing a `PolicyResult` and `PolicyContext` directly:

```python
from lackpy.policy import PolicyResult, PolicyContext
from lackpy.kit.registry import ResolvedKit
from lackpy.lang.grader import Grade


def test_my_source_restricts_tools():
    source = MySource(config={"blocked": ["edit_file"]})
    kit = ResolvedKit(tools={}, callables={}, grade=Grade(w=0, d=0), description="")
    current = PolicyResult(allowed_tools=frozenset({"read_file", "edit_file"}))

    result = source.resolve(current, {"kit": kit})

    assert result.allowed_tools == frozenset({"read_file"})
    assert "edit_file" in result.denied_tools
```

For chain-level integration tests, build a `PolicyLayer` with your source and the built-in sources, and verify the full resolution produces the expected result. See `tests/policy/test_chain_integration.py` for examples.
