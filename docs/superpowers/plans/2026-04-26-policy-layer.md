# PolicyLayer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a PolicyLayer abstraction that consolidates lackpy's scattered kit + Kibitzer policy logic into an ordered chain of immutable PolicySources, enabling progressive enrichment (kit → Kibitzer → umwelt).

**Architecture:** PolicyLayer holds ordered PolicySources. Each source receives the current PolicyResult and a PolicyContext (TypedDict with kit, principal, model, session history), and returns a new frozen PolicyResult. Sources run lowest-priority-first. The `resolved` flag on the result controls chain propagation. The service layer replaces its scattered `if self._kibitzer` conditionals with a single `self._policy.resolve(context)` call.

**Tech Stack:** Python 3.11+, dataclasses (frozen), typing (Protocol, TypedDict), pytest, MappingProxyType from types module.

**Spec:** `docs/superpowers/specs/2026-04-26-policy-layer-design.md`

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `src/lackpy/policy/__init__.py` | Public API re-exports |
| Create | `src/lackpy/policy/types.py` | PolicyResult, PolicyContext, ToolConstraints, Principal, ModelSpec |
| Create | `src/lackpy/policy/layer.py` | PolicyLayer class, PolicySource protocol |
| Create | `src/lackpy/policy/sources/__init__.py` | Empty init |
| Create | `src/lackpy/policy/sources/kit.py` | KitPolicySource |
| Create | `src/lackpy/policy/sources/kibitzer.py` | KibitzerPolicySource |
| Create | `src/lackpy/policy/sources/umwelt.py` | UmweltPolicySource |
| Create | `tests/policy/__init__.py` | Test package |
| Create | `tests/policy/test_types.py` | Tests for frozen types |
| Create | `tests/policy/test_layer.py` | Tests for PolicyLayer chain |
| Create | `tests/policy/test_kit_source.py` | Tests for KitPolicySource |
| Create | `tests/policy/test_kibitzer_source.py` | Tests for KibitzerPolicySource |
| Create | `tests/policy/test_umwelt_source.py` | Tests for UmweltPolicySource |
| Modify | `src/lackpy/service.py` | Replace Kibitzer conditionals with PolicyLayer |
| Modify | `src/lackpy/__init__.py` | Export PolicyLayer, PolicyResult |
| Create | `tests/policy/test_service_integration.py` | Service + PolicyLayer integration |

---

### Task 1: Core types — PolicyResult, ToolConstraints, Principal, ModelSpec

**Files:**
- Create: `src/lackpy/policy/__init__.py`
- Create: `src/lackpy/policy/types.py`
- Create: `tests/policy/__init__.py`
- Create: `tests/policy/test_types.py`

- [ ] **Step 1: Write failing tests for frozen types**

Create `tests/policy/__init__.py` (empty file) and `tests/policy/test_types.py`:

```python
"""Tests for policy core types."""

from __future__ import annotations

import dataclasses
import pytest
from types import MappingProxyType


class TestToolConstraints:
    def test_frozen(self):
        from lackpy.policy.types import ToolConstraints
        tc = ToolConstraints(max_level=2)
        with pytest.raises(dataclasses.FrozenInstanceError):
            tc.max_level = 3

    def test_defaults(self):
        from lackpy.policy.types import ToolConstraints
        tc = ToolConstraints()
        assert tc.max_level is None
        assert tc.allow_patterns == ()
        assert tc.deny_patterns == ()

    def test_with_patterns(self):
        from lackpy.policy.types import ToolConstraints
        tc = ToolConstraints(
            max_level=1,
            allow_patterns=("src/**/*.py",),
            deny_patterns=("*.secret",),
        )
        assert tc.max_level == 1
        assert tc.allow_patterns == ("src/**/*.py",)


class TestPrincipal:
    def test_human_default(self):
        from lackpy.policy.types import Principal
        p = Principal(id="teague")
        assert p.kind == "human"
        assert p.parent is None

    def test_subagent(self):
        from lackpy.policy.types import Principal
        p = Principal(id="worker-1", kind="subagent", parent="orchestrator")
        assert p.kind == "subagent"
        assert p.parent == "orchestrator"

    def test_frozen(self):
        from lackpy.policy.types import Principal
        p = Principal(id="teague")
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.id = "other"


class TestModelSpec:
    def test_defaults(self):
        from lackpy.policy.types import ModelSpec
        m = ModelSpec(name="qwen2.5-coder:1.5b")
        assert m.temperature == 0.0
        assert m.context_window is None
        assert m.tier is None

    def test_full(self):
        from lackpy.policy.types import ModelSpec
        m = ModelSpec(name="claude-haiku", temperature=0.2, context_window=200000, tier="large")
        assert m.tier == "large"


class TestPolicyResult:
    def test_empty_defaults(self):
        from lackpy.policy.types import PolicyResult
        r = PolicyResult()
        assert r.allowed_tools == frozenset()
        assert r.denied_tools == frozenset()
        assert r.tool_constraints == MappingProxyType({})
        assert r.grade is None
        assert r.namespace_desc is None
        assert r.prompt_hints == ()
        assert r.docs == ()
        assert r.resolved is False

    def test_frozen(self):
        from lackpy.policy.types import PolicyResult
        r = PolicyResult()
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.resolved = True

    def test_replace_returns_new_instance(self):
        from lackpy.policy.types import PolicyResult
        r1 = PolicyResult(allowed_tools=frozenset({"read_file"}))
        r2 = r1.replace(allowed_tools=frozenset({"read_file", "edit_file"}))
        assert r1.allowed_tools == frozenset({"read_file"})
        assert r2.allowed_tools == frozenset({"read_file", "edit_file"})
        assert r1 is not r2

    def test_replace_preserves_unmodified_fields(self):
        from lackpy.policy.types import PolicyResult
        r1 = PolicyResult(
            allowed_tools=frozenset({"read_file"}),
            namespace_desc="tools available",
            prompt_hints=("hint1",),
        )
        r2 = r1.replace(resolved=True)
        assert r2.allowed_tools == frozenset({"read_file"})
        assert r2.namespace_desc == "tools available"
        assert r2.prompt_hints == ("hint1",)
        assert r2.resolved is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/policy/test_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lackpy.policy'`

- [ ] **Step 3: Create the policy package and types module**

Create `src/lackpy/policy/__init__.py`:

```python
"""Policy layer: ordered resolution of tool constraints from multiple sources."""

from .types import (
    EMPTY_CONSTRAINTS,
    ModelSpec,
    PolicyContext,
    PolicyResult,
    Principal,
    ToolConstraints,
)

__all__ = [
    "EMPTY_CONSTRAINTS",
    "ModelSpec",
    "PolicyContext",
    "PolicyResult",
    "Principal",
    "ToolConstraints",
]
```

Create `src/lackpy/policy/types.py`:

```python
"""Core policy types: immutable result, context, and value objects."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Required, TypedDict

from types import MappingProxyType

if TYPE_CHECKING:
    from ..kit.registry import ResolvedKit
    from ..infer.context import StepContext
    from ..run.trace import Trace
    from ..lang.grader import Grade


@dataclass(frozen=True)
class ToolConstraints:
    """Per-tool constraints from policy resolution."""

    max_level: int | None = None
    allow_patterns: tuple[str, ...] = ()
    deny_patterns: tuple[str, ...] = ()


EMPTY_CONSTRAINTS: MappingProxyType[str, ToolConstraints] = MappingProxyType({})


@dataclass(frozen=True)
class Principal:
    """Identity of the requesting entity (S5)."""

    id: str
    kind: str = "human"
    parent: str | None = None


@dataclass(frozen=True)
class ModelSpec:
    """Inferencer properties that affect policy decisions (S4)."""

    name: str
    temperature: float = 0.0
    context_window: int | None = None
    tier: str | None = None


@dataclass(frozen=True)
class PolicyResult:
    """Immutable result of policy resolution.

    Sources produce new instances via replace(). The resolved flag
    controls chain propagation: True stops the chain.
    """

    allowed_tools: frozenset[str] = frozenset()
    denied_tools: frozenset[str] = frozenset()
    tool_constraints: MappingProxyType[str, ToolConstraints] = field(
        default_factory=lambda: EMPTY_CONSTRAINTS
    )
    grade: Any = None  # Grade, but Any to avoid circular import at runtime
    namespace_desc: str | None = None
    prompt_hints: tuple[str, ...] = ()
    docs: tuple[str, ...] = ()
    resolved: bool = False

    def replace(self, **changes: Any) -> PolicyResult:
        return dataclasses.replace(self, **changes)


class PolicyContext(TypedDict, total=False):
    """Shared request context passed through the policy chain.

    VSM-informed: S1 (kit), S2 (session_id), S4 (model),
    S5 (principal), S3* (history, trace).
    """

    kit: Required[Any]  # ResolvedKit, Any to avoid circular import
    principal: Principal
    model: ModelSpec
    session_id: str
    history: Any  # StepContext | None
    trace: Any  # Trace | None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/policy/test_types.py -v`
Expected: All 11 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/lackpy/policy/__init__.py src/lackpy/policy/types.py tests/policy/__init__.py tests/policy/test_types.py
git commit -m "feat(policy): core types — PolicyResult, ToolConstraints, Principal, ModelSpec"
```

---

### Task 2: PolicySource protocol and PolicyLayer

**Files:**
- Create: `src/lackpy/policy/layer.py`
- Create: `tests/policy/test_layer.py`
- Modify: `src/lackpy/policy/__init__.py`

- [ ] **Step 1: Write failing tests for PolicyLayer**

Create `tests/policy/test_layer.py`:

```python
"""Tests for PolicyLayer resolution chain."""

from __future__ import annotations

import pytest

from lackpy.policy.types import PolicyResult, PolicyContext
from lackpy.policy.layer import PolicyLayer, PolicySource
from lackpy.kit.registry import ResolvedKit
from lackpy.lang.grader import Grade


class StubSource:
    """A configurable stub PolicySource for testing."""

    def __init__(self, name: str, priority: int, transform=None, mark_resolved=False):
        self.name = name
        self.priority = priority
        self._transform = transform
        self._mark_resolved = mark_resolved
        self.called_with: list[tuple[PolicyResult, PolicyContext]] = []

    def resolve(self, current: PolicyResult, context: PolicyContext) -> PolicyResult:
        self.called_with.append((current, context))
        if self._transform:
            result = self._transform(current, context)
        else:
            result = current
        if self._mark_resolved:
            result = result.replace(resolved=True)
        return result


@pytest.fixture
def minimal_kit():
    """A minimal ResolvedKit for context construction."""
    return ResolvedKit(
        tools={},
        callables={},
        grade=Grade(w=0, d=0),
        description="",
    )


class TestPolicyLayerOrdering:
    def test_sources_run_lowest_priority_first(self, minimal_kit):
        call_order = []

        def make_transform(label):
            def transform(current, context):
                call_order.append(label)
                return current.replace(
                    prompt_hints=current.prompt_hints + (label,),
                )
            return transform

        layer = PolicyLayer()
        layer.add_source(StubSource("high", priority=100, transform=make_transform("high")))
        layer.add_source(StubSource("low", priority=0, transform=make_transform("low")))
        layer.add_source(StubSource("mid", priority=50, transform=make_transform("mid")))

        result = layer.resolve({"kit": minimal_kit})
        assert call_order == ["low", "mid", "high"]
        assert result.prompt_hints == ("low", "mid", "high")

    def test_resolved_stops_chain(self, minimal_kit):
        call_order = []

        def track(label):
            def transform(current, context):
                call_order.append(label)
                return current.replace(prompt_hints=current.prompt_hints + (label,))
            return transform

        layer = PolicyLayer()
        layer.add_source(StubSource("first", priority=0, transform=track("first")))
        layer.add_source(StubSource("stopper", priority=50, transform=track("stopper"), mark_resolved=True))
        layer.add_source(StubSource("skipped", priority=100, transform=track("skipped")))

        result = layer.resolve({"kit": minimal_kit})
        assert call_order == ["first", "stopper"]
        assert "skipped" not in result.prompt_hints
        assert result.resolved is True


class TestPolicyLayerEmpty:
    def test_no_sources_returns_empty_result(self, minimal_kit):
        layer = PolicyLayer()
        result = layer.resolve({"kit": minimal_kit})
        assert result.allowed_tools == frozenset()
        assert result.resolved is False

    def test_single_source(self, minimal_kit):
        def set_tools(current, context):
            return current.replace(allowed_tools=frozenset({"read_file"}))

        layer = PolicyLayer()
        layer.add_source(StubSource("only", priority=0, transform=set_tools))

        result = layer.resolve({"kit": minimal_kit})
        assert result.allowed_tools == frozenset({"read_file"})


class TestPolicyLayerAddSource:
    def test_add_source_maintains_sort(self, minimal_kit):
        layer = PolicyLayer()
        layer.add_source(StubSource("c", priority=100))
        layer.add_source(StubSource("a", priority=0))
        layer.add_source(StubSource("b", priority=50))

        names = [s.name for s in layer._sources]
        assert names == ["a", "b", "c"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/policy/test_layer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lackpy.policy.layer'`

- [ ] **Step 3: Implement PolicyLayer and PolicySource protocol**

Create `src/lackpy/policy/layer.py`:

```python
"""PolicyLayer: ordered resolution chain of PolicySources."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .types import PolicyContext, PolicyResult


@runtime_checkable
class PolicySource(Protocol):
    """A source of policy decisions in the resolution chain."""

    name: str
    priority: int

    def resolve(self, current: PolicyResult, context: PolicyContext) -> PolicyResult: ...


class PolicyLayer:
    """Ordered chain of PolicySources that produces a PolicyResult.

    Sources are sorted by priority ascending (lowest first).
    Each source receives the accumulated PolicyResult and the request
    context. If a source returns a result with resolved=True, the
    chain stops.
    """

    def __init__(self) -> None:
        self._sources: list[PolicySource] = []

    def add_source(self, source: PolicySource) -> None:
        self._sources.append(source)
        self._sources.sort(key=lambda s: s.priority)

    def resolve(self, context: PolicyContext) -> PolicyResult:
        result = PolicyResult()
        for source in self._sources:
            result = source.resolve(result, context)
            if result.resolved:
                break
        return result
```

- [ ] **Step 4: Update policy __init__.py exports**

Replace `src/lackpy/policy/__init__.py` with:

```python
"""Policy layer: ordered resolution of tool constraints from multiple sources."""

from .layer import PolicyLayer, PolicySource
from .types import (
    EMPTY_CONSTRAINTS,
    ModelSpec,
    PolicyContext,
    PolicyResult,
    Principal,
    ToolConstraints,
)

__all__ = [
    "EMPTY_CONSTRAINTS",
    "ModelSpec",
    "PolicyContext",
    "PolicyLayer",
    "PolicyResult",
    "PolicySource",
    "Principal",
    "ToolConstraints",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/policy/test_layer.py tests/policy/test_types.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/lackpy/policy/layer.py src/lackpy/policy/__init__.py tests/policy/test_layer.py
git commit -m "feat(policy): PolicyLayer chain with priority ordering and resolved-stops-propagation"
```

---

### Task 3: KitPolicySource

**Files:**
- Create: `src/lackpy/policy/sources/__init__.py`
- Create: `src/lackpy/policy/sources/kit.py`
- Create: `tests/policy/test_kit_source.py`

- [ ] **Step 1: Write failing tests for KitPolicySource**

Create `tests/policy/test_kit_source.py`:

```python
"""Tests for KitPolicySource."""

from __future__ import annotations

import pytest

from lackpy.policy.types import PolicyResult, PolicyContext
from lackpy.policy.sources.kit import KitPolicySource
from lackpy.kit.registry import ResolvedKit, resolve_kit
from lackpy.kit.toolbox import Toolbox, ToolSpec, ArgSpec
from lackpy.kit.providers.builtin import BuiltinProvider
from lackpy.lang.grader import Grade


@pytest.fixture
def toolbox():
    tb = Toolbox()
    tb.register_provider(BuiltinProvider())
    for name, desc, grade in [
        ("read_file", "Read file contents", 1),
        ("find_files", "Find files matching pattern", 1),
        ("edit_file", "Edit a file", 3),
    ]:
        tb.register_tool(ToolSpec(
            name=name, provider="builtin", description=desc,
            args=[ArgSpec(name="path", type="str")],
            returns="str", grade_w=grade, effects_ceiling=grade,
        ))
    return tb


@pytest.fixture
def kit(toolbox):
    return resolve_kit(["read_file", "find_files"], toolbox)


class TestKitPolicySource:
    def test_sets_allowed_tools_from_kit(self, toolbox, kit):
        source = KitPolicySource(toolbox)
        context: PolicyContext = {"kit": kit}
        result = source.resolve(PolicyResult(), context)
        assert result.allowed_tools == frozenset({"read_file", "find_files"})

    def test_sets_grade_from_kit(self, toolbox, kit):
        source = KitPolicySource(toolbox)
        context: PolicyContext = {"kit": kit}
        result = source.resolve(PolicyResult(), context)
        assert result.grade.w == 1
        assert result.grade.d == 1

    def test_sets_namespace_desc(self, toolbox, kit):
        source = KitPolicySource(toolbox)
        context: PolicyContext = {"kit": kit}
        result = source.resolve(PolicyResult(), context)
        assert result.namespace_desc is not None
        assert "read_file" in result.namespace_desc

    def test_never_marks_resolved(self, toolbox, kit):
        source = KitPolicySource(toolbox)
        context: PolicyContext = {"kit": kit}
        result = source.resolve(PolicyResult(), context)
        assert result.resolved is False

    def test_priority_is_zero(self, toolbox):
        source = KitPolicySource(toolbox)
        assert source.priority == 0

    def test_name_is_kit(self, toolbox):
        source = KitPolicySource(toolbox)
        assert source.name == "kit"

    def test_high_grade_kit(self, toolbox):
        kit = resolve_kit(["read_file", "edit_file"], toolbox)
        source = KitPolicySource(toolbox)
        context: PolicyContext = {"kit": kit}
        result = source.resolve(PolicyResult(), context)
        assert result.grade.w == 3
        assert result.allowed_tools == frozenset({"read_file", "edit_file"})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/policy/test_kit_source.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lackpy.policy.sources'`

- [ ] **Step 3: Implement KitPolicySource**

Create `src/lackpy/policy/sources/__init__.py`:

```python
"""Policy source implementations."""
```

Create `src/lackpy/policy/sources/kit.py`:

```python
"""KitPolicySource: baseline policy from resolved kit."""

from __future__ import annotations

from typing import Any

from ..types import PolicyContext, PolicyResult


class KitPolicySource:
    """Translates a ResolvedKit into the initial PolicyResult.

    Always present, lowest priority. Establishes the baseline
    allowed_tools, grade, and namespace_desc.
    """

    name = "kit"
    priority = 0

    def __init__(self, toolbox: Any) -> None:
        self._toolbox = toolbox

    def resolve(self, current: PolicyResult, context: PolicyContext) -> PolicyResult:
        kit = context["kit"]
        return PolicyResult(
            allowed_tools=frozenset(kit.tools.keys()),
            grade=kit.grade,
            namespace_desc=self._toolbox.format_description(list(kit.tools.keys())),
            resolved=False,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/policy/test_kit_source.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/lackpy/policy/sources/__init__.py src/lackpy/policy/sources/kit.py tests/policy/test_kit_source.py
git commit -m "feat(policy): KitPolicySource — baseline tool policy from resolved kit"
```

---

### Task 4: KibitzerPolicySource

**Files:**
- Create: `src/lackpy/policy/sources/kibitzer.py`
- Create: `tests/policy/test_kibitzer_source.py`

- [ ] **Step 1: Write failing tests for KibitzerPolicySource**

Create `tests/policy/test_kibitzer_source.py`:

```python
"""Tests for KibitzerPolicySource."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from lackpy.policy.types import PolicyResult, PolicyContext
from lackpy.policy.sources.kibitzer import KibitzerPolicySource
from lackpy.kit.registry import ResolvedKit
from lackpy.lang.grader import Grade
from lackpy.infer.context import StepContext, ProgramState, StepTrace


class FakeHints:
    def __init__(self, hints=None, doc_context=None):
        self.hints = hints or []
        self.doc_context = doc_context


class FakeKibitzerSession:
    def __init__(self, hints=None, doc_context=None, coaching=None):
        self._hints = FakeHints(hints, doc_context)
        self._coaching = coaching
        self.hint_calls: list[dict] = []

    def get_correction_hints(self, errors=None, model=None, attempt=None):
        self.hint_calls.append({"errors": errors, "model": model, "attempt": attempt})
        return self._hints

    def has_coaching(self):
        return self._coaching is not None

    def apply_coaching(self, desc):
        return desc + self._coaching


@pytest.fixture
def empty_kit():
    return ResolvedKit(
        tools={}, callables={}, grade=Grade(w=0, d=0), description="",
    )


def _make_step_trace():
    return StepTrace(
        step_name="test", provider_name=None, model=None,
        system_prompt=None, user_prompt=None, raw_output=None,
        duration_ms=0.0,
    )


class TestKibitzerPolicySourceBasic:
    def test_name_and_priority(self):
        source = KibitzerPolicySource(FakeKibitzerSession())
        assert source.name == "kibitzer"
        assert source.priority == 50

    def test_no_history_passes_through(self, empty_kit):
        source = KibitzerPolicySource(FakeKibitzerSession())
        current = PolicyResult(
            allowed_tools=frozenset({"read_file"}),
            namespace_desc="tools: read_file",
        )
        context: PolicyContext = {"kit": empty_kit}
        result = source.resolve(current, context)
        assert result.allowed_tools == frozenset({"read_file"})
        assert result.namespace_desc == "tools: read_file"
        assert result.resolved is False

    def test_never_modifies_allowed_tools(self, empty_kit):
        session = FakeKibitzerSession(hints=["use read_file instead"])
        source = KibitzerPolicySource(session)
        history = StepContext(intent="test", kit=empty_kit)
        history.programs.append(ProgramState(
            program="open('f')", intent="test", kit=empty_kit,
            valid=False, errors=["Forbidden name: 'open'"],
            trace=_make_step_trace(),
        ))
        current = PolicyResult(allowed_tools=frozenset({"read_file"}))
        context: PolicyContext = {"kit": empty_kit, "history": history}
        result = source.resolve(current, context)
        assert result.allowed_tools == frozenset({"read_file"})


class TestKibitzerPolicySourceHints:
    def test_adds_hints_on_failure(self, empty_kit):
        session = FakeKibitzerSession(hints=["use read_file instead of open"])
        source = KibitzerPolicySource(session)
        history = StepContext(intent="test", kit=empty_kit)
        history.programs.append(ProgramState(
            program="open('f')", intent="test", kit=empty_kit,
            valid=False, errors=["Forbidden name: 'open'"],
            trace=_make_step_trace(),
        ))
        current = PolicyResult(allowed_tools=frozenset({"read_file"}))
        context: PolicyContext = {"kit": empty_kit, "history": history}
        result = source.resolve(current, context)
        assert "use read_file instead of open" in result.prompt_hints

    def test_adds_doc_context(self, empty_kit):
        session = FakeKibitzerSession(doc_context="Signature: read_file(path: str) -> str")
        source = KibitzerPolicySource(session)
        history = StepContext(intent="test", kit=empty_kit)
        history.programs.append(ProgramState(
            program="open('f')", intent="test", kit=empty_kit,
            valid=False, errors=["Forbidden name: 'open'"],
            trace=_make_step_trace(),
        ))
        current = PolicyResult()
        context: PolicyContext = {"kit": empty_kit, "history": history}
        result = source.resolve(current, context)
        assert "Signature: read_file(path: str) -> str" in result.docs

    def test_no_hints_on_valid_program(self, empty_kit):
        session = FakeKibitzerSession(hints=["should not appear"])
        source = KibitzerPolicySource(session)
        history = StepContext(intent="test", kit=empty_kit)
        history.programs.append(ProgramState(
            program="x = 1", intent="test", kit=empty_kit,
            valid=True, errors=[],
            trace=_make_step_trace(),
        ))
        current = PolicyResult()
        context: PolicyContext = {"kit": empty_kit, "history": history}
        result = source.resolve(current, context)
        assert result.prompt_hints == ()


class TestKibitzerPolicySourceCoaching:
    def test_applies_coaching_to_namespace_desc(self, empty_kit):
        session = FakeKibitzerSession(coaching="\n- Never use open()")
        source = KibitzerPolicySource(session)
        current = PolicyResult(namespace_desc="tools: read_file")
        context: PolicyContext = {"kit": empty_kit}
        result = source.resolve(current, context)
        assert result.namespace_desc == "tools: read_file\n- Never use open()"

    def test_no_coaching_when_no_desc(self, empty_kit):
        session = FakeKibitzerSession(coaching="\n- coaching")
        source = KibitzerPolicySource(session)
        current = PolicyResult(namespace_desc=None)
        context: PolicyContext = {"kit": empty_kit}
        result = source.resolve(current, context)
        assert result.namespace_desc is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/policy/test_kibitzer_source.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lackpy.policy.sources.kibitzer'`

- [ ] **Step 3: Implement KibitzerPolicySource**

Create `src/lackpy/policy/sources/kibitzer.py`:

```python
"""KibitzerPolicySource: coaching, hints, and doc context from Kibitzer."""

from __future__ import annotations

from typing import Any

from ..types import PolicyContext, PolicyResult


class KibitzerPolicySource:
    """Adds prompt hints, doc context, and coaching from a Kibitzer session.

    Never modifies allowed_tools or denied_tools — Kibitzer is a
    coaching layer, not a policy authority.
    """

    name = "kibitzer"
    priority = 50

    def __init__(self, session: Any) -> None:
        self._session = session

    def resolve(self, current: PolicyResult, context: PolicyContext) -> PolicyResult:
        hints: list[str] = []
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

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/policy/test_kibitzer_source.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/lackpy/policy/sources/kibitzer.py tests/policy/test_kibitzer_source.py
git commit -m "feat(policy): KibitzerPolicySource — coaching and hints without policy authority"
```

---

### Task 5: UmweltPolicySource

**Files:**
- Create: `src/lackpy/policy/sources/umwelt.py`
- Create: `tests/policy/test_umwelt_source.py`

- [ ] **Step 1: Write failing tests for UmweltPolicySource**

Create `tests/policy/test_umwelt_source.py`:

```python
"""Tests for UmweltPolicySource."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from lackpy.policy.types import PolicyResult, PolicyContext, ToolConstraints
from lackpy.policy.sources.umwelt import UmweltPolicySource
from lackpy.kit.registry import ResolvedKit
from lackpy.lang.grader import Grade


class FakePolicyEngine:
    """Stub PolicyEngine that returns pre-configured tool entries."""

    def __init__(self, tool_entries: list[dict]):
        self._entries = tool_entries

    def resolve_all(self, type: str = "tool"):
        return self._entries


@pytest.fixture
def kit():
    return ResolvedKit(
        tools={}, callables={}, grade=Grade(w=0, d=0), description="",
    )


class TestUmweltPolicySourceBasic:
    def test_name_and_priority(self):
        source = UmweltPolicySource(FakePolicyEngine([]))
        assert source.name == "umwelt"
        assert source.priority == 100

    def test_restricts_to_kit_intersection(self, kit):
        engine = FakePolicyEngine([
            {"id": "read_file", "visible": "true"},
            {"id": "edit_file", "visible": "true"},
            {"id": "bash", "visible": "true"},
        ])
        source = UmweltPolicySource(engine)
        current = PolicyResult(allowed_tools=frozenset({"read_file", "edit_file"}))
        context: PolicyContext = {"kit": kit}
        result = source.resolve(current, context)
        assert result.allowed_tools == frozenset({"read_file", "edit_file"})
        assert "bash" not in result.allowed_tools

    def test_denies_tools_marked_invisible(self, kit):
        engine = FakePolicyEngine([
            {"id": "read_file", "visible": "true"},
            {"id": "edit_file", "visible": "false"},
        ])
        source = UmweltPolicySource(engine)
        current = PolicyResult(
            allowed_tools=frozenset({"read_file", "edit_file"}),
        )
        context: PolicyContext = {"kit": kit}
        result = source.resolve(current, context)
        assert result.allowed_tools == frozenset({"read_file"})
        assert "edit_file" in result.denied_tools

    def test_cannot_grant_tools_kit_lacks(self, kit):
        engine = FakePolicyEngine([
            {"id": "read_file", "visible": "true"},
            {"id": "bash", "visible": "true"},
        ])
        source = UmweltPolicySource(engine)
        current = PolicyResult(allowed_tools=frozenset({"read_file"}))
        context: PolicyContext = {"kit": kit}
        result = source.resolve(current, context)
        assert result.allowed_tools == frozenset({"read_file"})
        assert "bash" not in result.allowed_tools

    def test_never_marks_resolved(self, kit):
        source = UmweltPolicySource(FakePolicyEngine([]))
        context: PolicyContext = {"kit": kit}
        result = source.resolve(PolicyResult(), context)
        assert result.resolved is False


class TestUmweltPolicySourceConstraints:
    def test_sets_tool_constraints(self, kit):
        engine = FakePolicyEngine([
            {
                "id": "read_file",
                "visible": "true",
                "max_level": "2",
                "allow_patterns": ["src/**/*.py"],
                "deny_patterns": ["*.secret"],
            },
        ])
        source = UmweltPolicySource(engine)
        current = PolicyResult(allowed_tools=frozenset({"read_file"}))
        context: PolicyContext = {"kit": kit}
        result = source.resolve(current, context)
        assert "read_file" in result.tool_constraints
        tc = result.tool_constraints["read_file"]
        assert tc.max_level == 2
        assert tc.allow_patterns == ("src/**/*.py",)
        assert tc.deny_patterns == ("*.secret",)

    def test_no_constraints_when_not_specified(self, kit):
        engine = FakePolicyEngine([
            {"id": "read_file", "visible": "true"},
        ])
        source = UmweltPolicySource(engine)
        current = PolicyResult(allowed_tools=frozenset({"read_file"}))
        context: PolicyContext = {"kit": kit}
        result = source.resolve(current, context)
        assert "read_file" not in result.tool_constraints

    def test_merges_denied_with_existing(self, kit):
        engine = FakePolicyEngine([
            {"id": "bash", "visible": "false"},
        ])
        source = UmweltPolicySource(engine)
        current = PolicyResult(
            allowed_tools=frozenset({"read_file", "bash"}),
            denied_tools=frozenset({"rm_rf"}),
        )
        context: PolicyContext = {"kit": kit}
        result = source.resolve(current, context)
        assert "rm_rf" in result.denied_tools
        assert "bash" in result.denied_tools


class TestUmweltPolicySourcePreservesOtherFields:
    def test_preserves_hints_and_docs(self, kit):
        engine = FakePolicyEngine([
            {"id": "read_file", "visible": "true"},
        ])
        source = UmweltPolicySource(engine)
        current = PolicyResult(
            allowed_tools=frozenset({"read_file"}),
            prompt_hints=("use read_file for files",),
            docs=("docs/tools/read_file.md",),
            namespace_desc="read_file(path) -> str",
        )
        context: PolicyContext = {"kit": kit}
        result = source.resolve(current, context)
        assert result.prompt_hints == ("use read_file for files",)
        assert result.docs == ("docs/tools/read_file.md",)
        assert result.namespace_desc == "read_file(path) -> str"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/policy/test_umwelt_source.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lackpy.policy.sources.umwelt'`

- [ ] **Step 3: Implement UmweltPolicySource**

Create `src/lackpy/policy/sources/umwelt.py`:

```python
"""UmweltPolicySource: world-model policy from umwelt's PolicyEngine."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

from ..types import PolicyContext, PolicyResult, ToolConstraints


def _parse_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


class UmweltPolicySource:
    """Restricts tools based on umwelt's resolved capability-taxon policy.

    Highest priority. Can restrict the kit's tool set but cannot
    grant tools the kit doesn't have. Kit resolution (S1) is the
    ground truth for what's available.
    """

    name = "umwelt"
    priority = 100

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def resolve(self, current: PolicyResult, context: PolicyContext) -> PolicyResult:
        tool_entries = self._engine.resolve_all(type="tool")

        allowed: set[str] = set()
        denied: set[str] = set()
        constraints: dict[str, ToolConstraints] = {}

        for entry in tool_entries:
            name = entry["id"]
            if entry.get("visible") == "false":
                denied.add(name)
            else:
                allowed.add(name)

            if entry.get("max_level") or entry.get("allow_patterns") or entry.get("deny_patterns"):
                constraints[name] = ToolConstraints(
                    max_level=_parse_int(entry.get("max_level")),
                    allow_patterns=tuple(entry.get("allow_patterns", ())),
                    deny_patterns=tuple(entry.get("deny_patterns", ())),
                )

        effective_allowed = current.allowed_tools & frozenset(allowed)
        effective_denied = current.denied_tools | frozenset(denied)

        return current.replace(
            allowed_tools=effective_allowed,
            denied_tools=effective_denied,
            tool_constraints=MappingProxyType(constraints) if constraints else current.tool_constraints,
            resolved=False,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/policy/test_umwelt_source.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/lackpy/policy/sources/umwelt.py tests/policy/test_umwelt_source.py
git commit -m "feat(policy): UmweltPolicySource — world-model tool restrictions from PolicyEngine"
```

---

### Task 6: Full chain integration test

**Files:**
- Create: `tests/policy/test_chain_integration.py`

- [ ] **Step 1: Write integration test for the full chain**

Create `tests/policy/test_chain_integration.py`:

```python
"""Integration tests: full PolicyLayer chain with all three sources."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from lackpy.policy.layer import PolicyLayer
from lackpy.policy.types import PolicyResult, PolicyContext
from lackpy.policy.sources.kit import KitPolicySource
from lackpy.policy.sources.kibitzer import KibitzerPolicySource
from lackpy.policy.sources.umwelt import UmweltPolicySource
from lackpy.kit.registry import ResolvedKit, resolve_kit
from lackpy.kit.toolbox import Toolbox, ToolSpec, ArgSpec
from lackpy.kit.providers.builtin import BuiltinProvider
from lackpy.lang.grader import Grade
from lackpy.infer.context import StepContext, ProgramState, StepTrace


class FakeHints:
    def __init__(self, hints=None, doc_context=None):
        self.hints = hints or []
        self.doc_context = doc_context


class FakeKibitzerSession:
    def __init__(self, hints=None, doc_context=None, coaching=None):
        self._hints = FakeHints(hints, doc_context)
        self._coaching = coaching

    def get_correction_hints(self, errors=None, model=None, attempt=None):
        return self._hints

    def has_coaching(self):
        return self._coaching is not None

    def apply_coaching(self, desc):
        return desc + self._coaching


class FakePolicyEngine:
    def __init__(self, tool_entries):
        self._entries = tool_entries

    def resolve_all(self, type="tool"):
        return self._entries


@pytest.fixture
def toolbox():
    tb = Toolbox()
    tb.register_provider(BuiltinProvider())
    for name, desc, grade in [
        ("read_file", "Read file contents", 1),
        ("find_files", "Find files matching pattern", 1),
        ("edit_file", "Edit a file", 3),
    ]:
        tb.register_tool(ToolSpec(
            name=name, provider="builtin", description=desc,
            args=[ArgSpec(name="path", type="str")],
            returns="str", grade_w=grade, effects_ceiling=grade,
        ))
    return tb


def _make_step_trace():
    return StepTrace(
        step_name="test", provider_name=None, model=None,
        system_prompt=None, user_prompt=None, raw_output=None,
        duration_ms=0.0,
    )


class TestStandaloneConfig:
    """Kit source only — no Kibitzer, no umwelt."""

    def test_kit_only(self, toolbox):
        kit = resolve_kit(["read_file", "find_files"], toolbox)
        layer = PolicyLayer()
        layer.add_source(KitPolicySource(toolbox))

        result = layer.resolve({"kit": kit})
        assert result.allowed_tools == frozenset({"read_file", "find_files"})
        assert result.grade.w == 1
        assert result.namespace_desc is not None
        assert result.prompt_hints == ()
        assert result.resolved is False


class TestKitPlusKibitzer:
    """Kit + Kibitzer coaching."""

    def test_kibitzer_enriches_without_changing_tools(self, toolbox):
        kit = resolve_kit(["read_file", "find_files"], toolbox)
        layer = PolicyLayer()
        layer.add_source(KitPolicySource(toolbox))
        layer.add_source(KibitzerPolicySource(
            FakeKibitzerSession(coaching="\n- Always use find_files before read_file"),
        ))

        result = layer.resolve({"kit": kit})
        assert result.allowed_tools == frozenset({"read_file", "find_files"})
        assert "Always use find_files before read_file" in result.namespace_desc

    def test_kibitzer_adds_hints_on_failure(self, toolbox):
        kit = resolve_kit(["read_file"], toolbox)
        history = StepContext(intent="read a file", kit=kit)
        history.programs.append(ProgramState(
            program="open('test.txt')", intent="read a file", kit=kit,
            valid=False, errors=["Forbidden name: 'open'"],
            trace=_make_step_trace(),
        ))

        layer = PolicyLayer()
        layer.add_source(KitPolicySource(toolbox))
        layer.add_source(KibitzerPolicySource(
            FakeKibitzerSession(hints=["Use read_file() instead of open()"]),
        ))

        result = layer.resolve({"kit": kit, "history": history})
        assert "Use read_file() instead of open()" in result.prompt_hints
        assert result.allowed_tools == frozenset({"read_file"})


class TestFullStack:
    """Kit + Kibitzer + umwelt."""

    def test_umwelt_restricts_after_kibitzer_enriches(self, toolbox):
        kit = resolve_kit(["read_file", "find_files", "edit_file"], toolbox)

        layer = PolicyLayer()
        layer.add_source(KitPolicySource(toolbox))
        layer.add_source(KibitzerPolicySource(
            FakeKibitzerSession(coaching="\n- Be careful with edit_file"),
        ))
        layer.add_source(UmweltPolicySource(FakePolicyEngine([
            {"id": "read_file", "visible": "true"},
            {"id": "find_files", "visible": "true"},
            {"id": "edit_file", "visible": "false"},
        ])))

        result = layer.resolve({"kit": kit})
        assert result.allowed_tools == frozenset({"read_file", "find_files"})
        assert "edit_file" in result.denied_tools
        assert "Be careful with edit_file" in result.namespace_desc
        assert result.grade.w == 3  # grade from kit, not affected by umwelt

    def test_umwelt_cannot_add_tools_beyond_kit(self, toolbox):
        kit = resolve_kit(["read_file"], toolbox)

        layer = PolicyLayer()
        layer.add_source(KitPolicySource(toolbox))
        layer.add_source(UmweltPolicySource(FakePolicyEngine([
            {"id": "read_file", "visible": "true"},
            {"id": "edit_file", "visible": "true"},
            {"id": "bash", "visible": "true"},
        ])))

        result = layer.resolve({"kit": kit})
        assert result.allowed_tools == frozenset({"read_file"})


class TestKitPlusUmwelt:
    """Kit + umwelt, no Kibitzer."""

    def test_umwelt_restricts_kit(self, toolbox):
        kit = resolve_kit(["read_file", "edit_file"], toolbox)

        layer = PolicyLayer()
        layer.add_source(KitPolicySource(toolbox))
        layer.add_source(UmweltPolicySource(FakePolicyEngine([
            {"id": "read_file", "visible": "true", "max_level": "1"},
            {"id": "edit_file", "visible": "false"},
        ])))

        result = layer.resolve({"kit": kit})
        assert result.allowed_tools == frozenset({"read_file"})
        assert "edit_file" in result.denied_tools
        assert result.tool_constraints["read_file"].max_level == 1
```

- [ ] **Step 2: Run integration tests**

Run: `pytest tests/policy/test_chain_integration.py -v`
Expected: All tests PASS

- [ ] **Step 3: Run full policy test suite**

Run: `pytest tests/policy/ -v`
Expected: All tests PASS (types + layer + kit + kibitzer + umwelt + integration)

- [ ] **Step 4: Commit**

```bash
git add tests/policy/test_chain_integration.py
git commit -m "test(policy): full chain integration tests — standalone, +kibitzer, +umwelt, full stack"
```

---

### Task 7: Wire PolicyLayer into LackpyService

**Files:**
- Modify: `src/lackpy/service.py:0-15` (imports)
- Modify: `src/lackpy/service.py:102-114` (`__init__`)
- Modify: `src/lackpy/service.py:143-209` (remove `_init_kibitzer`, `_register_kibitzer_docs`, `_apply_kibitzer_hints`)
- Modify: `src/lackpy/service.py:253-336` (`generate`)
- Modify: `src/lackpy/service.py:370-512` (`delegate`)
- Modify: `src/lackpy/__init__.py`

This is the migration task. The existing behavior must be preserved — kit-only users see zero change.

- [ ] **Step 1: Run existing service tests to establish baseline**

Run: `pytest tests/test_service.py tests/test_integration.py -v`
Expected: All existing tests PASS (this is our safety net)

- [ ] **Step 2: Add PolicyLayer imports to service.py**

At the top of `src/lackpy/service.py`, add after line 18 (`from .lang.grammar import ALLOWED_BUILTINS`):

```python
from .policy.layer import PolicyLayer
from .policy.sources.kit import KitPolicySource
```

- [ ] **Step 3: Add PolicyLayer initialization to __init__**

In `src/lackpy/service.py`, in `__init__` (around line 102-114), after `self.toolbox` setup and `_init_inference_providers()`, add PolicyLayer construction. Replace lines 113-114:

```python
        self._kibitzer: Any = None
        self._init_kibitzer()
```

with:

```python
        self._policy = PolicyLayer()
        self._policy.add_source(KitPolicySource(self.toolbox))
        self._kibitzer: Any = None
        self._init_kibitzer()
```

- [ ] **Step 4: Wire Kibitzer into PolicyLayer during init**

In `_init_kibitzer()` (around line 143-161), after `self._kibitzer = KibitzerSession(...)` and `self._kibitzer.load()` succeed, add the KibitzerPolicySource. Change the method to:

```python
    def _init_kibitzer(self) -> None:
        """Initialize Kibitzer session if available."""
        if not _HAS_KIBITZER:
            return
        try:
            self._kibitzer = KibitzerSession(project_dir=self._workspace)
            self._kibitzer.load()
            self._kibitzer.register_tools([
                {
                    "name": spec.name,
                    "grade": {"w": spec.grade_w, "d": spec.effects_ceiling},
                    "description": spec.description,
                    "effects": "write" if spec.grade_w >= 3 else "read" if spec.grade_w >= 1 else "none",
                }
                for spec in self.toolbox.list_tools()
            ])
            self._register_kibitzer_docs()
            from .policy.sources.kibitzer import KibitzerPolicySource
            self._policy.add_source(KibitzerPolicySource(self._kibitzer))
        except Exception:
            self._kibitzer = None
```

- [ ] **Step 5: Replace hint application in generate() with PolicyLayer**

In `generate()`, replace the legacy Kibitzer hint block (lines ~316-330):

```python
        # Default: legacy dispatcher path
        dispatcher = InferenceDispatcher(providers=self._inference_providers)
        allowed = set(resolved.tools.keys()) | param_names

        # Kibitzer: get model-specific prompt hints from failure pattern tracker
        namespace_desc = resolved.description
        if self._kibitzer:
            # Find the model name from the first available LLM provider
            model_name = None
            for p in dispatcher.get_providers():
                m = getattr(p, "_model", None)
                if m:
                    model_name = m
                    break
            namespace_desc = self._apply_kibitzer_hints(namespace_desc, model=model_name)
```

with:

```python
        # Default: legacy dispatcher path
        dispatcher = InferenceDispatcher(providers=self._inference_providers)
        allowed = set(resolved.tools.keys()) | param_names

        # Resolve policy — kit baseline + optional kibitzer hints + optional umwelt
        from .policy.types import PolicyContext, ModelSpec
        model_name = None
        for p in dispatcher.get_providers():
            m = getattr(p, "_model", None)
            if m:
                model_name = m
                break
        policy_context: PolicyContext = {"kit": resolved}
        if model_name:
            policy_context["model"] = ModelSpec(name=model_name)
        policy = self._policy.resolve(policy_context)
        namespace_desc = policy.namespace_desc or resolved.description
```

- [ ] **Step 6: Run existing tests to verify no regression**

Run: `pytest tests/test_service.py tests/test_integration.py -v`
Expected: All existing tests still PASS

- [ ] **Step 7: Commit**

```bash
git add src/lackpy/service.py
git commit -m "refactor(service): wire PolicyLayer into init and generate, replace kibitzer hint conditionals"
```

---

### Task 8: Update public exports

**Files:**
- Modify: `src/lackpy/__init__.py`

- [ ] **Step 1: Add PolicyLayer and PolicyResult to package exports**

In `src/lackpy/__init__.py`, add after the existing imports (line 13, after `from .run.trace import Trace, TraceEntry`):

```python
from .policy import PolicyLayer, PolicyResult, PolicyContext
```

And add to `__all__`:

```python
    "PolicyLayer", "PolicyResult", "PolicyContext",
```

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ -v --ignore=tests/eval`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add src/lackpy/__init__.py
git commit -m "feat: export PolicyLayer, PolicyResult, PolicyContext from lackpy package"
```

---

### Task 9: Remove dead code — _apply_kibitzer_hints

**Files:**
- Modify: `src/lackpy/service.py:180-209`

After Task 7, `_apply_kibitzer_hints` is no longer called from `generate()`. It may still be called from the legacy dispatcher path. Check and remove if dead.

- [ ] **Step 1: Verify _apply_kibitzer_hints is not called anywhere**

Run: `grep -rn "_apply_kibitzer_hints" src/lackpy/`
Expected: Only the definition in `service.py` — no callers.

- [ ] **Step 2: Remove _apply_kibitzer_hints**

Delete the `_apply_kibitzer_hints` method (lines 180-209 of `src/lackpy/service.py`):

```python
    def _apply_kibitzer_hints(self, namespace_desc: str, model: str | None = None) -> str:
        ...  # entire method
```

- [ ] **Step 3: Run tests to verify no regression**

Run: `pytest tests/ -v --ignore=tests/eval`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/lackpy/service.py
git commit -m "refactor(service): remove dead _apply_kibitzer_hints — replaced by KibitzerPolicySource"
```

---

### Task 10: Service integration test with PolicyLayer

**Files:**
- Create: `tests/policy/test_service_integration.py`

- [ ] **Step 1: Write integration tests that verify the service uses PolicyLayer**

Create `tests/policy/test_service_integration.py`:

```python
"""Integration tests: LackpyService uses PolicyLayer correctly."""

from __future__ import annotations

import pytest
from pathlib import Path

from lackpy.service import LackpyService
from lackpy.kit.toolbox import ToolSpec, ArgSpec
from lackpy.policy.layer import PolicyLayer
from lackpy.policy.sources.kit import KitPolicySource


@pytest.fixture
def service(tmp_path):
    svc = LackpyService(workspace=tmp_path)
    svc.toolbox.register_tool(ToolSpec(
        name="read_file", provider="builtin",
        description="Read file contents",
        args=[ArgSpec(name="path", type="str", description="File path")],
        returns="str", grade_w=1, effects_ceiling=1,
    ))
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello world")
    return svc


class TestServiceHasPolicyLayer:
    def test_service_creates_policy_layer(self, service):
        assert hasattr(service, "_policy")
        assert isinstance(service._policy, PolicyLayer)

    def test_kit_source_always_registered(self, service):
        source_names = [s.name for s in service._policy._sources]
        assert "kit" in source_names

    def test_policy_resolves_with_kit(self, service):
        from lackpy.kit.registry import resolve_kit
        kit = resolve_kit(["read_file"], service.toolbox)
        result = service._policy.resolve({"kit": kit})
        assert "read_file" in result.allowed_tools
        assert result.grade.w == 1


class TestServiceDelegateUsesPolicyLayer:
    @pytest.mark.asyncio
    async def test_delegate_still_works(self, service):
        result = await service.delegate("read file test.txt", kit=["read_file"])
        assert result["success"]
        assert "read_file" in result["program"]

    @pytest.mark.asyncio
    async def test_validate_still_works(self, service):
        result = service.validate("x = read_file('test.txt')", kit=["read_file"])
        assert result.valid
```

- [ ] **Step 2: Run integration tests**

Run: `pytest tests/policy/test_service_integration.py -v`
Expected: All tests PASS

- [ ] **Step 3: Run full test suite one final time**

Run: `pytest tests/ -v --ignore=tests/eval`
Expected: All tests PASS — zero regressions

- [ ] **Step 4: Commit**

```bash
git add tests/policy/test_service_integration.py
git commit -m "test(policy): service integration tests — PolicyLayer wired correctly, delegate works"
```
