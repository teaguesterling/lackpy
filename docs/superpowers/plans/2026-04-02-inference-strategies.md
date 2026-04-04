# Inference Strategies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the inference pipeline into composable steps and combinators, then add SPM (solve-pick-restrict) mode alongside the existing 1-shot mode.

**Architecture:** Steps are pure fold functions over a `StepContext` accumulator. Two combinators — `Sequence` and `Fallback` — compose steps into strategies. Strategies are named configurations selected via `--mode`. The existing dispatcher is slimmed to provider selection only.

**Tech Stack:** Python 3.10+, dataclasses, pytest-asyncio

**Spec:** `docs/superpowers/specs/2026-04-02-inference-strategies-design.md`

---

## File Structure

```
src/lackpy/infer/
    context.py           # NEW: StepContext, ProgramState, StepTrace
    combinators.py       # NEW: Sequence, Fallback, Step protocol
    strategy.py          # NEW: InferenceStrategy protocol, STRATEGIES registry
    steps/               # NEW directory
        __init__.py
        generate.py      # Generate step (current 1-shot generation)
        validate.py      # Validate step (AST whitelist check)
        cleanup.py       # Cleanup step (wraps existing deterministic_cleanup)
        few_shot.py      # FewShotCorrect step
        fresh_fix.py     # FreshFix step
        solve.py         # Solve step (unconstrained generation)
        pick.py          # Pick step (AST analysis → derived kit)
        restrict.py      # Restrict step (rewrite to lackpy subset)
    dispatch.py          # MODIFY: slim to provider selection
    correction.py        # KEEP: still importable but no longer the primary path
    cleanup.py           # KEEP: deterministic_cleanup function used by Cleanup step
    sanitize.py          # KEEP: used by Generate/Solve steps
    prompt.py            # KEEP: used by Generate step
    fixer.py             # KEEP: used by FreshFix step
    hints.py             # KEEP: used by FewShotCorrect step
    providers/           # KEEP: unchanged
tests/infer/
    test_context.py      # NEW
    test_combinators.py  # NEW
    test_strategy.py     # NEW
    test_steps_generate.py  # NEW
    test_steps_validate.py  # NEW
    test_steps_cleanup.py   # NEW
    test_steps_few_shot.py  # NEW
    test_steps_fresh_fix.py # NEW
    test_steps_solve.py     # NEW
    test_steps_pick.py      # NEW
    test_steps_restrict.py  # NEW
```

---

### Task 1: StepContext and ProgramState dataclasses

**Files:**
- Create: `src/lackpy/infer/context.py`
- Test: `tests/infer/test_context.py`

- [ ] **Step 1: Write failing tests for context dataclasses**

```python
# tests/infer/test_context.py
"""Tests for StepContext and ProgramState."""

import pytest
from lackpy.infer.context import StepContext, ProgramState, StepTrace
from lackpy.kit.registry import ResolvedKit
from lackpy.kit.toolbox import ToolSpec
from lackpy.lang.grader import Grade


def _make_kit(tools=None):
    """Helper to build a minimal ResolvedKit for testing."""
    tools = tools or {"read": ToolSpec(name="read", provider="builtin", description="Read file")}
    return ResolvedKit(
        tools=tools,
        callables={n: lambda *a: None for n in tools},
        grade=Grade(w=1, d=1),
        description="read(path) -> str: Read file",
    )


class TestStepTrace:
    def test_create_trace(self):
        trace = StepTrace(
            step_name="generate",
            provider_name="ollama",
            model="qwen2.5:3b",
            system_prompt="You are...",
            user_prompt="find files",
            raw_output="files = glob('**/*.py')",
            duration_ms=123.4,
        )
        assert trace.step_name == "generate"
        assert trace.duration_ms == 123.4

    def test_trace_with_no_provider(self):
        trace = StepTrace(
            step_name="cleanup",
            provider_name=None,
            model=None,
            system_prompt=None,
            user_prompt=None,
            raw_output=None,
            duration_ms=0.5,
        )
        assert trace.provider_name is None


class TestProgramState:
    def test_create_program_state(self):
        trace = StepTrace(
            step_name="generate", provider_name="ollama", model="qwen2.5:3b",
            system_prompt=None, user_prompt="find files",
            raw_output="glob('**/*.py')", duration_ms=100.0,
        )
        state = ProgramState(
            program="glob('**/*.py')",
            intent="find files",
            kit=_make_kit(),
            valid=None,
            errors=[],
            trace=trace,
        )
        assert state.program == "glob('**/*.py')"
        assert state.valid is None

    def test_program_state_with_errors(self):
        trace = StepTrace(
            step_name="generate", provider_name="ollama", model="qwen2.5:3b",
            system_prompt=None, user_prompt="find files",
            raw_output="import glob", duration_ms=50.0,
        )
        state = ProgramState(
            program="import glob",
            intent="find files",
            kit=_make_kit(),
            valid=False,
            errors=["Forbidden AST node: Import at line 1"],
            trace=trace,
        )
        assert not state.valid
        assert len(state.errors) == 1


class TestStepContext:
    def test_create_context(self):
        kit = _make_kit()
        ctx = StepContext(intent="find files", kit=kit)
        assert ctx.intent == "find files"
        assert ctx.programs == []
        assert ctx.current is None

    def test_current_returns_last_program(self):
        kit = _make_kit()
        ctx = StepContext(intent="find files", kit=kit)
        trace = StepTrace(
            step_name="generate", provider_name=None, model=None,
            system_prompt=None, user_prompt=None, raw_output=None, duration_ms=0,
        )
        state1 = ProgramState(
            program="v1", intent="find files", kit=kit,
            valid=False, errors=["err"], trace=trace,
        )
        state2 = ProgramState(
            program="v2", intent="find files", kit=kit,
            valid=True, errors=[], trace=trace,
        )
        ctx.programs.append(state1)
        assert ctx.current.program == "v1"
        ctx.programs.append(state2)
        assert ctx.current.program == "v2"
        assert len(ctx.programs) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Projects/lackpy/main && python -m pytest tests/infer/test_context.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lackpy.infer.context'`

- [ ] **Step 3: Implement context.py**

```python
# src/lackpy/infer/context.py
"""Step context: the accumulator threaded through the inference fold."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepTrace:
    """Operational record of one step execution.

    Captures prompts, model, timing, and raw output for observability.
    """

    step_name: str
    provider_name: str | None
    model: str | None
    system_prompt: str | None
    user_prompt: str | None
    raw_output: str | None
    duration_ms: float


@dataclass
class ProgramState:
    """Semantic result of one generation/transform step.

    Each step in the fold pushes a ProgramState onto the context's
    programs list. The most recent entry is the current state.
    """

    program: str
    intent: str
    kit: Any  # ResolvedKit, but Any to avoid circular imports
    valid: bool | None
    errors: list[str]
    trace: StepTrace


@dataclass
class StepContext:
    """The accumulator threaded through the inference pipeline fold.

    Input fields (intent, kit, params_desc, extra_rules) are set at
    strategy entry and remain immutable across the fold. Steps read
    from `current` and push new ProgramState entries onto `programs`.
    """

    intent: str
    kit: Any  # ResolvedKit
    params_desc: str | None = None
    extra_rules: list | None = None
    programs: list[ProgramState] = field(default_factory=list)
    provider: Any = None

    @property
    def current(self) -> ProgramState | None:
        """The most recent ProgramState, or None if no programs yet."""
        return self.programs[-1] if self.programs else None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Projects/lackpy/main && python -m pytest tests/infer/test_context.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/lackpy/infer/context.py tests/infer/test_context.py
git commit -m "feat: add StepContext, ProgramState, StepTrace dataclasses"
```

---

### Task 2: Step protocol and combinators

**Files:**
- Create: `src/lackpy/infer/combinators.py`
- Test: `tests/infer/test_combinators.py`

- [ ] **Step 1: Write failing tests for combinators**

```python
# tests/infer/test_combinators.py
"""Tests for Step protocol, Sequence, and Fallback combinators."""

import pytest
from lackpy.infer.combinators import Step, Sequence, Fallback
from lackpy.infer.context import StepContext, ProgramState, StepTrace
from lackpy.kit.toolbox import ToolSpec
from lackpy.kit.registry import ResolvedKit
from lackpy.lang.grader import Grade


def _make_kit():
    tools = {"read": ToolSpec(name="read", provider="builtin", description="Read file")}
    return ResolvedKit(
        tools=tools,
        callables={n: lambda *a: None for n in tools},
        grade=Grade(w=1, d=1),
        description="read(path) -> str",
    )


def _make_trace(step_name="test"):
    return StepTrace(
        step_name=step_name, provider_name=None, model=None,
        system_prompt=None, user_prompt=None, raw_output=None, duration_ms=0,
    )


def _make_ctx():
    return StepContext(intent="test", kit=_make_kit())


class _AppendStep:
    """Test step that pushes a ProgramState with a given program string."""

    def __init__(self, program: str, valid: bool | None = None):
        self.name = f"append_{program}"
        self._program = program
        self._valid = valid

    async def run(self, ctx: StepContext) -> StepContext:
        ctx.programs.append(ProgramState(
            program=self._program, intent=ctx.intent, kit=ctx.kit,
            valid=self._valid, errors=[], trace=_make_trace(self.name),
        ))
        return ctx


class _SetValidStep:
    """Test step that sets valid on the current ProgramState."""

    def __init__(self, valid: bool, errors: list[str] | None = None):
        self.name = "set_valid"
        self._valid = valid
        self._errors = errors or []

    async def run(self, ctx: StepContext) -> StepContext:
        if ctx.current:
            ctx.current.valid = self._valid
            ctx.current.errors = self._errors
        return ctx


class TestSequence:
    @pytest.mark.asyncio
    async def test_runs_steps_in_order(self):
        seq = Sequence([_AppendStep("a"), _AppendStep("b"), _AppendStep("c")])
        ctx = _make_ctx()
        ctx = await seq.run(ctx)
        assert [p.program for p in ctx.programs] == ["a", "b", "c"]
        assert ctx.current.program == "c"

    @pytest.mark.asyncio
    async def test_empty_sequence(self):
        seq = Sequence([])
        ctx = _make_ctx()
        ctx = await seq.run(ctx)
        assert ctx.programs == []


class TestFallback:
    @pytest.mark.asyncio
    async def test_returns_first_valid_branch(self):
        fb = Fallback([
            Sequence([_AppendStep("bad", valid=None), _SetValidStep(False)]),
            Sequence([_AppendStep("good", valid=None), _SetValidStep(True)]),
            Sequence([_AppendStep("never_reached", valid=None), _SetValidStep(True)]),
        ])
        ctx = _make_ctx()
        ctx = await fb.run(ctx)
        assert ctx.current.program == "good"
        assert ctx.current.valid is True
        # "never_reached" should not appear
        programs = [p.program for p in ctx.programs]
        assert "never_reached" not in programs

    @pytest.mark.asyncio
    async def test_returns_last_failure_when_all_fail(self):
        fb = Fallback([
            Sequence([_AppendStep("a", valid=None), _SetValidStep(False)]),
            Sequence([_AppendStep("b", valid=None), _SetValidStep(False)]),
        ])
        ctx = _make_ctx()
        ctx = await fb.run(ctx)
        assert ctx.current.valid is False
        assert ctx.current.program == "b"

    @pytest.mark.asyncio
    async def test_all_attempts_in_programs_list(self):
        fb = Fallback([
            Sequence([_AppendStep("a", valid=None), _SetValidStep(False)]),
            Sequence([_AppendStep("b", valid=None), _SetValidStep(True)]),
        ])
        ctx = _make_ctx()
        ctx = await fb.run(ctx)
        programs = [p.program for p in ctx.programs]
        assert programs == ["a", "b"]

    @pytest.mark.asyncio
    async def test_empty_fallback(self):
        fb = Fallback([])
        ctx = _make_ctx()
        ctx = await fb.run(ctx)
        assert ctx.current is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Projects/lackpy/main && python -m pytest tests/infer/test_combinators.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lackpy.infer.combinators'`

- [ ] **Step 3: Implement combinators.py**

```python
# src/lackpy/infer/combinators.py
"""Step protocol and combinators for inference pipeline composition."""

from __future__ import annotations

from typing import Protocol

from .context import StepContext


class Step(Protocol):
    """A single fold function in the inference pipeline."""

    name: str

    async def run(self, ctx: StepContext) -> StepContext: ...


class Sequence:
    """Run steps in order, threading context through."""

    def __init__(self, steps: list) -> None:
        self.steps = steps
        self.name = "sequence"

    async def run(self, ctx: StepContext) -> StepContext:
        for step in self.steps:
            ctx = await step.run(ctx)
        return ctx


class Fallback:
    """Try branches in order. Return first where ctx.current.valid is True.

    Each branch starts from the context at Fallback entry. All ProgramState
    entries from all branches accumulate in the flat programs list.
    """

    def __init__(self, branches: list) -> None:
        self.branches = branches
        self.name = "fallback"

    async def run(self, ctx: StepContext) -> StepContext:
        if not self.branches:
            return ctx
        snapshot = len(ctx.programs)
        for branch in self.branches:
            ctx = await branch.run(ctx)
            if ctx.current and ctx.current.valid:
                return ctx
        return ctx
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Projects/lackpy/main && python -m pytest tests/infer/test_combinators.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/lackpy/infer/combinators.py tests/infer/test_combinators.py
git commit -m "feat: add Sequence and Fallback combinators with Step protocol"
```

---

### Task 3: Extract Generate step

**Files:**
- Create: `src/lackpy/infer/steps/__init__.py`
- Create: `src/lackpy/infer/steps/generate.py`
- Test: `tests/infer/test_steps_generate.py`

- [ ] **Step 1: Write failing tests for Generate step**

```python
# tests/infer/test_steps_generate.py
"""Tests for the Generate step."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from lackpy.infer.steps.generate import GenerateStep
from lackpy.infer.context import StepContext
from lackpy.kit.toolbox import ToolSpec
from lackpy.kit.registry import ResolvedKit
from lackpy.lang.grader import Grade


def _make_kit():
    tools = {
        "read": ToolSpec(name="read", provider="builtin", description="Read file",
                         args=[], returns="str", grade_w=1, effects_ceiling=1),
        "glob": ToolSpec(name="glob", provider="builtin", description="Find files",
                         args=[], returns="list[str]", grade_w=1, effects_ceiling=1),
    }
    return ResolvedKit(
        tools=tools,
        callables={n: lambda *a: None for n in tools},
        grade=Grade(w=1, d=1),
        description="read(path) -> str: Read file\nglob(pattern) -> list[str]: Find files",
    )


def _make_provider(output="files = glob('**/*.py')"):
    provider = MagicMock()
    provider.name = "test"
    provider.generate = AsyncMock(return_value=output)
    return provider


class TestGenerateStep:
    @pytest.mark.asyncio
    async def test_pushes_program_state(self):
        provider = _make_provider("files = glob('**/*.py')")
        step = GenerateStep(provider)
        ctx = StepContext(intent="find python files", kit=_make_kit())
        ctx = await step.run(ctx)
        assert ctx.current is not None
        assert ctx.current.program == "files = glob('**/*.py')"
        assert ctx.current.valid is None  # not yet validated
        assert ctx.current.trace.step_name == "generate"
        assert ctx.current.trace.provider_name == "test"

    @pytest.mark.asyncio
    async def test_sanitizes_and_cleans_output(self):
        raw = "```python\nimport os\nfiles = glob('**/*.py')\n```"
        provider = _make_provider(raw)
        step = GenerateStep(provider)
        ctx = StepContext(intent="find python files", kit=_make_kit())
        ctx = await step.run(ctx)
        # Should strip markdown fences and imports
        assert "```" not in ctx.current.program
        assert "import os" not in ctx.current.program

    @pytest.mark.asyncio
    async def test_records_timing(self):
        provider = _make_provider("glob('*.py')")
        step = GenerateStep(provider)
        ctx = StepContext(intent="find files", kit=_make_kit())
        ctx = await step.run(ctx)
        assert ctx.current.trace.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_provider_returns_none(self):
        provider = _make_provider(None)
        step = GenerateStep(provider)
        ctx = StepContext(intent="find files", kit=_make_kit())
        ctx = await step.run(ctx)
        # Should still push a state but with empty program
        assert ctx.current is not None
        assert ctx.current.program == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Projects/lackpy/main && python -m pytest tests/infer/test_steps_generate.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement Generate step**

```python
# src/lackpy/infer/steps/__init__.py
"""Inference pipeline steps."""
```

```python
# src/lackpy/infer/steps/generate.py
"""Generate step: produce a lackpy program from intent via a provider."""

from __future__ import annotations

import time
from typing import Any

from ..context import ProgramState, StepContext, StepTrace
from ..cleanup import deterministic_cleanup
from ..prompt import build_system_prompt
from ..sanitize import sanitize_output


class GenerateStep:
    """Generate a lackpy-constrained program from intent.

    Uses the provider's generate() method with the kit's namespace
    description as the system prompt context.
    """

    name = "generate"

    def __init__(self, provider: Any) -> None:
        self._provider = provider

    async def run(self, ctx: StepContext) -> StepContext:
        start = time.perf_counter()
        namespace_desc = ctx.kit.description
        system_prompt = build_system_prompt(namespace_desc, ctx.params_desc)

        raw = await self._provider.generate(ctx.intent, namespace_desc)
        elapsed = (time.perf_counter() - start) * 1000

        if raw is None:
            program = ""
        else:
            program = sanitize_output(raw)
            program = deterministic_cleanup(program)

        ctx.programs.append(ProgramState(
            program=program,
            intent=ctx.intent,
            kit=ctx.kit,
            valid=None,
            errors=[],
            trace=StepTrace(
                step_name=self.name,
                provider_name=self._provider.name,
                model=getattr(self._provider, '_model', None),
                system_prompt=system_prompt,
                user_prompt=ctx.intent,
                raw_output=raw,
                duration_ms=elapsed,
            ),
        ))
        return ctx
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Projects/lackpy/main && python -m pytest tests/infer/test_steps_generate.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/lackpy/infer/steps/__init__.py src/lackpy/infer/steps/generate.py tests/infer/test_steps_generate.py
git commit -m "feat: extract Generate step from dispatch pipeline"
```

---

### Task 4: Extract Validate step

**Files:**
- Create: `src/lackpy/infer/steps/validate.py`
- Test: `tests/infer/test_steps_validate.py`

- [ ] **Step 1: Write failing tests for Validate step**

```python
# tests/infer/test_steps_validate.py
"""Tests for the Validate step."""

import pytest
from lackpy.infer.steps.validate import ValidateStep
from lackpy.infer.context import StepContext, ProgramState, StepTrace
from lackpy.kit.toolbox import ToolSpec
from lackpy.kit.registry import ResolvedKit
from lackpy.lang.grader import Grade


def _make_kit():
    tools = {
        "glob": ToolSpec(name="glob", provider="builtin", description="Find files",
                         args=[], returns="list[str]", grade_w=1, effects_ceiling=1),
    }
    return ResolvedKit(
        tools=tools,
        callables={n: lambda *a: None for n in tools},
        grade=Grade(w=1, d=1),
        description="glob(pattern) -> list[str]",
    )


def _make_trace():
    return StepTrace(
        step_name="test", provider_name=None, model=None,
        system_prompt=None, user_prompt=None, raw_output=None, duration_ms=0,
    )


def _ctx_with_program(program: str):
    kit = _make_kit()
    ctx = StepContext(intent="test", kit=kit)
    ctx.programs.append(ProgramState(
        program=program, intent="test", kit=kit,
        valid=None, errors=[], trace=_make_trace(),
    ))
    return ctx


class TestValidateStep:
    @pytest.mark.asyncio
    async def test_valid_program(self):
        ctx = _ctx_with_program("files = glob('**/*.py')\nlen(files)")
        step = ValidateStep()
        ctx = await step.run(ctx)
        assert ctx.current.valid is True
        assert ctx.current.errors == []

    @pytest.mark.asyncio
    async def test_invalid_program_with_import(self):
        ctx = _ctx_with_program("import os\nglob('*.py')")
        step = ValidateStep()
        ctx = await step.run(ctx)
        assert ctx.current.valid is False
        assert len(ctx.current.errors) > 0

    @pytest.mark.asyncio
    async def test_invalid_unknown_function(self):
        ctx = _ctx_with_program("open('file.txt')")
        step = ValidateStep()
        ctx = await step.run(ctx)
        assert ctx.current.valid is False

    @pytest.mark.asyncio
    async def test_no_current_program(self):
        ctx = StepContext(intent="test", kit=_make_kit())
        step = ValidateStep()
        ctx = await step.run(ctx)
        assert ctx.current is None  # no-op when nothing to validate
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Projects/lackpy/main && python -m pytest tests/infer/test_steps_validate.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement Validate step**

```python
# src/lackpy/infer/steps/validate.py
"""Validate step: check current program against AST whitelist."""

from __future__ import annotations

from ..context import StepContext
from ...lang.validator import validate


class ValidateStep:
    """Validate the current program against allowed names and rules.

    Mutates ctx.current in place: sets valid and errors. Does not
    push a new ProgramState — validation is a gate, not a transform.
    """

    name = "validate"

    async def run(self, ctx: StepContext) -> StepContext:
        if ctx.current is None:
            return ctx

        allowed = set(ctx.current.kit.tools.keys()) if ctx.current.kit else set()
        if ctx.kit:
            allowed |= set(ctx.kit.tools.keys())

        result = validate(
            ctx.current.program,
            allowed_names=allowed,
            extra_rules=ctx.extra_rules,
        )
        ctx.current.valid = result.valid
        ctx.current.errors = result.errors
        return ctx
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Projects/lackpy/main && python -m pytest tests/infer/test_steps_validate.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/lackpy/infer/steps/validate.py tests/infer/test_steps_validate.py
git commit -m "feat: extract Validate step from correction chain"
```

---

### Task 5: Extract Cleanup, FewShotCorrect, and FreshFix steps

**Files:**
- Create: `src/lackpy/infer/steps/cleanup.py`
- Create: `src/lackpy/infer/steps/few_shot.py`
- Create: `src/lackpy/infer/steps/fresh_fix.py`
- Test: `tests/infer/test_steps_cleanup.py`
- Test: `tests/infer/test_steps_few_shot.py`
- Test: `tests/infer/test_steps_fresh_fix.py`

- [ ] **Step 1: Write failing tests for Cleanup step**

```python
# tests/infer/test_steps_cleanup.py
"""Tests for the Cleanup step."""

import pytest
from lackpy.infer.steps.cleanup import CleanupStep
from lackpy.infer.context import StepContext, ProgramState, StepTrace
from lackpy.kit.toolbox import ToolSpec
from lackpy.kit.registry import ResolvedKit
from lackpy.lang.grader import Grade


def _make_kit():
    tools = {"read": ToolSpec(name="read", provider="builtin", description="Read file")}
    return ResolvedKit(
        tools=tools, callables={n: lambda *a: None for n in tools},
        grade=Grade(w=1, d=1), description="read(path) -> str",
    )


def _make_trace():
    return StepTrace(
        step_name="test", provider_name=None, model=None,
        system_prompt=None, user_prompt=None, raw_output=None, duration_ms=0,
    )


def _ctx_with_program(program: str):
    kit = _make_kit()
    ctx = StepContext(intent="test", kit=kit)
    ctx.programs.append(ProgramState(
        program=program, intent="test", kit=kit,
        valid=False, errors=["has imports"], trace=_make_trace(),
    ))
    return ctx


class TestCleanupStep:
    @pytest.mark.asyncio
    async def test_strips_imports(self):
        ctx = _ctx_with_program("import os\ncontent = read('f.txt')")
        step = CleanupStep()
        ctx = await step.run(ctx)
        assert ctx.current.program == "content = read('f.txt')"
        assert ctx.current.trace.step_name == "cleanup"

    @pytest.mark.asyncio
    async def test_pushes_new_program_state(self):
        ctx = _ctx_with_program("import os\nread('f.txt')")
        step = CleanupStep()
        ctx = await step.run(ctx)
        assert len(ctx.programs) == 2  # original + cleaned

    @pytest.mark.asyncio
    async def test_no_current_is_noop(self):
        ctx = StepContext(intent="test", kit=_make_kit())
        step = CleanupStep()
        ctx = await step.run(ctx)
        assert ctx.current is None
```

- [ ] **Step 2: Write failing tests for FewShotCorrect step**

```python
# tests/infer/test_steps_few_shot.py
"""Tests for the FewShotCorrect step."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from lackpy.infer.steps.few_shot import FewShotCorrectStep
from lackpy.infer.context import StepContext, ProgramState, StepTrace
from lackpy.kit.toolbox import ToolSpec
from lackpy.kit.registry import ResolvedKit
from lackpy.lang.grader import Grade


def _make_kit():
    tools = {"glob": ToolSpec(name="glob", provider="builtin", description="Find files")}
    return ResolvedKit(
        tools=tools, callables={n: lambda *a: None for n in tools},
        grade=Grade(w=1, d=1), description="glob(pattern) -> list[str]",
    )


def _make_trace():
    return StepTrace(
        step_name="test", provider_name=None, model=None,
        system_prompt=None, user_prompt=None, raw_output=None, duration_ms=0,
    )


def _make_provider(output="glob('**/*.py')"):
    provider = MagicMock()
    provider.name = "test"
    provider.generate = AsyncMock(return_value=output)
    return provider


class TestFewShotCorrectStep:
    @pytest.mark.asyncio
    async def test_sends_errors_as_feedback(self):
        provider = _make_provider("glob('**/*.py')")
        step = FewShotCorrectStep(provider)
        kit = _make_kit()
        ctx = StepContext(intent="find files", kit=kit)
        ctx.programs.append(ProgramState(
            program="import glob", intent="find files", kit=kit,
            valid=False, errors=["Forbidden AST node: Import"], trace=_make_trace(),
        ))
        ctx = await step.run(ctx)
        # Should have called generate with error_feedback
        provider.generate.assert_called_once()
        call_kwargs = provider.generate.call_args
        assert call_kwargs[1].get("error_feedback") or len(call_kwargs[0]) > 2

    @pytest.mark.asyncio
    async def test_pushes_new_program_state(self):
        provider = _make_provider("glob('*.py')")
        step = FewShotCorrectStep(provider)
        kit = _make_kit()
        ctx = StepContext(intent="find files", kit=kit)
        ctx.programs.append(ProgramState(
            program="bad", intent="find files", kit=kit,
            valid=False, errors=["err"], trace=_make_trace(),
        ))
        ctx = await step.run(ctx)
        assert len(ctx.programs) == 2
        assert ctx.current.trace.step_name == "few_shot_correct"
```

- [ ] **Step 3: Write failing tests for FreshFix step**

```python
# tests/infer/test_steps_fresh_fix.py
"""Tests for the FreshFix step."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from lackpy.infer.steps.fresh_fix import FreshFixStep
from lackpy.infer.context import StepContext, ProgramState, StepTrace
from lackpy.kit.toolbox import ToolSpec
from lackpy.kit.registry import ResolvedKit
from lackpy.lang.grader import Grade


def _make_kit():
    tools = {"glob": ToolSpec(name="glob", provider="builtin", description="Find files")}
    return ResolvedKit(
        tools=tools, callables={n: lambda *a: None for n in tools},
        grade=Grade(w=1, d=1), description="glob(pattern) -> list[str]",
    )


def _make_trace():
    return StepTrace(
        step_name="test", provider_name=None, model=None,
        system_prompt=None, user_prompt=None, raw_output=None, duration_ms=0,
    )


class TestFreshFixStep:
    @pytest.mark.asyncio
    async def test_uses_fixer_prompt(self):
        provider = MagicMock()
        provider.name = "test"
        provider._chat = AsyncMock(return_value={"message": {"content": "glob('*.py')"}})
        step = FreshFixStep(provider)
        kit = _make_kit()
        ctx = StepContext(intent="find files", kit=kit)
        ctx.programs.append(ProgramState(
            program="import glob", intent="find files", kit=kit,
            valid=False, errors=["Forbidden: Import"], trace=_make_trace(),
        ))
        ctx = await step.run(ctx)
        assert ctx.current.trace.step_name == "fresh_fix"
        assert len(ctx.programs) == 2

    @pytest.mark.asyncio
    async def test_no_chat_method_tries_create_message(self):
        provider = MagicMock(spec=[])
        provider.name = "test"
        provider._create_message = AsyncMock(return_value="glob('*.py')")
        step = FreshFixStep(provider)
        kit = _make_kit()
        ctx = StepContext(intent="find files", kit=kit)
        ctx.programs.append(ProgramState(
            program="bad", intent="find files", kit=kit,
            valid=False, errors=["err"], trace=_make_trace(),
        ))
        ctx = await step.run(ctx)
        assert ctx.current.program == "glob('*.py')"
```

- [ ] **Step 4: Run all three test files to verify they fail**

Run: `cd ~/Projects/lackpy/main && python -m pytest tests/infer/test_steps_cleanup.py tests/infer/test_steps_few_shot.py tests/infer/test_steps_fresh_fix.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 5: Implement CleanupStep**

```python
# src/lackpy/infer/steps/cleanup.py
"""Cleanup step: deterministic AST rewrites (strip imports, rewrite open, etc.)."""

from __future__ import annotations

import time

from ..context import ProgramState, StepContext, StepTrace
from ..cleanup import deterministic_cleanup


class CleanupStep:
    """Apply deterministic cleanup transforms to the current program.

    Pushes a new ProgramState with the cleaned program. Does not validate.
    """

    name = "cleanup"

    async def run(self, ctx: StepContext) -> StepContext:
        if ctx.current is None:
            return ctx

        start = time.perf_counter()
        cleaned = deterministic_cleanup(ctx.current.program)
        elapsed = (time.perf_counter() - start) * 1000

        ctx.programs.append(ProgramState(
            program=cleaned,
            intent=ctx.current.intent,
            kit=ctx.current.kit,
            valid=None,
            errors=[],
            trace=StepTrace(
                step_name=self.name,
                provider_name=None,
                model=None,
                system_prompt=None,
                user_prompt=None,
                raw_output=None,
                duration_ms=elapsed,
            ),
        ))
        return ctx
```

- [ ] **Step 6: Implement FewShotCorrectStep**

```python
# src/lackpy/infer/steps/few_shot.py
"""FewShotCorrect step: re-invoke provider with error feedback."""

from __future__ import annotations

import time
from typing import Any

from ..context import ProgramState, StepContext, StepTrace
from ..cleanup import deterministic_cleanup
from ..hints import enrich_errors
from ..sanitize import sanitize_output


class FewShotCorrectStep:
    """Re-call the provider with the original intent plus enriched error feedback.

    Pushes a new ProgramState with the corrected program.
    """

    name = "few_shot_correct"

    def __init__(self, provider: Any) -> None:
        self._provider = provider

    async def run(self, ctx: StepContext) -> StepContext:
        if ctx.current is None:
            return ctx

        start = time.perf_counter()
        namespace_desc = ctx.kit.description
        enriched = enrich_errors(ctx.current.errors, namespace_desc)

        raw = await self._provider.generate(
            ctx.intent, namespace_desc, error_feedback=enriched,
        )
        elapsed = (time.perf_counter() - start) * 1000

        if raw is None:
            program = ""
        else:
            program = sanitize_output(raw)
            program = deterministic_cleanup(program)

        ctx.programs.append(ProgramState(
            program=program,
            intent=ctx.intent,
            kit=ctx.current.kit,
            valid=None,
            errors=[],
            trace=StepTrace(
                step_name=self.name,
                provider_name=self._provider.name,
                model=getattr(self._provider, '_model', None),
                system_prompt=None,
                user_prompt=ctx.intent,
                raw_output=raw,
                duration_ms=elapsed,
            ),
        ))
        return ctx
```

- [ ] **Step 7: Implement FreshFixStep**

```python
# src/lackpy/infer/steps/fresh_fix.py
"""FreshFix step: call provider with specialized fixer prompt."""

from __future__ import annotations

import time
from typing import Any

from ..context import ProgramState, StepContext, StepTrace
from ..cleanup import deterministic_cleanup
from ..fixer import build_fixer_messages
from ..sanitize import sanitize_output


async def _call_fixer(provider: Any, messages: list[dict]) -> str | None:
    """Try provider._chat() then fallback to provider._create_message()."""
    if hasattr(provider, "_chat"):
        try:
            response = await provider._chat(messages, temperature=0.4)
            content = response.get("message", {}).get("content", "")
            return content if content else None
        except Exception:
            pass

    if hasattr(provider, "_create_message"):
        try:
            system = next((m["content"] for m in messages if m["role"] == "system"), "")
            user_messages = [m for m in messages if m["role"] != "system"]
            response = await provider._create_message(system, user_messages)
            return response if isinstance(response, str) else None
        except Exception:
            pass

    return None


class FreshFixStep:
    """Call provider with a specialized fixer prompt to repair a broken program.

    Pushes a new ProgramState with the fixed program.
    """

    name = "fresh_fix"

    def __init__(self, provider: Any) -> None:
        self._provider = provider

    async def run(self, ctx: StepContext) -> StepContext:
        if ctx.current is None:
            return ctx

        start = time.perf_counter()
        namespace_desc = ctx.kit.description
        errors_text = "\n".join(ctx.current.errors)

        messages = build_fixer_messages(
            intent=ctx.intent,
            broken_program=ctx.current.program,
            errors=errors_text,
            namespace_desc=namespace_desc,
        )

        raw = await _call_fixer(self._provider, messages)
        elapsed = (time.perf_counter() - start) * 1000

        if raw is None:
            program = ""
        else:
            program = sanitize_output(raw)
            program = deterministic_cleanup(program)

        system_prompt = next((m["content"] for m in messages if m["role"] == "system"), None)
        user_prompt = next((m["content"] for m in messages if m["role"] == "user"), None)

        ctx.programs.append(ProgramState(
            program=program,
            intent=ctx.intent,
            kit=ctx.current.kit,
            valid=None,
            errors=[],
            trace=StepTrace(
                step_name=self.name,
                provider_name=self._provider.name,
                model=getattr(self._provider, '_model', None),
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                raw_output=raw,
                duration_ms=elapsed,
            ),
        ))
        return ctx
```

- [ ] **Step 8: Run all tests to verify they pass**

Run: `cd ~/Projects/lackpy/main && python -m pytest tests/infer/test_steps_cleanup.py tests/infer/test_steps_few_shot.py tests/infer/test_steps_fresh_fix.py -v`
Expected: All 8 tests PASS

- [ ] **Step 9: Commit**

```bash
git add src/lackpy/infer/steps/cleanup.py src/lackpy/infer/steps/few_shot.py src/lackpy/infer/steps/fresh_fix.py tests/infer/test_steps_cleanup.py tests/infer/test_steps_few_shot.py tests/infer/test_steps_fresh_fix.py
git commit -m "feat: extract Cleanup, FewShotCorrect, and FreshFix steps"
```

---

### Task 6: OneShotStrategy and strategy registry

**Files:**
- Create: `src/lackpy/infer/strategy.py`
- Test: `tests/infer/test_strategy.py`

- [ ] **Step 1: Write failing tests for OneShotStrategy**

```python
# tests/infer/test_strategy.py
"""Tests for InferenceStrategy and OneShotStrategy."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from lackpy.infer.strategy import OneShotStrategy, STRATEGIES
from lackpy.infer.context import StepContext
from lackpy.kit.toolbox import ToolSpec
from lackpy.kit.registry import ResolvedKit
from lackpy.lang.grader import Grade


def _make_kit():
    tools = {
        "glob": ToolSpec(name="glob", provider="builtin", description="Find files",
                         args=[], returns="list[str]", grade_w=1, effects_ceiling=1),
    }
    return ResolvedKit(
        tools=tools, callables={n: lambda *a: None for n in tools},
        grade=Grade(w=1, d=1), description="glob(pattern) -> list[str]: Find files",
    )


def _make_provider(output="glob('**/*.py')"):
    provider = MagicMock()
    provider.name = "test"
    provider.generate = AsyncMock(return_value=output)
    return provider


class TestOneShotStrategy:
    def test_name(self):
        assert OneShotStrategy.name == "1-shot"

    def test_build_returns_step(self):
        provider = _make_provider()
        strategy = OneShotStrategy()
        step = strategy.build(provider)
        assert hasattr(step, "run")

    @pytest.mark.asyncio
    async def test_valid_program_succeeds(self):
        provider = _make_provider("files = glob('**/*.py')\nlen(files)")
        strategy = OneShotStrategy()
        step = strategy.build(provider)
        ctx = StepContext(intent="find python files", kit=_make_kit())
        ctx = await step.run(ctx)
        assert ctx.current.valid is True
        assert "glob" in ctx.current.program

    @pytest.mark.asyncio
    async def test_invalid_program_tries_cleanup(self):
        # First call returns code with import, cleanup should strip it
        provider = _make_provider("import os\nglob('**/*.py')")
        strategy = OneShotStrategy()
        step = strategy.build(provider)
        ctx = StepContext(intent="find files", kit=_make_kit())
        ctx = await step.run(ctx)
        # Cleanup should have stripped the import, so it passes
        assert ctx.current.valid is True
        assert "import" not in ctx.current.program


class TestStrategiesRegistry:
    def test_one_shot_registered(self):
        assert "1-shot" in STRATEGIES

    def test_registry_values_are_classes(self):
        for name, cls in STRATEGIES.items():
            assert hasattr(cls, "name")
            assert hasattr(cls, "build")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Projects/lackpy/main && python -m pytest tests/infer/test_strategy.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement strategy.py**

```python
# src/lackpy/infer/strategy.py
"""Inference strategies: named fold configurations over steps and combinators."""

from __future__ import annotations

from typing import Any, Protocol

from .combinators import Fallback, Sequence
from .steps.generate import GenerateStep
from .steps.validate import ValidateStep
from .steps.cleanup import CleanupStep
from .steps.few_shot import FewShotCorrectStep
from .steps.fresh_fix import FreshFixStep


class InferenceStrategy(Protocol):
    """A named composition of steps that produces a valid lackpy program."""

    name: str

    def build(self, provider: Any) -> Any: ...


class OneShotStrategy:
    """Current behavior: generate → validate → correct on failure.

    Fallback chain: validate as-is, cleanup, few-shot correct, fresh fix.
    """

    name = "1-shot"

    def build(self, provider: Any) -> Any:
        return Sequence([
            GenerateStep(provider),
            Fallback([
                Sequence([ValidateStep()]),
                Sequence([CleanupStep(), ValidateStep()]),
                Sequence([FewShotCorrectStep(provider), ValidateStep()]),
                Sequence([FreshFixStep(provider), ValidateStep()]),
            ]),
        ])


STRATEGIES: dict[str, type] = {
    "1-shot": OneShotStrategy,
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Projects/lackpy/main && python -m pytest tests/infer/test_strategy.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/lackpy/infer/strategy.py tests/infer/test_strategy.py
git commit -m "feat: add OneShotStrategy and STRATEGIES registry"
```

---

### Task 7: Wire strategy into service layer and CLI

**Files:**
- Modify: `src/lackpy/service.py`
- Modify: `src/lackpy/cli.py`
- Modify: `src/lackpy/infer/dispatch.py`

- [ ] **Step 1: Run existing tests as baseline**

Run: `cd ~/Projects/lackpy/main && python -m pytest tests/ -v --ignore=tests/infer/test_steps_generate.py --ignore=tests/infer/test_steps_validate.py --ignore=tests/infer/test_steps_cleanup.py --ignore=tests/infer/test_steps_few_shot.py --ignore=tests/infer/test_steps_fresh_fix.py --ignore=tests/infer/test_context.py --ignore=tests/infer/test_combinators.py --ignore=tests/infer/test_strategy.py -x`
Expected: Existing tests PASS (capture count for comparison)

- [ ] **Step 2: Slim InferenceDispatcher to provider selection**

Add `get_provider()` and `get_providers()` methods to `src/lackpy/infer/dispatch.py`. Keep the existing `generate()` method for backward compatibility during migration.

```python
# Add to InferenceDispatcher class in dispatch.py:

    def get_provider(self) -> Any:
        """Return the first available provider."""
        for provider in self._providers:
            if provider.available():
                return provider
        raise RuntimeError("No inference providers available")

    def get_providers(self) -> list:
        """Return all available providers in priority order."""
        return [p for p in self._providers if p.available()]
```

- [ ] **Step 3: Add mode parameter to LackpyService.generate() and delegate()**

In `src/lackpy/service.py`, add `mode` parameter and strategy-based generation path:

```python
# Add import at top of service.py:
from .infer.strategy import STRATEGIES
from .infer.context import StepContext

# Modify generate() method signature and body:
    async def generate(self, intent: str, kit: str | list[str] | dict | None = None,
                       params: dict[str, Any] | None = None, rules: list | None = None,
                       mode: str | None = None) -> GenerationResult:
        resolved = self._resolve_kit(kit)
        _, params_desc, param_names = self._resolve_params(params, resolved)

        effective_mode = mode or self._config.inference_mode or "1-shot"

        if effective_mode in STRATEGIES:
            strategy_cls = STRATEGIES[effective_mode]
            strategy = strategy_cls()
            provider = self._get_strategy_provider()
            step = strategy.build(provider)
            ctx = StepContext(
                intent=intent, kit=resolved,
                params_desc=params_desc, extra_rules=rules,
            )
            ctx = await step.run(ctx)
            if ctx.current and ctx.current.valid:
                return GenerationResult(
                    program=ctx.current.program,
                    provider_name=ctx.current.trace.provider_name or "unknown",
                    generation_time_ms=sum(p.trace.duration_ms for p in ctx.programs),
                    correction_strategy=ctx.current.trace.step_name if len(ctx.programs) > 2 else None,
                    correction_attempts=max(0, len(ctx.programs) - 1),
                )
            raise RuntimeError(
                f"Strategy '{effective_mode}' failed. Last errors: {ctx.current.errors if ctx.current else 'no programs generated'}"
            )

        # Fallback to legacy dispatcher path
        dispatcher = InferenceDispatcher(providers=self._inference_providers)
        allowed = set(resolved.tools.keys()) | param_names
        return await dispatcher.generate(
            intent=intent, namespace_desc=resolved.description,
            allowed_names=allowed, params_desc=params_desc, extra_rules=rules,
        )

    def _get_strategy_provider(self):
        """Get first available provider for strategy-based generation."""
        dispatcher = InferenceDispatcher(providers=self._inference_providers)
        return dispatcher.get_provider()
```

- [ ] **Step 4: Add mode to delegate()**

In the `delegate()` method of `src/lackpy/service.py`, add `mode` parameter and pass it through to `generate()`:

```python
    async def delegate(self, intent: str, kit: str | list[str] | dict | None = None,
                       params: dict[str, Any] | None = None, sandbox: Any = None,
                       rules: list | None = None,
                       _program_override: str | None = None,
                       mode: str | None = None) -> dict[str, Any]:
```

Pass `mode=mode` in the `self.generate()` call within delegate.

- [ ] **Step 5: Add mode to config**

In `src/lackpy/config.py`, add `inference_mode` field to `LackpyConfig`:

```python
    inference_mode: str | None = None
```

And parse it from `[inference]` section of config.toml:

```python
    inference_mode=inference.get("mode"),
```

- [ ] **Step 6: Add --mode to CLI**

In `src/lackpy/cli.py`, add `--mode` argument to the parser:

```python
    parser.add_argument("--mode", default=None,
                        help="Inference mode: 1-shot, spm (default: from config or 1-shot)")
```

Pass `mode=args.mode` in the delegate/generate calls.

- [ ] **Step 7: Run ALL tests to verify no regression**

Run: `cd ~/Projects/lackpy/main && python -m pytest tests/ -v -x`
Expected: All tests PASS (same count as baseline + new tests)

- [ ] **Step 8: Commit**

```bash
git add src/lackpy/service.py src/lackpy/cli.py src/lackpy/config.py src/lackpy/infer/dispatch.py
git commit -m "feat: wire strategy-based generation into service layer and CLI"
```

---

### Task 8: Add Solve step (unconstrained generation)

**Files:**
- Create: `src/lackpy/infer/steps/solve.py`
- Test: `tests/infer/test_steps_solve.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/infer/test_steps_solve.py
"""Tests for the Solve step — unconstrained Python generation."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from lackpy.infer.steps.solve import SolveStep
from lackpy.infer.context import StepContext
from lackpy.kit.toolbox import ToolSpec
from lackpy.kit.registry import ResolvedKit
from lackpy.lang.grader import Grade


def _make_kit():
    tools = {
        "glob": ToolSpec(name="glob", provider="builtin", description="Find files",
                         args=[], returns="list[str]", grade_w=1, effects_ceiling=1),
        "read": ToolSpec(name="read", provider="builtin", description="Read file",
                         args=[], returns="str", grade_w=1, effects_ceiling=1),
    }
    return ResolvedKit(
        tools=tools, callables={n: lambda *a: None for n in tools},
        grade=Grade(w=1, d=1),
        description="glob(pattern) -> list[str]\nread(path) -> str",
    )


def _make_provider(output):
    provider = MagicMock()
    provider.name = "test"
    provider.generate = AsyncMock(return_value=output)
    return provider


class TestSolveStep:
    @pytest.mark.asyncio
    async def test_generates_unconstrained_python(self):
        # The solver should NOT use lackpy's restricted prompt
        code = "import glob\nfiles = glob.glob('**/*.py', recursive=True)\nlen(files)"
        provider = _make_provider(code)
        step = SolveStep(provider)
        ctx = StepContext(intent="find all python files and count them", kit=_make_kit())
        ctx = await step.run(ctx)
        assert ctx.current is not None
        assert ctx.current.trace.step_name == "solve"
        # Solver does NOT sanitize/cleanup — it preserves standard Python
        assert "import" in ctx.current.program or "glob" in ctx.current.program

    @pytest.mark.asyncio
    async def test_solve_prompt_differs_from_generate(self):
        provider = _make_provider("glob.glob('*.py')")
        step = SolveStep(provider)
        ctx = StepContext(intent="find files", kit=_make_kit())
        ctx = await step.run(ctx)
        # The system prompt should mention standard Python, not restricted
        assert ctx.current.trace.system_prompt is not None
        assert "standard" in ctx.current.trace.system_prompt.lower() or \
               "python" in ctx.current.trace.system_prompt.lower()

    @pytest.mark.asyncio
    async def test_provider_returns_none(self):
        provider = _make_provider(None)
        step = SolveStep(provider)
        ctx = StepContext(intent="find files", kit=_make_kit())
        ctx = await step.run(ctx)
        assert ctx.current.program == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Projects/lackpy/main && python -m pytest tests/infer/test_steps_solve.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement SolveStep**

```python
# src/lackpy/infer/steps/solve.py
"""Solve step: generate standard Python without lackpy restrictions."""

from __future__ import annotations

import time
from typing import Any

from ..context import ProgramState, StepContext, StepTrace
from ..sanitize import sanitize_output


_SOLVE_SYSTEM_PROMPT = """You write short Python scripts to accomplish tasks using pre-loaded helper functions.

Write standard Python. You may use imports, loops, classes — whatever you need.
Keep it short and direct. Output ONLY the code — no markdown, no explanation.

Available helper functions (already imported):
{namespace_desc}
"""


class SolveStep:
    """Generate a standard Python program from intent, unconstrained.

    Unlike GenerateStep, this does NOT apply deterministic_cleanup or
    restrict the prompt to the lackpy subset. The output may contain
    imports, classes, function defs — anything valid Python.
    """

    name = "solve"

    def __init__(self, provider: Any) -> None:
        self._provider = provider

    async def run(self, ctx: StepContext) -> StepContext:
        start = time.perf_counter()
        namespace_desc = ctx.kit.description
        system_prompt = _SOLVE_SYSTEM_PROMPT.format(namespace_desc=namespace_desc)

        raw = await self._provider.generate(ctx.intent, system_prompt)
        elapsed = (time.perf_counter() - start) * 1000

        if raw is None:
            program = ""
        else:
            program = sanitize_output(raw)
            # Intentionally NO deterministic_cleanup — keep standard Python

        ctx.programs.append(ProgramState(
            program=program,
            intent=ctx.intent,
            kit=ctx.kit,
            valid=None,
            errors=[],
            trace=StepTrace(
                step_name=self.name,
                provider_name=self._provider.name,
                model=getattr(self._provider, '_model', None),
                system_prompt=system_prompt,
                user_prompt=ctx.intent,
                raw_output=raw,
                duration_ms=elapsed,
            ),
        ))
        return ctx
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Projects/lackpy/main && python -m pytest tests/infer/test_steps_solve.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/lackpy/infer/steps/solve.py tests/infer/test_steps_solve.py
git commit -m "feat: add Solve step for unconstrained Python generation"
```

---

### Task 9: Add Pick step (AST analysis → derived kit)

**Files:**
- Create: `src/lackpy/infer/steps/pick.py`
- Test: `tests/infer/test_steps_pick.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/infer/test_steps_pick.py
"""Tests for the Pick step — AST analysis to derive kit from solver output."""

import pytest
from lackpy.infer.steps.pick import PickStep
from lackpy.infer.context import StepContext, ProgramState, StepTrace
from lackpy.kit.toolbox import ToolSpec
from lackpy.kit.registry import ResolvedKit
from lackpy.lang.grader import Grade


def _make_kit():
    tools = {
        "glob": ToolSpec(name="glob", provider="builtin", description="Find files",
                         args=[], returns="list[str]", grade_w=1, effects_ceiling=1),
        "read": ToolSpec(name="read", provider="builtin", description="Read file",
                         args=[], returns="str", grade_w=1, effects_ceiling=1),
        "write": ToolSpec(name="write", provider="builtin", description="Write file",
                          args=[], returns="bool", grade_w=3, effects_ceiling=3),
    }
    callables = {n: lambda *a: None for n in tools}
    return ResolvedKit(
        tools=tools, callables=callables,
        grade=Grade(w=3, d=3),
        description="glob(pattern)\nread(path)\nwrite(path, content)",
    )


def _make_trace():
    return StepTrace(
        step_name="solve", provider_name="test", model=None,
        system_prompt=None, user_prompt=None, raw_output=None, duration_ms=0,
    )


def _ctx_with_program(program: str):
    kit = _make_kit()
    ctx = StepContext(intent="test", kit=kit)
    ctx.programs.append(ProgramState(
        program=program, intent="test", kit=kit,
        valid=None, errors=[], trace=_make_trace(),
    ))
    return ctx


class TestPickStep:
    @pytest.mark.asyncio
    async def test_picks_glob_and_read(self):
        ctx = _ctx_with_program(
            "import glob\nfiles = glob.glob('**/*.py')\n"
            "for f in files:\n    content = read(f)\n    print(len(content))"
        )
        step = PickStep()
        ctx = await step.run(ctx)
        # Should derive a kit with glob and read but not write
        derived_tools = set(ctx.current.kit.tools.keys())
        assert "glob" in derived_tools
        assert "read" in derived_tools
        assert "write" not in derived_tools

    @pytest.mark.asyncio
    async def test_picks_from_standard_python_patterns(self):
        # Solver writes glob.glob() — Pick should recognize this maps to the glob tool
        ctx = _ctx_with_program("import glob\nglob.glob('src/**/*.cpp')")
        step = PickStep()
        ctx = await step.run(ctx)
        assert "glob" in ctx.current.kit.tools

    @pytest.mark.asyncio
    async def test_picks_from_open_pattern(self):
        # Solver writes open() — Pick should recognize this maps to read tool
        ctx = _ctx_with_program("with open('README.md') as f:\n    content = f.read()")
        step = PickStep()
        ctx = await step.run(ctx)
        assert "read" in ctx.current.kit.tools

    @pytest.mark.asyncio
    async def test_preserves_program_text(self):
        program = "files = glob('*.py')\nlen(files)"
        ctx = _ctx_with_program(program)
        step = PickStep()
        ctx = await step.run(ctx)
        assert ctx.current.program == program

    @pytest.mark.asyncio
    async def test_no_current_is_noop(self):
        ctx = StepContext(intent="test", kit=_make_kit())
        step = PickStep()
        ctx = await step.run(ctx)
        assert ctx.current is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Projects/lackpy/main && python -m pytest tests/infer/test_steps_pick.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement PickStep**

```python
# src/lackpy/infer/steps/pick.py
"""Pick step: analyze solver output AST to derive which tools are needed."""

from __future__ import annotations

import ast
import re
import time

from ..context import ProgramState, StepContext, StepTrace
from ...kit.registry import ResolvedKit
from ...lang.grader import Grade


# Patterns in standard Python that map to lackpy tools
_TOOL_PATTERNS: dict[str, list[str]] = {
    "glob": ["glob.glob", "glob.iglob", "glob("],
    "read": ["open(", ".read(", "readlines(", "read("],
    "write": [".write(", "write("],
    "edit": ["edit(", ".replace("],
}


def _extract_tool_names(program: str, available_tools: dict) -> set[str]:
    """Identify which available tools the program uses or implies.

    Checks both direct tool calls (e.g. glob('pattern')) and standard
    Python equivalents (e.g. glob.glob(), open().read()).
    """
    used = set()

    # Check direct calls via AST
    try:
        tree = ast.parse(program)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in available_tools:
                    used.add(node.func.id)
    except SyntaxError:
        pass

    # Check standard Python patterns
    for tool_name, patterns in _TOOL_PATTERNS.items():
        if tool_name in available_tools:
            for pattern in patterns:
                if pattern in program:
                    used.add(tool_name)
                    break

    return used


class PickStep:
    """Analyze the current program to derive which tools it needs.

    Pushes a new ProgramState with the same program text but a
    derived kit containing only the tools the program actually uses.
    Does not require an LLM — pure AST and pattern analysis.
    """

    name = "pick"

    async def run(self, ctx: StepContext) -> StepContext:
        if ctx.current is None:
            return ctx

        start = time.perf_counter()
        available = ctx.kit.tools
        used = _extract_tool_names(ctx.current.program, available)

        # Build derived kit with only the tools that were used
        derived_tools = {name: spec for name, spec in available.items() if name in used}
        derived_callables = {name: cb for name, cb in ctx.kit.callables.items() if name in used}

        # Compute aggregate grade from selected tools
        if derived_tools:
            max_w = max(s.grade_w for s in derived_tools.values())
            max_d = max(s.effects_ceiling for s in derived_tools.values())
        else:
            max_w, max_d = 0, 0

        derived_kit = ResolvedKit(
            tools=derived_tools,
            callables=derived_callables,
            grade=Grade(w=max_w, d=max_d),
            description="\n".join(
                f"{s.name}({', '.join(a.name for a in s.args)}) -> {s.returns}: {s.description}"
                for s in derived_tools.values()
            ),
        )

        elapsed = (time.perf_counter() - start) * 1000

        ctx.programs.append(ProgramState(
            program=ctx.current.program,
            intent=ctx.current.intent,
            kit=derived_kit,
            valid=None,
            errors=[],
            trace=StepTrace(
                step_name=self.name,
                provider_name=None,
                model=None,
                system_prompt=None,
                user_prompt=None,
                raw_output=None,
                duration_ms=elapsed,
            ),
        ))
        return ctx
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Projects/lackpy/main && python -m pytest tests/infer/test_steps_pick.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/lackpy/infer/steps/pick.py tests/infer/test_steps_pick.py
git commit -m "feat: add Pick step — AST analysis to derive tool kit"
```

---

### Task 10: Add Restrict step (rewrite to lackpy subset)

**Files:**
- Create: `src/lackpy/infer/steps/restrict.py`
- Test: `tests/infer/test_steps_restrict.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/infer/test_steps_restrict.py
"""Tests for the Restrict step — rewrite standard Python to lackpy subset."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from lackpy.infer.steps.restrict import RestrictStep
from lackpy.infer.context import StepContext, ProgramState, StepTrace
from lackpy.kit.toolbox import ToolSpec
from lackpy.kit.registry import ResolvedKit
from lackpy.lang.grader import Grade


def _make_kit():
    tools = {
        "glob": ToolSpec(name="glob", provider="builtin", description="Find files matching a glob pattern",
                         args=[], returns="list[str]", grade_w=1, effects_ceiling=1),
        "read": ToolSpec(name="read", provider="builtin", description="Read file contents",
                         args=[], returns="str", grade_w=1, effects_ceiling=1),
    }
    return ResolvedKit(
        tools=tools, callables={n: lambda *a: None for n in tools},
        grade=Grade(w=1, d=1),
        description="glob(pattern) -> list[str]: Find files\nread(path) -> str: Read file",
    )


def _make_trace():
    return StepTrace(
        step_name="pick", provider_name=None, model=None,
        system_prompt=None, user_prompt=None, raw_output=None, duration_ms=0,
    )


def _make_provider(output):
    provider = MagicMock()
    provider.name = "test"
    provider.generate = AsyncMock(return_value=output)
    return provider


class TestRestrictStep:
    @pytest.mark.asyncio
    async def test_rewrites_standard_python(self):
        # Provider returns a restricted version when asked to rewrite
        provider = _make_provider("files = glob('**/*.py')\nlen(files)")
        step = RestrictStep(provider)
        kit = _make_kit()
        ctx = StepContext(intent="find python files", kit=kit)
        ctx.programs.append(ProgramState(
            program="import glob\nfiles = glob.glob('**/*.py', recursive=True)\nlen(files)",
            intent="find python files", kit=kit,
            valid=None, errors=[], trace=_make_trace(),
        ))
        ctx = await step.run(ctx)
        assert ctx.current.trace.step_name == "restrict"
        assert ctx.current.program == "files = glob('**/*.py')\nlen(files)"

    @pytest.mark.asyncio
    async def test_restrict_prompt_contains_tool_signatures(self):
        provider = _make_provider("glob('*.py')")
        step = RestrictStep(provider)
        kit = _make_kit()
        ctx = StepContext(intent="find files", kit=kit)
        ctx.programs.append(ProgramState(
            program="import glob\nglob.glob('*.py')",
            intent="find files", kit=kit,
            valid=None, errors=[], trace=_make_trace(),
        ))
        ctx = await step.run(ctx)
        # The system prompt should contain the restricted tool set
        prompt = ctx.current.trace.system_prompt
        assert "glob" in prompt
        assert "read" in prompt

    @pytest.mark.asyncio
    async def test_restrict_prompt_contains_original_program(self):
        original = "import glob\nfiles = glob.glob('*.py')"
        provider = _make_provider("files = glob('*.py')")
        step = RestrictStep(provider)
        kit = _make_kit()
        ctx = StepContext(intent="find files", kit=kit)
        ctx.programs.append(ProgramState(
            program=original, intent="find files", kit=kit,
            valid=None, errors=[], trace=_make_trace(),
        ))
        ctx = await step.run(ctx)
        # User prompt should contain the original program
        user_prompt = ctx.current.trace.user_prompt
        assert original in user_prompt or "glob.glob" in user_prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Projects/lackpy/main && python -m pytest tests/infer/test_steps_restrict.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement RestrictStep**

```python
# src/lackpy/infer/steps/restrict.py
"""Restrict step: rewrite standard Python into the lackpy restricted subset."""

from __future__ import annotations

import time
from typing import Any

from ..context import ProgramState, StepContext, StepTrace
from ..cleanup import deterministic_cleanup
from ..sanitize import sanitize_output
from ...lang.grammar import ALLOWED_BUILTINS


_RESTRICT_SYSTEM_PROMPT = """You rewrite Python code to use ONLY a restricted set of pre-loaded functions.

Rules:
- No imports. All functions are pre-loaded.
- No function definitions, class definitions, while loops, try/except.
- For loops, if/else, list comprehensions, f-strings are allowed.
- Assign tool results to variables and reuse them.

Available functions (use ONLY these):
{namespace_desc}

Allowed builtins: {builtins}

Output ONLY the rewritten code. No explanation, no markdown fences."""

_RESTRICT_USER_PROMPT = """Rewrite this Python code using ONLY the available functions listed above.

Original code:
{program}

Rewritten code:"""


class RestrictStep:
    """Rewrite the current program into the lackpy restricted subset.

    Takes standard Python from a Solve step and rewrites it using
    only the available tool functions and allowed builtins.
    """

    name = "restrict"

    def __init__(self, provider: Any) -> None:
        self._provider = provider

    async def run(self, ctx: StepContext) -> StepContext:
        if ctx.current is None:
            return ctx

        start = time.perf_counter()

        # Use the derived kit from Pick if available, otherwise the input kit
        kit = ctx.current.kit or ctx.kit
        namespace_desc = kit.description
        builtins_str = ", ".join(sorted(ALLOWED_BUILTINS))

        system_prompt = _RESTRICT_SYSTEM_PROMPT.format(
            namespace_desc=namespace_desc,
            builtins=builtins_str,
        )
        user_prompt = _RESTRICT_USER_PROMPT.format(program=ctx.current.program)

        raw = await self._provider.generate(user_prompt, system_prompt)
        elapsed = (time.perf_counter() - start) * 1000

        if raw is None:
            program = ""
        else:
            program = sanitize_output(raw)
            program = deterministic_cleanup(program)

        ctx.programs.append(ProgramState(
            program=program,
            intent=ctx.intent,
            kit=kit,
            valid=None,
            errors=[],
            trace=StepTrace(
                step_name=self.name,
                provider_name=self._provider.name,
                model=getattr(self._provider, '_model', None),
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                raw_output=raw,
                duration_ms=elapsed,
            ),
        ))
        return ctx
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Projects/lackpy/main && python -m pytest tests/infer/test_steps_restrict.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/lackpy/infer/steps/restrict.py tests/infer/test_steps_restrict.py
git commit -m "feat: add Restrict step — rewrite standard Python to lackpy subset"
```

---

### Task 11: Add SPMStrategy and register it

**Files:**
- Modify: `src/lackpy/infer/strategy.py`
- Test: `tests/infer/test_strategy.py` (add SPM tests)

- [ ] **Step 1: Add failing tests for SPMStrategy**

Add to `tests/infer/test_strategy.py`:

```python
from lackpy.infer.strategy import SPMStrategy


class TestSPMStrategy:
    def test_name(self):
        assert SPMStrategy.name == "spm"

    def test_build_returns_step(self):
        provider = _make_provider()
        strategy = SPMStrategy()
        step = strategy.build(provider)
        assert hasattr(step, "run")

    def test_registered(self):
        assert "spm" in STRATEGIES

    @pytest.mark.asyncio
    async def test_spm_end_to_end_with_mock(self):
        # Solver returns standard Python, Restrict returns lackpy-compatible
        call_count = 0
        async def mock_generate(intent, namespace_desc, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Solve step: return standard Python
                return "import glob\nfiles = glob.glob('**/*.py')\nlen(files)"
            else:
                # Restrict step: return lackpy-compatible
                return "files = glob('**/*.py')\nlen(files)"

        provider = MagicMock()
        provider.name = "test"
        provider.generate = AsyncMock(side_effect=mock_generate)

        strategy = SPMStrategy()
        step = strategy.build(provider)
        ctx = StepContext(intent="find python files and count them", kit=_make_kit())
        ctx = await step.run(ctx)

        assert ctx.current.valid is True
        assert "import" not in ctx.current.program
        assert "glob(" in ctx.current.program
        assert call_count >= 2  # at least Solve + Restrict
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Projects/lackpy/main && python -m pytest tests/infer/test_strategy.py::TestSPMStrategy -v`
Expected: FAIL with `ImportError: cannot import name 'SPMStrategy'`

- [ ] **Step 3: Implement SPMStrategy**

Add to `src/lackpy/infer/strategy.py`:

```python
from .steps.solve import SolveStep
from .steps.pick import PickStep
from .steps.restrict import RestrictStep


class SPMStrategy:
    """Solve-Pick-Restrict: separate problem solving from constraint compliance.

    1. Solve: generate unconstrained standard Python
    2. Pick: analyze AST to determine which tools are needed
    3. Restrict: rewrite into lackpy subset (with fallback correction)
    """

    name = "spm"

    def build(self, provider: Any) -> Any:
        return Sequence([
            SolveStep(provider),
            PickStep(),
            Fallback([
                Sequence([RestrictStep(provider), ValidateStep()]),
                Sequence([RestrictStep(provider), CleanupStep(), ValidateStep()]),
                Sequence([RestrictStep(provider), FewShotCorrectStep(provider), ValidateStep()]),
            ]),
        ])
```

Add to STRATEGIES dict:

```python
STRATEGIES: dict[str, type] = {
    "1-shot": OneShotStrategy,
    "spm": SPMStrategy,
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Projects/lackpy/main && python -m pytest tests/infer/test_strategy.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/lackpy/infer/strategy.py tests/infer/test_strategy.py
git commit -m "feat: add SPMStrategy (solve-pick-restrict) and register it"
```

---

### Task 12: Full regression and integration test

**Files:**
- Test: run all existing + new tests

- [ ] **Step 1: Run full test suite**

Run: `cd ~/Projects/lackpy/main && python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: Manual smoke test with 1-shot mode (should match old behavior)**

Run: `cd ~/Projects/lackpy/main && lackpy delegate "find all test files" --kit glob --mode 1-shot`
Expected: Same behavior as before the refactor

- [ ] **Step 3: Manual smoke test with spm mode**

Run: `cd ~/Projects/lackpy/main && lackpy delegate "find all test files" --kit read,glob --mode spm`
Expected: Generates valid lackpy program via solve→pick→restrict pipeline

- [ ] **Step 4: Commit any fixes needed**

- [ ] **Step 5: Final commit with all tests passing**

```bash
git add -A
git commit -m "test: verify full regression and SPM integration"
```
