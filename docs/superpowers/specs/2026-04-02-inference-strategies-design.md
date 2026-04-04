# Inference Strategies: Composable Step-Based Pipeline

**Date:** 2026-04-02
**Status:** Design

## Problem

The current inference pipeline (`InferenceDispatcher`) conflates two concerns: provider fallback and generation strategy. All generation follows a single approach — generate a lackpy program directly from intent, then correct on failure. This makes it hard to add fundamentally different approaches like SPM (solve-pick-restrict) where an unconstrained model solves the problem first and a second pass conforms the solution to lackpy's restricted subset.

Small models (1.5B-3B) struggle to simultaneously understand the user's intent and comply with lackpy's AST restrictions. Separating problem-solving from constraint-compliance lets each step focus on one job.

## Design

### Core insight

The inference pipeline is a fold. Each step is a stateless function over managed state (`StepContext`). Steps are composed via two combinators — Sequence and Fallback — into strategies that define the shape of the pipeline. Validation is an explicit gate, not an implicit side effect, allowing intermediate steps to produce intentionally "invalid" output that later steps refine.

Reference: [Conversations Are Folds](https://judgementalmonad.com/blog/ma/06-conversations-are-folds) — the same architecture of stateless step functions iterated over managed state, with the strategy as the fold and the context as the accumulator.

### StepContext — the accumulator

```python
@dataclass
class StepTrace:
    """Operational record of one step execution."""
    step_name: str
    provider_name: str | None
    model: str | None
    system_prompt: str | None
    user_prompt: str | None
    raw_output: str | None
    duration_ms: float

@dataclass
class ProgramState:
    """Semantic result of one generation/transform step."""
    program: str
    intent: str                  # may differ per step (solver vs restrictor get different intents)
    kit: ResolvedKit | None      # may differ per step (Pick derives a new kit)
    valid: bool | None
    errors: list[str]
    trace: StepTrace

@dataclass
class StepContext:
    # Input (set at strategy entry, immutable across the fold)
    intent: str
    kit: ResolvedKit             # bundles tools, allowed_names, grade, namespace_desc
    params_desc: str | None = None
    extra_rules: list | None = None

    # Accumulated state
    programs: list[ProgramState] = field(default_factory=list)

    # Provider access
    provider: Any = None

    @property
    def current(self) -> ProgramState | None:
        return self.programs[-1] if self.programs else None
```

Each step reads from `ctx.current` and pushes a new `ProgramState` onto `ctx.programs`. The full history of generation attempts is preserved as a flat list — including failed Fallback branches — giving complete observability of everything that was tried.

`ResolvedKit` bundles tools, allowed names, grade, and namespace description. It is input context, not something steps mutate on `StepContext` directly. When the Pick step derives a new kit, it lives on the `ProgramState` it produces.

### Steps — the fold functions

```python
class Step(Protocol):
    name: str
    async def run(self, ctx: StepContext) -> StepContext: ...
```

Steps are pure transforms: read context, do one thing, return context. Steps that produce programs push a `ProgramState`. Steps that evaluate (Validate) mutate `ctx.current` in place.

| Step | Reads | Produces | Needs LLM |
|---|---|---|---|
| **Generate** | intent, kit namespace | ProgramState: lackpy-constrained program | Yes |
| **Solve** | intent (unconstrained prompt) | ProgramState: standard Python | Yes |
| **Pick** | current.program (AST analysis) | ProgramState: same program, derived kit | No |
| **Restrict** | current.program, current.kit | ProgramState: rewritten lackpy program | Yes |
| **Cleanup** | current.program | ProgramState: deterministic AST rewrites | No |
| **Validate** | current.program, kit.allowed_names | mutates current: sets valid, errors | No |
| **FewShotCorrect** | current.program, current.errors, intent | ProgramState: corrected program | Yes |
| **FreshFix** | current.program, current.errors | ProgramState: fixer-prompted program | Yes |

Each `ProgramState` captures its own intent (which may differ from `ctx.intent` — the Restrict step's "intent" is a rewriting instruction, not the user's original request), kit, and a `StepTrace` with operational details (prompts, model, timing, raw output).

### Combinators

**Sequence** — run steps in order, threading context through:

```python
class Sequence(Step):
    def __init__(self, steps: list[Step]): ...

    async def run(self, ctx: StepContext) -> StepContext:
        for step in self.steps:
            ctx = await step.run(ctx)
        return ctx
```

**Fallback** — try branches in order, return the first where `ctx.current.valid == True`. Each branch receives the same input context (the state at Fallback entry). All `ProgramState` entries from all branches — successful and failed — accumulate in the flat `programs` list, giving complete observability of everything tried:

```python
class Fallback(Step):
    def __init__(self, branches: list[Step]): ...

    async def run(self, ctx: StepContext) -> StepContext:
        snapshot = len(ctx.programs)  # remember entry point
        for branch in self.branches:
            branch_ctx = ctx  # each branch starts from the same input state
            branch_ctx = await branch.run(branch_ctx)
            if branch_ctx.current and branch_ctx.current.valid:
                return branch_ctx
            # failed branch's ProgramStates remain in the list
        return branch_ctx  # last failure, errors preserved
```

### Strategies — named fold configurations

```python
class InferenceStrategy(Protocol):
    name: str
    def build(self, provider: Any) -> Step: ...
```

**OneShotStrategy** — current behavior, refactored:

```python
class OneShotStrategy:
    name = "1-shot"

    def build(self, provider):
        return Sequence([
            Generate(provider),
            Fallback([
                Sequence([Validate()]),
                Sequence([Cleanup(), Validate()]),
                Sequence([FewShotCorrect(provider), Validate()]),
                Sequence([FreshFix(provider), Validate()]),
            ])
        ])
```

**SPMStrategy** — solve, pick, restrict:

```python
class SPMStrategy:
    name = "spm"

    def build(self, provider):
        return Sequence([
            Solve(provider),
            Pick(),
            Fallback([
                Sequence([Restrict(provider), Validate()]),
                Sequence([Restrict(provider), Cleanup(), Validate()]),
                Sequence([Restrict(provider), FewShotCorrect(provider), Validate()]),
            ])
        ])
```

Strategies can use different providers at different stages. SPM could use a fast model for Solve and a careful model for Restrict by constructing steps with specific providers.

### Provider fallback

Two complementary mechanisms:

**Transparent fallback:** The `InferenceDispatcher` is slimmed down to provider selection. It implements the same interface as a single provider but tries multiple backends (templates, rules, Ollama, Anthropic) under the hood. Steps see a single provider.

```python
class InferenceDispatcher:
    """Tries providers in priority order. Implements provider interface."""

    def __init__(self, providers: list[Any]): ...

    def get_provider(self) -> Any:
        """First available provider."""
        ...

    def get_providers(self) -> list[Any]:
        """All available providers in priority order."""
        ...
```

**Explicit fallback:** A strategy deliberately uses different providers at different stages via Fallback branches — e.g., try Ollama for Restrict, escalate to Anthropic if that fails. This is an intentional architectural choice, not a retry.

### Service layer integration

```python
STRATEGIES = {
    "1-shot": OneShotStrategy,
    "spm": SPMStrategy,
}

class LackpyService:
    async def generate(self, intent, kit, mode="1-shot", ...):
        provider = self._dispatcher.get_provider()
        strategy = STRATEGIES[mode]()
        step = strategy.build(provider=provider)
        ctx = StepContext(intent=intent, kit=resolved_kit, ...)
        ctx = await step.run(ctx)
        return self._build_generation_result(ctx)
```

### CLI integration

```
lackpy delegate "find all test files" --kit read,glob --mode spm
lackpy delegate "find all test files" --kit read,glob --mode 1-shot
```

Default mode is `1-shot` (preserves current behavior). Mode is also settable in `.lackpy/config.toml`:

```toml
[inference]
mode = "1-shot"
```

## Migration path

1. Add `StepContext`, `ProgramState`, `StepTrace` dataclasses
2. Extract existing logic into Step classes (Generate, Validate, Cleanup, FewShotCorrect, FreshFix)
3. Add Sequence and Fallback combinators
4. Build `OneShotStrategy` — reproduces current behavior exactly
5. Slim down `InferenceDispatcher` to provider selection only
6. Wire `--mode` into CLI and service layer
7. Verify refactor: existing tests pass against `OneShotStrategy`
8. Add new steps (Solve, Pick, Restrict) and `SPMStrategy`

Steps 1-7 are a refactor with no behavior change. Step 8 is the new feature.

## File layout

```
src/lackpy/infer/
    dispatch.py          # slimmed InferenceDispatcher (provider selection)
    strategy.py          # InferenceStrategy protocol, STRATEGIES registry
    context.py           # StepContext, ProgramState, StepTrace
    combinators.py       # Sequence, Fallback
    steps/
        __init__.py
        generate.py      # Generate step (current 1-shot generation)
        solve.py         # Solve step (unconstrained generation)
        pick.py          # Pick step (AST analysis -> derived kit)
        restrict.py      # Restrict step (rewrite to lackpy subset)
        cleanup.py       # Cleanup step (deterministic AST rewrites)
        validate.py      # Validate step (AST whitelist check)
        few_shot.py      # FewShotCorrect step
        fresh_fix.py     # FreshFix step
```
