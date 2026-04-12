# Kibitzer ↔ Lackpy Integration — Discussion Prompt

*For a Claude session working on the kibitzer side of the integration.*

## Context for the kibitzer dev

You are working on **kibitzer** (v0.2.0), a Claude Code extension that watches agent tool calls and suggests structured alternatives. The project lives at `~/Projects/kibitzer/`.

A parallel session has been working on **lackpy** (just released v0.6.0), a micro-inferencer that translates natural-language intents into programs executed by local LLMs (0.5b–7b). Lackpy is a **delegation target** — large models push structural read-and-fetch work down to fast local models so they don't waste context tokens on file reads, grep, and cross-referencing.

## What lackpy just learned (v0.6.0 eval findings)

A 1344-cell empirical sweep across 6 models × 4 interpreters × 4 prompt variants produced these findings:

### Interpreter-specialized prompts are transformative
- The generic Jupyter-cell prompt produces **11% pass rate** across all models
- Interpreter-specific prompts (`system_prompt_hint()`) produce **36-40%** — a 3-5× improvement
- ast-select goes from **0% → 93%** on smollm2 with just a syntax reference prompt
- qwen2.5-coder:7b goes from **14% → 86%** on python with an "orchestrate, don't implement" framing

### The top failure modes are prompt-addressable
1. **"Implement don't orchestrate" (40 failures)**: Models define `def find_def(name): ...` instead of calling the pre-loaded tool. The specialized prompt fixes this but kibitzer could also catch it via pattern detection.
2. **Path prefix errors (14 failures)**: Models use `toybox/app.py` instead of `app.py`. Addressable via prompt or via kibitzer intercepting path-prefixed calls.
3. **`open()` usage instead of `read_file()` (stdlib leak)**: Models default to Python stdlib. The prompt says "do NOT use open()" but kibitzer's before_call interception could redirect this.

### Model-specific prompt needs
- **Small models (smollm2, 1.5b)**: Work best with bare specialized prompts. Few-shot examples and constraints hurt them (context overload).
- **Large coder models (3b, 7b)**: Need few-shot examples to overcome code-completion priors. Constraints variant is best for 7b.
- **Non-coder models (qwen2.5:1.5b)**: Most prompt-obedient; work across all variants.

## What's already wired (service.py)

Lackpy already has a kibitzer integration in `LackpyService`:
- `_init_kibitzer()` creates a `KibitzerSession` and registers tools with grades
- `delegate()` calls `register_context()`, `validate_calls()`, `get_suggestions()`, `report_generation()`, and `save()`
- `RestrictedRunner.run()` accepts `kibitzer_session=` and calls `after_call()` on each tool invocation

This was built against the spec at `docs/superpowers/specs/2026-03-31-kibitzer-integration.md`.

## Integration directions to discuss

### 1. Kibitzer as prompt advisor

Lackpy now has `system_prompt_hint()` on each interpreter. But the hint is static — it doesn't adapt to the model or to observed failure modes. Kibitzer sees every generation outcome via `report_generation()`. Could kibitzer:
- Track which failure modes a model exhibits (implement-don't-orchestrate, path prefix, stdlib usage)?
- Feed that signal back to lackpy's prompt builder as dynamic negative constraints?
- Effectively close the loop: **model generates → kibitzer observes failure → prompt adjusts → next generation improves**?

This would make the prompt system adaptive rather than static. The eval harness proved that the RIGHT prompt makes a 3-5× difference; kibitzer could be the system that selects the right prompt dynamically.

### 2. Kibitzer interceptors for common model mistakes

The before_call API could intercept predictable model errors:
- `open("app.py")` → redirect to `read_file("app.py")` with a coaching suggestion
- `find_files("toybox/app.py")` → strip the `toybox/` prefix
- A generated program that defines `def find_def(name): ...` → kibitzer could flag the FunctionDef before execution (though lackpy's validator already catches this)

Question: should these be kibitzer interceptors, or should lackpy handle them in its own sanitization/correction pipeline? Where's the right boundary?

### 3. Grade-aware interpreter selection

Kibitzer already knows tool grades (`w` and `d`). If lackpy gains automatic interpreter selection (the deferred "quartermaster" feature), kibitzer's mode policy could influence which interpreter is chosen:
- In read-only mode → prefer ast-select or plucker (both read-only by nature)
- In write mode → allow the python interpreter with write tools
- Kibitzer's path guards → constrain which files the interpreter can target

### 4. Generation outcome tracking for agent-riggs

The `report_generation()` call already sends intent, program, provider, correction info, and success status. With v0.6.0's interpreter-aware prompts, this could also include:
- Which interpreter was used
- Which prompt variant was selected
- Whether the specialized prompt was used vs. baseline
- The model's score (0/1/2) if the eval harness was involved

Agent-riggs could then build a project-level model of "which (model, interpreter, prompt) works best for this project's tasks."

### 5. Kibitzer coaching for the correction chain

When a generated program fails validation, lackpy's `CorrectionChain` attempts deterministic cleanup, then a few-shot retry, then a fresh fixer. Kibitzer's `get_suggestions()` could feed into this chain:
- If kibitzer has seen the same failure pattern before, it could provide a targeted fix suggestion
- If the failure is a known model quirk (e.g., "qwen2.5-coder:3b always uses `open()` on first try"), kibitzer could tell the correction chain to apply the `open() → read_file()` substitution before retrying

## Questions for the kibitzer dev

1. **Is the before_call / after_call API stable enough** for lackpy to depend on for tool-call interception? Or is it still evolving?

2. **Does kibitzer have (or plan) a "failure pattern" tracker** that accumulates across sessions? The eval data shows failure modes are highly model-specific — a per-model pattern database would be extremely valuable.

3. **What's the right shape for kibitzer to return prompt hints?** Currently `get_suggestions()` returns string suggestions. For prompt adaptation, we'd need something more structured — e.g., `{"add_constraint": "do NOT use open()", "reason": "model has used open() in 3 of last 5 generations"}`.

4. **Should kibitzer own the model→prompt mapping?** Lackpy has the interpreter hints; the eval harness has the trial data; kibitzer sees the live generation outcomes. One of these three should be the authoritative source for "what prompt works for this model." Which?

5. **What's the integration testing story?** Lackpy's test suite currently mocks kibitzer out. Should we have a shared integration test that runs both?
