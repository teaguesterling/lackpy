# Design direction (intent, not yet realized)

!!! warning "This page describes intent, not current behavior"
    Everything below is **maintainer design direction** — where lackpy is *heading*,
    not what the code does today. For how lackpy actually works now, see
    [Architecture](../concepts/architecture.md), [Kits & Toolbox](../concepts/kits.md),
    and the [Inference Pipeline](../concepts/inference.md). Treat this page as a
    roadmap to bias decisions toward; flag where current structure diverges.

## What lackpy fundamentally is

lackpy started as a **safe Python interpreter for subagents** — Python with the
dangerous pieces *lacking*, so a subagent (a "lackey") can run generated code
safely. That safe-interpreter + program-generation pairing is the **core**.

The maintainer's framing: lackpy is *"actually 3 or 4 different programs in one"* —
roughly (1) the safe interpreter / execution model, (2) program generation
(intent → program), (3) kits / config, and (4) literate rendering. Generation and
execution are tightly coupled today; the long-term aim is to make these seams
explicit and separable.

## Directions

### 1. Split generation from execution
The interpreter/execution model and the intent→program generation layer should be
cleanly separable, even though they're coupled today. The `lackpy-lang` extraction
(its own PEP 420 namespace distribution) was the first step in this direction.

### 2. Generalize "kits" into a runtime config system
"kit" is a lackpy term the maintainer is **deliberately retiring**. The intent is to
generalize kits into a config system that configures the runtime *and the language
the runtime uses*, rather than kits being separate components.

!!! info "Current state"
    Kits are still **load-bearing today**: kit resolution is stage 1 of the
    `delegate()` pipeline, `kit_default` is a config field, and `kit` is a parameter
    on `delegate` / `run_program`. The generalization above is a goal, not done.

### 3. Literate lackpy as a first-class capability
The "literate" style — the agent writes a markdown-style document that is rendered
as it executes, then the rendered result is opened (e.g. via `xdg-open`) — is
considered valuable on its own, not just a presentation mode.

## Integration: cosmic-fabric

lackpy is meant to plug into **cosmic-fabric** (`cosmic-fabricd`), a canonical Rust
router that handles several routes to an inferencer consistently:

- cosmic-fabric is the canonical router and must have **no Python dependencies**.
- lackpy is invoked as a **subprocess** (for now) or **via its MCP server** — not
  linked in. The handoff seam is lackpy's **`delegate` tool**; lackpy may also
  expose its templates.
- lackpy works with both cosmic-fabric and raw Ollama. Tools handed across the seam
  should carry full specs (params, docs), and lackpy-specific terminology (e.g.
  "kit") should not bleed into cosmic-fabric.

## Model choice is local, not a default

The best model is a **per-machine / per-deployment** decision. The package default
stays generic (`qwen2.5-coder:1.5b`); the real choice lives in the (gitignored)
`.lackpy/config.toml`. See the note in [Inference Pipeline](../concepts/inference.md).
