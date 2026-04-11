---
title: Prompt Evaluation Harness — Design
date: 2026-04-11
status: draft
scope: research tooling (scripts/, tests/eval/); no production infer changes in v1
---

# Prompt Evaluation Harness — Design

## Motivation

Lackpy is a delegation target: larger, more capable models push structural read-and-fetch grunt work down to a fast local model so the orchestrator doesn't waste context tokens on grep, read, and cross-reference ceremony. The quality of that delegation hinges almost entirely on the **system prompt** paired with each interpreter.

Today, lackpy has one generic system prompt in `src/lackpy/infer/prompt.py`. It frames the task as a Jupyter kernel cell, which tests well for the `python` interpreter but is not adapted to `ast-select`, `pss`, or `plucker` — each of which targets a different output language with a different cognitive task. Prior evaluation work (the `scripts/pluckit-*` scripts, the findings captured in the auto-memory) touched only the `python` path and the fluent-chain assembler, and most findings are 10+ days old.

This spec defines an **empirical study**: an offline sweep across `{interpreter × prompt-variant × model × intent}`, scored by a hybrid (structural-gate + execution) rubric, emitting JSONL so results are queryable mid-run. The output is data plus a recommendations report — no changes to `src/lackpy/infer/` in this phase. Wiring the winning prompts back into production is explicitly a follow-up once we know what the winners are.

## Goals and non-goals

**In scope (v1):**
- Author a delegation-shaped intent corpus per interpreter, grounded in a realistic fixture repo.
- Sweep a matrix of prompts × models × intents × interpreters against live Ollama.
- Hybrid scoring: structural gate first, execution against the fixture only on gate-pass.
- Incremental JSONL output so partial results are queryable mid-run.
- Regression canaries under `tests/eval/` with loose initial thresholds.
- Committed report identifying the best (prompt, model) per interpreter and the failure modes observed.

**Out of scope (v1):**
- Changes to `src/lackpy/infer/prompt.py`, the dispatcher, or any production inference code.
- Wiring a per-interpreter prompt API into `LackpyService.generate()`.
- Testing correction/retry strategies (`spm`, `1-shot`, etc.) — the harness hits provider `generate()` directly and scores raw output.
- Mutation tasks (rename, addParam, refactor). Read-only intents only. Mutation evaluation is a deferred v2 harness.
- Cloud provider evaluation. Ollama only.

## Approach at a glance

Flat Python scripts under `scripts/prompt-eval/`, in the same style as `scripts/pluckit-quartermaster.py`: streaming Ollama calls with per-call timeouts, incremental JSONL, resumable runs, tqdm progress, graceful Ctrl+C. Regression canaries live under `tests/eval/`.

Three phases:

1. **Phase 1a — Qualifier (~15–25 min)**: all Ollama models × 1 interpreter (`python`) × baseline prompt × the 8 core `python` intents. Cheap filter that picks a top cohort for the full grid. Models that can't handle the easiest task on the most forgiving interpreter don't waste budget on the rest.
2. **Phase 1b — Grid (~2–4 hours)**: qualifier winners × 4 interpreters × 4 prompt variants × 14 intents. This is the main matrix.
3. **Phase 2 — Refinement (~1 hour)**: top 2 `(prompt, model)` per interpreter × axis variations (temperature, example count, constraint phrasing). Narrow sweep to characterize the winners.

Results are dumped to `results/prompt-eval-YYYY-MM-DD/` as JSONL; a `report.py` step consolidates to `report.md`.

## Layout

```
scripts/prompt-eval/
  __init__.py
  harness.py          # orchestrator — iterate matrix, emit JSONL row per cell
  runner.py           # ollama streaming client (timeout, exception catch, token counts)
  prompts.py          # variants dict: {interpreter: {variant_id: prompt_template}}
  intents.py          # corpora per interpreter: core[] + stretch[] with gates & assertions
  fixtures.py         # loader for tests/eval/fixtures/toybox/ + toybox-hash pinning
  scoring.py          # structural gates + execution harness (reuses lackpy.interpreters)
  query.py            # read a JSONL mid-run, print live per-model/per-prompt summary
  report.py           # JSONL -> consolidated JSON + report.md
  phase1a_qualifier.py  # phase 1a entry point
  phase1b_grid.py       # phase 1b entry point
  phase2_refinement.py  # phase 2 entry point
tests/eval/
  __init__.py
  fixtures/toybox/      # 9-file, ~700 LOC webapp with intentional code smells
  test_prompt_canaries.py  # @pytest.mark.slow, skip if no ollama, threshold checks
results/prompt-eval-YYYY-MM-DD/
  phase1a-qualifier.jsonl
  phase1b-grid.jsonl
  phase2-refinement.jsonl
  report.md
```

## The fixture repo: `tests/eval/fixtures/toybox/`

A small, realistic Flask-ish webapp (9 files, ~700 LOC) with **deliberate, auditable code smells**. The fixture is committed to git; every harness run pins the toybox's content hash in the JSONL header so cross-session comparisons are valid.

```
toybox/
  __init__.py
  app.py       # @route handlers; uses execute_sql; hardcoded paths; no error handling
  auth.py      # validate_token, hash_password, check_permissions;
               # @deprecated parse_legacy_token; SQL-string-concat bug
  models.py    # class User, Session, AuditLog; mutable default args bug; missing hints
  db.py        # execute_sql, get_connection, transaction; 1 resource-leak function
  api_v1.py    # older names: get_usr, save_usr, del_usr, list_usrs
  api_v2.py    # newer idiomatic names: get_user, save_user, delete_user, list_users
  utils.py     # parse_date (@deprecated), format_user, compute_hash (duplicate, dead)
  errors.py    # AuthError, ValidationError, DatabaseError classes
  config.py    # hardcoded DB path, SECRET_KEY literal, LOG_LEVEL literal
  tests/
    test_app.py    # test_login_flow, test_user_list
    test_auth.py   # test_validate_token, test_hash_password
    test_models.py # test_user_create
```

**Stable properties the toybox guarantees**:
- 2 `@deprecated` decorated functions (known names, known files).
- 4 `@route(...)` decorated functions in `app.py`.
- 3 SQL-building-via-string-concat call sites (security smell, auditable).
- 1 resource-leak function (opens connection, never closes).
- 2 mutable-default-args bugs.
- 4 `test_*` functions across `tests/`.
- 2 duplicate hash implementations across `auth.py` and `utils.py`.
- 4-to-4 rename correspondence between `api_v1.py` and `api_v2.py`.
- Stable line numbers so execution assertions can be literal constants.

## Corpus shape

Each intent is a structured dataclass:

```python
@dataclass
class Intent:
    id: str                          # stable identifier, e.g. "py.core.01"
    interpreter: str                 # "python" | "ast-select" | "pss" | "plucker"
    difficulty: str                  # "core" | "stretch"
    text: str                        # natural-language request given to the model
    return_shape: str                # "list[str]", "dict", "int", "markdown", etc.
    structural_gate: Callable[[str], GateResult]  # does the raw generation parse/validate?
    exec_assertion: Callable[[Any], bool]         # does the execution output match ground truth?
    notes: str = ""                  # what this intent is stressing
```

Per interpreter: **8 core + 6 stretch = 14 intents**, 4 interpreters × 14 = **56 intents total**. Core items establish baselines; stretch items test compositional depth and grammar frontier.

### `python` interpreter corpus (orchestration via kit tools)

The `python` interpreter orchestrates `read_file`, `find_files`, `find_definitions`, `find_callers`, and `find_files` from the default kit. Intents target the "Implement don't orchestrate" failure mode directly — they're phrased in ways that should lead the model to tool calls, not code synthesis.

**Core (8):** single- and small-composition lookups that return a usable artifact for an orchestrator.
1. Find the definition of `validate_token`, return file path and body.
2. Find all callers of `execute_sql`, return as a list of function names.
3. Find every file that defines `hash_password`, return file paths.
4. Read `app.py`, return full contents.
5. Find the definition of `User`, return the file it's in and the first 10 lines.
6. Find all callers of `validate_token`, return a dict mapping caller name → file.
7. Find every test file in the toybox, return as a list of paths.
8. Find the definition of `DatabaseError`, return the containing file contents.

**Stretch (6):** 2–3-call compositions that require result iteration.
1. For every caller of `execute_sql`, return file path and caller name as pairs.
2. Find all functions named `hash_password` across the codebase, return the count.
3. Find every test file under `tests/`, return a dict filename → contents.
4. Find callers of `validate_token`, return the union of files they live in.
5. Find the definition of `User` and all its callers, return a dict with keys `definition` and `callers`.
6. Find every function whose name starts with `test_`, return their defining files.

### `ast-select` interpreter corpus (bare CSS selector → markdown view)

The selector itself is the program. Every intent is phrased as "show me X as a view", matching the real delegation pattern: the orchestrator asks for a curated slice of the code and gets markdown back.

**Core (8):** single-axis selectors.
1. Show every function → `.fn`
2. Show the `User` class → `.cls#User`
3. Show the `validate_token` function → `.fn#validate_token`
4. Show every function whose name starts with `test_` → `.fn[name^="test_"]`
5. Show every `@deprecated` function → `.fn:has(.decorator#deprecated)` (or pluckit's decorator syntax equivalent)
6. Show every async function → `.fn:async`
7. Show every `@route` handler → `.fn:has(.decorator#route)`
8. Show every function whose name starts with an underscore → `.fn[name^="_"]`

**Stretch (6):** nested, combinator, and `:has`/`:not` selectors.
1. Show every private method of the `User` class → `.cls#User .fn[name^="_"]`
2. Show every function that calls `execute_sql` → `.fn:has(.call#execute_sql)`
3. Show every function that is not a test → `.fn:not([name^="test_"])`
4. Show every route handler that calls `execute_sql` → combined `:has`
5. Show every class that has an `__init__` method → `.cls:has(.fn#__init__)`
6. Show every function that raises an exception → `.fn:has(.raise)`

### `pss` interpreter corpus (selector sheets → curated multi-section views)

The real use case for pss is the **review view**: a multi-rule sheet that produces a structured markdown document combining several slices. Intents are "create a view of X, Y, Z" — exactly what a big model would push down when it wants review context without burning tokens on multiple file reads.

**Core (8):** 1–2 rule sheets.
1. Show `validate_token` with its body — 1-rule sheet.
2. Show the `User` class outline — 1-rule.
3. Show every route handler as a signature — 1-rule with filter.
4. Show every test function with signatures — 1-rule.
5. Show `validate_token` body and `hash_password` body — 2-rule.
6. Show the `User` class body and the `Session` class outline — 2-rule mixed.
7. Show every async function body.
8. Show every `@deprecated` function body.

**Stretch (6):** 2–3-rule curated views.
1. Security review: every function containing `execute_sql` with body, every `validate_*` with signature, every route handler with signature — 3-rule.
2. Documentation view of the auth module: every function with body, every class with outline — 2-rule.
3. Test-surface view: every test function as signature, every class under test as outline — 2-rule.
4. Public functions as signatures and private functions as bodies — 2-rule with `:not` and attr.
5. Show every function in `api_v2.py` with its body — file-scoped 1-rule.
6. Deprecated-code view: every `@deprecated` function body and its callers as signatures — 2-rule with relationship (hardest, grammar-contingent).

### `plucker` interpreter corpus (fluent chain, read-only in v1)

Intents make the **terminal operation explicit** in the text ("return the names of…", "count…", "view…"). This both sets return-shape expectations and gives the model a directional hint.

**Core (8):** single-chain with simple terminal.
1. Count every function → `source().find('.fn').count()` → `int`
2. Return the names of every class → `.names()` → `list[str]`
3. Count the matches for `validate_token` → `int == 1`
4. Return the names of every async function → `list[str]`
5. View the `User` class → `.view()` → markdown
6. Count every test function → `int`
7. Return the names of every method inside the `User` class → `list[str]`
8. View every `@deprecated` function → markdown

**Stretch (6):** deeper chains and relationship traversal.
1. Return the names of every caller of `validate_token` → `.callers().names()`
2. Return the names of every function whose complexity is greater than 5 → `.filter(...)`
3. View every route handler.
4. Return the names of every function that calls `execute_sql`.
5. View every function defined in `api_v2.py`.
6. Return the count of test functions and non-test functions as a tuple — tests plucker-as-Python composition.

### Corpus caveats

- **Pluckit grammar-contingent items**: several stretch items for `ast-select`, `pss`, and `plucker` depend on whether pluckit's selector grammar actually supports the construct (e.g. `:has(.decorator#foo)`, `:has(.raise)`, `.fn:not(...)` with attribute combo, `.callers()` on a `Selection`). These will be verified against the installed pluckit source during implementation. Unsupported items are either dropped from the corpus or filed as issues against pluckit. The report will note any drops.
- **Seeding**: the plucker corpus seeds from the 12 intents in `scripts/pluckit-model-compare.py` and `scripts/pluckit-quartermaster.py`, adapted to the toybox. The `python` corpus seeds from examples in `tests/infer/test_retrieval.py`. `ast-select` and `pss` are hand-authored against the toybox since no prior corpus exists.

## Prompt variants

Four variants per interpreter, forming a ladder. The delta between each rung isolates one axis.

| # | Variant ID | Description |
|---|------------|-------------|
| 1 | `baseline` | The current production `build_system_prompt()` verbatim — the Jupyter-cell framing. Shared across interpreters (this is what makes it baseline). |
| 2 | `specialized` | A prompt written specifically for the interpreter's output language and cognitive task. E.g., for `ast-select`: "You write a single CSS selector, nothing else. No code fences, no explanation, no Python, no chain — just the selector." For `python`: a reframe of the baseline that explicitly says "orchestrate tools, do not implement" to target the memory-captured failure mode. |
| 3 | `specialized + few-shot` | Variant 2 plus 3–5 hand-picked examples matching the expected output shape. Leverages the retrieval-beats-stuffing finding: fewer, more relevant examples > many generic ones. |
| 4 | `specialized + few-shot + constraints` | Variant 3 plus an explicit constraints list ("never emit code fences", "never define new functions", "never use imports", "output must be a single line"). Tests whether explicit negative guidance helps or primes the forbidden constructs — the auto-memory flags this as an open question. |

All four variants are evaluated against all models in the cohort. Variant 4 existing is important because the prior findings note that explicit constraint listing **may be counterproductive** — this is the variant that tests that claim directly.

**Prompt authoring notes**:
- `python` variant 2 targets "Implement don't orchestrate" failure — the framing should make tool calls feel inevitable.
- `ast-select` variant 2 must be uncompromising about single-selector output. No Python, no chain, no multi-line.
- `pss` variant 2 should anchor the declaration vocabulary explicitly (`show: body | signature | outline`) and clarify multi-rule syntax.
- `plucker` variant 2 should anchor on `source(...).chain.terminal` shape and enumerate chainable methods.

## Scoring

Hybrid two-stage rubric. Each cell in the matrix produces an integer score in `{0, 1, 2}`:

- **Stage 1 — structural gate** (fast, always run):
  - `python`: parse via `ast.parse`, then run `lackpy.lang.validator.validate(program, allowed_names=kit_names)`. Pass if `result.valid`.
  - `ast-select`: non-empty, single line, no `{`/`}`, and (optionally) parseable by pluckit's selector parser if available without executing. A selector is also considered gate-pass if the raw shape is syntactically plausible (`.<nodekind>` or `.<nodekind>#<name>` etc.) — we don't want to penalize a correct selector because of a pluckit-version mismatch.
  - `pss`: non-empty, balanced braces, ≥ 1 rule.
  - `plucker`: parse via `ast.parse`, expect a single `Expression` or `Expr` starting with a `Call` to `source`. Gate pass if the AST shape matches `source(...).chain.terminal`.
  - Gate fail → score 0.

- **Stage 2 — execution** (only on gate pass):
  - `python`: run the generated program through `PythonInterpreter.execute()` against the toybox as `base_dir`, with the default kit.
  - `ast-select`: run `AstSelectInterpreter.execute()` with `config={"code": "<toybox>/**/*.py"}`.
  - `pss`: run `PssInterpreter.execute()` with `config={"code": "<toybox>/**/*.py"}`.
  - `plucker`: run `PluckerInterpreter.execute()` with `config={"code": "<toybox>/**/*.py"}`.
  - If execution fails (exception, interpreter-level error, timeout) → score 1.
  - If execution succeeds **and** the intent's `exec_assertion(output)` returns `True` → score 2.
  - If execution succeeds but the assertion fails → score 1.

Score 0 means "didn't even produce something plausible". Score 1 means "produced something plausible that didn't work". Score 2 means "produced the right answer". The three-bucket distribution lets the report distinguish prompts that fail upstream (at generation) from those that fail downstream (at execution).

**Additional metrics captured per cell** (recorded in JSONL regardless of score):
- `duration_ms_generation` — wall-clock for the Ollama call.
- `duration_ms_execution` — wall-clock for the interpreter run.
- `tokens_eval` — tokens produced by the model (from Ollama's eval_count).
- `tokens_prompt` — prompt tokens (from Ollama's prompt_eval_count).
- `raw_generation` — the raw model output before any sanitization, for failure-mode analysis.
- `sanitized_program` — the program after stripping fences etc., as actually scored.
- `gate_errors`, `exec_error` — error strings when applicable.
- `assertion_fn_id` — which assertion was checked (for re-scoring under different assertions later).

**Deferred oracle — sitting_duck**: the `sitting_duck` DuckDB extension exposes `ast_select(path_pattern, selector)` which returns matched AST rows as a SQL resultset. This would be an excellent lackpy-independent oracle for `ast-select` and for individual selectors within a `pss` sheet: run the selector via `duckdb -c "FROM ast_select('<toybox>/**/*.py', '<selector>')"` and count rows. At the time of writing, the binary has an extension-load issue (`pattern_matching.sql` at startup uses operators from `core_functions` before it's loaded). When that's fixed, `scoring.py` can add a `sitting_duck_oracle_count()` helper as a cross-check against pluckit's count — disagreement between the two would be valuable signal (possibly a pluckit bug, possibly a sitting_duck bug). For v1, scoring uses pluckit only; sitting_duck oracle is a documented TODO.

## Runner, resumability, and progress

- **Streaming Ollama calls** with per-call timeouts, adapted from `scripts/pluckit-quartermaster.py`'s `_chat_with_timeout` helper. Timeout defaults to 60s, overridable per run.
- **Incremental JSONL output**: one row per cell, flushed after every generation. Mid-run files are valid JSONL.
- **Resumable runs**: at startup, the harness reads any existing JSONL at the output path and builds a set of `(interpreter, prompt_id, model, intent_id)` keys already completed. Those cells are skipped. If the process is killed or Ollama crashes, re-running picks up exactly where it left off.
- **Incremental query**: `scripts/prompt-eval/query.py <jsonl>` reads a live file and prints per-model / per-prompt / per-interpreter pass rates, total cells done vs total expected, median latency. Can be run in a second terminal or as an `Agent` tool call against the filesystem while the harness runs in the first.
- **tqdm** outer bar over total cells; description updates to show current `model / interpreter / prompt_id / intent_id`. Phase 1a uses a single flat bar since the matrix is tiny.
- **Graceful Ctrl+C**: `KeyboardInterrupt` flushes any in-progress row, prints run stats, exits 0. The JSONL is never left in a corrupt state.
- **No parallelism in v1**: Ollama's `keep_alive=30m` handles warm model reuse. Parallel model execution would fight for VRAM. Serial run, real wall-clock number.
- **Toybox hash pinning**: every JSONL file's first line is a header row `{"_meta": {...}}` containing the toybox content hash (hash of the concatenated sorted file contents). When `report.py` consolidates results from multiple JSONL files, it refuses to merge rows with different toybox hashes.

## Phase structure

### Phase 1a — Qualifier

- **Matrix**: 19 Ollama models × `python` interpreter × `baseline` prompt × 8 `python` core intents.
- **Volume**: ~152 runs, estimated wall-clock 15–25 minutes.
- **Goal**: filter the model pool down to a cohort for the full grid. Models that score 0 on more than half the core items are dropped.
- **Output**: `results/prompt-eval-2026-04-11/phase1a-qualifier.jsonl` + a rank table printed to stdout.
- **Selection rule**: top 6 models by total score, with a minimum floor of ≥ 50% core gate-pass. Ties broken by median generation latency. We pre-commit to the qualifier's output without re-running it.

### Phase 1b — Main grid

- **Matrix**: 6 qualifier-winning models × 4 interpreters × 4 prompt variants × 14 intents (8 core + 6 stretch).
- **Volume**: `6 × 4 × 4 × 14 = 1344` runs. At 5s median generation per cell plus ~100ms execution: `~2 hours`.
- **Goal**: identify the best `(prompt, model)` per interpreter on real delegation tasks.
- **Output**: `results/prompt-eval-2026-04-11/phase1b-grid.jsonl`.
- **Selection rule per interpreter**: the `(prompt, model)` cell with the highest total score, with ties broken by (1) fewest stretch-item 0s, (2) median latency. Record the top 2 cells per interpreter for Phase 2.

### Phase 2 — Refinement

- **Matrix**: for each interpreter, take the top 2 `(prompt, model)` cells from Phase 1b and sweep:
  - temperature: `{0.0, 0.2, 0.4}`
  - example count (for variants with few-shot): `{3, 5, 7}`
  - constraint-list phrasing: `{whitelist-only, whitelist+negative, none}`
- **Volume**: 4 interpreters × 2 cells × ~9 axis combinations × 14 intents ≈ 1000 runs. At ~5s each: ~1.4 hours.
- **Goal**: characterize the winners. Is the winning prompt robust across temperature? Does adding more examples hurt past a threshold? Does the explicit-constraint addition help on any variant or uniformly hurt (matching the prior finding)?
- **Output**: `results/prompt-eval-2026-04-11/phase2-refinement.jsonl`.

Phase runtime estimates assume Ollama's keep_alive keeps models hot between cells of the same model; worst case (cold cache on every model switch) could double the wall-clock, which the phase scripts will report.

## Deliverables

1. **JSONL rows** for all three phases under `results/prompt-eval-2026-04-11/`.
2. **Consolidated JSON** per phase (`phase1b-grid.json` etc.) written by `report.py` from the JSONL.
3. **Markdown report** (`report.md`) containing:
   - Executive summary: best `(prompt, model)` per interpreter with aggregate score and example output.
   - Per-interpreter section: scoring matrix (model × prompt heatmap), top 3 cells, top failure modes observed for the best cell.
   - Cross-interpreter comparison: does any single model dominate? Is any prompt variant robustly best across interpreters?
   - Phase 2 findings: temperature sensitivity, example-count sweet spot, constraint-list impact.
   - Pluckit grammar gaps: any stretch items dropped, linked to any issues filed.
   - Recommendations for production wiring (feeding a later work order to actually change `src/lackpy/infer/prompt.py`).
4. **Regression canaries** under `tests/eval/test_prompt_canaries.py`:
   - Marked `@pytest.mark.slow` and `@pytest.mark.skipif(no_ollama)`.
   - Per interpreter: the winning model × winning prompt × 3 canary intents (1 core, 1 stretch, 1 baseline).
   - Asserts `score == 2` for each canary.
   - Thresholds start at the Phase 1b winners and are tightened as subsequent research runs confirm stability. When a canary starts failing, we re-run the relevant phase 2 axis and decide whether prompt or implementation regressed.
5. **Auto-memory update**: a new `prompt_eval_2026-04-11.md` memory entry summarizing the winners and the key deltas from the prior (10-day-old) findings.

## Model cohort

From live `ollama list` on `localhost:11435`:

- **Tiny (<1GB)**: `qwen2.5-coder:0.5b`, `qwen3:0.6b`
- **Small (1–2GB)**: `qwen2.5-coder:1.5b`, `qwen2.5:1.5b`, `llama3.2:1b`, `codegemma:2b`, `granite3.1-dense:2b`, `granite3.3:2b`, `smollm2:latest`
- **Medium (2–3GB)**: `llama3.2:latest` (3b), `phi4-mini:latest` (3.8b), `qwen2.5-coder:3b`, `qwen2.5:3b`, `granite-code:3b`, `phi3:latest` (3.8b)
- **Large (5–8GB)**: `qwen2.5-coder:7b`, `qwen2.5:7b`, `gemma:latest` (9b), `qwen2:latest` (7b)

Excluded: `llama3.2-vision:11b` and `:90b` (vision models; text sweep treats image-conditioned context as weight noise).

Total cohort for Phase 1a: **19 models**.

## Risks and open questions

1. **Pluckit grammar gaps**: several stretch items may turn out to be inexpressible. Mitigation: verify during implementation, document drops in the report, file issues where appropriate.
2. **Model memory pressure**: swapping across 19 models in Phase 1a on a 32 GB laptop may cause Ollama OOMs or slow swaps. Mitigation: Phase 1a sorts models by size so we go small → large and can bail early if memory pressure bites. The qualifier's sort-by-size also means the bar for "did this even start to work" is set by the small end of the cohort first.
3. **Fixture drift**: if the toybox changes between runs, old results become incomparable. Mitigation: hash pinning in the JSONL header; `report.py` refuses cross-hash merges.
4. **"Specialized" prompt authoring quality**: the entire point of Phase 1b is to measure interpreter-aware prompts, but the quality of the "specialized" variant is gated on how well I (the author) write it. Mitigation: iterate the specialized variants during implementation against a tiny 1-model × 3-intent dry-run before committing to the full Phase 1b run. If a specialized variant is clearly worse than baseline, revise before the full sweep.
5. **Sanitization leak-through**: lackpy's real pipeline runs generations through `sanitize_output()` before validating (strips fences, preambles, trailing commentary). The harness should sanitize with the same function so findings transfer to production — but we should also record the raw un-sanitized output so the report can quantify how much sanitization is doing.
6. **Ollama cache warmth non-determinism**: the first call to a cold model is slower than subsequent calls. Running cells in `(model, prompt, interpreter, intent)` outer-to-inner order means every model has at least one cold call; the latency metric should be computed as median (not mean) to de-weight warm-up.
7. **Sitting_duck oracle deferred**: see "Deferred oracle" note above. v1 uses pluckit only; when the sitting_duck extension-load bug is fixed, adding the oracle is a small follow-up patch to `scoring.py`.
8. **Mutation tasks deferred**: v1 is read-only, which cleanly covers the delegation-for-fetch pattern but leaves "can a local model rename something correctly?" unmeasured. Once v1 findings are in, v2 can add a tempdir-copy mutation track.

## Execution plan (session-level)

This spec is the deliverable for this session's brainstorm. Once the user approves, the next step is to invoke the `superpowers:writing-plans` skill to turn this into an implementation plan with ordered steps and verification checkpoints.

The implementation itself, per user guidance, happens in phases:
1. **Author** the toybox, corpus, prompts, scoring, runner, query, report scripts, and the regression canary test file. No Ollama calls yet.
2. **Dry-run** the harness with a single model × single prompt × 3 intents to shake out bugs.
3. **Phase 1a** — I can run this in-session; it's short. Report qualifier winners.
4. **Phase 1b** — Teague runs this; I query the JSONL mid-run as useful.
5. **Phase 2** — Teague runs this; same pattern.
6. **Report and canaries** — I write `report.md`, commit, file any pluckit/sitting_duck issues, and update auto-memory.

Wiring the winners into `src/lackpy/infer/prompt.py` and the dispatcher is **explicitly out of scope** for this spec. It becomes a follow-up plan after the report lands and the user confirms the findings are actionable.
