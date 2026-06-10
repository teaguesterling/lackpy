# Plan: rename to put each name where its meaning points

> **Status: plan only — NOT started.** Captured for a later, deliberate refactor.
> The current names (`lackpy` = harness/runtime dist, `lackpy-lang` = language
> leaf) are untouched; the woollama-core integration continues against them.

## Motivation

`lackpy` literally means *"Python that lacks most of Python"* — a description of
the **language**, not of the generation/inference harness that produces and runs
it. Today the names are inverted:

- `lackpy` (the dist) is the **harness**: service, inference / program-generation,
  interpreters, kit, sources, MCP, CLI.
- `lackpy-lang` (the leaf, imported `lackpy.lang`) is the **language**: grammar,
  validator, grader, spec — stdlib-only, no runtime deps.

And the project is growing toward **multiple languages** (lackpy + shlack). So the
clean naming swaps them:

- **`lackpy`** = the lackpy **language runtime** (today's `lackpy-lang`).
- **`shlack`** = a sibling language runtime (future, its own dist).
- **`lacklangster`** = the **harness** that wrangles any lack-language (today's
  `lackpy` runtime dist).

Each name then lands where its meaning already points: `lackpy` *is* the restricted
language; `lacklangster` *is* the multi-language harness.

## Target structure

| Today | After |
|---|---|
| dist `lackpy-lang`, import `lackpy.lang` (grammar/validator/grader/spec) | dist `lackpy`, import `lackpy.*` — the language runtime |
| (none) | dist `shlack`, import `shlack.*` — sibling language (future) |
| dist `lackpy`, import `lackpy.*` (service/infer/interpreters/kit/sources/mcp/CLI) | dist `lacklangster`, import `lacklangster.*` — the harness |

## The one structural decision: where does *execution* live?

"Language runtime**s**" could mean the language owns its executor. But lackpy's
interpreters (`plucker`, `ast_select`, `pss`) execute programs against the
**kit/tools**, which is harness territory.

**Recommendation (keep the clean leaf boundary):** `lackpy` stays a *pure*
language — grammar + validation + grader + spec, stdlib-only, no upward deps (the
property `tests/lang/test_no_upward_deps.py` already guards). `lacklangster` owns
execution: interpreters, tools, inference, the agentic loop. "Runtime" here means
*the formal definition the harness runs*, not the runner. This keeps the swap
mostly a **rename**, not a re-layering, and preserves the leaf's purity guard.

*Alternative (defer unless wanted):* each language ships its own executor — a
deeper split that moves interpreter machinery into the language packages.

## Namespace mapping (confirmed direction)

- harness: `lackpy.<mod>` → `lacklangster.<mod>` (service, infer, interpreters,
  kit, sources, mcp, run, policy, cli, ctl, config, …).
- language: `lackpy.lang.<mod>` → `lackpy.<mod>` (the leaf loses the `.lang`
  segment and *becomes* the `lackpy` top-level).

## Scope / blast radius (why this is bigger than the earlier idea)

The earlier `lackpy-lang → lacklangster` idea touched ~37 `lackpy.lang` import
sites. **This swap renames the main package**, so the blast radius is most of the
tree:

- Every internal import in the harness (`from lackpy.x` / `from .x` → `lacklangster`).
- Every `from lackpy.lang import …` (~37 sites) → `from lackpy import …`.
- CLI entry points: `lackpy` / `lackpyctl` → `lacklangster` / `lacklangsterctl`
  (decide whether a `lackpy` command survives as a language-runner).
- Both `pyproject.toml`s (dist names, `[project.scripts]`, the pytest
  `pythonpath`, `[tool.uv.sources]`), the docs, and effectively every test module.
- The woollama integration relocates: `lackpy.infer.providers.woollama` →
  `lacklangster.infer.providers.woollama` (no logic change).

## Migration sequence (phased, shim-backed — mirrors the woollama-core extraction)

The woollama extraction proved this pattern: move behind alias shims, keep every
import working, flip call sites, then delete shims. Apply it here:

1. **Language leaf first** (smaller, isolated). New dist `lackpy` from today's
   `lackpy-lang`, importing `lackpy.*` (drop `.lang`). Leave `lackpy.lang` as an
   alias shim (`sys.modules` re-export) so the harness keeps importing it.
   Update the leaf's own pyproject + the purity guard test. Ship green.
2. **Harness rename.** New dist `lacklangster` from today's `lackpy` runtime,
   importing `lacklangster.*`. The trickiest step: the harness currently *is*
   `lackpy`, and the leaf now *wants* to be `lackpy` — so this must land together
   with, or right after, step 1's namespace freed up. Move the harness modules to
   `lacklangster/`, repoint internal imports, keep `lackpy` (harness) import paths
   as alias shims during the transition.
3. **Flip the language references.** Harness imports of the leaf:
   `from lackpy.lang import …` → `from lackpy import …`.
4. **CLI + packaging.** Rename `[project.scripts]`; update both pyprojects, the
   pytest `pythonpath`, `[tool.uv.sources]`, docs, READMEs.
5. **Delete the shims.** Repoint all remaining imports; remove the compat aliases.
6. **shlack** (separate, later): a new `shlack` dist alongside `lackpy`, driven by
   the `lacklangster` harness via the same language-runtime interface.

Each phase keeps the suite green; ship per phase.

## Status-quo bridge (what's in place now, no rename)

So the integration can continue without the rename, the worktree resolves the
monorepo locally (see `pyproject.toml` `[tool.uv.sources]`):

- `lackpy-lang` → local path `packages/lackpy-lang` (editable). (Un-masked — it
  isn't published, so the registry pin `==0.12.0` is satisfied from the tree.)
- `woollama` → editable sibling worktree `../woollama-core-extraction`.
- `requires-python` bumped to `>=3.12` (the universal lock pulls the `full`
  extra's `nsjail-python`, which is 3.12-only).

These are scoped to the integration branch and unwind naturally when the packages
publish.

## Open questions

- Execution placement (above) — pure-leaf `lackpy` vs. per-language executors.
- Does a `lackpy` CLI survive (run a lackpy program) distinct from the
  `lacklangster` harness CLI?
- Is `shlack` one dist with `lacklangster`, or its own dist? (Plan assumes its own,
  for a clean per-language boundary.)
- Publish order: language (`lackpy`) before harness (`lacklangster`), since the
  harness depends on the language.
