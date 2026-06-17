# RFC 0002 — Increment 5: kits → profiles

!!! note "Status: design plan — NOT started"
    This is the increment-5 writeup deferred by [RFC 0002 §10](tool-sources.md#10-migration--sequencing):
    the **kits→profile layer that sits on top of sources**, retiring "kit" per
    [direction.md §2](direction.md#directions). Increments 1–4 (the source substrate) are
    implemented. This doc is a plan to react to — decisions are marked
    **`Decision needed (rec: …)`** for the maintainer to redline. Nothing here is built.

---

## 1. Problem & goals

[direction.md §2](direction.md#directions) calls for retiring "kit" as a primitive and
generalizing it into a runtime-config system that "configures the runtime **and the
language** the runtime uses." Today "kit" only selects **tools**; the *other* per-task
knobs — inference mode, model, interpreter, sandbox — live in **service-global**
`LackpyConfig` (one value per service) or are threaded as a growing pile of `delegate()`
arguments. There is no named, reusable, per-task bundle.

A **profile** is that bundle: a named, per-task selection of *tools + inference config +
which language/interpreter + policy defaults*. A kit becomes the **degenerate profile**
that only sets tools.

Goals:

1. Promote the service-global per-task knobs into named, selectable **profiles**.
2. Retire "kit" as the user-facing primitive (deprecate, don't break, the `kit=` API).
3. Build *on top of* the source substrate (increments 1–4) — never re-introduce
   hard-coded tools or a parallel resolution path.
4. Keep the door open for the deferred capabilities (Quartermaster, multi-language,
   session-scoping) without designing them now.

**Non-goal for v1:** intent→profile inference (Quartermaster), per-language/interpreter
auto-selection, session-scoped policy/sandbox, backward-derivation of profiles from
output, auto sandbox-strategy. All deferred — see §6.

---

## 2. The grounded "why" (use cases, reasons, invariants)

This section exists so the design is anchored in what actually depends on "kit." It is
the acceptance surface: a profile must serve every use case and preserve every invariant.

### 2.1 Use cases a kit/profile serves (today)

1. **Allowed-namespace for validation** — the tool names the AST validator checks every
   `ast.Call` against (the kit's primary job).
2. **Callable namespace for execution** — the same names bound to callables at run time
   (late-bound via `toolbox.resolve`).
3. **Security-grade aggregation** — `grade = join(tool grades)` (`compute_grade`).
4. **System-prompt grounding** — kit → `namespace_desc` injected into the inference prompt.
5. **Template/rules gating** — tier-0/1 deterministic shortcuts only fire if the matching
   tool is in the namespace.
6. **Doc references for coaching** — per-tool `docs` collected for Kibitzer hints.
7. **Per-step / derived kit** — the "Pick" step *derives* a tighter kit from a generated
   program's AST (no LLM). A kit is not always input — it can be **computed from output**.
8. **Saved-program (Lackey) bundling** — tools + params + returns + creation-log + pattern
   around a validated `run()` body. The existing proto-profile.
9. **CLI / MCP delegation surface** — `--kit`, `delegate(kit=…)`, MCP `kit_info/list/create`.
10. **Eval-harness scoping** — a harness-local kit adds helpers without touching builtins.
11. **Policy baseline (S1 ground truth)** — `KitPolicySource` turns the kit into the
    baseline `allowed_tools`/`grade`/`namespace_desc` that later sources only *narrow*.
12. **Per-task inference selection (the new one a profile formalizes)** — mode, interpreter,
    model, temperature, prompt variant, sandbox strategy. Today scattered across global
    config + a few per-call args; `generate()`/`delegate()` already take `mode`/`interpreter`
    per call, and `model`+`mode` already thread into `PolicyContext` — so the plumbing
    partly exists; a profile just **names and bundles** it.

### 2.2 Reasons for the change

- **Config bundling** — `delegate(intent, kit, params, mode, interpreter, extra_tools, …)`
  is a growing arg list; a named profile collapses it.
- **Per-task vs global mismatch** — `inference_mode`/`kit_default`/`sandbox_*` are
  one-per-service; real tasks want different mode/model/sandbox per intent.
- **Reduce agent burden of picking kits (#6)** — toward intent→profile (Quartermaster),
  with least-privilege as the design goal (deferred to a later phase, but the model must
  not preclude it).
- **Interpreter = language-profile × execution-model** — choosing *what language/interpreter
  to run* is a first-class per-task choice the profile should own.
- **Lackey already proves the bundle is wanted** — a profile is Lackey's reusable runtime
  sibling.
- **Ratchet / promotion** — successful runs promote to named reusable units; profiles give
  those a home.

### 2.3 Invariants (these become the migration acceptance checklist — §8)

A profile **must**:

1. **Never hard-code tool names.** Tools are *referenced* by name and resolved against the
   source-populated `Toolbox`; a profile never *defines* a tool. (`tests/lang/test_no_upward_deps.py` stays green.)
2. **Be S1 ground truth, narrow-only.** A profile's tool set is the maximum scope; policy
   (umwelt/kibitzer) can deny but never grant beyond it.
3. **Derive grade from its tool set, immutable through the chain.** No hard-coded grade;
   policy does not alter it.
4. **Not touch the AST whitelist.** `ALLOWED_NODES`/`ALLOWED_BUILTINS` are fixed; a profile
   cannot loosen them (custom rules may only *tighten*).
5. **Keep the leaf pure.** Profiles are **runtime** constructs atop `lackpy-lang`; they must
   never leak into the language package.
6. **Stay small-model-friendly.** Lean prompts; param **metadata, not values**, in prompts;
   no few-shot bloat. A profile carrying params/interpreter hints must preserve this split.
7. **Preserve late-binding / call-time gating.** Resolve against *live* availability;
   a withdrawn (virtual/MCP) tool fails cleanly.
8. **Keep zero overhead for the bare case.** A profile with no policy/kibitzer must cost no
   more than today's kit-only path.
9. **Not assume one kit per delegation.** Per-step kits differ (Pick); incremental/literate
   sessions are session-scoped. The abstraction must not bake in "one profile = one call."
10. **Carry, not enforce, the grade.** Grade is informational in stock lackpy; enforcement
    (mode→max-grade) lives in the Kibitzer/policy layer. The profile's job is an accurate,
    tool-derived grade — not gating.

---

## 3. The profile model

### 3.1 Headline decision — thin vs fat

> **Decision needed (rec: THIN).** A profile is a **named bundle of *references*** —
> not a new fat resolved object. It references (a) a tool selection (the existing
> kit grammar: name/list/dict), (b) an inference config (model/mode/order/temperature),
> (c) an interpreter/language, and (d) optional policy defaults. It **composes from pieces
> that already exist**: resolution still goes tools → `Toolbox` → `ResolvedKit` → grade →
> `KitPolicySource`, unchanged. The "fat" alternative (a `ResolvedProfile` that subsumes
> `ResolvedKit`) duplicates the grade/policy machinery and weakens invariants 2–3. Thin
> keeps "a kit is the degenerate profile" literally true and the blast radius small.

Under the thin model, `ResolvedKit` stays the resolution core (rename it `ResolvedTools`
or leave it — terminology, §7). A `Profile` is the *input* that says which tools, which
inference config, which interpreter; resolving a profile = resolve its tools into the
existing `ResolvedKit` **plus** carry its inference/interpreter selection into the
`StepContext`/`PolicyContext` that `delegate()` already builds.

### 3.2 What a profile contains (v1)

```toml
# .lackpy/config.toml
[profiles.fast]
tools = ["read_file", "find_files"]      # the kit grammar: name | list | dict | "none"
model = "ollama/qwen2.5-coder:3b"        # was service-global inference; now per-profile
mode  = "1-shot"                         # 1-shot | spm
# interpreter = "python"                 # python | ast-select | plucker | … (language × model)
# temperature = 0.2
# sandbox = "subprocess"                 # deferred constraints — §6

[profiles.careful]
tools = "filesystem"                     # a named tool-set (today's .kit file, see §7)
model = "ollama/qwen2.5-coder:7b"
mode  = "spm"
```

- **`tools`** is exactly today's `kit=` grammar (name/list/dict/`"none"`), resolved against
  the toolbox — invariant 1 holds by construction.
- **inference fields** (`model`/`mode`/`order`/`temperature`) are the service-global knobs,
  now per-profile; absent → fall back to the service default (invariant 8: zero overhead).
- **`interpreter`** names the language×execution-model. v1: pass-through to the existing
  per-call `interpreter` arg; auto-selection deferred (§6).
- **policy defaults** (optional): the default `PolicyContext` seed; absent → today's behavior.

### 3.3 Where profiles live

> **Decision needed (rec: config tables, with `.kit`/`.profile` files as a *named tool-set*
> sub-case).** A profile bundles config, so `[profiles.<name>]` TOML tables are the natural
> home (vs. the line-list `.kit` file, which can only express a tool list). Keep `.kit`/
> `.profile` *files* as a way to name a **tool-set** that a profile's `tools =` can reference
> — i.e. files describe tool sets; config tables describe profiles. This preserves the
> existing `.kit` files as the degenerate (tools-only) case.

### 3.4 How it threads through (no new resolution path)

```
profile name/inline
   ├─ tools      → resolve_kit(tools, toolbox) → ResolvedKit (tools, callables, grade, desc, docs)   [unchanged]
   ├─ model/mode → StepContext / PolicyContext  (already per-call; profile just supplies them)        [removes the getattr(p,"_model") reflection]
   ├─ interpreter→ existing per-call interpreter selection                                            [pass-through in v1]
   └─ policy     → default PolicyContext seed                                                          [optional]
```

The profile is resolved **before** `_resolve_kit`/`_gate_kit`, feeding their inputs. Grade,
policy, validation, and the source substrate are untouched.

---

## 4. Backward compatibility

> **Decision needed (rec: deprecated alias, not hard cutover).** Unlike the internal
> woollama provider cutover, `kit=` is on **four public methods** (`delegate`/`generate`/
> `validate`/`run_program`) **and the MCP tool surface** (`delegate`, `kit_info`, …) — these
> are external clients. Keep `kit=` working as a deprecated alias for `profile=` for ≥1
> release (a bare kit value = a tools-only profile), with a deprecation warning. Hard-cutover
> only the internal call sites.

**Reconcile the CLI while here (two real bugs, see §2.1.9 / research):**

- `--kit` (and a new `--profile`) currently accept **only** a comma-list — the named/dict/
  `"none"` forms that the API supports are unreachable from the CLI. Make `--profile` accept
  a profile **name** (the common case) and the full tool grammar.
- `--kit`'s help advertises `@file`, which **was never implemented**. Implement it or drop
  the claim.

---

## 5. v1 scope (tight)

1. `Profile` input type + a `resolve_profile()` that yields `(ResolvedKit, inference-config,
   interpreter, policy-seed)` by composing existing machinery.
2. `[profiles.<name>]` config parsing in `LackpyConfig` (raw dicts, like `[[tools]]`).
3. `profile=` parameter on `delegate`/`generate`/`validate`/`run_program`, with `kit=` kept
   as a deprecated alias. `profile_default` config field (alias of `kit_default`).
4. Per-profile inference (`model`/`mode`/`temperature`) and `interpreter` pass-through wired
   into the `StepContext`/`PolicyContext` that already exist (delete the `getattr(p,"_model")`
   reflection — invariant/surprise #8).
5. CLI/`lackpyctl`: `--profile` + `lackpyctl profile {list,info,create}` (keep `kit`
   subcommands as deprecated aliases). Fix the `@file`/grammar gaps.
6. Docs: a "Profiles" concept page; deprecate "kit" language; `config.toml.example` profile.

Everything in v1 is **composition over existing pieces** — no change to sources, grade,
policy, or validation.

---

## 6. Explicit deferrals (Phase N+, acknowledged not missed)

| Deferred | Why later | What v1 must not preclude |
|---|---|---|
| **Quartermaster** (intent→profile) | Already a scored prototype (`scripts/pluckit-quartermaster.py`, `qm-*.json`); generalizes tools→profile selection. | A profile must be *inferable* — keep the input shape model-producible (names, not opaque objects). |
| **Multi-language / interpreter auto-selection** | Needs the interpreter registry (`interpreter-types.md`) and the lacklangster rename. | The `interpreter`/language field exists in the model from v1 (pass-through). |
| **Session-scoped policy/sandbox** (literate/incremental) | Open question in `interpreter-types.md`; breaks "one profile = one call". | Invariant 9 — don't bake call-scoping into the type. |
| **Backward-derivation** (Pick → least-privilege profile) | "Pick" derives a kit from output; a profile-capture API is a separate feature. | Resolution stays one-directional in v1; the type allows a derived profile later. |
| **Auto sandbox-strategy** | Constrained by interpreter serializability + bridged tools (nsjail design). | Carry an explicit `sandbox` field; don't auto-pick yet. |
| **Grade-ceiling enforcement** | Lives in Kibitzer/policy, not lackpy core. | Carry an accurate tool-derived grade (invariant 10). |

---

## 7. Terminology retirement

"kit" appears in: `kit/` package, `ResolvedKit`, `KitPolicySource`, `kit_default`,
`kit=`/`--kit`, `kit_info/list/create`, `.kit` files, `_resolve_kit`/`_gate_kit`, the
"Kits & Toolbox" concept doc. Retire it **last** (final migration phase) via alias shims, so
no phase both renames *and* changes behavior. The internal `kit/toolbox.py` (the `Toolbox`)
is arguably mis-named too, but it's the source-aggregation registry — rename to
`tools/` is optional and can trail.

---

## 8. Migration plan (phased, shim-backed, green per phase)

Mirrors the proven woollama-core extraction / the rename plan's shim pattern: move behind
aliases, flip call sites, retire terminology last; **the §2.3 invariants are the regression
checklist each phase must keep green.**

1. **Profile model + resolution (additive).** Add `Profile` + `resolve_profile()` composing
   existing machinery; `[profiles.<name>]` parsing. No call site changed yet. *Green: all
   existing kit tests pass untouched; new profile tests cover the degenerate (tools-only)
   case = a kit.*
2. **Wire per-profile inference/interpreter.** Thread profile-supplied `model`/`mode`/
   `interpreter` into `StepContext`/`PolicyContext`; delete the `_model` reflection. *Green:
   delegate with a profile selecting a model/mode behaves identically to today's global
   config when the profile omits them (invariant 8).*
3. **`profile=` public param + `kit=` deprecated alias** on the 4 methods + MCP surface;
   `profile_default`. *Green: every existing `kit=` call still works (alias) with a
   deprecation warning; MCP contract unchanged.*
4. **CLI/ctl `--profile` + `profile` subcommands** (kit aliases retained); fix `@file` +
   the grammar gap. *Green: CLI reference doc matches the real interface.*
5. **Docs + terminology deprecation.** "Profiles" concept page; mark "kit" deprecated.
6. **Retire "kit" internals.** Rename `ResolvedKit`→`ResolvedTools` (or keep),
   `KitPolicySource`, `_resolve_kit`, etc., behind shims; remove shims last. *Green:
   `test_no_upward_deps` + full suite.*

Each phase ships independently and keeps the suite green.

> **Sequencing decision needed (rec: profiles BEFORE the lacklangster rename).** Both touch
> most of the tree; running them concurrently is a merge nightmare. Profiles is the
> behavioral change; the [rename](rename-to-lacklangster.md) is mechanical and should follow.

---

## 9. Acceptance checklist (the invariants, as gates)

Each migration PR must keep all of these green:

- [ ] No hard-coded tool names; tools resolve via sources (`test_no_upward_deps` green).
- [ ] Profile tool set is narrow-only S1 ground truth (policy can't expand it).
- [ ] Grade derived from tools, unchanged by policy.
- [ ] AST whitelist untouched.
- [ ] `lackpy-lang` leaf stays pure (no profile/runtime imports).
- [ ] Prompts carry param metadata, not values; no few-shot bloat.
- [ ] Late-binding / call-time gating preserved (virtual/MCP withdrawal fails cleanly).
- [ ] Bare profile (no policy/kibitzer) costs no more than today's kit-only path.
- [ ] No "one profile = one delegation" assumption baked into the type.
- [ ] Grade carried accurately; enforcement left to Kibitzer/policy.
- [ ] `kit=` / `--kit` / MCP `kit_*` still work (deprecated alias) for ≥1 release.

---

## 10. Open questions / decisions for the maintainer

1. **Thin vs fat profile** (§3.1) — rec: thin. *Headline.*
2. **Where profiles live** (§3.3) — rec: config tables; files = named tool-sets.
3. **`kit=` deprecation** (§4) — rec: alias for ≥1 release, not hard cutover.
4. **Sequencing vs rename** (§8) — rec: profiles first.
5. **`ResolvedKit` rename** (§7) — `ResolvedTools`, or keep the name and only retire the
   user-facing "kit"?
6. **`interpreter`/language in v1** — pass-through only (rec), or design the registry now?
7. **Profile inheritance/composition** — do profiles compose (`extends = "fast"`)? Likely
   Phase N+, but the config shape should not preclude it.

---

*Siblings: [RFC 0002 (sources)](tool-sources.md) · [Direction](direction.md) ·
[Interpreter types](interpreter-types.md) · [Rename to lacklangster](rename-to-lacklangster.md).*
