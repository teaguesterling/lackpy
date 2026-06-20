# RFC 0002 — Increment 5: kits → profiles

!!! note "Status: design plan — NOT started"
    This is the increment-5 writeup deferred by [RFC 0002 §10](tool-sources.md#10-migration--sequencing):
    the **kits→profile layer that sits on top of sources**, retiring "kit" per
    [direction.md §2](direction.md#directions). Increments 1–4 (the source substrate) are
    implemented. The headline design decisions are **resolved** (maintainer, 2026-06 — see
    §10): thin profile, config tables, hard removal of "kit" (no alias), MCP-routed
    provider-agnostic tools, profiles before the rename. A few items remain open (§10.7–10).
    Nothing here is built yet — this is the plan the implementation phases (§8) follow.

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
9. **Lifecycle-agnostic — not "one profile = one synchronous call."** Per-step kits differ
   (Pick); literate sessions are session-scoped; and the *same* profile must be drivable
   one-shot, conversational, eventful, scheduled, background, or fanned-out (§3.6).
   `resolve_profile` yields a driver-agnostic, emit-capable, session-able unit — never a
   call-locked or sync-return-locked object.
10. **Carry, not enforce, the grade.** Grade is informational in stock lackpy; enforcement
    (mode→max-grade) lives in the Kibitzer/policy layer. The profile's job is an accurate,
    tool-derived grade — not gating.

---

## 3. The profile model

### 3.1 Headline decision — thin vs fat

> **Decided: THIN** (maintainer, 2026-06). A profile is a **named bundle of *references*** —
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
tools = ["read_file", "find_files"]      # the tool grammar: name | list | dict | "none"
model = "ollama/qwen2.5-coder:3b"        # was service-global inference; now per-profile
mode  = "1-shot"                         # 1-shot | spm
# language  = "restricted-python"        # grammar/validator/spec: lackpy-lang | shlack (§3.5)
# execution = "one-shot"                 # one-shot | literate (composite/incremental) (§3.5)
# temperature = 0.2
# sandbox = "subprocess"                 # deferred constraints — §6

[profiles.careful]
tools = "filesystem"                     # a named tool-set (a .profile file, see §7)
model = "ollama/qwen2.5-coder:7b"
mode  = "spm"

[profiles.notebook]                      # both axes set — a literate run of a shlack document
tools     = "filesystem"
language  = "shlack"                     # the other language; same toolbox (§3.5)
execution = "literate"                   # session-scoped, incremental (§3.5)
```

- **`tools`** is exactly today's `kit=` grammar (name/list/dict/`"none"`), resolved against
  the toolbox — invariant 1 holds by construction.
- **inference fields** (`model`/`mode`/`order`/`temperature`) are the service-global knobs,
  now per-profile; absent → fall back to the service default (invariant 8: zero overhead).
- **`language` × `execution`** are the **two orthogonal axes** an interpreter is built from
  (per [interpreter-types.md](interpreter-types.md)): the *language profile*
  (grammar/validator/spec) and the *execution model* (one-shot / incremental / composite —
  *literate* is the latter). Kept as separate fields so the combinations fall out (§3.5).
  v1 accepts only the defaults (`restricted-python`, `one-shot`); the fields exist from day one.
- **policy defaults** (optional): the default `PolicyContext` seed; absent → today's behavior.

### 3.3 Where profiles live

> **Decided: config tables, with `.profile` files as a *named tool-set* sub-case**
> (maintainer, 2026-06). A profile bundles config, so `[profiles.<name>]` TOML tables are the
> natural home (vs. a line-list file, which can only express a tool list). A `.profile` *file*
> names a **tool-set** that a profile's `tools =` can reference — files describe tool sets;
> config tables describe profiles.

### 3.3a Tools are provider-agnostic — route, don't reimplement

> **Decided: a profile's tools come from *any* source kind, and routing favors MCP**
> (maintainer, 2026-06). Tools must **not** be python-function-centric. The source substrate
> (increments 1–4) already discovers tools from `ConfigToolSource` (python), `McpToolSource`
> (MCP), and `VirtualToolSource` (harness); a profile references tool *names*, and resolution
> goes through whichever source owns each name — invariant 1 unchanged. The design bias is
> **MCP-as-the-routing-mechanism**: prefer expressing tools as MCP (or MCP-shaped) endpoints
> over hand-written python providers.
>
> **Resolved (§10.8 = B): lackpy keeps its own MCP client; it does *not* route through woollama.**
> Investigation showed `woollama-core`'s only tool path is an LLM *orchestrate* loop where the
> caller provides dispatch — a category mismatch with lackpy's program-driven execution — and
> the MCP machinery lives in the woollama *daemon*, not the importable core. So v1 uses the
> existing `McpToolSource`, unchanged. The woollama seam is the **inverse** (B): lackpy's MCP
> server is registered as a woollama backend (§10.8, §11). Routing *through* woollama (option A)
> is deferred pending a woollama raw-dispatch proxy.

### 3.4 How it threads through (no new resolution path)

```
profile name/inline
   ├─ tools             → resolve_kit(tools, toolbox) → ResolvedKit (tools, callables, grade, desc, docs)  [unchanged; language-agnostic]
   ├─ model/mode        → StepContext / PolicyContext  (already per-call; profile supplies them)            [removes the getattr(p,"_model") reflection]
   ├─ language×execution→ interpreter registry → concrete interpreter (+ language spec → prompt/tiers)      [v1: defaults only — §3.5]
   └─ policy            → default PolicyContext seed                                                        [optional]
```

The profile is resolved **before** `_resolve_kit`/`_gate_kit`, feeding their inputs. Grade,
policy, validation, and the source substrate are untouched.

### 3.5 Languages, execution models, and sessions — handling literate & shlack

The plan must hold up for **literate lackpy** (a document driven cell-by-cell through a
persistent kernel) and **shlack** (a sibling lack-language). It does — because the profile
keeps the **two interpreter axes orthogonal** ([interpreter-types.md](interpreter-types.md):
*interpreter = language profile × execution model*):

- **`language`** — the language profile (grammar/validator/grader/spec): `restricted-python`
  (lackpy-lang) today, `shlack` later.
- **`execution`** — the execution model: `one-shot` today; `literate` = composite-decomposing
  over the incremental kernel; `spm`/others slot in here too.

A concrete interpreter is `(language × execution)`, resolved via the interpreter registry.

**Why two axes is the elegance.** Keeping them independent makes every combination *fall out*
for free: `literate × restricted-python`, `one-shot × shlack`, and **`literate × shlack`** all
work with no new profile shape — you register an interpreter and (for a new language) a spec.
A single flat `interpreter = "plucker"` string would instead force a named interpreter per
combination, and shlack-in-a-notebook would be a retrofit.

**What is language-agnostic vs language-specific** (the seam that makes shlack cheap):

| Shared across *all* languages | Selected by the profile's `language` |
|---|---|
| toolbox / sources, tool specs, **grade** (derived from tools), policy. *A tool is a callable; the language is only how a program composes the calls.* One toolbox serves lackpy **and** shlack. | grammar / validator / spec, the **generation prompt** (`build_system_prompt` from that language's spec), and the deterministic template/rules tiers (shlack needs shlack templates). |

So adding shlack adds a language profile + its prompt/tiers; it touches **none** of the
profile/tool/grade/policy machinery. Invariants 1–3 hold unchanged across languages.

**Sessions (literate / incremental).** For an incremental `execution`, the profile scopes a
**session** (the whole document run), not one call:

- The profile is the session's **ground truth** — max tool scope, grade *ceiling*, policy seed
  (invariant 2). It supplies session **defaults**; it does **not** answer per-cell
  re-resolution — that's the interpreter's job.
- The incremental interpreter owns the per-cell concerns: validation against the *live*
  namespace, optional per-cell narrowing (the "Pick" pattern lowers the *effective* set without
  raising the ceiling), and the open per-cell-vs-session policy/sandbox question
  (interpreter-types.md). The profile doesn't pre-decide these.
- This is exactly why **invariant 9** ("no 'one profile = one call'") exists: `resolve_profile`
  must yield something an interpreter can run *once* (one-shot) **or** drive *across cells*
  (literate). Concretely: resolution returns the session ground-truth (ResolvedKit + grade +
  inference/policy seed); a one-shot interpreter consumes it once, a literate interpreter holds
  it across cells. (A *cell* naming a narrower profile is a possible later refinement, not v1.)

**v1 stays tight, the type stays open.** v1 implements only `language = restricted-python`,
`execution = one-shot`, call-scoped — today's path. But the profile **type carries both axes
and session-capability from day one** (defaulted), so literate and shlack land by registering
interpreters + per-language specs/tiers, with **no reshaping of the profile or
`resolve_profile`**. That is the elegance test this section exists to pass.

### 3.6 Execution is an open registry — lackpy owns *steps*, the harness owns *lifecycle*

`one-shot` and `incremental` are not the whole `execution` space. Surveying the harnesses
(Claude Code, woollama, cosmic-fabric) and runtime features against it surfaces at least nine
modes, each adding a dimension the one-shot baseline lacks:

| Mode | New dimension | Grounded in |
|---|---|---|
| one-shot *(have)* | — (sync, stateless, single program) | `delegate` |
| incremental / literate *(have)* | **state** (persistent namespace) | literate kernel |
| **conversational / REPL** | continuation = a **live turn** (arrives at runtime) | woollama conversation handles (`/v1/responses`); cosmic-fabric sessions |
| **eventful / reactive** | lifecycle = **persistent watcher** on a signal | Monitor; hooks; webhooks; MCP `notification` |
| **scheduled / recurring** | lifecycle = **time-trigger** | cron / `/schedule` / `/loop` |
| **background / long job** | lifecycle = **submit → poll → complete** | background tasks |
| **streaming** | output = a **stream**, not a value | woollama `complete_stream`; literate render |
| **parallel / fan-out** | **concurrency** → merge | Workflow `parallel`/`pipeline`; subagent fan-out |
| **agentic loop** | continuation = the **model** picks next | woollama `orchestrate`; cosmic-fabric tool-loop |

The `execution` axis spans six dimensions: *state, lifecycle, continuation-driver, concurrency,
output, emission.*

**The boundary (load-bearing):** lackpy owns the **step-execution primitives** — `one-shot`
and `incremental` (run a program/cell safely, with optional persistent state). Everything else
is an **orchestration lifecycle a harness composes** from those primitives — and that's exactly
what Claude Code / woollama / cosmic-fabric already do. This *generalizes §10.8 = B*: the harness
drives; lackpy provides the safe step + the profile. "Eventful lackpy" = Monitor fires → harness
runs a profile; "scheduled" = cron → same; "conversational" = the harness holds the session and
feeds turns; "parallel" = a Workflow fans out N profile invocations. **lackpy should not *become*
eventful/scheduled/agentic — it should be a clean unit a harness drives in any of those modes.**

**Three requirements this puts on the model (bake in before Phase 1):**

1. **`execution` is an open registry, not a closed enum.** `one-shot`/`incremental` are
   lackpy-native; conversation/streaming may become native later; the rest resolve to
   "harness-driven". Never branch on a fixed `execution in {…}` set.
2. **The profile is lifecycle-decoupled.** The *same* profile (tools + language + inference +
   policy) must run one-shot, in a conversation, on a trigger, as a job, or fanned-out — only the
   **driver** changes. So `resolve_profile` must **not** assume `delegate`'s synchronous
   return-a-value contract; it yields a **driver-agnostic, session-able unit** (invariant 9,
   generalized from "not call-locked" to "not *lifecycle*-locked").
3. **Name the emission seam.** conversation, eventful, and streaming need the run to **emit**
   mid-execution (a partial result, an event, a notification) — not only return a final value.
   literate's render-as-execute is the existing case. The execution result must be an
   **event/emit channel**, not only a return — so those drivers have a hook.

**v1 still ships only `one-shot × restricted-python`** — but its `Profile` + `resolve_profile`
contract is **driver-agnostic, emit-capable, and registry-backed**, so the other eight modes
plug in as drivers (mostly harness-side) without touching the profile.

---

## 4. Backward compatibility — hard removal, no alias

> **Decided: remove "kit" entirely; no alias** (maintainer, 2026-06; overrides the earlier
> "deprecated alias" recommendation). `kit` is replaced by `profile` everywhere — the public
> params, the CLI, the MCP surface, the config field, the file extension. There is **no
> `kit=` compatibility alias**.
>
> This is a **breaking change to the MCP tool surface**: `delegate(kit=…)`/`generate(kit=…)`
> become `profile=`, and `kit_info`/`kit_list`/`kit_create` become `profile_info`/`profile_list`/
> `profile_create`. lackpy is pre-1.0 (0.13.x); the maintainer accepts the break. Ship it under
> a clear release note (a minor bump), and update the lackpy MCP plugin manifest in lockstep.
>
> Internally the move still uses shims **within a phase** to keep the suite green during the
> rename (e.g. `ResolvedKit` → `ResolvedTools`), but no shim survives into the public API.

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
3. `profile=` parameter replacing `kit=` on `delegate`/`generate`/`validate`/`run_program`
   and the MCP surface — **no `kit=` alias** (§4). `kit_default` → `profile_default`.
4. Per-profile inference (`model`/`mode`/`temperature`) and `interpreter` pass-through wired
   into the `StepContext`/`PolicyContext` that already exist (delete the `getattr(p,"_model")`
   reflection — invariant/surprise #8).
5. CLI/`lackpyctl`: `--profile` + `lackpyctl profile {list,info,create}` replacing the `kit`
   flag/subcommands (no alias). Fix the `@file`/grammar gaps while reworking the surface.
6. Docs: a "Profiles" concept page; deprecate "kit" language; `config.toml.example` profile.

Everything in v1 is **composition over existing pieces** — no change to sources, grade,
policy, or validation.

---

## 6. Explicit deferrals (Phase N+, acknowledged not missed)

| Deferred | Why later | What v1 must not preclude |
|---|---|---|
| **Quartermaster** (intent→profile) | Already a scored prototype (`scripts/pluckit-quartermaster.py`, `qm-*.json`); generalizes tools→profile selection. | A profile must be *inferable* — keep the input shape model-producible (names, not opaque objects). |
| **shlack (a 2nd language)** | Needs the shlack language profile (grammar/validator/spec) + its prompt/tiers. *The profile machinery is ready* — adding a language touches no profile/tool/grade/policy code (§3.5). | The `language` axis exists from v1 (defaulted `restricted-python`); the per-language prompt/tier seam is identified. |
| **Interpreter auto-selection** (intent → execution model) | Needs the interpreter registry (`interpreter-types.md`); related to Quartermaster. | The `execution` axis exists from v1 (defaulted `one-shot`). |
| **Harness-driven execution modes** (conversation, eventful, scheduled, background, parallel, agentic — §3.6) | These are *orchestration lifecycles a harness owns* (Claude Code / woollama / cosmic-fabric), not lackpy step-execution. lackpy supplies the profile + safe step; the harness drives. | The three §3.6 requirements: `execution` is an open registry; the profile is lifecycle-decoupled; the execution result is an emit/event channel (invariant 9). |
| **Literate / session-scoped policy/sandbox** | The session model is sketched (§3.5) but the per-cell-vs-session policy/sandbox question is open in `interpreter-types.md`. | Invariant 9 — `resolve_profile` yields a session ground-truth a literate interpreter can drive across cells, not a call-locked object (§3.5). |
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
3. **`profile=` replaces `kit=`** on the 4 methods + the MCP surface (no alias);
   `kit_default`→`profile_default`; `kit_info/list/create`→`profile_*`. *Breaking MCP change
   — bump + release note + update the MCP plugin manifest. Green: full suite migrated to
   `profile=`; MCP server advertises the new tool names.*
4. **CLI/ctl `--profile` + `profile` subcommands** (kit aliases retained); fix `@file` +
   the grammar gap. *Green: CLI reference doc matches the real interface.*
5. **Docs + terminology deprecation.** "Profiles" concept page; mark "kit" deprecated.
6. **Retire "kit" internals.** Rename `ResolvedKit`→`ResolvedTools` (or keep),
   `KitPolicySource`, `_resolve_kit`, etc., behind shims; remove shims last. *Green:
   `test_no_upward_deps` + full suite.*

Each phase ships independently and keeps the suite green.

> **Decided: profiles BEFORE the lacklangster rename** (maintainer, 2026-06). Both touch
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
- [ ] "kit" fully removed by the final phase — no `kit=` / `--kit` / MCP `kit_*` / `.kit`
      remain (no alias); the only public vocabulary is "profile".

---

## 10. Decisions & open questions

**Resolved (maintainer, 2026-06):**

1. **Thin vs fat profile** (§3.1) — ✅ **thin** (references, not a fat resolved object).
2. **Where profiles live** (§3.3) — ✅ **config tables**; `.profile` files = named tool-sets.
3. **Tools are provider-agnostic** (§3.3a) — ✅ tools come from any source kind; **route via
   MCP**, not python-function-centric.
4. **`kit` removal** (§4) — ✅ **hard removal, no alias** (breaking MCP change accepted, pre-1.0).
5. **Sequencing vs rename** (§8) — ✅ **profiles before** the lacklangster rename.
6. **CLI reconciliation** (§4) — ✅ rework to `--profile`; fix the `@file`/grammar gaps.

**Still open:**

7. **`ResolvedKit` rename** (§7) — `ResolvedTools`, or keep the internal name and only retire
   the user-facing "kit"?
8. **woollama ↔ lackpy direction** (§3.3a) — ✅ **resolved: B (woollama uses lackpy)**
   (maintainer, 2026-06). lackpy **keeps its own `sources/mcp/` client** for its own tool
   execution (its program-driven model doesn't fit woollama's LLM orchestrate loop). The
   integration seam is the **inverse**: lackpy's **existing MCP server** is registered as a
   woollama backend — once woollama gains MCP-client dispatch, woollama can call lackpy's
   `delegate` (etc.) as a peer node. *A* (lackpy routing its tool calls *through* woollama)
   is deferred pending a woollama raw-dispatch **proxy** endpoint — a separate, larger bet.
   **Consequence:** lackpy's MCP tool surface becomes the integration contract → re-examine it
   (rename `kit_*`→`profile_*`, split actions-vs-data into tools-vs-resources, add tool
   annotations). See §11.
9. **`interpreter`/language in v1** — pass-through only (rec), or design the interpreter
   registry now?
10. **Profile inheritance/composition** — do profiles compose (`extends = "fast"`)? Likely
    Phase N+, but the config shape should not preclude it.

---

## 11. MCP surface re-examination (consequence of §10.8 = B)

With **B** decided, lackpy's MCP server is the integration contract a woollama-class
orchestrator consumes — an LLM picks tools from names + descriptions + schemas + annotations.
The current surface (`mcp/server.py`, 14 flat tools) was built as a thin dev wrapper and
isn't fit for that. Re-examination findings:

1. **Rename `kit`→`profile` (forced by §4).** `kit_info`/`kit_list`/`kit_create` →
   `profile_*`; the `kit=` param on `delegate`/`generate`/`run_program`/`validate`/`create` →
   `profile=`. No alias. This rides the §8 migration.
2. **Split actions (tools) from data (resources).** ~9 of the 14 are read-only introspection
   that clutter the model-facing tool list and invite mis-selection. Move the *data* to MCP
   **resources**: `config`, `language_spec`, `toolbox_list`, `provider_list`, `docs_index`,
   `resolve_doc`. Keep as **tools** only the *actions*: `delegate`, `generate`, `run_program`,
   `validate`, `create`, `profile_create`. Shrinks the tool surface ~14 → ~6 — the headline
   for an orchestrator is `delegate`.
3. **Add tool annotations (`readOnly`/`destructive`/`idempotent`).** Today none. `validate`/
   `generate` are read-only; `delegate`/`run_program` **execute generated code** and are
   non-readonly + potentially destructive (depends on the profile's tools); `create` writes a
   file. This is the same `annotations`→grade mapping lackpy *consumes* from other MCP servers
   (RFC §6) — lackpy should *produce* it on its own surface so woollama/grade-mapping can gate.
4. **Write descriptions/schemas for an LLM, not a dev.** Tool docstrings should say *when to
   use* + inputs + outputs; params (`profile`, `params`, `rules`, `extra_tools`) currently have
   **no schema descriptions** (FastMCP derives bare types) — an orchestrator can't use them
   correctly. Give each a `Field(description=…)`.
5. **Trim the `delegate` result payload.** It returns the full service dict (program + entire
   trace + stdout) — heavy for an orchestrator. Default to a lean result (success/output/grade),
   trace opt-in.
6. **Drop or implement the `sandbox` param.** It's exposed on `delegate`/`run_program`/`create`
   but is reserved/unwired in the service — a misleading contract; remove until real.

These land **with the §8 migration** (the rename is shared) plus the tools-vs-resources +
annotations work as its own MCP-contract PR. Tracking: this supersedes the surface a future
`woollama`-registers-`lackpy` integration depends on.

---

*Siblings: [RFC 0002 (sources)](tool-sources.md) · [Direction](direction.md) ·
[Interpreter types](interpreter-types.md) · [Rename to lacklangster](rename-to-lacklangster.md).*
