# lackpy STE (stepy100) — Plan 2: the runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every program the Plan 1 validator accepts run as plain Python inside lackpy. Do that safely, with workspace-confined file access governed by Claude-Code-style permissions. Then switch lackpy's generation, surfaces and docs to the new language, so the old kit model can be retired.

**Architecture:** One runtime namespace replaces the kit namespace. It holds the classified builtins, a restricted `__import__` that returns allowlist proxies for approved modules, a `Path` stand-in that checks every I/O call against a `PathPolicy`, the `type` stand-in, `__name__`/`__file__`, and an uncatchable `LackpyRefusal`. The validator stays the first gate, and the runtime proxies are the backstop for what static checks cannot see. Generation switches to the measured pre-prompt.

**Tech Stack:** Python 3.11+ for `lackpy` and stdlib only. `lackpy-lang` stays at ≥3.10. Uses `fnmatch`/`pathlib` for gitignore-style matching (no new dependency). pytest via blq.

**Spec:** `docs/superpowers/specs/2026-10-18-lackpy-ste-design.md`. Read §5.3 (the plain-Python invariant), §5.1 (safety notes), and all of §8. §8 lists Plan 1's changes, the post-Plan-1 decisions (nesting, Path permissions, `safety.attr-store`), and the nine "Carried into Plan 2" gate requirements. Background on lackpy's runtime as it is today is in this plan's Task notes; you need no other document.

## Global Constraints

- Work on branch `mcp-tool-metadata`. Plan 1 is merged there, at 6855fa1 and later. Do not push without the controller's instruction.
- **The plain-Python invariant (design §5.3):** a conforming program runs unchanged under CPython with only the standard library, from the workspace directory, and gives the same output there as inside lackpy. Inside lackpy it can only do less: a refusal is an error, never a different result.
- `LackpyRefusal` derives from `BaseException`, never from `Exception`. Programs cannot name it. Every refusal message names the approved alternative when `dictionary.UNAPPROVED` has one.
- The safety tier stays non-waivable. No task loosens a `safety.*` rule except Task 1's ruled nesting relaxation, which applies to `style.*` rules only.
- **Golden rule (CLAUDE.md):** change expectations first, run them, see them fail as predicted, then change code. When an existing test encodes the retired kit model, say in the commit which test changed and why.
- `lackpy-lang` stays stdlib-only and at Python ≥3.10. Runtime code lives in `src/lackpy/`, not in `packages/lackpy-lang`.
- Run tests only through blq (`test-lang`, `test-all`, `corpus`). Do not use shell pipes.
- Every change to what the validator admits records `(corpus: admitted A→B, structural S→T)` in its commit subject.

## Review Focus

1. **A path that leaves the workspace through `..`, a symlink, or an absolute path, built at run time.** Static checks cannot see it, so the `PathPolicy` must refuse it. Task 4 tests `Path("../x")`, `Path("/etc/passwd")`, a symlink in the workspace pointing out, and `Path(params["root"]) / ".."`.
2. **A proxy that leaks a real module or host class through a return value**, for example `json.loads.__module__` (blocked statically), or `Path("x").parent` returning a real `pathlib.Path`. Task 4 tests that every `Path` member that returns a path returns the stand-in.
3. **A program that catches the refusal by accident** through a handler the validator allows. Task 3 tests that `except ValueError`, `except KeyError` and `except OSError` around a refused call all let the refusal propagate.
4. **Generated programs from the new prompt still using the kit.** Task 8 re-runs the 30-task eval through `LackpyService.delegate` and asserts that no generated program calls `read_file` or `find_files`.
5. **A long-lived service process contaminated across runs.** The MCP server runs many programs. Task 6 tests that two consecutive `run_program` calls share no module or class state, for example a `Counter` subclass attribute set in run 1 is not visible in run 2. `safety.attr-store` (Task 2) blocks the write statically. The test proves the runtime does not depend on that.

---

## Today's runtime, in brief (the facts tasks rely on)

- `LackpyService.run_program` (src/lackpy/service.py:728) resolves a profile, gates tools, resolves params, and calls `_execute_program` (:573). That validates with `allowed_names = tools | param_names` (:594) and executes through `_execute` (:532) → `_run_sync` (:560), which does `os.chdir(workspace)` then `RestrictedRunner.run` (src/lackpy/run/runner.py:22).
- `RestrictedRunner.run(program, namespace, params, kibitzer_session)`:
  - binds kit tools wrapped by `make_traced` (trace.py:162, which catches `Exception` only)
  - binds `ALLOWED_BUILTINS` and a capturing `print`
  - injects params as bare globals (:65-67)
  - sets `exec_globals = {"__builtins__": {}}`
  - rewrites the last bare expression to `__result__`
  - compiles and `exec`s
  - turns errors into `ExecutionResult` (run/base.py:21), catching `Exception` only
- `delegate` (service.py:756) generates via `build_system_prompt` (infer/prompt.py:38), then validates and executes like `run_program`.
- `AsyncBridge.call_sync` (bridge.py:257) raises a plain `TimeoutError`.
- The policy layer (src/lackpy/policy/) governs tool sets only. The `sandbox` and `policy` profile keys are parsed and never enforced.

---

### Task 1: The ruled nesting relaxation, and `safety.attr-store`

**Files:**
- Modify: `packages/lackpy-lang/src/lackpy/lang/rules_style_shape.py` (`style.call-nesting`, `style.io-position`), `packages/lackpy-lang/src/lackpy/lang/rules_safety.py` (new `safety.attr-store`), `packages/lackpy-lang/src/lackpy/lang/spec.py` (pre-prompt habit row and shape text), `docs/reference/style-guide.md` (regenerate)
- Test: `tests/lang/test_style_shape.py`, `tests/lang/test_attr_store.py` (new), `tests/lang/test_preprompt.py`

**Interfaces:**
- `style.call-nesting`: the maximum depth is 3, not 2. `print(len(sorted(x)))` is valid. `print(len(sorted(list(x))))` (depth 4) is invalid.
- `style.io-position`: an I/O call may be an argument (`json.loads(Path(p).read_text())` is valid). An I/O call is still never the receiver of an I/O call. `Path(p).read_text().read_text()` is invalid, with the message "an I/O call must be the first method call of its chain".
- `safety.attr-store`: an `ast.Attribute` with `ctx` `Store` or `Del` is a violation. Message: `Assignment to an attribute is not allowed: '<expr>' at line N`. Hint: `build a new value and assign it to a name`.

- [ ] **Step 1: Flip and add expectations.**
  - `print(len(sorted(x)))` and `d = json.loads(Path(p).read_text())` become valid. Those programs need their imports: add `from pathlib import Path` and `import json`.
  - `print(len(sorted(list(x))))` is invalid (`style.call-nesting`).
  - Attribute-store cases, each invalid with `safety.attr-store`: `from collections import Counter\nC = Counter\nC.most_common = len`, `from pathlib import Path\n[Path][0].read_text = print`, and `x = []\nx.y = 1`. `del x.y` is invalid too.
  - Add `test_waive_everything` cases for the two host-mutation programs.
  - The pre-prompt test: replace the habit row "an I/O call inside another call" with nothing, because it is now allowed. The shape line becomes "A call nests at most 3 deep".
- [ ] **Step 2: Run, and expect failures** in exactly those tests (blq `test-lang`).
- [ ] **Step 3: Implement. Update each changed rule's `bad`/`good` examples. Regenerate the guide** (`.venv/bin/python scripts/gen_style_guide.py`).
- [ ] **Step 4: blq `test-all`, then blq `corpus`.** Record admitted/structural before and after. Expected: structural rises by roughly the 307 programs that `call-nesting` alone blocked, plus `io-position`'s 104. Report the measured figure, not this estimate.
- [ ] **Step 5: Commit** with the message `lang: nesting to depth 3, I/O calls as arguments, no attribute stores (corpus: …)`.

### Task 2: `LackpyRefusal` and the runtime namespace

**Files:**
- Create: `src/lackpy/run/refusal.py`, `src/lackpy/run/namespace.py`
- Modify: `src/lackpy/run/runner.py`
- Test: `tests/run/test_namespace.py` (new), `tests/run/test_runner.py`

**Interfaces:**
- `class LackpyRefusal(BaseException)` with fields `what: str`, `hint: str | None`. `str()` gives `"<what> is not allowed in lackpy" + (": " + hint if hint)`.
- `refuse(what: str, hint_key: str | None = None) -> NoReturn` looks up `dictionary.UNAPPROVED[hint_key]` for the hint.
- `build_namespace(*, program_file: str | None, path_policy: "PathPolicy", print_fn) -> dict` returns the exec globals:
  - `__builtins__`: a dict of every name in `builtins_policy.ALLOWED`, from the real `builtins` module, plus `__import__` (Task 3) and `__build_class__` absent
  - `__name__ = "__main__"`
  - `__file__`: `program_file`, or `"<stdin>"` when `program_file` is None
  - `type`: the stand-in, a callable whose result exposes only `__name__`. Any other attribute raises `LackpyRefusal`. Calling the result refuses. `type(x) is int` compares the stand-in object, so it is False and harmless, and static `safety.type-form` rejects it anyway.
  - `print`: `print_fn`
- `RestrictedRunner.run(program, namespace=None, params=None, kibitzer_session=None, *, program_file=None, path_policy=None)`. The kit `namespace` argument stays for backward compatibility until Task 7. When it is None, the runner uses `build_namespace`.
- **The runner never raises.** It catches `Exception`, `LackpyRefusal`, `RecursionError` and `MemoryError` around parse, rewrite, compile and exec. It reports `ExecutionResult(success=False, error=...)`. `KeyboardInterrupt` and `SystemExit` still propagate (programs cannot raise them, and the host owns them).

- [ ] **Step 1: Write the failing tests:**
  - `KeyError`, `json.JSONDecodeError` (after Task 3) and `repr` are available
  - `__name__ == "__main__"`, and `__file__` is `"<stdin>"` or the given path
  - `type(3).__name__ == "int"`
  - `type(3).mro` gives `LackpyRefusal`
  - a 3,000-deep unary expression returns `success=False` and does not raise (design §8, gate 4)
  - `LackpyRefusal` is not a subclass of `Exception`
- [ ] **Step 2: Run, and expect failure** (ImportError).
- [ ] **Step 3: Implement.** Keep the `__result__` rewrite and the capturing print.
- [ ] **Step 4: blq `test-all`.** The existing runner tests that pass a kit namespace must still pass.
- [ ] **Step 5: Commit.**

### Task 3: Restricted `__import__` and module proxies

**Files:**
- Create: `src/lackpy/run/modules.py`
- Modify: `src/lackpy/run/namespace.py`
- Test: `tests/run/test_modules.py` (new). Replace `tests/run/test_import_fail_closed.py` with `tests/run/test_import_proxies.py`; that test's docstring says Plan 2 replaces it.

**Interfaces:**
- `restricted_import(name, globals=None, locals=None, fromlist=(), level=0)` is bound as `__builtins__["__import__"]`. It returns a `ModuleProxy` for a name in `dictionary.MODULES`, and refuses anything else with the hint key `name` (for example `os` → "use pathlib.Path").
  - Design call (Claude), replacing the AST import rewrite of design §4.3: the restricted `__import__` is simpler, and has the same effect. The program keeps its `import` statements, so the plain-Python invariant holds trivially.
- `class ModuleProxy` has attributes `MODULES[name]` only, each the real member. Exceptions:
  - `pathlib.Path` → the `Path` stand-in (Task 4)
  - `collections.Counter` / `defaultdict` / `deque` / `OrderedDict` → a fresh subclass per run, so no state crosses runs (Review Focus 5). Each subclass keeps the base's `__name__` and `__qualname__`, so its repr reads `Counter({...})` exactly as under plain Python. The equivalence test depends on that.
  - `__getattr__` of anything else refuses
  - `__setattr__` and `__delattr__` always refuse
  - `__version__` returns the real module's value when it has one, else raises `AttributeError`, as plain Python would
- `from m import x` works through `fromlist`.

- [ ] **Step 1: Write the failing tests.**
  - The four alias escape routes from design §5.1: `statistics.sys`, `json.codecs`, `re.functools` and `collections.abc`, each also through an alias. Each refuses at run time even with the validator bypassed (call the runner directly).
  - `import os` refuses with the hint `use pathlib.Path`.
  - `json.loads('[1]') == [1]`.
  - `from collections import Counter` works.
  - Review Focus 3: wrap each of these programs in `try: ... except ValueError/KeyError/OSError:`. The refusal still propagates, and the result is `success=False`.
- [ ] **Step 2: Run, and expect failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: blq `test-all`.**
- [ ] **Step 5: Commit.**

### Task 4: `PathPolicy` from Claude-style permissions, and the `Path` stand-in

**Files:**
- Create: `src/lackpy/run/paths/__init__.py`, `src/lackpy/run/paths/rules.py` (pattern parsing and matching), `src/lackpy/run/paths/sources.py` (settings loading), `src/lackpy/run/paths/proxy.py` (the stand-in)
- Test: `tests/run/paths/test_rules.py`, `tests/run/paths/test_sources.py`, `tests/run/paths/test_proxy.py` (all new)

**Interfaces:**
- `@dataclass(frozen=True) class PathRule`: `kind: Literal["read", "edit"]`, `effect: Literal["allow", "deny"]`, `pattern: str`, `anchor: Path` (the directory a `/p` pattern is relative to), `source: str` (the file it came from).
- `parse_rule(text: str, *, effect, anchor: Path, source: str) -> PathRule | None`. It accepts `Read(...)` and `Edit(...)`. Any other tool rule returns None; `Write(...)` is accepted by Claude Code but never consulted, and this mirrors that. `ask` rules are parsed with `effect="deny"`.
- `class PathPolicy`:
  - `__init__(workspace: Path, rules: list[PathRule], extra_dirs: list[Path] = [])`
  - `check(path: Path, op: Literal["read", "write"]) -> None` raises `LackpyRefusal` on a refusal
  - `describe() -> list[str]` returns one line per effective rule, for the trace and for docs
- Semantics. Each item is a test case; source: `https://code.claude.com/docs/en/permissions`, fetched 2026-10-18.
  1. Order is deny, then allow. A deny from any source wins. `Read` denies block reads and writes. `Edit` denies block writes.
  2. Anchors:
     - `//p` is absolute.
     - `~/p` is under home.
     - `/p` is relative to `anchor`. For workspace settings that is the workspace. For `~/.claude/settings.json` it is `~/.claude`.
     - `p` and `./p` are relative to the workspace.
  3. Globs: `*` matches within one path segment, and `**` matches across segments. A bare filename such as `.env` matches at any depth. A single-segment directory pattern such as `src/**` matches at any depth for deny, but only `<workspace>/src` for allow.
  4. Default when no rule matches: read and write are allowed inside `workspace` and `extra_dirs`. Everything else is refused.
  5. Symlinks: an allow must match both the requested path and `resolve()`d target. A deny matches either. A write refuses when the final component is a symlink.
  6. Case sensitivity is not documented upstream. Match case-sensitively and say so in the docstring.
- `load_policy(workspace: Path, *, home: Path = Path.home(), managed: Path | None = MANAGED_SETTINGS) -> PathPolicy` reads, and unions (Teague's ruling 4, 2026-10-18: managed settings included):
  - Claude Code's managed settings, from `https://code.claude.com/docs/en/managed-settings` (fetched 2026-10-18). `MANAGED_SETTINGS` is chosen by `sys.platform`:
    - Linux and WSL: `/etc/claude-code/managed-settings.json`
    - macOS: `/Library/Application Support/ClaudeCode/managed-settings.json`
    - Windows: `C:\Program Files\ClaudeCode\managed-settings.json`. Do not read the legacy `C:\ProgramData` path.
  - The `managed-settings.d/` directory beside that file is merged after it: every `*.json` file in alphabetical order. Hidden files are skipped. Permission lists union, with duplicates removed.
  - When any managed source sets `allowManagedPermissionRulesOnly: true`, only managed rules apply. The user, project and local `allow`, `ask` and `deny` rules, and the `[paths]` table, are ignored for rules. `additionalDirectories` from them still define the default-readable directories. That last part is a design call (Claude); the docs speak of rules only.
  - Tests use a temporary `managed` path; they never read the real system file.
  - `home/.claude/settings.json`
  - `workspace/.claude/settings.json`
  - `workspace/.claude/settings.local.json`
  - `workspace/.lackpy/config.toml`, a `[paths]` table with `allow = [...]`, `deny = [...]` and `additional_directories = [...]`, using the same rule strings
  It also takes `permissions.additionalDirectories` from the settings files. A missing file is skipped. A malformed file raises `ValueError`, with the file name in the message; it is never silently ignored.
- `make_path_class(policy: PathPolicy) -> type`: a class named `Path` whose instances wrap a real `pathlib.Path`.
  - Members, exactly `dictionary.PATH_MEMBERS` plus `__truediv__`, `__str__`, `__fspath__`, `__eq__`, `__hash__` and `__lt__`, so `sorted()` works.
    - Reads: `read_text`, `read_bytes`, `exists`, `is_file`, `is_dir`, `glob`, `rglob`, `iterdir`. Each calls `policy.check(p, "read")` first.
    - Writes: `write_text`, `write_bytes`, `mkdir`. Each calls `check(p, "write")`.
    - Every member that returns paths returns stand-in instances.
  - Any other attribute refuses with the `UNAPPROVED` hint when one exists. `setattr` refuses.
  - Relative paths resolve against the workspace, not the process cwd. `__str__` and `__fspath__` return the path as the program built it: `str(Path("src") / "a.cpp") == "src/a.cpp"`, matching plain `pathlib`. The resolved absolute path is used only for checks and I/O.
  - `name`, `stem`, `suffix`, `parent` and `parts` are properties, as on `pathlib.Path`. `parent` returns a stand-in.

- [ ] **Step 1: Write the failing tests.**
  - One test per semantics item, 1–6 above.
  - The four Review Focus 1 inputs: `Path("../x")`, `Path("/etc/passwd")`, a symlink in the workspace pointing outside it, and a params-built `..`.
  - A `.claude/settings.json` with a `Read(./secrets/**)` deny makes `Path("secrets/a").read_text()` refuse, and makes writes there refuse too.
  - An `Edit(./out/**)` deny allows reads in `out/` and refuses writes.
  - `Path("x").parent` is a stand-in (Review Focus 2).
  - `Path.home()`, `Path("x").unlink()`, `Path("x").open()` and `Path("x").parser` each refuse. On Python 3.13, `parser` is the real `os` (design §8, gate 1).
- [ ] **Step 2: Run, and expect failure.**
- [ ] **Step 3: Implement.** Use `fnmatch.translate` per segment for `*`, and implement `**` yourself. Do not add a dependency.
- [ ] **Step 4: blq `test-all`.**
- [ ] **Step 5: Commit.**

### Task 5: Typed `params`, tracing through `Path`, refusals in the trace

**Files:**
- Modify: `src/lackpy/run/runner.py`, `src/lackpy/run/trace.py`, `src/lackpy/run/bridge.py`, `src/lackpy/service.py` (`_resolve_params`)
- Test: `tests/run/test_params.py` (new), `tests/run/test_trace.py`

**Interfaces:**
- Params (design §4.1, D15). The program's top-level `params = {...}` literal holds the defaults. The runner replaces values in that literal (an AST rewrite of the dict's value nodes) with the passed values before compile.
  - A passed key that the literal lacks is an error, reported in `ExecutionResult.error`.
  - A passed value whose type differs from the default's type is an error. Types are compared with `type(default)`, where `int` is accepted for a `float` default.
  - When a program has no `params` literal and params are passed, it is an error.
  - Bare-name injection is removed. The `[params]` docblock types arrive in Plan 3; this task infers types from the defaults.
- The trace records each `Path` read and write as `TraceEntry(tool="Path.read_text", args=[str(path)], ...)`.
- `make_traced` and the Path trace both record a `LackpyRefusal` as a failed entry, then re-raise it (design §8, gate 3).
- `AsyncBridge.call_sync` raises `LackpyRefusal("a tool call timed out")` in place of `TimeoutError` (gate 5). Design call (Claude): a program must not catch and retry a hung tool.
- `ExecutionResult.variables` excludes callables, classes, modules and proxies (gate 7). Design call (Claude): results carry data only.

- [ ] **Step 1: Write the failing tests.**
  - `params={"n": 5}` overrides the program's own `params = {"n": 3}`. This is the opposite of today's measured behaviour, where the program's `n = 3` silently won.
  - An unknown key, a type mismatch, and a program without a `params` literal each give `success=False`, with a message that names the key.
  - Trace entries for `Path` I/O.
  - A refused call appears in the trace.
  - A bridge timeout is uncatchable.
  - `variables` holds no function.
- [ ] **Step 2: Run, and expect failure.**
- [ ] **Step 3: Implement.** Update `_resolve_params` in the service: the old bare-name collision check goes, because params no longer bind names.
- [ ] **Step 4: blq `test-all`.** Tests that relied on bare-name params flip. Each flipped test is listed in the commit message.
- [ ] **Step 5: Commit.**

### Task 6: The equivalence test, and the gate suite

**Files:**
- Create: `tests/run/test_equivalence.py`, `tests/run/test_gate.py`
- Uses: `scripts/eval/ste_usability/tasks.py` and `fixtures/`. Commit the eval scaffolding in this task if the controller has not already.

**Interfaces:**
- `test_equivalence.py`, parametrized over the 30 eval tasks. Each case runs `task["reference"]` twice in fresh copies of `fixtures/`: once with `python3 prog.py` (subprocess, cwd = copy), and once through `LackpyService(workspace=copy).run_program(reference)`. It asserts equal stdout, equal written files, and success on both. This is design §5.3's invariant, measured.
- `test_gate.py` has one test per design §8 "Carried into Plan 2" item 1–9, each named after its item number. Item 9 is satisfied by Task 3's replacement of the fail-closed test.
  - It includes Review Focus 5: two consecutive `run_program` calls on one service share no module or class state.
  - Item 8 (`datetime.UTC`): the runtime requires Python ≥3.11, where `UTC` exists. The test asserts that `from datetime import UTC` runs. The static validator's 3.10 claim belongs to `lackpy-lang` and is out of scope here; record that in the test docstring.

- [ ] **Step 1: Write both test files.**
- [ ] **Step 2: Run them.** Every gate item implemented by Tasks 2–5 must pass. A failure is a defect in the owning task: report it, do not paper over it.
- [ ] **Step 3: Commit.**

### Task 7: Remove the kit from the agent's view

**Files:**
- Modify: `src/lackpy/service.py` (`run_program`, `delegate`, `validate`, `generate`, `create`, `_gate_tools`, `_planned_tool_calls`), `src/lackpy/mcp/server.py`, `src/lackpy/cli.py`, `src/lackpy/interpreters/python.py`
- Test: `tests/test_service.py`, `tests/test_mcp_tools.py`, `tests/test_mcp_contract.py`, `tests/test_cli.py`, `tests/interpreters/test_python.py`

**Interfaces:**
- **Default path:** `run_program(program, *, params=None, profile=None, ...)` with no profile runs in the new namespace (Task 2), with `load_policy(workspace)` as its `PathPolicy`. It validates with `allowed_names=set()`. No kit tools are bound.
- **The kit is removed from the runtime (Teague's ruling 5, 2026-10-18).**
  - Delete the four builtin kit tools. That means `read_file`, `find_files`, `write_file` and `edit_file` go from `src/lackpy/sources/builtins.py`, `src/lackpy/tools/providers/builtin.py` and `sources/default_tools.toml`.
  - `RestrictedRunner`, `LackpyService` and `PythonInterpreter` bind no tool callables. A profile now selects a model, its parameters and rules only. A profile that names tools is an error, with the message `tool kits were removed in lackpy stepy100; use pathlib.Path (docs/concepts/paths.md)`.
  - Anything that depended on kit binding, such as the literate interpreter's tool layer, plucker or pluckit, breaks until it migrates, as ruled. Do not hide that breakage:
    - Mark each newly failing test outside the files this task owns `xfail(strict=True, reason="kit removed in stepy100 Plan 2; <interpreter> migration pending")`.
    - Write `~/.claude/inbox/lackpy-interpreter-migration.md`, listing every xfailed test by interpreter.
  - The MCP and CLI drop `extra_tools` and the tool half of `profile`.
  - `tests/test_mcp_contract.py` pins the tool names and descriptions. Change them in this task's Step 1.
- **`PythonInterpreter`** uses the same namespace and policy (it shares `RestrictedRunner`).
- **The `os.chdir` model stays** for tools that use relative paths. The `Path` stand-in resolves against the workspace explicitly, whatever the cwd.

- [ ] **Step 1: Flip the expectations.**
  - The tests that assert kit tools exist, or are the default namespace.
  - The MCP and CLI parameters and description text.
  - Add `test_profile_naming_tools_is_an_error`.
  - Name each flipped test in the commit.
- [ ] **Step 2: Run, and expect failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: blq `test-all`.**
- [ ] **Step 5: Commit.**

### Task 8: Generation teaches the new language

**Files:**
- Modify: `src/lackpy/infer/prompt.py`, `src/lackpy/interpreters/python.py` (`system_prompt_hint`), `src/lackpy/infer/hints.py`, `src/lackpy/infer/correction.py` (`_FORBIDDEN_TO_TOOL`), `src/lackpy/infer/cleanup.py`, `src/lackpy/infer/providers/rules.py`, `src/lackpy/infer/steps/pick.py`
- Test: `tests/infer/test_prompt.py`, `test_hints.py`, `test_rules.py`, `test_cleanup.py`, `test_steps.py`, `test_templates.py`, `test_fixer.py`, `test_strategy.py`, `tests/test_prompt_examples.py`, `tests/interpreters/test_python.py`
- Create: `scripts/eval/ste_usability/run_lackpy.py`

**Interfaces:**
- `build_system_prompt(...)`, without a kit profile, returns `format_preprompt()` + the skeleton + `format_style_guide()` + the task framing. This matches the eval prompt that measured Qwen3-30B at 27/30 valid. The old "Jupyter notebook cell generator" / "Kernel namespace:" / "orchestrate tools" text goes.
- `PythonInterpreter.system_prompt_hint` drops "Do NOT use open(). Use read_file()", "ORCHESTRATE, DO NOT IMPLEMENT" and "Do NOT define functions, classes, or use import". The pre-prompt replaces all three.
- `hints.py` and `correction.py` map forbidden names to the `Path` forms from `dictionary.UNAPPROVED`. The source of truth is the dictionary, not a second table.
- `cleanup.py` stops rewriting `open().read()` into `read_file(...)`. Rewriting `open(p).read()` into `Path(p).read_text()` is a deterministic fix, so keep that rewrite, adding the `from pathlib import Path` import when it is missing.
- The rules provider and `pick.py` emit `Path` programs, not kit calls.
- `run_lackpy.py RUN_DIR LABEL ROUTE`: for each eval task, call `LackpyService(workspace=<fixtures copy>).generate(task["goal"], model=<route via woollama>)`, then save the program the way `run_local.py` does.

- [ ] **Step 1: Flip the pinned prompt-text tests to the new text.** Add `test_no_kit_names_in_default_prompt`.
- [ ] **Step 2: Run, and expect failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: blq `test-all`.**
- [ ] **Step 5: Measure, as the controller's step: device time is serialized, and the controller runs it.** Run `run_lackpy.py` with Qwen3-30B-A3B-Instruct on the explicit route. Score it with `score.py`, and compare it with the harness's `qwen3-30b-pre` r1. Review Focus 4: assert that no generated program calls `read_file` or `find_files`. Record the figures in `scripts/eval/ste_usability/FINDINGS.md`.
- [ ] **Step 6: Commit.**

### Task 9: Docs, and the skills handoff

**Files:**
- Modify: `README.md`, `docs/index.md`, `docs/getting-started.md`, `docs/tutorial.md`, `docs/concepts/language-spec.md`, `docs/concepts/profiles.md`, `docs/concepts/policy-layer.md` (fix the `PolicyContext["kit"]` key error while there), `docs/tools/*.md`, `mkdocs.yml`
- Create: `docs/concepts/paths.md` (the PathPolicy: sources, rule syntax, defaults, examples)
- Create: `~/.claude/inbox/lackpy-skills-stepy100.md` (the skills handoff; skills live in the retritis plugin repo)

**Interfaces:**
- **README:**
  - It leads with "lackpy runs plain Python that a validator has checked", and shows one `run_program` example and one `delegate` example.
  - It links the style guide and `docs/concepts/paths.md`.
  - The tagline "Python that lacks most of Python" stays; it is still true.
- **docs/index.md:** remove the false claims that `import`, `try` and `while` are "structurally impossible". `while` still is, but imports and `try` are allowed now.
- **The tutorial** walks through the guarded patch, the JSON read with `try`, and a scan with `def`, using the pre-prompt's three worked examples.
- **The tool pages** become `Path` member pages, or are removed, with redirects in `mkdocs.yml`.
- **The inbox note** follows the claude-inbox format. Its `**Agent:** retritis (repo: ~/Projects/retritis)` line names the skills to update:
  - `lackpy-workflow`: the tool surface and the no-kit default
  - `lackpy-delegation`: the caller contract, the pre-prompt, `Path` permissions and the `params` dict
  It links this plan and the style guide.

- [ ] **Step 1: Write the docs.** Every code block in the docs must validate. Add `tests/test_docs_examples.py`, which extracts each python fence from those files, runs `validate`, and asserts that it is valid.
- [ ] **Step 2: Run blq `test-all`, then `mkdocs build --strict` (through blq `exec`).**
- [ ] **Step 3: Commit.** Write the inbox note last.

### Task 10: Close-out

- [ ] **Step 1: blq `test-all` and `corpus`.** Record the figures in `scripts/corpus/README.md`.
- [ ] **Step 2: Check every design §8 gate item against `tests/run/test_gate.py`.** Mark them done in the design, with the commit of each.
- [ ] **Step 3: Commit.** The controller then asks Teague whether `mcp-tool-metadata` goes to `main`.
