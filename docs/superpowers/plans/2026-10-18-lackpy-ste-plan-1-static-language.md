# lackpy STE — Plan 1: the static language Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `lackpy-lang`'s validator into the STE rule set from the approved design: a registry of named rules in two tiers, the loosened language the corpus needs, the new STE rules, `# ste:` waivers, a generated style guide, and a corpus harness that measures every change.

**Architecture:** Every check becomes a `Rule` in a registry (`id`, `tier`, guide text, a non-conforming and a conforming example, a `check` function). `validate()` runs the registry and returns structured `Violation`s plus the old `errors` strings. The guide's Part 1 and Part 2 are generated from the registry and the dictionary, and a test proves that each rule's own examples behave as the guide says.

**Tech Stack:** Python ≥3.10 stdlib only in `lackpy-lang` (`ast`, `tokenize`, `builtins`, `dataclasses`). DuckDB (Python package) for the corpus scripts only. pytest. Builds and tests run through blq.

**Spec:** `docs/superpowers/specs/2026-10-18-lackpy-ste-design.md` (approved 2026-10-18, commit `c750b63`). Read sections 2.3–2.4, 4.3, 4.4, 5.1 and 5.3 before starting.

## This plan in the series

The design splits into four plans, and each one ends with working, tested software:

1. **This plan: the static language.** `lackpy-lang` only, plus the corpus harness. It also includes two small runtime edits that the static changes force (removing `sort_by` and deriving `ALLOWED_BUILTINS`).
2. **The runtime.** Proxies for modules and `Path`, the uncatchable refusal, the `type` stand-in, the `params` dict, the dunder values, tracing through `Path`, and the equivalence test.
3. **The intent.** The `# /// intent` parser, the `[params]` rules, the derivation, and the comparison. Before it starts, decide this: TOML parsing needs `tomllib` (Python 3.11 and later), but `lackpy-lang` supports Python 3.10 and later.
4. **The surfaces.** The MCP and CLI bundle, the skeleton, the inference prompts, guide Part 3, and the translation measurement.

## Global Constraints

- Branch `lackpy-ste`, which is stacked on the unmerged `mcp-tool-metadata`. Do not merge `lackpy-ste` before Plan 2 lands. After this plan, some programs validate but cannot run yet. Examples: `import json` (the runtime has no `__import__`) and `type(x).__name__` (no stand-in yet).
- `lackpy-lang` stays stdlib-only, with `requires-python = ">=3.10"`. Use no 3.11-only features in `packages/lackpy-lang`. `tests/lang/test_no_upward_deps.py` must stay green.
- **Golden rule (CLAUDE.md):** never change implementation and tests in one step. Every task first changes the expectations, runs them, and sees them fail as predicted. Only then does it change the code.
- A safety-tier rule is never waivable. A style-tier rule is waivable only with `# ste: <rule-id>, <why>`.
- Keep the existing `errors` strings of every ported rule byte-identical. `src/lackpy/infer/distill.py` and `tests/test_distill.py` match on them, for example `Forbidden name: 'open' at line 1`.
- Run tests through blq (`mcp__plugin_blq_blq_mcp__run`), never through raw pytest in Bash. Use no shell pipes. Filter the output with `output(...)`.
- Every task that changes what the validator admits records the corpus delta in its commit message, as `corpus: admitted A→B, structural S→T`.
- Write prose (the guide text and the rule summaries) to the ASD-STE100 subset in `~/.CLAUDE.md`.

## Review Focus

1. **A loosened construct that reopens an escape.** Allowing `try` makes exception classes reachable, and `KeyError.mro()` reaches `object`. Task 3 adds `mro` to `DENIED_ATTRIBUTES` and tests it.
2. **An alias that defeats a static module check.** `s = statistics` followed by `s.sys` passes any static check. Task 4 adds a test that records this gap as an expected pass (`xfail(strict=True)`, reason "runtime proxy, Plan 2"), so that Plan 2 must turn it into a refusal.
3. **A waiver that names a safety rule.** It must produce the violation `meta.waiver-safety`, and the safety violation must remain (Task 10).
4. **`except BaseException` or a bare `except`, with a waiver.** These must stay rejected, because a handler that catches `BaseException` would swallow Plan 2's refusals. They belong to the safety rule `safety.except-base` (Task 8).
5. **A program that defines a function with the same name as a builtin** (`def sorted(...)`). Shadowing breaks "one definition site per concept". `style.def-shadow` rejects it (Task 7).

---

## Rule ids (the contract for every task)

Every rule carries a `bad` and a `good` example, and `test_rule_examples_behave_as_documented` (Task 2) runs both for every rule. A task that adds or changes a rule writes or updates its examples in the same step as the rule. Each example must be a complete program, with its imports.

| id | tier | task | rule |
|---|---|---|---|
| `safety.parse` | safety | 2 | the program parses |
| `safety.node` | safety | 2 (then 6, 7, 8 extend it) | only allowed AST node types |
| `safety.forbidden-name` | safety | 2 | no name from `FORBIDDEN_NAMES` (with a hint) |
| `safety.attr-private` | safety | 2 (then 5) | no attribute beginning with `_`, except the allowlisted forms |
| `safety.attr-denied` | safety | 2 (then 3) | no attribute in `DENIED_ATTRIBUTES` |
| `safety.call-target` | safety | 2 | only named calls and method calls |
| `safety.unknown-call` | safety | 2 (then 4, 7) | a called name is an allowed builtin, an allowed name, a program `def`, or an import binding |
| `safety.import` | safety | 4 | imports only of dictionary modules, in the approved forms |
| `safety.module-attr` | safety | 4 | an attribute of a module binding is in that module's approved set |
| `safety.dunder-name` | safety | 5 | no name beginning with `__`, except `__name__` and `__file__` as loads |
| `safety.type-form` | safety | 5 | `type` only as `type(<expr>).__name__` |
| `safety.except-base` | safety | 8 | no bare `except`, no `except BaseException` |
| `style.assert-message` | style | 6 | `assert` has a string message |
| `style.for-iter` | style | 6 | a `for` iterates over a call, a name, or an expression with no call in it |
| `style.lambda-position` | style | 6 | `lambda` only as `key=`, or as an argument to `sorted`, `min` or `max` |
| `style.lambda-assign` | style | 6 | no `name = lambda ...` |
| `style.def-top-level` | style | 7 | `def` only at the top level, with no decorators |
| `style.def-before-use` | style | 7 | a function calls only functions defined above it |
| `style.def-signature` | style | 7 | no `*args` or `**kwargs`; defaults are constants |
| `style.def-return-mode` | style | 7 | a value on every return, or on none |
| `style.def-returns-function` | style | 7 | a function never returns a function |
| `style.def-length` | style | 7 | at most 12 statements in a body |
| `style.def-shadow` | style | 7 | a `def` never reuses a builtin name or an import binding |
| `style.try-named` | style | 8 | no `except Exception`; every handler names classes |
| `style.try-body` | style | 8 | at most 3 statements in a `try` body |
| `style.try-else-finally` | style | 8 | no `else:` and no `finally:` on a `try` |
| `style.try-handler` | style | 8 | each handler ends in an approved shape (definite assignment) |
| `style.genexp-consumed` | style | 9 | a generator expression is a call argument, and never stored |
| `style.attr-chain` | style | 9 | at most 3 attribute links (`a.b.c`) |
| `style.method-chain` | style | 9 | at most 3 chained method calls |
| `style.call-nesting` | style | 9 | a call nests at most 2 deep in arguments |
| `style.io-position` | style | 9 | an I/O call is never an argument, and is only the root of a chain |
| `style.comprehension-nested` | style | 9 | no comprehension inside a comprehension |
| `style.comprehension-multi-for` | style | 9 | one `for` per comprehension |
| `style.elif-chain` | style | 9 | an `if`/`elif` chain of 3 or more branches becomes a dict dispatch |
| `meta.waiver-safety` | (reported) | 10 | a `# ste:` that names a safety rule |
| `meta.waiver-unknown` | (reported) | 10 | a `# ste:` that names no known rule |

`meta.*` violations make the program invalid. They are not rules in the registry.

---

### Task 1: Corpus harness and blq commands

**Files:**
- Create: `scripts/corpus/extract.py`, `scripts/corpus/coverage.py`, `scripts/corpus/select_fixtures.py`, `scripts/corpus/README.md`
- Create: `scripts/corpus/baseline.json`
- Modify: `pyproject.toml` (dev extra: add `"duckdb>=1.1"`)

**Interfaces:**
- Produces: `python scripts/corpus/coverage.py [--snapshot PATH] [--json]`. It prints `{"parsed": int, "admitted": int, "admitted_pct": float, "structural": int, "structural_pct": float, "by_rule": {key: programs}}`. `structural` ignores the vocabulary keys `safety.forbidden-name`, `safety.unknown-call`, `safety.import` and `safety.module-attr`. Until Task 2 lands, the keys are the message-derived keys from the design's Appendix A (`name:open`, `node:Assert`, ...).
- Produces: the snapshot at `~/.local/share/lackpy/corpus/heredocs-2026-10-18.parquet`, with columns `session_file VARCHAR, description VARCHAR, body VARCHAR`. It holds distinct bodies only.

- [ ] **Step 1: Register the blq commands.** `register_command(name="test-lang", cmd="python -m pytest tests/lang -q")` and `register_command(name="corpus", cmd="python scripts/corpus/coverage.py --json")`. Verify with `commands()`: both are listed.
- [ ] **Step 2: Write `extract.py`.** It runs the two SQL statements in the design's Appendix A through the `duckdb` Python package. It deduplicates on `body` and writes the parquet file. If the snapshot already exists, it refuses to overwrite it unless given `--force`.
- [ ] **Step 3: Run the extraction.** Run `python scripts/corpus/extract.py` through `exec`. Expected: it prints a row count of distinct non-empty bodies of about 7,003. The count can be higher, because the history has grown since 2026-10-18. Record the exact count in `README.md`.
- [ ] **Step 4: Write `coverage.py`.** When the snapshot is missing, it must print `corpus not found: <path>` to stderr and exit with status 2. It must never report zeros.
- [ ] **Step 5: Verify the failure path.** Run `coverage.py --snapshot /nonexistent.parquet`. Expected: exit 2, with the message.
- [ ] **Step 6: Record the baseline.** Run blq `corpus`. Expected: `admitted_pct` near 0.1 and `structural_pct` near 36.5. The figures differ slightly if the snapshot count differs. Write the output to `baseline.json`.
- [ ] **Step 7: Write `select_fixtures.py`.** It selects 50 candidates: 15 guarded patches, 10 JSON reads, 10 scans, 5 regex programs, 5 with `try`, and 5 with `def`. It uses the shape regexes of design section 2.2. It writes them to `~/.local/share/lackpy/corpus/fixture-candidates/NN-<family>.py`, outside git. It commits nothing.
- [ ] **Step 8: HUMAN GATE.** Ask Teague to review the candidates for private content. Commit nothing into `tests/fixtures/corpus/` until Teague approves. If you are a subagent, report the candidate path and stop here. The rest of this plan does not depend on the fixtures.
- [ ] **Step 9: Commit** `scripts/corpus/`, `pyproject.toml`, and `baseline.json`, with the message `corpus: harness and baseline (admitted 0.1%, structural 36.5%)`, using the real figures.

### Task 2: Rule registry; port today's rules with no change in behaviour

**Files:**
- Create: `packages/lackpy-lang/src/lackpy/lang/registry.py`, `packages/lackpy-lang/src/lackpy/lang/rules_safety.py`
- Modify: `packages/lackpy-lang/src/lackpy/lang/validator.py`, `packages/lackpy-lang/src/lackpy/lang/__init__.py`, `scripts/corpus/coverage.py`
- Test: `tests/lang/test_registry.py`

**Interfaces:**
- Produces (in `registry.py`):
  ```python
  Tier = Literal["safety", "style"]
  @dataclass(frozen=True)
  class Violation: rule_id: str; line: int | None; message: str; hint: str | None = None
  @dataclass(frozen=True)
  class Waiver: rule_id: str; line: int; why: str
  @dataclass
  class Context:
      tree: ast.Module; source: str
      parents: dict[int, ast.AST]          # id(child) -> parent
      allowed_names: frozenset[str]        # caller-supplied (today: kit tool names)
      defined_functions: dict[str, int]    # top-level def name -> line
      module_bindings: dict[str, str]      # local name -> dictionary module ("json", "pathlib", ...)
      member_bindings: dict[str, tuple[str, str]]  # local name -> (module, member), from `from m import x`
  @dataclass(frozen=True)
  class Rule:
      id: str; tier: Tier; summary: str; rationale: str; bad: str; good: str
      check: Callable[[Context], list[Violation]]
  RULES: list[Rule]
  def register(id: str, tier: Tier, summary: str, rationale: str, bad: str, good: str) -> Callable  # decorator; appends to RULES; duplicate id raises ValueError
  def rule_by_id(rule_id: str) -> Rule
  ```
- Produces (in `validator.py`): `ValidationResult` keeps `valid`, `errors`, `calls` and `variables`. It gains `violations: list[Violation]` and `waivers: list[Waiver]`. `errors == [v.message for v in violations]`. `validate(program, allowed_names=None, extra_rules=None)` keeps its signature. `extra_rules` callables still return `list[str]`, which become `Violation(rule_id="extra", ...)`.
- `calls` keeps its meaning: named calls of builtins and allowed names only. It never includes program `def` names or import bindings, because the kibitzer pre-check reads it.

- [ ] **Step 1: Write the failing tests.** In `test_registry.py`:
  - `test_every_rule_id_is_unique_and_tiered`: ids are unique, and each tier is `safety` or `style`.
  - `test_rule_examples_behave_as_documented`, parametrized over `RULES`: `validate(rule.bad)` has a violation with `rule_id == rule.id`, and `validate(rule.good).valid` is true.
  - `test_errors_mirror_violations`: for `validate("import os\nopen('x')")`, `errors == [v.message for v in violations]`, and `errors` contains `Forbidden name: 'open' at line 2`.
  - `test_ported_rules_have_ids`: the ids `safety.parse`, `safety.node`, `safety.forbidden-name`, `safety.attr-private`, `safety.attr-denied`, `safety.call-target` and `safety.unknown-call` exist.
- [ ] **Step 2: Run them, and expect failure.** blq `run(command="test-lang", extra=["tests/lang/test_registry.py"])`. Expected: `ModuleNotFoundError: lackpy.lang.registry`.
- [ ] **Step 3: Implement the registry, and port each existing step of `validate()` into `rules_safety.py`** as a registered rule. Keep each message string exactly. Move the lambda rule (Step 5.5) and the for-loop rule (Step 5) unchanged for now, with the ids `style.lambda-position` and `style.for-iter`. Tasks 6 changes them. `validate()` builds `Context` once and runs `RULES` in order.
- [ ] **Step 4: Run the lang tests, then the full suite.** `run(command="test-lang")`, then `run(command="test")`. Expected: everything passes that passed on `c750b63`. The suite must not change: this task changes structure only.
- [ ] **Step 5: Switch `coverage.py` to group by `Violation.rule_id`, and re-run blq `corpus`.** Expected: the same `admitted` and `structural` counts as the baseline.
- [ ] **Step 6: Commit** with the message `lang: rule registry; port the validator's checks with no change in behaviour (corpus: unchanged)`.

### Task 3: Classified builtins; remove `sort_by`; deny `mro`

**Files:**
- Create: `packages/lackpy-lang/src/lackpy/lang/builtins_policy.py`
- Modify: `packages/lackpy-lang/src/lackpy/lang/grammar.py` (`ALLOWED_BUILTINS` derived; `"mro"` added to `DENIED_ATTRIBUTES`), `src/lackpy/run/runner.py:14-50,100-102` (delete `_sort_by` and its branch), `src/lackpy/infer/hints.py:39`
- Test: `tests/lang/test_builtins_policy.py`. Update `tests/lang/test_grammar.py:59`, `tests/lang/test_validator.py:193-212`, `tests/run/test_runner.py:134`, `tests/infer/test_hints.py:27` and the `sort_by` cases in `tests/lang/test_sandbox_escape.py`.

**Interfaces:**
- Produces: `ALLOWED: dict[str, str]` and `DENIED: dict[str, str]` (name → reason), and `allowed_builtin_names() -> frozenset[str]`.
- `ALLOWED`, exactly (from the corpus counts): `print len sorted str sum set dict range max list enumerate isinstance int any repr min next bool zip round all chr tuple float abs ord iter reversed frozenset hex`, plus every name `n` in `dir(builtins)` where `isinstance(getattr(builtins, n), type)` and `issubclass(..., Exception)` holds. Plus `ExceptionGroup`. It excludes `BaseException`, `SystemExit`, `KeyboardInterrupt`, `GeneratorExit` and `BaseExceptionGroup`.
- `DENIED`: every other name in `dir(builtins)` on the Python that runs the tests. Each has a reason category: `"reflection"`, `"code-execution"`, `"io"`, `"interpreter-control"`, `"type-machinery"`, `"use-comprehension"` (`map`, `filter`), `"binary"`, or `"dunder"`. Give `open` the reason `"io"`, and give `type` the reason `"type-machinery"`. Plan 2 adds the `type` stand-in.
- `grammar.ALLOWED_BUILTINS = allowed_builtin_names()`. The runner keeps injecting `getattr(builtins, name)` for each name, so exception classes become available at run time as a side effect.

- [ ] **Step 1: Change the expectations.**
  - Rewrite every `sort_by` test listed above to assert rejection: `validate("sort_by([], 'a')")` is invalid with `safety.unknown-call`, and `test_hints` no longer suggests `sort_by`.
  - Add `test_every_builtin_is_classified`: `set(dir(builtins)) <= set(ALLOWED) | set(DENIED)`, and `ALLOWED.keys() & DENIED.keys() == set()`.
  - Add `test_no_dangerous_name_is_allowed`: none of `open eval exec compile __import__ getattr setattr delattr hasattr globals locals vars dir type super object breakpoint input exit quit BaseException SystemExit KeyboardInterrupt GeneratorExit` is in `ALLOWED`.
  - Add `test_exceptions_allowed`: `KeyError`, `ValueError`, `FileNotFoundError` and `UnicodeDecodeError` are in `ALLOWED`.
  - Add `test_mro_denied`: `validate("x = KeyError.mro()")` has `safety.attr-denied`.
- [ ] **Step 2: Run them, and expect failure.** blq `test-lang`, then `test`. Expected: the new tests and the flipped `sort_by` tests fail. Nothing else fails.
- [ ] **Step 3: Implement `builtins_policy.py`. Then derive `ALLOWED_BUILTINS`, add `mro`, delete `_sort_by` from `runner.py`, and delete the hint line.**
- [ ] **Step 4: Run the full suite.** blq `test`. Expected: all pass.
- [ ] **Step 5: Run blq `corpus`, and commit** with the message `lang: classify every builtin; remove sort_by; deny mro (corpus: …)`.

### Task 4: The dictionary; imports; module attributes

**Files:**
- Create: `packages/lackpy-lang/src/lackpy/lang/dictionary.py`
- Modify: `grammar.py` (remove `"pathlib"` from `FORBIDDEN_NAMES`; remove `Import`, `ImportFrom` and `alias` from `FORBIDDEN_NODES` and add them to `ALLOWED_NODES`), `rules_safety.py`, `validator.py` (build `module_bindings` and `member_bindings`)
- Test: `tests/lang/test_imports.py`. Update `tests/lang/test_validator.py::TestForbiddenNodes::test_rejects_import*` (`import os` stays rejected, now under `safety.import`), the `pathlib` case at `tests/test_distill.py:107`, and `tests/eval/test_intents_python.py:46` (`import os` stays rejected, so check that it still is).

**Interfaces:**
- Produces in `dictionary.py`:
  - `MODULES: dict[str, frozenset[str]]`. These are the approved members, from the corpus counts in this plan's research:
    - `json`: `loads dumps JSONDecodeError`
    - `re`: `sub subn findall finditer search match fullmatch split compile escape error S I M X DOTALL IGNORECASE MULTILINE VERBOSE`
    - `collections`: `Counter defaultdict deque OrderedDict`
    - `math`: `comb sqrt exp`
    - `statistics`: `median mean stdev pstdev quantiles`
    - `hashlib`: `sha256 sha1`
    - `datetime`: `datetime date timedelta timezone UTC`
    - `pathlib`: `Path`
  - `PATH_MEMBERS: frozenset[str]`: `read_text write_text read_bytes write_bytes exists is_file is_dir glob rglob iterdir mkdir name stem suffix parent parts with_suffix relative_to joinpath`
  - `PATH_IO_METHODS: frozenset[str]`: `read_text write_text read_bytes write_bytes glob rglob iterdir mkdir exists is_file is_dir`
  - `UNAPPROVED: dict[str, str]` maps an unapproved word to its hint. Entries: `open` → `use Path(p).read_text() or Path(p).write_text(s)`; `json.load` → `use json.loads(Path(p).read_text())`; `json.dump` → `use Path(p).write_text(json.dumps(obj))`; `glob` → `use Path(d).glob(pattern)`; `os` → `use pathlib.Path`; `os.path` → `use pathlib.Path`; `map` → `use a comprehension`; `filter` → `use a comprehension with if`; `format` → `use an f-string`.
- Import forms accepted: `import m`, `import m as a`, `from m import x`, `from m import x as y`, where `m` is in `MODULES` and `x` is in `MODULES[m]`. Rejected under `safety.import`: any other module, `import m.sub`, relative imports, and `from m import *`.
- `safety.module-attr`: `<binding>.<attr>`, where `<binding>` is in `module_bindings`, requires `attr` to be in `MODULES[module]`. `safety.unknown-call` also accepts calls to `member_bindings` names (for example `Path(...)` and `Counter(...)`).
- Hints: a violation whose subject is in `UNAPPROVED` sets `Violation.hint`. The `errors` string stays unchanged.

- [ ] **Step 1: Write the failing tests** in `test_imports.py`:
  - `test_approved_imports_validate`, parametrized over `import json`, `import re as regex`, `from pathlib import Path`, `from collections import Counter as C`, and `import statistics`.
  - `test_unapproved_imports_rejected_with_hint`: `import os` gives `safety.import` with hint `use pathlib.Path`; `import subprocess` gives `safety.import`; `import os.path` gives `safety.import`; `from json import *` gives `safety.import`; `from . import x` gives `safety.import`.
  - `test_module_attribute_checked`: `import statistics\nx = statistics.sys` gives `safety.module-attr`; `import json\njson.codecs` gives `safety.module-attr`; `import json\njson.loads('1')` is valid.
  - `test_from_import_binding_is_callable`: `from pathlib import Path\np = Path('a')` is valid, and `calls` does not contain `Path`.
  - `test_alias_gap_is_runtime_work`, marked `xfail(strict=True, reason="runtime proxy, Plan 2")`: `import statistics\ns = statistics\nx = s.sys` is invalid.
  - `test_open_hint`: `validate("open('f')").violations[0].hint == "use Path(p).read_text() or Path(p).write_text(s)"`.
- [ ] **Step 2: Flip the existing expectations listed under Files. Run them, and expect failure.** blq `test-lang`, then `test`. Expected: only the new and flipped tests fail.
- [ ] **Step 3: Implement** `dictionary.py`, the binding collection in `validator.py`, `safety.import`, `safety.module-attr`, the `unknown-call` extension, and the hints.
- [ ] **Step 4: Run the full suite.** blq `test`. Expected: all pass, and `test_alias_gap_is_runtime_work` reports `xfail`.
- [ ] **Step 5: Run blq `corpus`, and commit** with the message `lang: approved imports and the dictionary (corpus: …)`.

### Task 5: Underscore names and the dunder allowlist

**Files:**
- Modify: `rules_safety.py`
- Test: `tests/lang/test_dunders.py`

**Interfaces:**
- `safety.dunder-name`: an `ast.Name` whose id begins with `__` is rejected. The exceptions are `__name__` and `__file__` with `ctx=Load`. A `Store` of either is rejected.
- `safety.attr-private` allows exactly two forms. The first is `<b>.__version__`, where `<b>` is in `module_bindings`. The second is `.__name__`, where the value is `Call(func=Name("type"), args=[one expr], keywords=[])`.
- `safety.type-form`: a `Name("type")` that is not the `func` of such a call, or whose call's parent is not `Attribute(attr="__name__")`, is rejected. Remove `type` from `FORBIDDEN_NAMES`, because this rule now owns it.
- Single-underscore names (`_`, `_tmp`) stay allowed.

- [ ] **Step 1: Write the failing tests**, parametrized:
  - Valid: `if __name__ == "__main__":\n    print(1)`; `p = __file__`; `import json\nv = json.__version__`; `n = type(3).__name__`; `for _, v in [(1, 2)]:\n    print(v)`.
  - Invalid: `__name__ = "x"` (`safety.dunder-name`); `__import__("os")` (`safety.dunder-name`); `x = __builtins__` (`safety.dunder-name`); `t = type(3)` (`safety.type-form`); `type(3)()` (`safety.type-form`); `x = (3).__class__.__name__` (`safety.attr-private`); `x = "a".__version__` (`safety.attr-private`); `t = type` (`safety.type-form`).
- [ ] **Step 2: Run them, and expect failure.** blq `test-lang`. Also re-run `tests/lang/test_sandbox_escape.py`: it must stay green before and after this task.
- [ ] **Step 3: Implement the three rules.**
- [ ] **Step 4: Run the full suite.** blq `test`. Expected: all pass, including every sandbox-escape test.
- [ ] **Step 5: Run blq `corpus`, and commit** with the message `lang: deny dunder names except an allowlist; type(x).__name__ form (corpus: …)`.

### Task 6: Loosen the simple constructs

**Files:**
- Create: `packages/lackpy-lang/src/lackpy/lang/rules_style_basic.py`
- Modify: `grammar.py` (add `Assert`, `Continue`, `Break`, `Pass`, `BitAnd`, `BitOr` and `BitXor` to `ALLOWED_NODES`; remove `Assert` from `FORBIDDEN_NODES`), `rules_safety.py` (remove the moved rules)
- Test: `tests/lang/test_style_basic.py`. Update `tests/lang/test_validator.py::test_rejects_assert` and `::test_rejects_standalone_lambda` (it stays rejected, but now under `style.lambda-assign`).

**Interfaces:**
- `style.assert-message`: `ast.Assert.msg` is a `Constant` str, or a `JoinedStr`.
- `style.for-iter`: `iter` is a `Call` or a `Name`, or it is an expression that contains no `Call` node.
- `style.lambda-position`: a `Lambda` parent is `keyword(arg="key")`, or the lambda is a positional argument of a `Call` whose `func` is `Name` in `{"sorted", "min", "max"}`.
- `style.lambda-assign`: `Assign` whose value is a `Lambda`.

- [ ] **Step 1: Write the failing tests:**
  - Valid: `assert x == 1, "x is one"`; `for k in [1, 2]:\n    print(k)`; `for k in d["items"]:\n    print(k)`; `for i in range(3):\n    if i:\n        continue\n    break`; `s = {1} & {2} | {3} ^ {4}`; `m = max(xs, key=lambda r: r[1])`; `m = sorted(xs, key=lambda r: r[1])`.
  - Invalid: `assert x` (`style.assert-message`); `for k in sorted(x)[:3]:\n    pass` (`style.for-iter`); `f = lambda x: x` (`style.lambda-assign`); `y = list(map(lambda x: x, xs))` (`safety.forbidden-name` for `map`, with the hint `use a comprehension`).
- [ ] **Step 2: Flip the two existing expectations. Run, and expect failure.** blq `test-lang`, then `test`.
- [ ] **Step 3: Implement.** `style.for-iter` and `style.lambda-position` were ported in Task 2 with the old behaviour. Update their `bad` and `good` examples in this same step, because the examples are part of the rule's definition, and `test_rule_examples_behave_as_documented` checks them.
- [ ] **Step 4: Run the full suite.** blq `test`.
- [ ] **Step 5: Run blq `corpus`, and commit** with the message `lang: allow assert with message, continue/break, set operators; loosen for and lambda (corpus: …)`. Expected: `structural` rises from the Task 5 figure toward the design's 73.5% for `assert` alone. If it does not rise, stop and investigate before you commit.

### Task 7: Function rules

**Files:**
- Create: `packages/lackpy-lang/src/lackpy/lang/rules_style_def.py`
- Modify: `grammar.py` (add `FunctionDef` and `Return` to `ALLOWED_NODES`, remove them from `FORBIDDEN_NODES`; `Yield`, `YieldFrom`, `AsyncFunctionDef`, `Global` and `Nonlocal` stay forbidden), `validator.py` (fill `defined_functions`), `rules_safety.py` (`unknown-call` accepts `defined_functions`)
- Test: `tests/lang/test_style_def.py`. Update `tests/lang/test_validator.py::test_rejects_function_def` and `tests/eval/test_intents_python.py:47`. That `def` case is now valid, so move it to the accepted examples there.

**Interfaces:** These rules implement the ids in the table above, using the design's section 4.3 "Rules for `def`".
- `style.def-length`: `len([n for n in ast.walk(fn) if isinstance(n, ast.stmt)]) - 1 <= 12`.
- `style.def-returns-function`: a `return` whose value is a `Lambda`, or a `Name` in `defined_functions`.
- `style.def-before-use`: every call of a name in `defined_functions` happens at a line after that function's `def`. A call to itself, or to any later function, is a violation. This is the rule that forbids recursion.
- `style.def-shadow`: the `def` name is in `ALLOWED_BUILTINS`, or in `member_bindings`, or in `module_bindings`.

- [ ] **Step 1: Write the failing tests:**
  - Valid: `def total(xs):\n    return sum(xs)\nprint(total([1]))`; `def f(a, b=2, c="x", d=None, e=(1, 2)):\n    return a`.
  - Invalid, one test per id:
    - nested `def` (`style.def-top-level`)
    - `@dec\ndef f():\n    return 1` (`style.def-top-level`)
    - `def f():\n    return f()` (`style.def-before-use`)
    - `def f():\n    return g()\ndef g():\n    return 1` (`style.def-before-use`)
    - `def f(*a):\n    return 1` (`style.def-signature`)
    - `def f(a=[]):\n    return a` (`style.def-signature`)
    - `def f(a=g()):\n    return a` (`style.def-signature`)
    - `def f(x):\n    if x:\n        return 1\n    return` (`style.def-return-mode`)
    - `def f():\n    return lambda: 1` (`style.def-returns-function`)
    - a 13-statement body (`style.def-length`)
    - `def sorted(x):\n    return x` (`style.def-shadow`)
    - `def f():\n    yield 1` (`safety.node`)
    - `def f():\n    global x` (`safety.node`)
- [ ] **Step 2: Flip the two existing expectations. Run, and expect failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the full suite.** blq `test`.
- [ ] **Step 5: Run blq `corpus`, and commit** with the message `lang: top-level def with STE function rules (corpus: …)`.

### Task 8: `try` rules

**Files:**
- Create: `packages/lackpy-lang/src/lackpy/lang/rules_style_try.py`
- Modify: `grammar.py` (add `Try` and `ExceptHandler`; `TryStar` stays forbidden), `rules_safety.py` (`safety.except-base`), and the comment above `DENIED_ATTRIBUTES` in `grammar.py`, which says "Try / Raise / sys remain excluded". Rewrite it to say that `Try` is now allowed, and that `__traceback__` (by the underscore rule), `tb_frame`, `tb_next` and `mro` are the guards.
- Test: `tests/lang/test_style_try.py`. Update `tests/lang/test_validator.py::test_rejects_try`: `try:\n    x = 1\nexcept:\n    pass` stays rejected, now by `safety.except-base`.

**Interfaces:** These implement the design's section 4.3 "Rules for `try`".
- `safety.except-base`: `handler.type is None`, or a handler names `BaseException`.
- `style.try-named`: a handler names `Exception`, or names something that is neither in `builtins_policy.ALLOWED` as an exception class nor a module attribute in `MODULES` (`json.JSONDecodeError`, `re.error`).
- `style.try-handler`: the last statement of each handler is `continue` (inside a loop), `return <expr>` (inside a `def`), `pass` (only when the `try` body assigns no name), or a set of assignments that covers every name the body assigns. The statements before it may be `print(...)` calls only. Definite assignment: a handler that falls through must assign every name that the body stores.

- [ ] **Step 1: Write the failing tests**, one per handler shape from the design table (skip, return, ignore, default), each valid. Then the invalid cases:
  - `try:\n    x = d[k]\nexcept Exception:\n    x = 0` (`style.try-named`)
  - `try:\n    x = d[k]\nexcept BaseException:\n    x = 0` (`safety.except-base`)
  - the same with `# ste: safety.except-base, test` (still `safety.except-base`; full waiver behaviour arrives in Task 10)
  - `try:\n    x = d[k]\nexcept KeyError:\n    pass\nprint(x)` (`style.try-handler`)
  - a 4-statement body (`style.try-body`)
  - `try:\n    x = 1\nexcept KeyError:\n    x = 0\nelse:\n    x = 2` (`style.try-else-finally`)
  - `try:\n    x = 1\nexcept KeyError:\n    x = 0\nfinally:\n    print(1)` (`style.try-else-finally`)
  - `try:\n    x = 1\nexcept KeyError as e:\n    t = e.__traceback__` (`safety.attr-private`)
  - `try:\n    x = 1\nexcept KeyError as e:\n    t = e.with_traceback(None).tb_frame` (`safety.attr-denied`)
- [ ] **Step 2: Flip `test_rejects_try`. Run, and expect failure.**
- [ ] **Step 3: Implement, and rewrite the `DENIED_ATTRIBUTES` comment.**
- [ ] **Step 4: Run the full suite.** blq `test`.
- [ ] **Step 5: Run blq `corpus`, and commit** with the message `lang: try with named errors and approved handler shapes (corpus: …)`.

### Task 9: STE shape rules

**Files:**
- Create: `packages/lackpy-lang/src/lackpy/lang/rules_style_shape.py`
- Test: `tests/lang/test_style_shape.py`. Check `tests/lang/test_sandbox_escape.py::TestGeneratorFrameEscapes::test_idiomatic_generator_forms_accepted`: its programs must still pass `style.genexp-consumed`. If one stores a generator, flip that case first, and say so in the commit message.

**Interfaces:** These implement the shape ids in the table above.
- `style.call-nesting`: the depth of `Call` nodes nested through `args` and `keywords` must be at most 2. `print(len(x))` is depth 2 and passes; `print(len(sorted(x)))` fails.
- `style.io-position`: a `Call` whose `func` is `Attribute` with `attr` in `PATH_IO_METHODS` must not be in another call's `args` or `keywords`. When it is the receiver of a method chain, it must be the innermost call of that chain.
- `style.method-chain`: count the consecutive `Call(func=Attribute(value=Call(...)))` links. More than 3 fails.
- `style.attr-chain`: `a.b.c.d` (4 `Attribute` links on a `Name`) fails.
- `style.elif-chain`: an `If` whose `orelse` is a single `If` that in turn has an `orelse` (3 or more branches) fails.

- [ ] **Step 1: Write the failing tests.** Prefix each program with the imports it uses (`from pathlib import Path`, `import json`). Without them, the program fails `safety.unknown-call` or `safety.module-attr` instead of the rule under test.
  - Valid: `text = Path(p).read_text().splitlines()`; `print(len(x))`; `n = sum(1 for x in xs if x)`; `a = x.b.c`; `y = s.strip().lower().split()`.
  - Invalid, one per id:
    - `d = json.loads(Path(p).read_text())` (`style.io-position`)
    - `g = (x for x in xs)` (`style.genexp-consumed`)
    - `print(len(sorted(x)))` (`style.call-nesting`)
    - `a = x.b.c.d` (`style.attr-chain`)
    - `y = s.strip().lower().split().pop()` (`style.method-chain`)
    - `y = [[a for a in r] for r in rows]` (`style.comprehension-nested`)
    - `y = [a for r in rows for a in r]` (`style.comprehension-multi-for`)
    - a 3-branch `if`/`elif`/`else` (`style.elif-chain`)
- [ ] **Step 2: Run, and expect failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the full suite.** blq `test`. Any existing test that now fails because of a shape rule is a decision point. Report it. Do not edit it quietly.
- [ ] **Step 5: Run blq `corpus`.** Compare with the design's figures: the four unconditional rules kept 97.5% conforming, and `call-nesting` and `io-position` were not measured at that time. Record the new per-rule counts in the commit message. Commit with the message `lang: STE shape rules (corpus: …)`.

### Task 10: `# ste:` waivers

**Files:**
- Create: `packages/lackpy-lang/src/lackpy/lang/waivers.py`
- Modify: `validator.py`
- Test: `tests/lang/test_waivers.py`

**Interfaces:**
- Produces: `parse_waivers(source: str) -> list[Waiver]`. It reads comments with `tokenize`. The form is `# ste: <rule-id>, <why>`. `why` is required and must not be empty. A malformed waiver gives `meta.waiver-unknown`.
- A waiver applies to violations on its own line, and on the line after it when the comment stands alone on its line.
- A waived `style.*` violation is removed from `violations` and appears in `ValidationResult.waivers`. A waiver that names a `safety.*` id gives `meta.waiver-safety`, and the safety violation stays. A waiver that names an unknown id gives `meta.waiver-unknown`.

- [ ] **Step 1: Write the failing tests:**
  - `g = (x for x in xs)  # ste: style.genexp-consumed, read twice below` is valid, with one waiver.
  - The same waiver on the line above, standing alone on its line, is also valid.
  - `open('f')  # ste: safety.forbidden-name, need it` is invalid. It has both `safety.forbidden-name` and `meta.waiver-safety`.
  - `# ste: style.nope, x` gives `meta.waiver-unknown`.
  - `# ste: style.genexp-consumed` with no reason gives `meta.waiver-unknown`.
  - A waiver does not leak to unrelated lines.
- [ ] **Step 2: Run, and expect failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the full suite.** blq `test`.
- [ ] **Step 5: Extend `coverage.py` to count waivers by rule id. Commit** with the message `lang: # ste: waivers for style rules; safety is never waivable`.

### Task 11: The generated style guide (Parts 1 and 2)

**Files:**
- Modify: `packages/lackpy-lang/src/lackpy/lang/spec.py`. Add `format_style_guide() -> str`, and make `get_spec()` include `"rules": [{"id", "tier", "summary"}]`.
- Create: `scripts/gen_style_guide.py` (it writes the file below), `docs/reference/style-guide.md`
- Modify: `mkdocs.yml` (add the page under Reference)
- Test: `tests/lang/test_style_guide.py`, and update `tests/lang/test_spec.py`

**Interfaces:**
- Guide structure (fixed copy):
  - Title: `# lackpy Style Guide`
  - A one-paragraph preface. It states the plain-Python rule (design section 5.3), the two tiers, and the waiver form.
  - `## Part 1. Program rules`, with one `### <id>` section per rule in registry order. Each section has the summary, the tier, `**Why:**` with the rationale, `**Not approved:**` with `bad` in a python fence, and `**Approved:**` with `good` in a python fence.
  - `## Part 2. Dictionary`, with these tables: approved builtins (from `ALLOWED`, grouped as functions and exceptions), approved modules and members (from `MODULES`), `Path` members, and unapproved words with their approved alternatives (from `UNAPPROVED`).
  - A closing note: `Part 3 (intent rules) arrives with Plan 3.`

- [ ] **Step 1: Write the failing tests.**
  - `test_guide_file_matches_generator`: `Path("docs/reference/style-guide.md").read_text() == format_style_guide()`.
  - `test_guide_lists_every_rule`: every `rule.id` appears as a heading.
  - `test_spec_exposes_rules`: `get_spec()["rules"]` has the same ids as `RULES`.
- [ ] **Step 2: Run, and expect failure.**
- [ ] **Step 3: Implement `format_style_guide`, run `scripts/gen_style_guide.py`, and read the generated file once, in full, as a reviewer would.** Check each rule's summary against the ASD-STE100 subset: at most 25 words, active voice, one idea.
- [ ] **Step 4: Run the full suite** (blq `test`) and lint (blq `lint`).
- [ ] **Step 5: Commit** with the message `docs: generated lackpy style guide, Parts 1 and 2`.

### Task 12: Close-out measurement

- [ ] **Step 1: Run blq `corpus`, and record the final figures in `scripts/corpus/README.md`.** Include a table of admitted and structural counts per task, taken from the commit messages.
- [ ] **Step 2: Compare with the design.** The design predicted 93.9% structural admission with `try` included (section 4.3). Report the measured figure. If it differs by more than 2 points, name the rules responsible, from `by_rule`. A gap here is a finding to report. It is not a failure to hide.
- [ ] **Step 3: Commit** with the message `corpus: plan 1 close-out figures`.
