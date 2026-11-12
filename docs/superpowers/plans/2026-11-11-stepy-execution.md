# Stepy Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `stepy` command that runs a program or a literate document like `python` runs a file; every document stops at its first failure; each cell can forbid its re-run; a script declares its interface and its writes in a `# /// intent` block, and `stepy` enforces the writes.

**Architecture:** Stop-on-failure is a change to `LiterateInterpreter._run_document`, and the binding layer and the reactive re-run are then removed. The session channel records the failed cell so a resume starts no later than it, and `LiterateRun` refuses a submit that would run a `rerun=no` cell again. A new `lackpy.intent` module parses the block. A write-glob layer on `PathPolicy` enforces declared writes. A new `lackpy.stepy` module is the command.

**Tech Stack:** Python 3.11+ (lackpy), `tomllib`, `argparse`, pytest. Linux only (fork).

**Spec:** `docs/superpowers/specs/2026-11-10-stepy-execution-design.md` (f36f00a, approved by Teague 2026-11-11). Where this plan and the spec disagree, the spec wins.

## Global Constraints

- Repo: `/srv/logical/workspace/home/teague/Projects/lackpy-worktrees/snape-literate-stepy`, branch `stepy-exec`. Never push.
- Tests: `blq run test-all` when blq works. The blq MCP server failed to connect on 2026-11-11, so the substitute is `.venv/bin/python -m pytest <paths> -q` through Bash. The full suite must stay green after each task.
- Golden rule: change a test's expectation first, run it and see it fail for the predicted reason, then change the implementation. Never change both in one step.
- No new runtime dependency (spec E5).
- Tests never read the real `~/.claude` or `/etc/claude-code`. Use the `tmp_policy_home` fixture (`tests/test_ctl.py`) or a `policy_loader` stub.
- Error text names the cell (its index and its first line), the cause, and the remedy when one exists. ASD-STE100 at about 80%: 25 words a sentence at most.
- Glossary terms from the spec, one per concept: program, document, script, cell, failure, refusal, intent block, declared writes, resume, re-run.
- Commit with `git commit -F <message file>`. End each message with:
  `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01SM9Rz4KCre34wobnTkmrfu`.
- Do not commit `.kibitzer/state.json` or `.riggs/store.duckdb`.

## Review Focus

1. A document whose failed cell printed something before it raised: the print reaches `printed`, and nothing from a later cell does. Owned by Task 5.
2. A resume after a failure where the next submit changes only a cell after the failed one: the resume still starts at the failed cell. Owned by Task 3.
3. `writes = ["out/{name}.txt"]` with `--name "../x"`: the filled-in glob must not let the write leave `out/`. Owned by Task 7.
4. A `.py` file with a shebang line above the intent block: the block is still found. Owned by Task 6.
5. A `rerun=no` cell after a changed prefix (a fresh run inside the same `LiterateRun`): still refused. Owned by Task 4.

---

### Task 1: Stop at the first failure in `_run_document` (spec E1)

**Files:**
- Modify: `src/lackpy/interpreters/literate/__init__.py` (the loop at about 299-378, `_run_resolved_cell` at 466)
- Create: `tests/literate/test_stop_on_failure.py`
- Modify: the tests that pin running past a failure. Find them by running the suite after Step 3. The spec names `tests/literate/test_dirty_subgraph.py`, `test_l1_conformance.py`, `test_annotations_path.py`, `test_checkpoint.py`, `test_selftest.py`.

**Interfaces:**
- Produces: `ExecutionResult.success is False` for a document with a failure. `error` starts with `cell {index} ({first line}): ` and then the cause. The index counts from 0, as `Cell` indices do. `error_phase` is `"timeout"` for a timeout, `"refused"` for a refusal, and None otherwise. Cells after the failed one are not run.
- Produces: `ExecutionResult.variables` holds the bindings from before the failed cell only.

Failure means each of these, at the cell where it happens: an exception, an unbound name (the hole pre-pass), the effect-ceiling gate (`gate_violations`), a timeout (a `Resume` of kind `"timeout"`), and a prose interpolation that raises. A refusal keeps today's behaviour: the document ends and the session closes (D4).

- [ ] **Step 1: Write the failing tests** in `tests/literate/test_stop_on_failure.py`, using the in-process interpreter (`LiterateInterpreter().execute`), as `test_l1_conformance.py` does:
  - `test_a_raising_cell_stops_the_document`: cells `a = 1`, `1/0`, `print("later")`. Assert `not result.success`, `"later" not in result.stdout`, `result.error.startswith("cell 1 (1/0): ")`, `"ZeroDivisionError" in result.error`.
  - `test_an_unbound_name_stops_the_document`: cells `x = y + 1`, `print("later")`. Assert failure, `"y" in result.error`, `"later"` not printed.
  - `test_a_gated_cell_stops_the_document`: a read-only `grade_ceiling` config and a write cell, then `print("later")`. Assert failure and no `"later"`.
  - `test_a_raising_prose_expression_stops_the_document`: prose with an interpolation of `1/0`, then a code cell `print("later")`. Assert failure and no `"later"`.
  - `test_a_refusal_still_ends_the_document`: keep a refusal case, with `error_phase == "refused"`.
  - `test_cells_before_the_failure_keep_their_bindings`: cells `a = 1`, `1/0`. Assert `result.variables == {"a": 1}`.
- [ ] **Step 2: Run them.** `.venv/bin/python -m pytest tests/literate/test_stop_on_failure.py -q`. Expected: every test fails except the refusal one, because the document runs on.
- [ ] **Step 3: Change the expectations of the old tests.** Run the full literate suite. For each test that asserts a run past a failure (an `ErrorValue`, a `Hole`, an `Unavailable` binding, a heal, or a re-run), choose one:
  - rewrite it to assert the stop, when its subject is still real (an error message, or a ledger entry for the failure);
  - mark it for deletion in Task 2, when its subject is only the binding layer or the heal. Add `pytest.mark.skip(reason="binding layer retired: deleted in Task 2")`. Do not delete it yet.
  Record the list (test id, then rewrite or delete) in the task report.
- [ ] **Step 4: Implement the stop.** In the loop, end the document at the first failure. Record a ledger entry for it, as today, and set the result's `error` in the form above. A timeout `Resume` ends the document with `error_phase="timeout"` instead of reifying and going on (`pending_timeout`). Do not delete the reify helpers yet: Task 2 removes them.
- [ ] **Step 5: Run the tests.** The new file passes. The full suite passes, apart from the tests skipped in Step 3.
- [ ] **Step 6: Commit** `feat(literate): stop a document at its first failure (spec E1)`.

### Task 2: Remove the binding layer and the reactive re-run (spec E3)

**Files:**
- Modify: `src/lackpy/interpreters/literate/__init__.py` (remove `_reify_error_for_cell`, `_reify_timeout_for_cell`, `_reify_unavailable_for_cell`, `_reify_holes_for_cell`, `_reexecute_dirty_subgraph` and its call, `_reactive_mode`)
- Modify: `src/lackpy/interpreters/literate/kernel/forgiveness.py`, `kernel/reactive.py`, `kernel/__init__.py`, `kernel/versions.py`, `kernel/persistence.py`, `selftest.py`: remove what only the binding layer used
- Modify: `src/lackpy/run/cells.py` (remove `rerun_cell` if no caller is left)
- Modify: `src/lackpy/interpreters/literate/session.py:20` (the docstring that says "a corrected round simply rebinds the holes")
- Delete: the tests skipped in Task 1 Step 3

**Interfaces:**
- Consumes: Task 1's stop.
- Produces: nothing new. `StreamingDriver` (`kernel/driver.py`) and `kernel/recovery.py` are not changed. If they import a name you remove, keep that name.

- [ ] **Step 1: Delete the skipped tests,** and only those. Run the suite. Expected: green.
- [ ] **Step 2: Remove the code.** Use `grep` for each removed name across `src`, `scripts` and `tests` before you remove it. A removed name must have no caller left. `round_is_left` has no source caller. Remove it with its tests unless the streaming path uses it.
- [ ] **Step 3: Correct the fold docstring** in `session.py` so that it says: a round stops at its first failure and returns Left, and literate_agent answers a Left with a full resubmission (`scripts/literate_agent.py`).
- [ ] **Step 4: Run the full suite and ruff** (`.venv/bin/ruff check src tests`). Expected: green, and no unused imports.
- [ ] **Step 5: Commit** `refactor(literate): retire the binding layer, holes and the reactive re-run (spec E3)`. The message lists the deleted tests.

### Task 3: A resume starts no later than the failed cell (spec E2)

**Files:**
- Modify: `src/lackpy/run/session_worker.py` (the result it sends: add `"failed_cell": int | None`)
- Modify: `src/lackpy/run/process.py` (`SessionChannel`: add the failed cell to `stopped`, and keep its checkpoint like the one in `_protect`)
- Test: `tests/literate/test_resume.py`

**Interfaces:**
- Consumes: Task 1. The worker knows the index of the failed cell.
- Produces: after a submit or a resume that ends with a failure at cell `i`, `SessionChannel.stopped` contains `i` and `can_resume(i)` is True. `LiterateRun._submit`'s ruling A (`run/sessions.py:178`) then needs no change.

- [ ] **Step 1: Write the failing tests** in `tests/literate/test_resume.py`, through `LiterateRun` with the existing fixtures of that file:
  - `test_a_resume_after_a_failure_starts_at_the_failed_cell`: submit cells `a = 1`, `b = a / 0`, `c = 3`. Then submit the same document with only cell 2 changed to `c = 4`. Assert that the second result fails again in cell 1, because cell 1 runs again. Assert that cell 0 did not run again. Use the trace or a print in cell 0, as the file's other tests do.
  - `test_a_fixed_failed_cell_resumes_from_its_checkpoint`: the same first submit, then change cell 1 to `b = a / 1`. Assert success, `variables["b"] == 1.0`, and that cell 0 did not run again.
- [ ] **Step 2: Run them.** Expected: the first test fails, because the resume starts at cell 2.
- [ ] **Step 3: Implement.** The worker's final result carries `failed_cell`. `SessionChannel` adds it to `stopped` when the document ends, and keeps the checkpoint before it. `max_checkpoints` must still hold that checkpoint and the latest (`process.py:944`).
- [ ] **Step 4: Run** the resume and checkpoint tests, then the full suite. Expected: green.
- [ ] **Step 5: Commit** `feat(run): a resume after a failure starts at the failed cell (spec E2)`.

### Task 4: `rerun=no` forbids any second run in a session (spec E4)

**Files:**
- Modify: `src/lackpy/interpreters/literate/parser.py` (validate the `rerun` option: `yes` or `no`)
- Modify: `src/lackpy/run/sessions.py` (`LiterateRun._submit`, `_fresh`)
- Test: `tests/literate/test_rerun_option.py`

**Interfaces:**
- Consumes: Task 3. Also `Cell.options` (`parser.py:129`), which already holds `key=value` options.
- Produces: `LiterateRun` keeps `self._ran_no_rerun: set[int]`, the indices of the `rerun=no` cells that have run in this `LiterateRun`. A submit whose start index (resume or fresh) is at or before one of them returns `ExecutionResult(success=False, error_phase="refused", error=...)` and runs nothing. The error says: `cell {i} ({first line}) has rerun=no and has already run in this session; start a new session to run it again`.

The rule is deliberately broad: a submit that starts at or before a `rerun=no` cell that has run is refused, even if the new document removed that cell. This follows "fail loud".

- [ ] **Step 1: Write the failing tests:**
  - `test_rerun_no_refuses_a_resume_after_a_later_failure`: cells `a = 1`, `` ```lackpy rerun=no `` `print("write")`, `1/0`. Then submit with cell 0 changed. Assert `error_phase == "refused"` and that `"rerun=no"` is in the error. Assert that `"write"` was printed once only, across both submits.
  - `test_rerun_no_allows_a_resume_after_it`: the same first submit, then change cell 2 only. Assert that the resume runs, starting at cell 2.
  - `test_rerun_no_refuses_after_its_own_failure`: a `rerun=no` cell that raises, then resubmit it fixed. Assert refused.
  - `test_rerun_no_refuses_a_fresh_run_in_the_same_session`: change `params` (which forces `_fresh`). Assert refused (Review Focus 5).
  - `test_rerun_yes_is_the_default`: no option, and the cell runs again on a resume.
  - `test_a_bad_rerun_value_is_a_validation_error`: `rerun=maybe` fails validation, and nothing runs.
- [ ] **Step 2: Run them.** Expected: they fail.
- [ ] **Step 3: Implement.** Record each `rerun=no` cell that reached `cell_start` (the channel sees each `cell_start` index). Check before both `channel.resume` and `_fresh`.
- [ ] **Step 4: Run** the new file and the full suite. Expected: green.
- [ ] **Step 5: Commit** `feat(literate): rerun=no forbids a second run of a cell in a session (spec E4)`.

### Task 5: `printed`, the output of print only (spec E11)

**Files:**
- Modify: `src/lackpy/run/base.py` (`ExecutionResult`: add `printed: str = ""`)
- Modify: `src/lackpy/interpreters/literate/__init__.py` (join each cell's captured stdout, `run/cells.py:351`, in cell order)
- Modify: `src/lackpy/run/session_worker.py:317` and `src/lackpy/run/process.py:856` (carry `printed` across the wire, and check that it is a `str`)
- Modify: the program path (`run/runner.py` result, and the worker result for programs): `printed` equals `stdout`
- Test: `tests/literate/test_printed.py`

**Interfaces:**
- Produces: `ExecutionResult.printed`. For a program, the same text as `stdout`. For a document, what the cells printed, in cell order, including what a failed cell printed before it failed (Review Focus 1). Prose and rendered output are never in it. `stdout` is unchanged for every caller.

- [ ] **Step 1: Write the failing tests:**
  - `test_printed_holds_only_cell_prints`: prose `# Title`, then cells `print("a")` and `x = 2`, then prose `{x}`. Assert `result.printed == "a\n"`.
  - `test_printed_keeps_the_failed_cells_output`: cells `print("one")`, `print("two"); 1/0`, `print("three")`. Assert `result.printed == "one\ntwo\n"`.
  - `test_printed_crosses_the_session_worker`: the same document through `LiterateRun`. Assert the same `printed`.
  - `test_program_printed_equals_stdout`: `print("hi")` through `run_program`. Assert `result.printed == result.stdout == "hi\n"`.
- [ ] **Step 2: Run them.** Expected: `AttributeError` or an empty `printed`.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the new file and the full suite. Expected: green.
- [ ] **Step 5: Commit** `feat(run): ExecutionResult.printed carries only what a script printed (spec E11)`.

### Task 6: The intent block (spec E5, E6)

**Files:**
- Create: `src/lackpy/intent.py`
- Test: `tests/test_intent.py`

**Interfaces:**
- Produces:
  - `class IntentError(ValueError)`.
  - `@dataclass(frozen=True) class Param: name: str; type: str; doc: str; default: Any`.
  - `@dataclass(frozen=True) class Intent: goal: str = ""; reads: tuple[str, ...] = (); writes: tuple[str, ...] | None = None; checks: tuple[str, ...] = (); output: str = "none"; params: tuple[Param, ...] = ()`. `writes is None` means the block has no `writes` key.
  - `def parse_intent(source: str) -> Intent | None`. It returns None when there is no block, and raises `IntentError` for a malformed one.
  - `def program_intent(program: str) -> Intent | None`: the block at the top of a program. Only a shebang line, blank lines and comments may come before it.
  - `def document_intent(document: str) -> Intent | None`: the block in the first code cell of a document. Use the literate parser.
  - `def check_params(intent: Intent | None, program: str) -> list[str]`: the error texts from the §4.1 checks, empty when they pass. It fills each `Param.default` from the `params` literal (`run/runner.py:_params_literal`).

Rules, from the spec and §4.1:
- The block is `# /// intent` … `# ///`. Strip `# ` from each line, then parse with `tomllib`.
- The allowed keys are `goal`, `reads`, `writes`, `checks`, `output` and the table `params`. Any other key is an `IntentError` that names the key.
- A second block is an `IntentError`.
- `output` is one of `none`, `printed` or `value`.
- Each `[params]` entry is `{ type = ..., doc = ... }`. The type is one of `str`, `int`, `float`, `bool`, `list[str]`.
- `check_params` checks the keys of `[params]` against the keys of the literal in both directions, and each default against its declared type. A `[params]` with no literal is an error. So is a literal with no `[params]`, but only when a block exists. A program with no block is valid.

- [ ] **Step 1: Write the failing tests** (one test each): the §4.1 example parses to the expected `Intent`; no block returns None; a shebang line above the block (Review Focus 4); an unknown key; a second block; a bad `output`; a bad param type; a key in `[params]` that is missing from the literal; a key in the literal that is missing from `[params]`; a default whose type does not match (`limit = "20"` with `type = "int"`); a document whose first code cell has the block; a document whose block is in the second cell, which is not found (returns None); `writes` absent gives `writes is None`, and `writes = []` gives `()`.
- [ ] **Step 2: Run them.** Expected: `ModuleNotFoundError`.
- [ ] **Step 3: Implement** `src/lackpy/intent.py`.
- [ ] **Step 4: Run** the tests. Expected: green.
- [ ] **Step 5: Commit** `feat: parse and check the # /// intent block (spec E5, E6)`.

### Task 7: Enforce declared writes (spec E7)

**Files:**
- Modify: `src/lackpy/run/paths/rules.py` (`PathPolicy`: a new field `write_globs: tuple[str, ...] | None = None`, kept by `to_data`/`from_data`; `check` refuses a write that matches no glob; add `def with_write_globs(policy: PathPolicy, globs: Sequence[str]) -> PathPolicy` beside `read_only_policy`)
- Modify: `src/lackpy/run/paths/proxy.py` (`mkdir`: the ancestor rule; rename and replace: check both ends)
- Modify: `src/lackpy/service.py` (`run_program(..., declared_writes: Sequence[str] | None = None)`, passed down to `_load_path_policy(read_only=..., declared_writes=...)` through `_execute_program`, `_execute` and `_execute_literate`)
- Test: `tests/run/test_write_globs.py`, and a service test in `tests/test_service_writes.py`

**Interfaces:**
- Consumes: Task 6's `Intent.writes`, filled in by the caller (Task 8).
- Produces: `run_program(..., declared_writes=None)` keeps today's behaviour. `declared_writes=[]` allows no write. A list allows writes that match one of its globs and that the settings policy also allows. A refused write is a `LackpyRefusal` whose text names the path and lists the declared globs.

Rules:
- The globs use the `PathRule` pattern semantics, relative to the workspace (spec E7).
- The caller fills in `{name}` before the call. `with_write_globs` refuses a glob that, after it is made absolute and normalised, is not under the workspace or an allowed directory. So a parameter value like `../x` cannot move a glob out (Review Focus 3).
- `mkdir(p)` is allowed when `p` is an ancestor of the fixed prefix of some glob, the part before its first wildcard.
- A rename is checked at both its source and its target.

- [ ] **Step 1: Write the failing tests:** a write that matches is allowed; a write that matches no glob is refused, and the text names the globs; `()` refuses every write; `None` keeps today's behaviour; the deny rules of the settings policy still win over a matching glob; `mkdir("out", parents=True)` with `out/*.txt`; a rename into a declared glob from outside it is refused; a glob that leaves the workspace after normalisation is refused by `with_write_globs`; `to_data`/`from_data` round-trips `write_globs`; and `run_program` with `declared_writes` refuses on both axes, program and literate.
- [ ] **Step 2: Run them.** Expected: they fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the new tests and the full suite. Expected: green.
- [ ] **Step 5: Commit** `feat(paths): enforce declared write globs on a run (spec E7)`.

### Task 8: The `stepy` command for programs (spec E8-E11)

**Files:**
- Create: `src/lackpy/stepy.py` (`def main(argv: list[str] | None = None) -> int`)
- Modify: `pyproject.toml` `[project.scripts]`: `stepy = "lackpy.stepy:main"`
- Test: `tests/test_stepy.py`

**Interfaces:**
- Consumes: `lackpy.intent` (Task 6), `run_program(..., declared_writes=...)` (Task 7), `ExecutionResult.printed` (Task 5).
- Produces: `main` for `.py` files and `-c`. Task 9 adds `.md`.

Behaviour:
- `argv[0]` is the script path, or `-c` and the program text. Every option after it belongs to the script. stepy's own options are `--help`, `--check`, `--timeout SECONDS`, `--profile NAME` and `--workspace DIR`. A `[params]` key may not use one of these names: that is an `IntentError` at check time.
- A parameter option is `--<key> VALUE`, converted to the declared type. A `bool` is `--<key>` or `--no-<key>`. A `list[str]` repeats the option. An underscore in a key is written as an underscore.
- Before a run, validate the intent block and `check_params`. An error exits 2, and nothing runs.
- Declared writes: `intent.writes` with each `{name}` filled in from the final parameter values. No block, or no `writes` key, gives `[]`.
- Output: write `result.printed` to stdout. On failure, write `stepy: ` and the error to stderr.
- Exit codes: 0 success; 1 failure or refusal; 2 usage or validation error with nothing run; 124 when `error_phase == "timeout"`.
- `--check`: validate (`service.validate`), the intent block and `check_params`. Exit 0 or 2. Print the errors to stderr.
- `--help`: print the goal, then each parameter (`--key TYPE (default: …) doc`), then `writes: declared, enforced`, `reads: declared, not checked`, and `checks`. With no block, print `this script declares nothing, so it may write nothing`.
- The workspace defaults to the current directory. A profile is resolved the way `lackpy` resolves it (`cli.py:_file_entrypoint`).

- [ ] **Step 1: Write the failing tests.** Call `main([...])` with `capsys` and a temporary workspace, and use the policy fixture so the real `~/.claude` is never read:
  - `test_runs_a_program_and_prints_only_its_prints`: exit 0, stdout `"hi\n"`, empty stderr.
  - `test_named_options_set_params`: the §4.1 TODO-counting example with `--root src --limit 1`.
  - `test_bool_and_list_options`.
  - `test_unknown_option_exits_2_and_runs_nothing`. The script prints, and nothing is printed.
  - `test_wrong_type_exits_2`: `--limit many`.
  - `test_no_block_means_no_writes`: a write is refused, exit 1, and stderr names the path.
  - `test_declared_write_is_allowed` and `test_write_outside_the_globs_is_refused`.
  - `test_param_in_write_glob_is_filled_in`, including `--name ../x`, which is refused (Review Focus 3).
  - `test_timeout_exits_124`: `--timeout 0.5` with `for i in iter(int, 1): pass`.
  - `test_check_runs_nothing`: exit 0, and nothing printed or written.
  - `test_help_lists_the_interface`.
  - `test_dash_c_runs_program_text`.
  - `test_other_extension_exits_2`.
  - `test_shebang_file_runs`: a `#!/usr/bin/env stepy` first line.
- [ ] **Step 2: Run them.** Expected: `ModuleNotFoundError`.
- [ ] **Step 3: Implement** `src/lackpy/stepy.py` and the entry point. Reinstall with `.venv/bin/python -m pip install -e . -q`, so the `stepy` script exists.
- [ ] **Step 4: Run** the tests. Then run `stepy` itself once on a temporary script, to check the installed entry point.
- [ ] **Step 5: Commit** `feat: the stepy command for programs (spec E8-E11)`.

### Task 9: `stepy` for documents (spec E8, E10, E11)

**Files:**
- Modify: `src/lackpy/stepy.py`
- Test: `tests/test_stepy.py`

**Interfaces:**
- Consumes: Task 8, `document_intent` (Task 6), Task 1's stop.
- Produces: `stepy doc.md [options]` runs the document with `execution="literate"`. The profile is resolved as `ctl._render` resolves it (`ctl.py:51-62`): the named profile, else the configured default, with the execution forced to literate. Move that resolution into one helper that both commands call, rather than copying it. `--help` for a document says that `--timeout` is the limit for each cell.

- [ ] **Step 1: Write the failing tests:**
  - `test_document_prints_only_cell_output`: prose plus two printing cells. stdout is the two prints only.
  - `test_document_stops_at_the_first_failure`: exit 1, a later cell's print is absent, and stderr names the cell.
  - `test_document_params_from_options`: a first-cell block with `[params]`, and a cell that prints the parameter.
  - `test_document_declared_writes_are_enforced`.
  - `test_document_timeout_is_per_cell_and_exits_124`.
  - `test_render_execute_is_unchanged`: `lackpyctl render --execute` still prints the rendered document.
- [ ] **Step 2: Run them.** Expected: exit 2 for the `.md` extension, or failures.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `tests/test_stepy.py`, `tests/test_ctl.py`, and the full suite. Expected: green.
- [ ] **Step 5: Commit** `feat: stepy runs literate documents (spec E8)`.

### Task 10: Replace the `_BIND_THROUGH_HOLES` clause (spec, "Decided: unbound names")

**Files:**
- Modify: `src/lackpy/interpreters/literate/prompt.py:54` (the clause and its name)
- Test: the prompt tests that quote the clause (find them with `grep -rn "BIND_THROUGH\|Bind through" tests`)

**This task stops for Teague.** The clause is Teague's parked wording. The new text must not be committed until he approves it.

- [ ] **Step 1: Draft the new clause** in the task report only, not in the source. Name it `_CELLS_IN_ORDER`. It must say, in the format's STE style: a cell must come after the cells whose names it reads; the first failure stops the document; to fix one, correct the failed cell and resubmit, and the cells before it do not run again. Keep it to four sentences or fewer.
- [ ] **Step 2: Stop and report** `NEEDS_CONTEXT` with the draft. The controller asks Teague, and gets his approved text.
- [ ] **Step 3: Change the prompt test expectations** to the approved text, and run them. Expected: they fail.
- [ ] **Step 4: Replace the clause** in `prompt.py` with the approved text, and run the tests. Expected: green.
- [ ] **Step 5: Commit** `feat(literate): the prompt says cells run in order and stop at a failure`.
