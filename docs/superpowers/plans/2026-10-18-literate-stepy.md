# Literate stepy implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** lackpy's literate documents run on the stepy100 runtime. Every cell is validated and runs in the restricted namespace, with `Path` under `PathPolicy` and no kit. The `LITERATE_PENDING` refusal is then removed.

**Architecture:** A new `CellSession` in `src/lackpy/run/cells.py` owns one namespace from `build_namespace` per document. It validates each cell against the cumulative program, and it execs and evaluates through the same refusal and data handling as `RestrictedRunner`. `LightweightKernel` executes only through a `CellSession`. Prose interpolation leaves the exec'd code, and the annotations compile to `Path` forms. The tool layer and the host-exec paths are deleted.

**Tech Stack:** Python 3.11+, lackpy-lang (`lackpy.lang.validate`), `src/lackpy/run/*`, pytest via blq.

**Spec:** docs/superpowers/specs/2026-10-18-literate-stepy-design.md (decisions D1–D10, gates 1–5). The parent spec is docs/superpowers/specs/2026-10-18-lackpy-ste-design.md (§5.3, §8).

## Global Constraints

- Branch `literate-stepy`, made from `main` 0682519. Never push.
- Run tests through blq: `test-all`, plus focused runs with `extra=[path]`. Commit with jetsam. End each commit message with the session's Co-Authored-By and Claude-Session lines.
- Golden rule: write or flip the expectations first, run them and see RED, then implement.
- Never call the Tiiny device. Tests never read the real `~/.claude` or `/etc/claude-code`: use a tmp home, and set `managed=None`.
- A literate cell can do nothing that a lackpy program cannot do. No host-exec path stays, not even for tests.
- `LackpyRefusal` stays uncatchable. A refusal in a cell ends the document (D4).

## Review Focus

1. A route from a cell or a prose expression to host builtins, a real module, a frame, or an unconfined path. Probe with the Plan 2 Task 7 C1 escape document (gate 1).
2. Cumulative validation that differs from validating the concatenated program. For example: an import in cell 1 and an alias escape in cell 3, or a def in one cell called from another.
3. Reactive re-execution or recovery "inspect" that re-runs code without a validation it never received.
4. Forgiveness swallowing a refusal: a `Hole` or an `ErrorValue` must never stand in for a `LackpyRefusal`.
5. The concatenated code cells differing from `python3` on the equivalence fixtures.

---

### Task 1: `CellSession`

**Files:**
- Create: `src/lackpy/run/cells.py`.
- Modify: `src/lackpy/run/runner.py`. Extract the helpers for error text, data filtering and refusal catching, so both modules share them.
- Test: `tests/run/test_cells.py`.

**Interfaces:**
- `CellSession(*, path_policy: PathPolicy | None, trace: Trace | None = None, program_file: str | None = None)` holds one namespace.
- `run_cell(source: str) -> CellOutcome` validates the cumulative program, which is every executed cell plus this one. It reports only the violations on this cell's lines, then execs. `CellOutcome` has these fields:
  - `ok: bool`
  - `stdout: str`
  - `error: str | None`
  - `phase: Literal["validate", "runtime", "refused"] | None`
  - `delta: dict` (the names this cell bound or rebound, data only)
- `eval_expr(source: str) -> Any` validates `source` as a one-expression program in the cumulative context, then evaluates it in the namespace.
- `bound_names() -> set[str]` returns the user names bound so far.
- A `LackpyRefusal` gives `phase="refused"` and is never re-raised to the caller. A cell that catches it in `try` still gives `phase="refused"` (gate 4).

- [ ] **Step 1: Write the failing tests.**
  - Names, imports and defs flow across cells.
  - Imports and `json.codecs` across cells are refused, statically or at runtime (Review Focus 2).
  - A violation is reported on this cell's lines only.
  - A refusal inside `try`/`except Exception` gives `phase="refused"`.
  - `eval_expr("len(rows)")` works, and `eval_expr("rows.__class__")` is rejected.
  - Stdout is captured per cell.
  - `Path` works under a tmp policy.
- [ ] **Step 2: Run the tests and expect them to fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run blq `test-all`.**
- [ ] **Step 5: Commit.**

### Task 2: The kernel runs only through `CellSession`

**Files:**
- Modify: `src/lackpy/interpreters/literate/kernel/lightweight.py`, `compiler.py`, `display.py`, `__init__.py` (`_build_namespace`, `_run_document`, `_reexecute_dirty_subgraph`), `kernel/driver.py`, `kernel/recovery.py` (inspect), and `selftest.py`.
- Delete: `src/lackpy/interpreters/literate/journal.py` and its test.
- Test: `tests/literate/kernel/*`, `tests/literate/test_interpreter.py`, `test_compiler.py`, `test_forgiveness.py`, `test_dirty_subgraph.py` and `test_selftest.py`.

**Interfaces:**
- `LightweightKernel(session: CellSession)`. There is no namespace argument and no `base_dir`. `_cwd` and `_do_exec` are gone.
- Prose (D3): the compiler emits segments, not code. The kernel renders each text segment directly. It renders each `{expr}` segment as `display(session.eval_expr(expr))`.
- D4: a `phase="refused"` outcome stops `_run_document`. The cell gets `error_phase="refused"`, and the `[kernel]` note carries the hint. The ledger records it, and the run's success is False.
- D5: holes open before validation, as they do today.
- Reactive re-execution reruns through `session.run_cell`, so the cumulative check runs again. Recovery "inspect" uses `eval_expr`.

- [ ] **Step 1: Flip the expectations.** Tests that inject arbitrary namespaces or callables now build a `CellSession`. Delete a test that pins host exec, and name each deletion in the commit message. Add Review Focus 4: a refusal is never reified as a `Hole` or an `ErrorValue`.
- [ ] **Step 2: Run the tests and expect them to fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run blq `test-all`.**
- [ ] **Step 5: Commit.**

### Task 3: Annotations on `Path`; the tool layer goes

**Files:**
- Modify: `src/lackpy/interpreters/literate/parser.py` (the annotation compile targets), `compiler.py`, and `effects.py` (remove the grade map; strict mode reads the session trace).
- Delete: `literate/tools.py`, `literate/tool_effects.toml`, and their tests.
- Test: `tests/literate/test_effects.py`, `test_parser.py`, `test_unavailable.py`, and a new `tests/literate/test_annotations_path.py`.

**Interfaces:**
- `@read(p)` compiles to `Path(p).read_text()`, and `@write(p)` compiles to `Path(p).write_text(...)`. Each compiled cell imports `Path` when it needs it, so the concatenated program stays plain Python.
- `@diff(p)` runs on the host side (D6). Read through the session's `Path` stand-in, patch in pure Python, and write through the stand-in.
- The grade ceiling becomes a policy deny (D7). A read-only literate profile adds `Edit(//**)` to the policy's deny list.
- Strict reactive mode marks a cell world-effecting when its trace shows a `Path` write.

- [ ] **Step 1: Write the failing tests.**
  - `@read` and `@write` work under a tmp policy.
  - `@diff` against a denied path refuses.
  - A read-only profile refuses `@write`.
  - Strict mode withholds a dependent of a cell that wrote a file.
- [ ] **Step 2: Run the tests and expect them to fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run blq `test-all`.**
- [ ] **Step 5: Commit.**

### Task 4: Service, MCP and CLI seam, and the gates

**Files:**
- Modify: `src/lackpy/service.py` (`_resolve_profile`, `_execute_literate`), `src/lackpy/profiles.py` (remove `LITERATE_PENDING`), `src/lackpy/ctl.py` (`render --execute`), and `src/lackpy/mcp/server.py` if the result shape needs it.
- Test: `tests/test_service.py` (remove the `_LITERATE` xfails), `tests/test_mcp_contract.py`, `tests/test_ctl.py`, and new `tests/literate/test_gate.py` and `tests/literate/test_equivalence.py`.
- Create: `tests/literate/fixtures/`, holding 5 documents that mirror eval tasks: a guarded patch, a JSON read with `try`, a scan with `def`, a regex task, and a document that writes a file.

**Interfaces:**
- `_execute_literate` builds a `CellSession` with `load_policy(workspace, home=…, managed=…)`, through the service's existing policy seam. The result is an `ExecutionResult` with these fields:
  - `stdout`: the rendered document;
  - `output`: the last cell's value, made data-safe;
  - `trace`;
  - `error`: the first refusal or failure.
- `lackpyctl render --execute` runs the document through the same path.
- The gates come from the spec, items 1 to 5. Gate 5 is an AST scan of `src/lackpy/interpreters/literate/`. Every `exec`, `eval` and `compile` call there is either parse-only (an allowlist) or goes through `CellSession`.
- Equivalence: for each fixture, the concatenated code cells under `python3 -I` give the same stdout of code cells and the same files as the code-cell output of the document run. Prose output is excluded.

- [ ] **Step 1: Write the failing tests, and flip the refusal-pinning tests.**
- [ ] **Step 2: Run the tests and expect them to fail.**
- [ ] **Step 3: Implement.** Update `~/.claude/inbox/lackpy-interpreter-migration.md`: remove the literate entries and keep the rest. Load the claude-inbox skill before you edit it.
- [ ] **Step 4: Run blq `test-all`, and run the literate and run suites on Python 3.13 with the uv command from Plan 2 Task 6.**
- [ ] **Step 5: Commit.**

### Task 5: Generation, and the agent script

**Files:**
- Modify: `src/lackpy/interpreters/literate/prompt.py`, `service.py` (`_generate_literate`), and `scripts/literate_agent.py`.
- Test: `tests/literate/test_prompts.py` and `tests/test_service.py` (literate generation routing).

**Interfaces:**
- The literate system prompt is `format_preprompt()`, then the literate format (fences, prose interpolation, annotations), then one worked document. It contains no tool list, no `import os`, and no `run_command`. A test, `test_no_kit_or_subprocess_names_in_literate_prompt`, checks this.
- `scripts/literate_agent.py` runs each step through the service path, or through a `CellSession` built with `load_policy`. It no longer uses a bare `LiterateSession` over a host namespace.

- [ ] **Step 1: Flip the pinned prompt text, and add the no-kit test.**
- [ ] **Step 2: Run the tests and expect them to fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run blq `test-all`.**
- [ ] **Step 5: Commit.**

### Task 6: Docs and close-out

**Files:**
- Modify: `docs/concepts/literate.md` (rewrite it for stepy, with the D1–D10 behaviour stated in ASD-STE100), `tests/test_docs_examples.py` (also validate the literate fences in `literate.md`, cell by cell, as cumulative programs), and `docs/concepts/paths.md` (one line: documents use the same policy).
- Close-out: run blq `test-all` and record the counts in the ledger. Mark the gates done in the spec, with commits.

- [ ] **Step 1: Write the docs.** Load the ste-writing skill first.
- [ ] **Step 2: Run blq `test-all`, then `mkdocs build --strict` through the uvx route.**
- [ ] **Step 3: Commit.**
