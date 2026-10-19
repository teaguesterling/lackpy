# Literate stepy: design

Date: 2026-10-18. Status: written without a question round. Teague said "Start on the literate version", so every decision below is Claude's call unless it is marked as Teague's.

## Intent

lackpy has a literate format: a markdown document whose fenced cells run in order, with prose that interpolates values. Plan 2 of stepy100 refused it (`LITERATE_PENDING`), because its cells ran as unrestricted host Python. They had full builtins, `os`, `subprocess`, and a process-wide `os.chdir`. In the Plan 2 Task 7 review, a document read a file outside the workspace and reached host `os`.

Teague called the literate format "a really useful, safe format" and wants it back (2026-10-18). This design puts it on the stepy100 runtime:

- every code cell is a validated lackpy program;
- every cell runs in the restricted namespace;
- file access goes through `Path` under the `PathPolicy`;
- there is no kit.

## Goals

1. A literate document runs through `run_program` / `delegate` with `execution="literate"`, through MCP, and through `lackpyctl render --execute`. No `LITERATE_PENDING` remains.
2. Each code cell is validated, on the cumulative program as described in §Validation. It then runs in a namespace built by `lackpy.run.namespace.build_namespace`, which all the cells of one document share.
3. No code from a document reaches host builtins, a real module, `subprocess`, or a path outside the policy. The Plan 2 Task 7 escape document is a gate test, and it must refuse.
4. Plain-Python invariant (spec §5.3 extended): the concatenation of a document's code cells, in order, is a lackpy program. It runs unchanged under `python3`, with the same written files.
5. The kernel features that are pure text and data logic keep working. These are the parser and its annotations, prose interpolation, holes and forgiveness, the ledger, versions, reactive re-execution, streaming, and the session fold.

## Non-goals

- New document features. The intent block (`# /// intent`) belongs to Plan 3.
- A program time limit. It is Plan 3's first task (Teague, 2026-10-18). A literate document gets it when programs do.
- Subprocess tools. `run_command`, `run_tests` and `search_content` are removed, not sandboxed. The stepy100 language forbids subprocess.
- Notebook (`.ipynb`) execution. `kernel/formats.py` may still render one.

## Anti-goals

- A second, looser runtime for documents. A literate cell must not be able to do anything a lackpy program cannot do.
- Any path that still execs document code with host builtins. That includes test-only and library-only paths: `LightweightKernel` must not keep a host-exec mode.

## Decisions

### D1. The cell runtime is new code in `src/lackpy/run/cells.py`
`CellSession(workspace, path_policy, trace)` owns one namespace from `build_namespace`, and owns the print capture for each cell. It exposes four methods:
- `run_cell(source) -> CellOutcome`, which validates the source, then execs it;
- `eval_expr(source) -> value`, which validates the source as an expression program, then evaluates it;
- `bound_names()`;
- `snapshot()`.

It reuses `RestrictedRunner`'s pieces: error text, `_is_data`, and refusal handling. It does not reuse `RestrictedRunner.run`, because that builds a fresh namespace for each run.

### D2. Validation is cumulative
A cell is valid when the code of every earlier cell, followed by this cell, validates as one lackpy program. The validator then tracks imports, defs and names across cells exactly as it does in one program. Only violations on this cell's lines are reported against it. An earlier cell's violation has already stopped that cell.

A cell that the reactive kernel re-runs was validated on its first run.

### D3. Prose leaves the exec'd code
Today prose compiles to `print(__literate_display__(...))`, and the validator rejects dunder names. In the new design, the kernel splits prose into text and `{expr}` segments. It validates each expression as a one-line lackpy program in the session's namespace, and evaluates it with `eval_expr`. The display helper (`display.py`) formats the value on the host side.

### D4. A refusal ends the document
`LackpyRefusal` stays uncatchable. The kernel's `except BaseException` today turns every error into a cell error, so a refusal would become an ordinary cell error. The new rule is that a refusal stops the run. The cell gets `error_phase="refused"`, its hint is rendered through the `[kernel]` channel, and later cells do not run. The overall result is `success=False`.

Forgiveness still reifies an ordinary `Exception` as an `ErrorValue`, as it does today.

### D5. Holes come before validation
A name that no cell binds yet opens a typed `Hole`, and the cell is skipped. This is unchanged from today. Validation runs only on a cell that is about to execute, so the validator never sees a hole name.

### D6. Annotations use Path, not tool calls
- `@read(p)` compiles to `Path(p).read_text()`.
- `@write(p)` compiles to `Path(p).write_text(...)`.
- `@diff(p)` is applied on the host side. The kernel reads `p` through the session's `Path` stand-in, applies the diff with the existing patcher in pure Python, and writes the result back through the same stand-in. So the policy checks both the read and the write. No patch function is exposed to the program.
- `@scratch`, `@hidden`, `@gather` and `@continue` are unchanged.

### D7. The tool layer goes
`literate/tools.py` is deleted, together with its six shims. `tool_effects.toml` and the grade map in `effects.py` go too, since they key on kit names.

The grade ceiling becomes a policy: a read-only literate profile adds a deny on every write (`Edit(//**)`) to the policy. Reactive strict mode stops using the `sys.addaudithook` observer. To find "world effects", it reads the session trace for `Path` write entries.

### D8. No cwd change
`lightweight.py` stops calling `os.chdir`, because the `Path` stand-in already resolves against the workspace. `journal.py` is unwired today, and it touches files outside the policy, so it is deleted.

### D9. The service seam
`_execute_literate` builds a `CellSession` with `load_policy(workspace)`, then runs the document. It validates per D2, and returns an `ExecutionResult` with `stdout` (the rendered document), `output`, `trace` and `error`.

`LITERATE_PENDING` is removed from `profiles.py`, `service.py` and `ctl.py`. The strict xfails tagged "literate migration pending" turn back into passing tests, or into rewritten tests. The inbox note `~/.claude/inbox/lackpy-interpreter-migration.md` drops them.

### D10. Generation
The literate prompt (`literate/prompt.py`) teaches stepy. It combines `format_preprompt()`, the literate format and the annotations. It uses no tool list and does not mention `import os`.

`scripts/literate_agent.py` runs documents through the new session, which closes the parked residual from Plan 2. The staged-dsl refusal stays.

## What is left to the builder

- How `CellSession` shares code with `RestrictedRunner`. Do not duplicate logic: extract helpers.
- How the 24 test files in `tests/literate` change. Delete a test that pins host exec or a kit tool. Rewrite it when the behaviour it pins survives.
- The text of the `[kernel]` refusal annotation.

## Gates

1. The Plan 2 Task 7 C1 escape document refuses through every entry point. That document read a file outside the workspace through `(lambda: 0).__globals__`, walked `object.__subclasses__()`, and reached host `os`.
2. A cell that writes `.claude/settings.json` refuses, which shows the Plan 2 write protection holds inside a document.
3. For the equivalence fixtures, the concatenated code cells match `python3` in stdout and in written files.
4. A refusal inside a `try`, in a cell, stops the document.
5. No `exec`, `eval` or `compile` call in `src/lackpy/interpreters/literate/` runs document code outside `CellSession`. A test checks this, with an allowlist of parse-only `compile` calls.
