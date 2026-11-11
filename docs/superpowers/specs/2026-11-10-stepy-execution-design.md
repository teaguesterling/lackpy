# Stepy execution: the `stepy` command, stop on failure, and the intent block

Date: 2026-11-10. Status: draft from the brainstorm with Teague. Each decision says whose it is. "Teague" means he decided it in the brainstorm. "Claude's call" means Claude decided it and Teague did not object, or has not yet seen it.

## Intent

**The goal (Teague).** "I want agents to write stepy scripts that are reusable and clean." A script is run again by the same agent later, by other agents, by Teague by hand, and by hooks and CI (Teague: "All").

**What is missing today.**

- There is no command that runs a stepy program the way `python` runs a script. `lackpy` (`src/lackpy/cli.py`) treats `-c` as a natural-language intent and passes parameters as `--param key=value` text. `lackpyctl render --execute` (`src/lackpy/ctl.py:11`) runs a literate document but prints the interpreter's whole output, not only what the cells print.
- A caller cannot learn a script's interface. The `# /// intent` block is designed (`2026-10-18-lackpy-ste-design.md` §4.1) but nothing parses it. It exists only as prompt text (`src/lackpy/infer/prompt.py:23`).
- A literate document runs past a failed cell. The binding layer (`kernel/forgiveness.py`) turns the failure into a value, and later cells still run and print. A reader must then work out which output is still valid.

**The principle behind the failure rules (Teague: "think through this from asdste100 first principles").** A literate document is a procedure, and a cell is a step. ASD-STE100 procedures give four rules that apply here:

1. Do the steps in the written sequence. A step's precondition is that the steps before it succeeded.
2. A condition comes before its action and is never implied. "Run this cell if it does not depend on the failed one" is an implied condition.
3. The reader must not have to infer what is valid.
4. After a fault, correct it and do the failed step again. The resume path already does this.

## Glossary

One term per concept, for this whole document.

- **program**: a `.py` stepy file, or the text given to `-c`.
- **document**: a literate `.md` file.
- **script**: a program or a document.
- **cell**: one ```` ```lackpy ```` code block in a document.
- **failure**: a cell or program that raises an exception, times out, references a name that is not bound, or is stopped by the effect ceiling gate. A refusal is not a failure (see E1).
- **refusal**: a `LackpyRefusal`. It ends the document and closes the session, as today (literate stepy D4).
- **intent block**: the `# /// intent` TOML comment block of §4.1.
- **declared writes**: the `writes` globs in the intent block.
- **resume**: a later submit in the same session that starts at a cell after the first, from that cell's checkpoint (`LiterateRun._submit`, `src/lackpy/run/sessions.py:170`).
- **re-run**: running a cell a second time in the same session, whether its first run succeeded or failed.

## Goals

1. `stepy` runs a script like `python` runs a file: a shebang, stdout, stderr and an exit code.
2. Every way a document runs stops at the first failure. No later cell runs or prints.
3. A session can resume at the failed cell. Each cell can forbid its own re-run.
4. A script states its interface in its intent block. `stepy --help` shows it.
5. Under `stepy`, a script can write only what its declared writes allow.

## Non-goals

- The §4.2 derivation of intent from the AST, and the comparison of declared against derived fields. These stay in project 2.
- Enforcing `reads`. v1 does not enforce reads (effect-model decision, 2026-10-25).
- Stdin as input to a script (Teague, agreed 2026-11-10). Stepy has no `sys`. A script that needs input takes a path parameter.
- A REPL, `-i`, or `-m`.
- Changing what `lackpy` or the MCP tools allow a program to write. They keep today's behaviour until project 2.
- The time limit for literate_agent's first document and its `@continue` rounds (the open gate 4 item). It is a separate item.
- Packs (project 3). Windows.

## Anti-goals

- Output after a failure that a reader could take as valid.
- A write outside the declared writes that succeeds under `stepy`.
- A session that runs a cell again after that cell forbade its re-run.
- `--help` that shows `reads` or `writes` as verified. They are declared. Only `writes` is enforced, and only under `stepy`.
- A recovery mechanism in the spec. What to do after a failure belongs to the document's own prose (Teague: "I don't want to over specify in the spec itself").

## Decisions: failure in a document

### E1. Stop at the first failure, everywhere (Teague)

A run stops at the first failure. This applies to every way a document runs: `stepy`, `lackpyctl render --execute`, the MCP session tools, and literate_agent. It is not only a CLI rule.

- Later cells do not run and print nothing.
- The run reports failure. Its error names the cell (its index and its first line) and the cause.
- A refusal is unchanged: it ends the document and closes the session. A failure leaves the session open for a resume. A refusal does not.
- A prose expression that raises is a failure at its place in the document, and the document stops there (Claude's call).
- **literate_agent** (Claude's call). Its first document and its `@continue` rounds run in the in-process fold (`LiterateSession.step`, `interpreters/literate/session.py:427`), which calls the same `_run_document`. So E1 applies there with no extra work: the round stops at its first failure and returns Left. literate_agent already answers a Left with a full resubmission through its `LiterateRun` (`scripts/literate_agent.py:236`), and that path has the checkpoints of E2. The fold keeps no checkpoints, so E2 and E4 apply only through `LiterateRun`. Moving the fold onto `LiterateRun` is the open gate 4 item, not this project. The fold's docstring that says "a corrected round simply rebinds the holes" (`session.py:20`) must be corrected.

This reverses the binding-layer decision ("design conflict #2, option (c)" in `kernel/forgiveness.py:4`), which said execution continues and nothing rolls back. It also reverses literate stepy D5 for unbound names (see "Decided: unbound names" below).

### E2. The state after a failure is the state before the failed cell (Claude's call)

The checkpoint fork already holds the state from before each cell (`_Hook`, `src/lackpy/run/session_worker.py:100`). After a failure, the session keeps that checkpoint. The failed cell's in-process effects are gone with its child process. Its effects outside the process (a file it wrote before it failed) are not undone. E4 exists for that case.

The failed cell joins `channel.stopped`, as a timed-out cell does today. `_submit`'s "ruling A" (`sessions.py:178`) then makes the next resume start no later than the failed cell. Timeouts already work this way. E2 extends the rule to every failure.

### E3. Retire the binding layer and the reactive re-run (Teague, via E1)

With E1 nothing after a failure has run, so nothing is left to heal. The resume of E2 replaces the reactive re-run.

Remove from the batch/session path (`LiterateInterpreter._run_document`, `src/lackpy/interpreters/literate/__init__.py:142`):

- binding `ErrorValue`, timeout holes and `Unavailable` (`_reify_error_for_cell`, `_reify_timeout_for_cell`, `_reify_unavailable_for_cell`);
- the chained-hole pre-pass (`_reify_holes_for_cell`, `__init__.py:1062`) (Teague: "Retire holes");
- `_reexecute_dirty_subgraph` (`__init__.py:592`) and its call at `__init__.py:389`;
- `rerun_cell` in `src/lackpy/run/cells.py:204`, if nothing else uses it.

The ledger and versions stay. A failure is still a ledger entry. The streaming driver (`kernel/driver.py`) has no production caller and already stops on an unrecovered failure. It is not changed.

**Test order (golden rule).** Many tests pin the old behaviour: `tests/literate/test_dirty_subgraph.py` (23 tests), `test_l1_conformance.py` (24), and parts of `test_annotations_path.py`, `test_checkpoint.py`, `test_selftest.py` and `tests/run/test_cells.py`. Change the expectations first, confirm that each one fails for the predicted reason, and only then change the interpreter. A test that pinned only a heal is deleted, and the plan lists each deletion.

### E4. Each cell can forbid its re-run (Teague; the syntax is Claude's call)

The fence info string already parses `key=value` options into `Cell.options` (`_ANNOTATION_RE`, `parser.py:33`). The new option is `rerun`:

````markdown
```lackpy rerun=no
Path(params["out"]).write_text(report)
```
````

- `rerun=yes` is the default (Teague: "allow by default"). A resume may run the cell again.
- With `rerun=no`, once the cell has run in a session, whether it succeeded or failed, no later submit in that session may run it again (Teague, 2026-11-10: "Any second run"). Such a submit fails before it runs anything. The error names the cell and says that its re-run is forbidden.
- The case this covers: a cell that writes a file succeeds, a later cell fails, and the next submit changes a cell before the write cell. Without `rerun=no`, the resume would write the file a second time.
- A new session is a new run. `rerun=no` does not reach across sessions (Claude's call).
- `stepy` has no resume, so `rerun` changes nothing there.
- Any other value of `rerun` is a validation error.
- What to do next is not defined by lackpy. The document's prose says it (Teague).

## Decisions: the intent block

### E5. Parse the block of §4.1 (Teague: "bring it forward")

The block's form and fields are §4.1 of `2026-10-18-lackpy-ste-design.md`: `goal`, `reads`, `writes`, `checks`, `output`, and `[params]`. Section 4.1 is the authority. This section adds only what §4.1 leaves open.

- **Where it goes.** In a program, at the top, before any code. In a document, in the first code cell (Claude's call from 2026-10-25). The YAML front matter stays for rendering options.
- **One block at most.** A second block is a validation error.
- **Unknown keys** are a validation error (Claude's call: fail loud, so a typo does not become an empty field).
- **TOML.** lackpy needs Python 3.11 and has `tomllib`. lackpy-lang supports Python 3.10 (`packages/lackpy-lang/pyproject.toml:15`). Where the parser lives, and how 3.10 is served, is the builder's call, with one limit: no new runtime dependency without Teague.
- **A program with no block** is valid. It has no description and no declared writes.

### E6. Check `[params]` against the `params` literal (Teague, from §4.1)

The §4.1 rules apply as written. The keys of `[params]` and of the `params` literal are the same set, checked in both directions. Each default matches its declared type. The v1 types are `str`, `int`, `float`, `bool` and `list[str]`. The existing key and type checks are in `_apply_params` (`src/lackpy/run/runner.py:293`) and `_apply_document_params` (`__init__.py:1535`).

### E7. Enforce the declared writes under `stepy` (Teague)

Under `stepy`, the Path stand-in allows a write only when the path matches a declared writes glob. This is on top of the Claude-settings policy, never instead of it.

- **Where.** A narrowing layer on the PathPolicy that `LackpyService._load_path_policy` (`src/lackpy/service.py:789`) builds. `read_only_policy` (`run/paths/rules.py:377`) is the precedent. "Allow only these globs" is not a deny rule, so the layer may need a new form. That is the builder's call.
- **Parameters in globs.** A `{name}` pattern is filled in with the parameter's value at the start of the run.
- **No block, or no `writes` key: no writes** (Claude's proposal, approved by Teague). An agent that wants a write must declare it.
- **A refused write** is a refusal, as the policy's refusals are today. It names the path and the declared globs.
- **For a document,** the declared writes of the first cell's block cover the whole session.
- **Which operations count** (Claude's call). Every path the policy checks as a write (`PathPolicy.check(path, "write")`) must also match a declared glob. A rename checks both ends. `mkdir` is allowed for a directory that is an ancestor of a path some declared glob can match, so `writes = ["out/*.txt"]` allows `mkdir("out", parents=True)`.
- **Glob form** (Claude's call). The existing `PathRule` pattern semantics, relative to the workspace.
- **Only `stepy`.** `lackpy script.py`, `lackpyctl` and the MCP tools are unchanged. The cost: for a while, one program can write under one command and be refused under another.

## Decisions: the `stepy` command

### E8. One command; the extension chooses how it runs (Teague)

- `stepy foo.py [options]` runs a program. `#!/usr/bin/env stepy` works, so `./foo.py --root src` also runs it.
- `stepy doc.md [options]` runs a document.
- Any other extension exits 2 with a message that names `.py` and `.md`.
- `stepy -c 'program text' [options]` runs program text, as `python -c` does. Stepy generates nothing, so `-c` keeps its python meaning. This gives `-c` two meanings across the two commands. Teague accepted that cost.
- The entry point is `stepy = "lackpy.stepy:main"` in `[project.scripts]` (the module name is Claude's call).

### E9. Parameters are named options only (Teague)

- `stepy count_todos.py --root src --limit 5`. Each option is a `[params]` key, and its value is converted to the declared type.
- A `bool` takes `--name` or `--no-name`. A `list[str]` takes the option once per item.
- No positional arguments, and no `--param`. Under the one-way rule, each parameter has one way to be set.
- A script with no `[params]` takes no parameter options.
- An unknown option, a missing value, or a value that does not convert exits 2 before anything runs.

### E10. The other options (Teague)

- `--help` prints the interface from the intent block: `goal`, each parameter with its type, default and `doc`, `writes` as "declared, enforced", `reads` as "declared, not checked", and `checks`. With no block, it says that the script declares nothing and may write nothing.
- `--check` validates and runs nothing. It covers syntax, vocabulary, the intent block, and the `[params]` match. It exits 0 or 2.
- `--timeout SECONDS` sets the time limit, capped by the profile's `max_timeout` (`profiles.py:81`), as in Plan 3. For a document it is the limit for each cell, as the literate axis defines it today (Claude's call). `--help` says so. A caller that needs a limit on the whole run uses `timeout(1)`.
- `--profile NAME` and `--workspace DIR` work as they do in `lackpy` (Claude's call). The workspace defaults to the current directory, as `python` does.

### E11. Output and exit codes (Teague; the codes are Claude's guess, accepted)

- stdout carries only what the program or the cells print (Teague: "only what the cells print is the right answer"). It is the captured print output (`run/cells.py:351`), not the interpreter's rendered output: today the session worker sends `str(result.output)` (`run/session_worker.py:317`). Prose and its expressions never reach stdout. A program's return value is not printed.
- What the failed cell printed before it failed is printed, then the error goes to stderr, as `python` does with a traceback (Claude's call). The anti-goal is output from cells after the failure. The order of the output is required. Live streaming is not; the builder may print each cell's output when the cell ends.
- stderr carries the error: the failed cell or line, and the cause.
- Exit codes:
  - 0: the run succeeded;
  - 1: the program, or a cell, failed or was refused;
  - 2: a usage or validation error, and nothing ran;
  - 124: a timeout, as `timeout(1)` uses.

## Decided: unbound names and the `_BIND_THROUGH_HOLES` wording (Teague, 2026-11-10: "Retire holes")

Today an unbound name binds a typed `Hole`, the cell is skipped, and a later binding heals it (literate stepy D5). The model is told so in `_BIND_THROUGH_HOLES` (`src/lackpy/interpreters/literate/prompt.py:54`): "you may reference results before you define them". That wording was parked by Teague.

E1 wins. An unbound name is a failure, and the document stops at that cell. The fix is to add the binding cell and resume. The reason: a forward reference is an implied condition (STE rule 2 in Intent).

- Holes are retired from the batch/session path, with the rest of E3.
- The `_BIND_THROUGH_HOLES` clause is replaced. Its new text says that a cell must come after the cells whose names it reads, and that a failure stops the document. **The new text needs Teague's approval before it is committed.** The plan drafts it and stops for that approval.

## Left to the builder

- Where the intent parser lives, and how lackpy-lang on Python 3.10 handles a block (within E5's limit).
- The form of the writes-narrowing layer (E7).
- How `_run_document` reports the failed cell to `channel.stopped` (E2).
- The error text, within the house rules: name the cell, the cause, and the remedy when one exists.
