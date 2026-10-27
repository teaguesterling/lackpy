# Stepy Plan 3, project 1: the time limit and the core vocabulary

Date: 2026-10-25. Status: draft from the Plan 3 brainstorm with Teague. Each decision says whose it is. "Teague" means he decided it in the brainstorm. "Claude's call" means he did not object, but he did not decide it either.

Plan 3 is three projects, built in this order:

1. this document: the time limit, and the additions to the core vocabulary;
2. the intent block and the effect derivation (a separate design);
3. packs, the extension mechanism (a separate design).

The decisions for projects 2 and 3 are recorded in Claude's memory file `lackpy-plan3-vocabulary-decisions.md`. They are not part of this document.

## Intent

**The problem.** A lackpy program has no time limit today. `config.sandbox_timeout` (120 s) appears in a report only (`service.py:1328`). `RestrictedRunner.run` calls `exec` inside the caller's process. So a runaway program freezes the session that called it, and with it the MCP server.

**The programs that hang.** The language forbids `while` and recursion, and these programs still pass validation (measured at 929480e):

- `for x in xs: xs.append(x)`, a loop over a list that the loop extends;
- `for i in iter(int, 1): ...`, an iterator that never reaches its sentinel;
- `re.search(r"(a+)+$", "a" * 26 + "b")`, which takes 7.8 s, and each 2 more characters cost about 4 times more;
- `collections.deque(iter(int, 1))`, which grows until memory runs out;
- `Path("fifo").read_text()` on a named pipe in the workspace, or any read while an NFS mount stalls.

Whether a program finishes cannot be decided from its text in general. So the limit has to be enforced while the program runs.

**Why it comes first.** Teague: the time limit is Plan 3's first task (2026-10-18), for two reasons: a runaway program must not freeze its caller, and a program must not hang on an outside service. Later work depends on it. The ban on recursion exists only because there was no limit. Literate documents get the limit when programs do. Pack calls, in project 3, are new places where a program can spend time.

**The vocabulary.** The same project adds the stdlib members that Teague approved for the core dictionary. It also fixes four places where the dictionary breaks the "one way" rule. Both changes are small, they touch the same files, and they need the same surface tests.

## Goals

1. Every program run and every literate cell has a wall-clock limit. lackpy enforces it by killing a process. A program cannot catch it, delay it or turn it off.
2. A timeout is its own outcome, `error_phase="timeout"`. It is not a refusal. The caller can retry: the model rewrites the program or cell and submits it again. The caller may also ask for a longer limit, up to the profile's maximum (Teague).
3. In a literate document, a timed-out cell leaves the document's state as it was before that cell. A retry starts from that state. No earlier cell runs a second time (Teague: "Timeout allows attempting again").
4. The core dictionary gains the members listed in §Vocabulary, and the one-way changes listed there are in effect.
5. Each new member's returned objects have passed the attribute-surface test that the spec requires (design §5.1).

## Non-goals

- Packs, the intent block and the effect derivation. These are projects 2 and 3.
- Host facts and subprocess. Teague deferred subprocess. Host facts will probably become a pack.
- Windows. Fork does not exist there. lackpy runs on Linux for Teague (accepted by Teague).
- Lifting the ban on recursion. The limit makes it possible. Whether to lift the ban is a separate decision for later.
- Filesystem confinement at the OS level (Landlock). This is a later option.

## Anti-goals

- A limit that a program can stop. A watchdog thread is not a limit (see D1).
- A result channel that runs code in the parent. If the child is ever compromised, the channel must not let it run code in the parent (see D3).
- A retry that runs an earlier cell's writes again.

## Decisions: the time limit

### D1. Kill a process; do not interrupt a thread (measured)
In a thread, a Python-level interrupt cannot stop a regex that runs in C, or a read that is blocked in the OS. Measured on longbottom: a thread ran the regex above, and `PyThreadState_SetAsyncExc` was sent at 0.2 s. The regex still ran its full 7.5 s. Only killing a process enforces the limit.

### D2. A zygote, and one forked child for each run (Teague accepted; the details are Claude's call)
- lackpy starts a zygote: a fresh `python -I` process. It imports the runner, the namespace and the policy code, and then waits. It never starts a thread.
- The zygote is started as a new process, not forked from the service. The service may already have threads (the MCP bridge thread), and a fork after threads have started is unsafe.
- For each run, the zygote forks a child. The child runs the program and reports back. Measured on longbottom: a fork from a process that has already imported the runner costs about 1.7 ms. A fresh interpreter that imports the runner costs about 100 ms.
- The zygote's environment is scrubbed, and it holds no secrets and no open sockets. Every child inherits whatever the zygote has.
- In the child, the program runs with the workspace as its working directory. The child sets it with `os.chdir`, so the service no longer changes its own cwd. Then `_exec_lock` no longer has to serialize runs (the builder decides whether to remove it).

### D3. The result channel carries JSON only (Claude's call; a hard rule)
The child sends stdout, the value, the trace entries and the error as JSON over a pipe. The parent never unpickles anything from the child. A test checks that the channel code does not import `pickle`, `marshal` or `shelve`, in the style of gate 5.

### D4. The trace streams as the program runs (Claude's call)
The child sends each trace entry when it happens, not at the end. If the child is killed, the parent still knows what the program read and wrote before the limit. The timeout result says so, for example: "this cell wrote `out.json` before it was stopped". The killed child's own copy of the trace is lost, so without streaming this would be unknown.

### D5. OS limits in the child (Claude's call)
Before the program runs, the child sets:
- `RLIMIT_AS`, a memory cap. This stops the `deque(iter(int, 1))` case. The default value is the builder's call. Measure a normal run first.
- `RLIMIT_CPU`, a backstop above the wall-clock limit.
- `PR_SET_NO_NEW_PRIVS`.

`RLIMIT_NPROC` is **not** set. The builder check measured that `os.fork()` fails with EAGAIN after `setrlimit(RLIMIT_NPROC, (0, 0))`, and an unprivileged process cannot raise the limit again. The checkpoint forks of D6 need a session process that can fork. Process creation stays blocked by the language: no `os`, no `subprocess`, no real modules. A seccomp filter is a later option (Claude's call, after the builder check; an earlier draft wrongly listed NPROC).

### D6. A literate document: one session process, and a checkpoint fork for each cell (Claude's call, from Teague's retry rule)
```
zygote ─fork→ S0 (session: empty namespace)
              S0 ─fork→ C1 runs cell 1 ─ok→ C1 becomes S1, S0 exits
                        S1 ─fork→ C2 runs cell 2 ─timeout→ kill C2
                        S1 is untouched: a retry of cell 2 forks from S1
```
- The process that holds the session never runs a cell itself. For each cell, it forks a child. A fork is a copy-on-write checkpoint of the whole namespace.
- If the cell does not time out, the child takes over as the session process. This includes a cell that raised an ordinary error. The kernel already handles errors with `ErrorValue` and forgiveness.
- If the cell times out, the parent kills the child. The session process still holds the state from before the cell.
- `CellSession` lives in the session process. The kernel calls `run_cell`, `rerun_cell`, `eval_expr`, `data` and `edit_text` over the pipe. Earlier rulings keep every reply JSON-safe: `eval_expr` returns data or repr text, and `data()` returns data.
- `@diff` is applied in the child, through that session's `Path` stand-in. Its patcher is host code that runs in the child (`edit_text` takes a callable today, and a callable cannot cross the pipe).
- The session process lives as long as the session, `literate_agent`'s rounds included. It is reaped when the session closes, or after it has been idle for a set time (the builder's call).
- A plain program is the one-cell case of the same mechanism.

### D7. A timeout is not a refusal (Teague)
- `error_phase="timeout"` sits next to `"refused"` in `CellOutcome` and `ExecutionResult`.
- A refusal still latches the session (literate stepy D4). A timeout does not.
- The caller retries, not lackpy (Teague: the model "rewrites the cell and resubmits"). `delegate` gets no new loop after the run: it returns the timeout result, and the caller decides. A refusal is still never worth a retry, and the result says which of the two it is.
- In a document, a timed-out cell is reported in the `[kernel]` channel. Later cells that read its names see a `Hole`, as they do after an error. The document continues.

### D8. The limit is a parameter, capped by the profile (Teague: the model "can add a parameter for longer timeout"; the cap is Claude's call)
- `run_program`, `delegate` and each literate submission take an optional `timeout` in seconds. The MCP tools and the CLI expose it.
- The default is `config.sandbox_timeout` (120 s).
- A profile may set `max_timeout`, the default being the same value. A request above the cap refuses before the run, and the message names the cap and the profile.
- In a literate document, the limit applies to each cell.
- The limit is not in the program text. It is not part of what the program means, and the python3 twin does not have it.

### D9. The process runner wraps the engines; it does not replace them (Claude's call, after the builder check)
- `RestrictedRunner` and `CellSession` stay as the in-process engines. The child runs them.
- A new process runner wraps them. Every service entry point goes through it: `run_program`, `delegate`, the MCP tools, `lackpyctl render --execute` and `literate_agent`. So no route that a model or a user takes runs without the limit.
- Direct library use of `RestrictedRunner` or `CellSession` (tests, `selftest`) still runs in-process, with no limit. The docs say so. This is not the anti-goal "a limit a program can stop": the program cannot choose the route. The caller chooses it, and the caller is host code.
- Static validation, caller rules included, runs in the parent, before the code is sent. Caller rules are Python callables (`validator.py:100`), and a callable cannot cross the pipe. Inside `CellSession`, the builder separates validation from execution, so the cumulative check can run in the parent.
- The `PathPolicy` crosses the pipe as data. `edit_text` patchers are named, not passed (D6).
- **Amendment (Teague, 2026-10-25): `literate_agent` is only partly limited.** In `scripts/literate_agent.py`, only a full-document resubmission runs in the session worker, with the limit. The first document and its `@continue` rounds still run in-process, with no limit, as they did before Plan 3. So gate 4 is only partly met through `literate_agent`: a first document whose cell 2 loops still hangs. Teague deferred the fix to a later project. The docs say so (`docs/concepts/time-limits.md`).
- **Amendment (as built):** static validation runs in the worker, not in the parent (Ruling D). Rules cross the pipe as names, and a callable rule is refused.

### D10. A retry resumes from the checkpoint, in literate_agent and over MCP (Teague: "1 and 2")
- **Resume on a matching prefix.** A resubmitted document resumes from the checkpoint before its first changed cell, when every cell before that point is textually unchanged since the last submission. This is a full-document resubmission, as Teague ruled in 297eb0c. The cells that are unchanged and already ran do not run again. If the prefix differs, the document runs in a fresh session, as 297eb0c says. The match compares cell source text, and the comparison is exact.
- **The checkpoints to keep.** To resume before any cell, the session must keep a process for each point it may resume from. The builder keeps the checkpoint before the last cell that did not succeed, and the latest one. Other cells resume by a fresh run. Memory is shared copy-on-write, so the cost is small. The builder measures it.
- **literate_agent.** The `LiterateSession` keeps its session process between rounds, and applies the prefix rule to each round.
- **Persistent sessions over MCP.** A literate `run_program` returns a `session_id`. A later call that passes the `session_id` with a resubmitted document applies the prefix rule to that session. A call without one starts fresh, as today. The service keeps a registry of live sessions. A session ends on an explicit close, after an idle time, or when the registry is full: the oldest idle session goes first. The limits are the builder's call, with measured defaults. An unknown or expired `session_id` gives a clear error, never a silent fresh run.
- **A session belongs to one workspace and one profile.** A call with the same `session_id` and a different profile or workspace refuses.

## Decisions: the core vocabulary

Teague approved these members for the core dictionary (`packages/lackpy-lang/src/lackpy/lang/dictionary.py`):

| Job | Members |
|---|---|
| Read config | `tomllib.loads` |
| Read tables | `csv.reader`, `csv.DictReader`, over `text.splitlines()` |
| Read XML | `xml.etree.ElementTree.fromstring` |
| File facts | `Path.stat()` with `.st_size` and `.st_mtime`; `Path.with_name`, `.match`, `.as_posix`, `.is_symlink` |
| Text | `textwrap.dedent`, `indent`, `fill`; `html.unescape`, `escape`; `unicodedata.normalize` |
| Compare text | `difflib.unified_diff`, `get_close_matches` |
| Encode | `base64.b64encode`, `b64decode` |
| Shape data | `itertools.groupby`, `chain`, `islice`, `pairwise`, `batched`, `product`, `combinations`, `accumulate` |
| Arithmetic | `math.floor`, `ceil`, `log`, `log10`, `isclose`, `isfinite`, `prod`, `gcd`, `fsum`, `inf`, `nan` |

Rules for the table:
- **No infinite iterators.** `itertools.count`, `cycle` and `repeat` stay out (Claude's call; Teague kept them out of the table).
- **Python versions.** `lackpy-lang` supports 3.10. `tomllib` needs 3.11, and `itertools.batched` needs 3.12. `ModuleProxy` already skips a member that the running Python does not have (`modules.py:65`). The builder decides how the validator and the error text treat such a member on an older Python.
- **The `Path` additions** are members of the stand-in, not of the real `Path`. `stat()` returns an object whose attributes must pass the surface test. `stat` is a read.
- **CSV and XML take text.** `csv.reader` takes an iterable of lines, and `ElementTree.fromstring` takes a string. The program reads the file through `Path` first. No member takes a path or a file object.

**One-way changes** (Teague, from the review page):
- Regex flags: keep both the short and the long names (`re.S` and `re.DOTALL`). This is Teague's exception to the one-way rule.
- Regex calls: keep both `re.search(p, s)` and `re.compile(p).search(s)`. This is another exception from Teague.
- UTC: `datetime.UTC` only. `datetime.timezone` stays for other offsets. `timezone.utc` becomes an unapproved word whose hint names `datetime.UTC` (Claude's call on how to keep `timezone`).
- `collections.OrderedDict` is removed. It becomes an unapproved word whose hint says "use dict; a dict keeps insertion order".

**The one-way rule** (Teague adopted Claude's proposal): each operation has one approved way, the most specific name for the intent that is also correct. Every other way becomes an `UNAPPROVED` entry with a hint. The two regex items above are Teague's exceptions.

## Constraints

- The safety tier of design §5.1 applies to every new member. The builder lists the public attributes of each type that a new member returns. Examples are a `csv` reader, an `Element`, a `stat_result`, the `difflib` generators, and the `itertools` objects and their groups. The builder then confirms that no route reaches a module, a frame, a file or a namespace. This is a test, not an assumption.
- `lackpy-lang` stays free of runtime dependencies. The zygote, the fork and the channel belong in `lackpy`, not in `lackpy-lang`.
- The plain-Python invariant holds. Every new member exists in the standard library, so the python3 twin still runs.
- Tests never read the real `~/.claude` or `/etc/claude-code`.

## Gates

1. The regex case above stops at the limit. The result has `error_phase="timeout"` and arrives within the limit plus 1 s.
2. A read of a FIFO in the workspace stops at the limit.
3. `collections.deque(iter(int, 1))` stops at the memory cap, and the error says so.
4. In a literate document, cell 2 times out. The document is resubmitted with cell 2 rewritten, both through `literate_agent` and through MCP with the `session_id`. Cell 2 sees cell 1's names. A file that cell 1 wrote is written once, not twice. A resubmission that changes cell 1 runs fresh.
5. A cell that writes a file and then loops forever: the timeout result names the written file (D4).
6. The channel code imports no `pickle`, `marshal` or `shelve` (D3).
7. A `timeout` above the profile's `max_timeout` refuses before the run (D8).
8. Each new core member passes the attribute-surface test.
9. The full suite and the 3.13 legs pass as they do at 929480e. On longbottom that is 2880 passed, 49 skipped and 77 xfailed.

## What is left to the builder

- The pipe protocol: framing, and the message types.
- How the service finds, starts and restarts the zygote. What happens when the zygote itself dies.
- The default memory cap and the idle time before a session process is reaped. Measure first.
- Whether `_exec_lock` goes once the cwd change moves into the child.
- How the CLI and MCP surfaces spell the `timeout` parameter.
