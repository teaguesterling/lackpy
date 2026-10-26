# Time limits

Every lackpy program and every literate cell has a time limit. lackpy enforces
the limit by killing a process. A program cannot catch the stop, delay it or turn
it off. This page also covers retries that resume, and the stdlib members added
to the core vocabulary in the same change.

Linux only. The limit uses `fork`, `prctl` and `setrlimit`, and the memory budget
reads `/proc`. The limit does not exist on Windows.

## Terms

| Term | Meaning |
| --- | --- |
| time limit | The number of seconds that a run, or one cell, may take. |
| timeout | The outcome when lackpy stops a run at its time limit. The result has `error_phase="timeout"`. |
| cap | The most seconds that a profile allows. It is the `max_timeout` of the profile. |
| session | A literate run that stays alive for retries. It has a `session_id`. |
| checkpoint | A forked process that holds the state of a session before a segment. |

## Why a program needs a limit

The language forbids `while` and recursion. Some programs still run for a very
long time, or forever. A regex such as `(a+)+$` takes seconds on a short input
and much more on a longer one. A loop over `iter(int, 1)` never ends. A read from
a named pipe blocks. Whether a program ends cannot be decided from its text, so
lackpy enforces the limit while the program runs.

A thread cannot stop a regex that runs in C. So lackpy runs each program in a
separate worker process and kills the process at the limit.

## Set the limit

The `timeout` parameter is a number of seconds. These routes take it:

- `run_program` and `delegate`, in the library and over MCP;
- each literate submission;
- `lackpyctl render --execute --timeout SECONDS`.

The default is the smaller of two values: `[sandbox] timeout_seconds` (120 s)
and the `max_timeout` of the profile. The profile sets its cap in
`.lackpy/config.toml`:

```toml
[profiles.small]
max_timeout = 5
```

A request that is not acceptable is refused before the run starts. The result
has `error_phase="refused"`. These refusals were measured on 2026-10-25 (run
`python:176`):

```text
--- timeout=30: success=False error_phase=refused
    timeout 30 s is over the profile's max_timeout (5 s, profile small)
--- timeout=0: success=False error_phase=refused
    timeout must be a positive number of seconds, not 0
--- timeout=nan: success=False error_phase=refused
    timeout must be a positive number of seconds, not nan
```

An explicit `timeout` must be a finite number above 0. A `max_timeout` in the
profile must also be a positive number of seconds. Otherwise lackpy raises a
profile error.

In a literate document, the limit applies to each cell, not to the whole
document.

The limit is not part of the program text. The python3 twin of a program has no
limit.

## What a timeout looks like

A program that takes too long gets a timeout. This run used
`re.search(r"(a+)+$", "a" * 30 + "b")` with `timeout=2`. It was measured on
2026-10-25 (run `python:176`):

```text
--- regex, timeout=2: success=False error_phase=timeout
    timeout: the run took longer than 2 s and was stopped
```

`error_phase` has three values that matter here:

| `error_phase` | Meaning | Retry? |
| --- | --- | --- |
| `"timeout"` | The run took longer than its limit and was stopped. | Yes. Rewrite the program, or ask for a longer limit. |
| `"refused"` | The path policy or a parameter check refused the run. | No. The same input is refused again. |
| `null` | The run ended, or failed in an ordinary way. | Depends on the error. |

A timeout is not a refusal. A refusal latches a literate session. A timeout does
not.

The caller retries. `delegate` does not retry for the caller: it returns the
timeout result and the caller decides.

The stop takes effect at the limit. A cell can overrun the limit by up to 0.1 s
(`DEADLINE_GRACE`) before the result arrives.

## Other limits in the worker

The worker process also has these limits. A program cannot change them:

- A memory cap. The default is `[sandbox] memory_mb`, 512 MiB. A run that passes
  it ends with an error that starts `memory:`.
- A CPU-time limit, set above the wall-clock limit as a backstop. In a literate
  session, it comes from the profile cap.
- `PR_SET_NO_NEW_PRIVS`.

lackpy does not set `RLIMIT_NPROC`. It stops `fork` in the worker, and the
checkpoints need `fork`. The language stops a program from creating a process.

## Literate documents

A timeout in a literate document stops one cell. The document goes on, and the
names that the cell would bind become holes. See
[Time limits and retries](literate.md#time-limits-and-retries) for a measured
example, the rule that decides where a retry resumes, and the `session_id`
rules.

A short summary of the rules:

- The document runs in a session worker. The worker takes one checkpoint for each
  segment. A segment ends at any cell that can run code: a code cell, or a prose
  cell with an interpolation.
- A retry resumes at `k`, the smaller of the first changed cell and the first
  cell stopped in the last run. The cells before `k` do not run again.
- A longer `timeout` keeps the session. The session limit on CPU time comes from
  the profile cap.
- Over MCP, `run_program` returns `session_id` for a literate profile. Pass it
  back with the corrected document. Call `close_session(session_id)` when done.
- A session ends at `close_session`, after 600 s without a call (lackpy reaps it
  on the next registry call), or by eviction. The defaults are 8 sessions, 32
  checkpoints for each session, and 1024 MiB (Pss) of checkpoints for each
  session.
- These runs are not resumed, so they run fresh: an append-only resubmission
  after a run that completed all its cells, and a resubmission that pauses at
  `@continue` in `literate_agent`. Rounds of `@continue` have no time limit.

### What the timeout note says about writes

The `[kernel]` note names a file that the stopped cell wrote before the stop. The
worker streams each trace entry as it happens, so the parent knows what the cell
did even though the cell's own process is gone. A write that had started but was
not confirmed is named in the note. It is not in `trace.files_modified`.

## Direct library use has no limit

`RestrictedRunner` and `CellSession` are the in-process engines. A caller who
uses them directly runs the program in the caller's process, with no limit. The
caller chooses this route, and the caller is host code. A program cannot choose
it.

Every service route goes through the process runner: `run_program`, `delegate`,
the MCP tools, `lackpyctl render --execute` and `literate_agent`.

## The core vocabulary

The same change adds members to the core dictionary, and fixes four places that
broke the rule of one way. The generated [style guide](../reference/style-guide.md)
lists every approved member.

New modules and members:

| Job | Members |
| --- | --- |
| Read config | `tomllib.loads` |
| Read tables | `csv.reader`, `csv.DictReader`, over `text.splitlines()` |
| Read XML | `xml.etree.ElementTree.fromstring` (write `import xml.etree.ElementTree as ET`) |
| File facts | `Path.stat()`, with `.st_size` and `.st_mtime`; `Path.with_name`, `.match`, `.as_posix`, `.is_symlink` |
| Text | `textwrap.dedent`, `indent`, `fill`; `html.unescape`, `escape`; `unicodedata.normalize` |
| Compare text | `difflib.unified_diff`, `get_close_matches` |
| Encode | `base64.b64encode`, `b64decode` |
| Shape data | `itertools.groupby`, `chain`, `islice`, `pairwise`, `batched`, `product`, `combinations`, `accumulate` |
| Arithmetic | `math.floor`, `ceil`, `log`, `log10`, `isclose`, `isfinite`, `prod`, `gcd`, `fsum`, `inf`, `nan` |

Four rules apply to the new members:

- `csv.reader` and `ElementTree.fromstring` take text. Read the file with `Path`
  first.
- The infinite iterators `itertools.count`, `cycle` and `repeat` are not
  approved.
- `tomllib` needs Python 3.11, and `itertools.batched` needs Python 3.12. On an
  older Python, the validator reports `<module>.<member> needs Python <version>`.
- Each type that a new member returns passed the attribute-surface test.

### The four one-way changes

The one-way rule gives each operation one approved way. Four decisions apply it.
The hints below were measured on 2026-10-25 (run `python:178`):

| Operation | Decision | Hint |
| --- | --- | --- |
| Regex flags | Both names stay approved: `re.S` and `re.DOTALL`. | none |
| Regex calls | Both forms stay approved: `re.search(p, s)` and `re.compile(p).search(s)`. | none |
| UTC | `datetime.UTC` only. `datetime.timezone` stays for other offsets. | `use datetime.UTC` |
| Ordered dict | `collections.OrderedDict` is removed. | `use dict; a dict keeps insertion order` |

The two regex decisions are deliberate exceptions to the one-way rule.
`timezone.utc` and `OrderedDict` are now unapproved words with these hints. The
infinite iterators have a hint of the same kind: `not approved: an infinite
iterator; use range(n) or a list`.

The validator output for `datetime.timezone.utc`:

```text
--- datetime.timezone.utc: valid=False
    Unknown module member: 'timezone.utc' at line 2 (not in the dictionary)
    --- Suggestions ---
    For timezone.utc: use datetime.UTC.
```
