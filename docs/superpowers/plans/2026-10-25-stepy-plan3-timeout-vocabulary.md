# Stepy Plan 3, project 1: time limit and core vocabulary — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every program run and every literate cell that goes through a service route has a wall-clock limit that lackpy enforces by killing a process. A timeout can be retried, and a literate retry resumes from a checkpoint. The core dictionary gains the approved stdlib members and the one-way changes.

**Architecture:** A zygote (a fresh `python -I` process, with no threads) forks one worker for each run. A worker talks to the service over an AF_UNIX socket, in newline-delimited tagged JSON, and streams trace entries as they happen. A literate session worker forks a checkpoint at the top of each document cell. A killed cell leaves its checkpoint, which continues or resumes. The in-process engines (`RestrictedRunner`, `CellSession`, the literate kernel) do not change their behaviour. They run inside the worker.

**Tech Stack:** Python 3.10+ stdlib only (`os.fork`, `socket.send_fds`/`recv_fds`, `resource`, `ctypes` for `prctl`), pytest, blq.

**Spec:** `docs/superpowers/specs/2026-10-25-stepy-plan3-timeout-vocabulary-design.md` (commit 440214d; approved by Teague 2026-10-25). Read it before your task. The decision ids D1–D10 and the gate numbers 1–9 below are the spec's.

## Global Constraints

- Linux only for the process runner. On a platform without `os.fork`, the process runner raises `RuntimeError("the process runner needs os.fork (Linux)")` when it starts. It never falls back silently to in-process (spec non-goal: Windows).
- `lackpy-lang` gains no runtime dependency. The zygote, the worker, the wire and the limits live in `src/lackpy/run/`.
- The channel carries tagged JSON only. No module under `src/lackpy/run/` that touches the channel imports `pickle`, `marshal` or `shelve` (D3, gate 6).
- The default limit is `config.sandbox_timeout` (120 s). The memory cap is `config.sandbox_memory_mb` (512, already in `config.py:26`). Both are existing config keys.
- `error_phase` values: `"timeout"`, `"refused"`, or `None`. A timeout never latches a session. A refusal still does (literate stepy D4).
- Tests never read the real `~/.claude` or `/etc/claude-code`. Use the existing `_path_policy_loader` seam and the fixture policies in `tests/run/paths/`.
- Run builds and tests through blq (`mcp__plugin_blq_blq_mcp__run`, commands `test-all`, `test-lang`, `test-mcp`, `lint`). Do not run pytest through Bash. For a single test file, register no new command. Use `run(command="test-all", args=...)` if the tool accepts extra args, else `mcp__plugin_blq_blq_mcp__exec`.
- The suite at the base, measured on longbottom: 2880 passed, 49 skipped, 77 xfailed. Gate 9 compares against this.
- Change implementation or tests, never both at once in one step, unless the step says why.

## Review Focus

1. **Orphans.** The service process exits or is killed while a worker or session is running. Expected: no zygote, worker or checkpoint process outlives it. Set `PR_SET_PDEATHSIG(SIGKILL)` in the zygote and in each worker. Test in Task 4.
2. **Three ways to die, three messages.** A worker killed at the wall-clock limit, a worker that hit `RLIMIT_AS` (a `MemoryError`, or death), and a worker that crashed (a signal, or an exit without a result). Each gets its own text, and only the first gets `error_phase="timeout"`. Test in Task 4.
3. **Huge output.** A program that prints without end, for example `for i in iter(int, 1): print(i)`. Expected: the parent never holds more than the stdout cap (10 MB). The result says the output was cut. The limit still applies. Test in Task 4.
4. **Concurrent runs.** Two `run_program` calls at once, from MCP or the library. Expected: each runs in its own worker with the workspace as cwd, the results never cross, and the service's own cwd never changes. Test in Task 5.
5. **A write cut in the middle.** The limit falls while `Path.write_text` is running. Expected: the timeout result names the file as "write started, not confirmed". The stand-in streams a start entry before the write and a done entry after it. Test in Task 6.

---

### Task 1: Core vocabulary — modules, members and one-way changes

**Files:**
- Modify: `packages/lackpy-lang/src/lackpy/lang/dictionary.py` (`MODULES`, `UNAPPROVED`)
- Modify: `src/lackpy/infer/prompt.py:18-31` (`SKELETON`'s approved-imports line)
- Test: `tests/lang/test_dictionary_plan3.py` (new), `tests/run/test_import_proxies.py`

**Interfaces:**
- Produces: `dictionary.MODULES` entries for `tomllib`, `csv`, `xml.etree.ElementTree`, `textwrap`, `html`, `unicodedata`, `difflib`, `base64`, `itertools`, and the wider `math`. The import name is the dotted module name, so `import xml.etree.ElementTree as ET` and `from xml.etree.ElementTree import fromstring` both work. Check how `restricted_import` handles a dotted name (`modules.py:110`). If it does not handle one, extend it here.
- Produces: `dictionary.approved_imports_line() -> str`, the text `"pathlib (Path), json, re, ..."` built from `MODULES`, in `MODULES` order.

Exact members (spec §Decisions: the core vocabulary):
- `tomllib`: `loads`, `TOMLDecodeError`
- `csv`: `reader`, `DictReader`, `Error`
- `xml.etree.ElementTree`: `fromstring`, `ParseError`
- `textwrap`: `dedent`, `indent`, `fill`
- `html`: `unescape`, `escape`
- `unicodedata`: `normalize`
- `difflib`: `unified_diff`, `get_close_matches`
- `base64`: `b64encode`, `b64decode`
- `itertools`: `groupby`, `chain`, `islice`, `pairwise`, `batched`, `product`, `combinations`, `accumulate`
- `math`: add `floor`, `ceil`, `log`, `log10`, `isclose`, `isfinite`, `prod`, `gcd`, `fsum`, `inf`, `nan` to `comb`, `sqrt`, `exp`
- `collections`: remove `OrderedDict`
- `datetime`: keep `timezone`
- The exception classes (`TOMLDecodeError`, `csv.Error`, `ParseError`) are my addition, so that `except` works as it does for `JSONDecodeError`. Each one must be added to the approved exception classes the same way `JSONDecodeError` is. Find where Plan 1 did that.

`UNAPPROVED` additions (key → hint, exact text):
- `"collections.OrderedDict"` → `"use dict; a dict keeps insertion order"`
- `"timezone.utc"` → `"use datetime.UTC"`
- `"itertools.count"`, `"itertools.cycle"`, `"itertools.repeat"` → `"not approved: an infinite iterator; use range(n) or a list"`

- [ ] **Step 1: Write the failing tests** in `tests/lang/test_dictionary_plan3.py`:
  - `test_new_modules_validate`: for each new module, a one-line program that imports it and calls one member validates (`validate(src).valid is True`).
  - `test_ordered_dict_is_unapproved`: `validate("import collections\nd = collections.OrderedDict()\n")` is not valid, and the violation's hint is `"use dict; a dict keeps insertion order"`.
  - `test_timezone_utc_hint`: `datetime.timezone.utc` is not valid, with the hint `"use datetime.UTC"`. `datetime.timezone(datetime.timedelta(hours=-7))` is valid.
  - `test_infinite_itertools_refused`: `itertools.count` is not valid, with the infinite-iterator hint.
  - `test_regex_both_ways_still_valid`: `re.S`, `re.DOTALL`, `re.compile(p).search(s)` and `re.search(p, s)` all validate (Teague's exceptions).
  - `test_xml_entity_attacks_refused`: both checks run through `RestrictedRunner`, each in a `try`/`except ET.ParseError`.
    - A billion-laughs document (9 levels of 10× entity expansion) gives `ParseError` in under 2 s, with text containing `"amplification"`.
    - A document with `<!ENTITY x SYSTEM "file:///etc/hostname">` and `&x;` gives `ParseError` `"undefined entity"`. The file's content is never in the result.

    Both were measured on longbottom with expat 2.6.1 (0.21 s, and "undefined entity"). The test pins them so that a Python with an older expat fails loudly. Stdlib `ElementTree` stays, rather than `defusedxml`: that is third-party, and it would break the python3 twin.
  - In `tests/run/test_import_proxies.py`, `test_new_module_members_run`: run `ET.fromstring("<a><b/></a>").find("b").tag` through `RestrictedRunner` and get `"b"`. Run `list(itertools.batched([1, 2, 3], 2))` and get `[(1, 2), (3,)]`, skipped below Python 3.12.
- [ ] **Step 2: Run them and see them fail** (blq `test-lang`, and `test-all` for the runner test).
- [ ] **Step 3: Implement** the `MODULES` and `UNAPPROVED` changes. The `timezone.utc` hint needs a static rule on the attribute `utc` of a name bound to `datetime.timezone`. Use the hint mechanism the validator uses for `json.load` (`hint_for`). On a Python that lacks a member (`tomllib` below 3.11, `batched` below 3.12), the validator reports `safety.module-attr` with the text `"<module>.<member> needs Python <version>"`. Keep a small table for this in `dictionary.py`.
- [ ] **Step 4: Replace** the hand-written approved list in `SKELETON` with `approved_imports_line()`. Change the comment at `prompt.py:18-19`. The eval's copy in `scripts/eval/ste_usability/prompts.py` is a frozen, measured prompt: leave it, and say so in the comment.
- [ ] **Step 5: Run** blq `test-lang` and `test-all`. Expected: the new tests pass. Every test that pins the old skeleton text or the old `OrderedDict` member is updated in this step, and the report names each one.
- [ ] **Step 6: Commit** `feat(lang): Plan 3 core vocabulary and one-way changes`.

### Task 2: Path stand-in members and the surface test

**Files:**
- Modify: `packages/lackpy-lang/src/lackpy/lang/dictionary.py` (`PATH_MEMBERS`, `PATH_IO_METHODS`)
- Modify: `src/lackpy/run/paths/proxy.py` (the stand-in)
- Test: `tests/run/paths/test_proxy.py`, `tests/lang/test_return_surface.py` (new)

**Interfaces:**
- Produces: stand-in members `stat`, `with_name`, `match`, `as_posix`, `is_symlink`.
  - `stat()` is a read: the policy checks it as a read, and the trace records it as `Path.stat`.
  - `stat()` returns an object with exactly `st_size` and `st_mtime`. It is not the real `os.stat_result`: keep that surface to two names.
  - `is_symlink` is a read. `with_name`, `match` and `as_posix` are pure.
- Produces: `tests/lang/test_return_surface.py::SURFACES`, a table that maps each new member to the type it returns and the set of public attribute names that a program may reach on that type. The test computes the real public attributes of each returned type, and asserts each one is either denied by the static rules or harmless. Harmless means data, a pure method, or a method that returns another type in the table. This is spec §5.1 as a test.

- [ ] **Step 1: Write the failing tests:**
  - `test_stat_reads_through_policy`: under a policy that denies reading `secret.txt`, `Path("secret.txt").stat()` refuses. Under one that allows it, `st_size` is the file's length.
  - `test_stat_surface_is_two_names`: `[n for n in dir(Path("a.txt").stat()) if not n.startswith("_")] == ["st_mtime", "st_size"]`.
  - `test_pure_members`: `Path("a/b.txt").with_name("c.txt").as_posix() == "a/c.txt"`, and `Path("a/b.py").match("*.py") is True`.
  - `test_return_surface` (parametrized over `SURFACES`): a csv reader, a `DictReader` row, an `Element`, the `difflib.unified_diff` generator, `get_close_matches` (a list), the `itertools` objects including a `groupby` group, and the `stat` result. For each, every public attribute is classified. An unclassified attribute fails with its name.
- [ ] **Step 2: Run and see them fail.**
- [ ] **Step 3: Implement** the members in `proxy.py`. Add the five names to `PATH_MEMBERS`, and `stat` and `is_symlink` to `PATH_IO_METHODS`. Fill in `SURFACES` until the test passes. If a returned type exposes a route to a module, a frame, a file or a namespace, stop and report it. Do not add a deny entry and continue.
- [ ] **Step 4: Run** blq `test-all`. Expected: all pass.
- [ ] **Step 5: Commit** `feat(run): Path stat and pure members; return-surface test (spec §5.1)`.

### Task 3: The wire, trace streaming, the policy as data, and error_phase

**Files:**
- Create: `src/lackpy/run/wire.py`
- Modify: `src/lackpy/run/trace.py` (`Trace`, `record_call`, `TraceEntry`)
- Modify: `src/lackpy/run/paths/rules.py` (`PathPolicy`)
- Modify: `src/lackpy/run/base.py` (`ExecutionResult`)
- Test: `tests/run/test_wire.py` (new), `tests/run/test_trace.py`, `tests/run/paths/test_rules.py`

**Interfaces:**
- Produces, in `wire.py`:
  - `encode(obj: Any) -> str`: one line of JSON, no newline.
  - `decode(line: str) -> Any`.
  - `send(sock: socket.socket, msg: dict[str, Any]) -> None`: writes `encode(msg) + "\n"`.
  - `Reader(sock)` with `.recv(timeout: float | None) -> dict[str, Any] | None`. It returns None at EOF and raises `TimeoutError` at the timeout. It never buffers more than `max_line` bytes (default 16 MB): a longer line raises `WireError`.
  - `class WireError(Exception)`.
  - Tagging: `{"__t": "bytes", "v": <base64>}`, `"tuple"`, `"set"`, `"frozenset"`, and `"dict"` (a dict with a non-str key or a `"__t"` key, as `[[k, v], ...]`). `decode` raises `WireError` on an unknown tag. Nothing outside the data types of `runner._is_data` encodes. Anything else raises `WireError`, never a repr fallback.
- Produces, in `trace.py`:
  - `Trace.listener: Callable[[TraceEntry], None] | None`, a dataclass field with `default=None, repr=False, compare=False`. `record_call` calls it after it appends an entry.
  - `TraceEntry.to_data() -> dict` and `TraceEntry.from_data(d) -> TraceEntry`.
- Produces, in `rules.py`: `PathPolicy.to_data() -> dict` and `PathPolicy.from_data(d: dict) -> PathPolicy`. A round trip gives a policy whose `check` decides every path the same way. A read-only policy stays read-only.
- Produces, in `base.py`: `ExecutionResult.error_phase: str | None = None`. `RestrictedRunner.run` sets it to `"refused"` when its result comes from a `LackpyRefusal`.

- [ ] **Step 1: Write the failing tests:**
  - `test_round_trip`, parametrized over `None`, `True`, `3`, `2.5`, `"s"`, `b"\x00\xff"`, `(1, 2)`, `{1, 2}`, `frozenset({"a"})`, `{1: "a"}`, `{"__t": "x"}`, and nested mixes: `decode(encode(v)) == v`, with the type kept.
  - `test_unknown_tag_rejected`.
  - `test_non_data_rejected`: `encode(object())` and `encode(len)` raise `WireError`.
  - `test_reader_eof_timeout_and_cap`, using a `socket.socketpair()`.
  - `test_channel_imports_no_pickle` (gate 6): parse every module under `src/lackpy/run/` named `wire.py`, `zygote.py`, `process.py`, `limits.py`, `session_worker.py` or `sessions.py` that exists. Assert that none imports `pickle`, `marshal` or `shelve`, by AST, at any depth.
  - `test_listener_sees_each_entry` in `test_trace.py`.
  - `test_policy_round_trip` in `test_rules.py`, over the existing fixture policies, read-only included.
  - `test_refusal_sets_error_phase`, in `tests/run/test_runner.py`.
- [ ] **Step 2: Run and see them fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** blq `test-all`. Expected: all pass.
- [ ] **Step 5: Commit** `feat(run): tagged-JSON wire, streaming trace listener, PathPolicy as data, error_phase`.

### Task 4: Zygote, worker and ProcessRunner for plain programs

**Files:**
- Create: `src/lackpy/run/limits.py`, `src/lackpy/run/zygote.py`, `src/lackpy/run/process.py`
- Test: `tests/run/test_process_runner.py` (new)

**Interfaces:**
- Consumes: Task 3 (`wire`, `Trace.listener`, `PathPolicy.to_data/from_data`, `ExecutionResult.error_phase`).
- Produces, in `limits.py`: `apply_child_limits(*, memory_mb: int | None, cpu_seconds: int | None) -> None`.
  - It sets `RLIMIT_AS` (memory_mb MiB), `RLIMIT_CPU` (cpu_seconds) and `PR_SET_NO_NEW_PRIVS`, and `PR_SET_PDEATHSIG(SIGKILL)`. The two prctl calls go through `ctypes`.
  - It never sets `RLIMIT_NPROC` (spec D5, as corrected).
- Produces, in `zygote.py`: `main(argv: list[str]) -> int`, run as `python -I -m lackpy.run.zygote <control-fd>`.
  - The zygote imports the runner, the namespace and the policy code, then waits on the control socket.
  - A `{"op": "spawn"}` message arrives with one fd attached (`socket.recv_fds`). The zygote forks. The child calls `apply_child_limits`, serves exactly one run on that fd, and exits with `os._exit`.
  - The zygote reaps children and never starts a thread.
  - It scrubs `os.environ` to `PATH`, `HOME`, `LANG` and `TZ`, then drops each of those that is unset.
- Produces, in `process.py`:
  - `class ProcessRunner` with `__init__(self, *, memory_mb: int = 512, python: str = sys.executable)`.
  - `run(self, program: str, *, params: dict[str, Any] | None, path_policy: PathPolicy, workspace: Path, timeout: float) -> ExecutionResult`.
  - `close(self) -> None`.
  - The zygote starts lazily on the first `run`, over a `socket.socketpair()` passed with `pass_fds`. If the zygote is dead when a call needs it, the runner starts a new one. Each run sends `{"op": "spawn"}` with one end of a new socketpair.
- Worker protocol, one run. Parent to worker: `{"op": "run", "program", "params", "policy", "workspace"}`. Worker to parent:
  - `{"event": "start", "pid"}`;
  - `{"event": "trace", "entry"}`, one for each entry;
  - `{"event": "stdout", "text"}`, in chunks;
  - `{"event": "done", "result": {...}}`.
- The worker `os.chdir(workspace)`s before the run. `RestrictedRunner.run` does the run, unchanged.
- Result texts. These are exact, and later tasks and docs quote them:
  - timeout: `error_phase="timeout"`, `error=f"timeout: the run took longer than {timeout:g} s and was stopped"`.
  - memory: `error_phase=None`, `error=f"memory: the run used more than {memory_mb} MiB and was stopped"`.
  - crash: `error_phase=None`, `error=f"the run process ended without a result ({how})"`, where `how` is `"signal N"` or `"exit status N"`.
  - stdout over 10 MB: the result's `stdout` ends with `"\n[lackpy: output cut at 10 MB]"`, and the parent stops reading stdout chunks.

- [ ] **Step 1: Write the failing tests** (each with `timeout=2` unless it says otherwise):
  - `test_simple_run`: `x = 1 + 1` then `x` gives `output == 2`, `success is True`.
  - `test_regex_stops_at_limit` (gate 1): the spec's regex gives `error_phase == "timeout"`, within `timeout + 1` s of wall time.
  - `test_fifo_read_stops_at_limit` (gate 2): `os.mkfifo` in a tmp workspace, then the program `Path("p").read_text()`.
  - `test_memory_cap` (gate 3): `memory_mb=256`, the program `import collections` then `d = collections.deque(iter(int, 1))`. The error starts with `"memory:"`, and `error_phase is None`.
  - `test_trace_survives_kill` (gate 5, plain-program form): `Path("out.json").write_text("{}")` then `for i in iter(int, 1): pass`. The timeout result's `trace.files_modified` contains `out.json`.
  - `test_crash_message` (Review Focus 2): kill the worker with SIGSEGV from the test, using the pid from the `start` event, exposed for tests as `ProcessRunner.last_pid`. The error contains `"signal 11"`.
  - `test_stdout_cap` (Review Focus 3).
  - `test_no_orphans` (Review Focus 1): start a `ProcessRunner` in a subprocess, start a long run, and SIGKILL that subprocess. Within 2 s, no process whose parent chain leads to it is left. Check with `/proc`.
  - `test_zygote_restart`: kill the zygote, and the next `run` still succeeds.
  - `test_zygote_env_scrubbed`: set `LACKPY_TEST_SECRET` in the parent. Read the worker's environ with a test-only op `{"op": "environ"}`, served only when the zygote is started with `--test-ops`. The variable is absent.
- [ ] **Step 2: Run and see them fail.**
- [ ] **Step 3: Implement** `limits.py`, `zygote.py` and `process.py`. The parent loop reads events with `Reader.recv` against the deadline. At the deadline it sends SIGKILL to the worker's pid, then drains what the socket already holds.
- [ ] **Step 4: Run** blq `test-all`. Expected: the new tests pass and the base count holds.
- [ ] **Step 5: Commit** `feat(run): zygote and ProcessRunner, a time limit enforced by kill (D1-D5)`.

### Task 5: Service routing, the timeout parameter and max_timeout

**Files:**
- Modify: `src/lackpy/service.py` (`_execute`, `_run_sync`, `run_program`, `delegate`, `_exec_lock`)
- Modify: `src/lackpy/profiles.py` (`_PROFILE_KEYS`, `Profile`, `ResolvedProfile`, `resolve_profile`)
- Modify: `src/lackpy/mcp/server.py` (`run_program`, `delegate`), `src/lackpy/ctl.py`
- Test: `tests/test_service_timeout.py` (new), `tests/test_mcp_tools.py`

**Interfaces:**
- Consumes: Task 4's `ProcessRunner`.
- Produces:
  - `LackpyService.run_program(..., timeout: float | None = None)` and `delegate(..., timeout: float | None = None)`.
  - The MCP tools take `timeout: float | None = None` and return `"error_phase"` in their dict.
  - `lackpyctl run` and `render --execute` take `--timeout SECONDS`.
- Produces: a profile key `max_timeout: float | None`. With None, the cap is `config.sandbox_timeout`.
- Produces: the refusal text for a request over the cap: `f"timeout {t:g} s is over the profile's max_timeout ({cap:g} s, profile {name})"`. It is refused before validation, with `error_phase="refused"`.
- Ruling: the service owns one `ProcessRunner`, which it creates lazily. `_run_sync` no longer calls `os.chdir`: the worker does that. `_exec_lock` is removed if nothing else needs it. Say which in the report.
- Ruling: literate execution keeps its current in-process path in this task. Task 6 moves it.

- [ ] **Step 1: Write the failing tests:**
  - `test_run_program_timeout` (gate 1 through the service).
  - `test_timeout_over_cap_refuses` (gate 7).
  - `test_default_timeout_is_sandbox_timeout`: the config has `sandbox_timeout=1`, and a program that loops forever stops near 1 s.
  - `test_concurrent_runs` (Review Focus 4): two `run_program` calls run together under `asyncio.gather`. One writes `a.txt` and prints `"a"`, the other writes `b.txt` and prints `"b"`. Each result has its own stdout, and `os.getcwd()` in the test process is unchanged.
  - `test_mcp_timeout_param` in `test_mcp_tools.py`. This runs under blq `test-mcp`.
- [ ] **Step 2: Run and see them fail.**
- [ ] **Step 3: Implement.** `run_program` resolves the effective timeout as `timeout or sandbox_timeout`, checks it against the cap, validates in the parent (with caller rules, as today), then calls `ProcessRunner.run`. The run is synchronous and blocks, so call it with `asyncio.to_thread`. This is the parent, not the worker, so the zygote's no-threads rule is not affected.
- [ ] **Step 4: Run** blq `test-all` and `test-mcp`. Expected: all pass, at the base count plus the new tests.
- [ ] **Step 5: Commit** `feat(service): run_program and delegate run in a worker with a timeout (D8)`.

### Task 6: Literate session worker, with a checkpoint for each cell

**Files:**
- Create: `src/lackpy/run/session_worker.py`
- Modify: `src/lackpy/interpreters/literate/__init__.py` (`_run_document`'s cell loop at line 288; a new `_reify_timeout_for_cell`)
- Modify: `src/lackpy/run/paths/proxy.py` (a start entry before a write)
- Modify: `src/lackpy/service.py` (`_execute_literate`)
- Test: `tests/literate/test_checkpoint.py` (new)

**Interfaces:**
- Consumes: Tasks 3–5.
- Ruling, which departs from spec D6's wording, and why. The spec says the kernel calls `CellSession` over the pipe. Instead, **the whole literate interpreter runs in the session worker**. The checkpoint forks inside `_run_document`. The kernel's state (the ledger, versions, the dependency graph, rendered parts, holes) lives in the same coroutine as the namespace. A fork of the worker checkpoints both together. A pipe between the kernel and `CellSession` would leave the kernel's state behind on the host. A rollback to "before cell k" would then need that state rebuilt. This also keeps `@diff` and validation in the worker. Caller rules cross the pipe as names: the `rules: list[str]` that `resolve_rules` already accepts. A callable rule is refused by the process path, with `ValueError("the process runner takes rule names, not callables")`.
- Produces: a checkpoint hook. `context.config["checkpoint"]` is a callable `(index: int) -> Resume | None`, called at the top of each iteration of the cell loop (line 288). When there is no hook (in-process), nothing changes.
- Produces: `Resume`, a dataclass in `session_worker.py`. `kind` is `"timeout"` or `"resume"`. It also carries `document: str | None`.
  - `"timeout"`: the kernel reifies cell `index` as timed out and goes to the next cell.
  - `"resume"`: the kernel parses `document` and continues its loop at `index`, over the new document's cells.
- Produces: `_reify_timeout_for_cell(...)`. It binds a `Hole` for each name the cell would bind (from `_static_ref_bind_sets`). It writes a ledger entry `timed_out`, and a `[kernel]` note `f"timeout: cell {n} took longer than {t:g} s and was stopped"`. If the trace shows writes, the note adds `"; it wrote {paths} before it was stopped"`. A write that has a start entry but no done entry is listed as `"{path} (write started, not confirmed)"` (Review Focus 5). The document's `success` is False, and `error_phase` is `"timeout"`.
- The hook in the worker, at index i:
  1. It forks.
  2. The child sends `{"event": "cell_start", "index": i, "pid": <child pid>}` and returns None, so the loop runs the cell.
  3. The parent becomes the checkpoint for i. It blocks in `os.waitpid(child)`. When the child dies, it reads one instruction (`timeout` or `resume`) from the channel and returns the matching `Resume`.
  4. A child that finishes the document sends `done` and then waits for the next instruction. It does not exit, because it is the newest checkpoint. All processes share the one inherited channel fd. Only the deepest live process reads it.
- The parent side: `_execute_literate` runs the document in a session worker under the per-cell limit. Each `cell_start` re-arms the deadline. At a deadline, it SIGKILLs that cell's pid, then sends `{"op": "timeout"}`.
- Produces, in `process.py`:
  - `ProcessRunner.open_session(self, *, path_policy: PathPolicy, workspace: Path, cell_timeout: float, rule_names: list[str] | None, context_config: dict[str, Any]) -> SessionChannel`. It spawns a session worker through the zygote: `{"op": "spawn_session"}`.
  - `SessionChannel` has `send(msg)`, `events(deadline) -> Iterator[dict]` and `kill(pid)`, plus a `checkpoints` attribute: the list of live `(index, pid)`, kept from `cell_start` events.
  - Task 7's `LiterateRun` is built on `SessionChannel`.
  - In this task, `_execute_literate` opens a channel, submits once, and closes it.
- Produces: in `proxy.py`, `Path.write_text` and `write_bytes` record a `Path.write_text:start` entry before the write, then the normal entry after it.

- [ ] **Step 0: Spike, and keep its test.** `test_fork_inside_running_coroutine`: in a fresh process, a coroutine under `asyncio.run` calls `os.fork()` at its midpoint. The child finishes the coroutine and writes a marker. The parent waits for the child, then finishes its own copy. Assert that both markers are present and the values are right. If this fails, stop and report: the checkpoint design needs another shape.
- [ ] **Step 1: Write the failing tests:**
  - `test_cell_timeout_does_not_end_document`: cell 1 is `a = 1`, cell 2 loops forever, cell 3 is `b = a + 1`, and the prose is `{b}`. The run has `error_phase == "timeout"`. The render shows `2` for `{b}`, and cell 2's `[kernel]` note has the timeout text.
  - `test_timeout_names_written_file` (gate 5, literate form).
  - `test_write_started_not_confirmed` (Review Focus 5): a cell writes a 200 MB string with a limit too short for the write to finish. Measure the size so the write takes longer than the limit on longbottom. If that cannot be made reliable, use a test-only hook that blocks inside the stand-in's write.
  - `test_refusal_still_latches`: a refusal in cell 2 still ends the document (D4 unchanged).
  - `test_callable_rule_refused_by_process_path`.
- [ ] **Step 2: Run and see them fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** blq `test-all` and `test-mcp`. Expected: all pass, `tests/literate/test_gate.py` included.
- [ ] **Step 5: Commit** `feat(literate): session worker with a checkpoint fork per cell; a timed-out cell does not end the document (D6, D7)`.

### Task 7: Resume on a matching prefix — literate_agent and MCP sessions

**Files:**
- Create: `src/lackpy/run/sessions.py`
- Modify: `src/lackpy/service.py`, `src/lackpy/mcp/server.py`, `scripts/literate_agent.py`
- Test: `tests/literate/test_resume.py` (new), `tests/test_mcp_tools.py`

**Interfaces:**
- Consumes: Task 6's worker, hook and `Resume`.
- Produces, in `sessions.py`:
  - `class LiterateRun`, the parent's handle on one session worker. It has `submit(document: str, params: dict | None) -> ExecutionResult`, `close() -> None`, `session_id: str`, `workspace: Path`, `profile_name: str` and `last_used: float`.
  - `submit` applies the prefix rule. It compares the new document's cell sources (from the literate parser) with the last submission's, and finds the first index k where they differ. If there is a live checkpoint for k, it kills the process for cell k and every newer one, then sends `{"op": "resume", "document"}` to checkpoint k. Otherwise it closes the worker and runs the document fresh.
  - The first `submit` is always fresh.
- Produces: `class SessionRegistry` with `__init__(self, *, max_sessions: int = 8, idle_seconds: float = 600, max_checkpoints: int = 32)`. It has `open(...) -> LiterateRun`, `get(session_id: str, *, workspace: Path, profile_name: str) -> LiterateRun` and `close(session_id) -> None`. The defaults are starting values: measure the memory of a session with 32 checkpoints, and report the number.
  - `get` raises `SessionError(f"unknown or expired session_id {sid!r}")` for an id it doesn't hold.
  - It raises `SessionError(f"session {sid!r} belongs to profile {p!r} in {w}")` on a mismatch.
  - When the registry is full, `open` closes the session that has been idle longest.
  - Beyond `max_checkpoints`, the oldest checkpoint process exits. A resume before it runs fresh.
- Produces, in MCP:
  - `run_program(..., session_id: str | None = None)`. On a literate profile it returns `"session_id"`.
  - A new tool `close_session(session_id: str) -> dict`.
- Produces, in `literate_agent.py`: a full-document resubmission calls `LiterateRun.submit` on the agent's one `LiterateRun`, instead of opening a fresh session. Update the comment at lines 212–217: this replaces the "runs ONCE in a fresh session" text with the prefix rule.

- [ ] **Step 1: Write the failing tests** (gate 4):
  - `test_resume_after_timeout_skips_unchanged_prefix`, through `LiterateRun`: cell 1 appends a line to `log.txt` and binds `a`, and cell 2 loops forever. The resubmission has the same cell 1 and a fixed cell 2 that uses `a`. Afterwards `log.txt` has one line, not two, and cell 2's value is right.
  - `test_resume_through_mcp_session_id`: the same case through `run_program` with the returned `session_id`.
  - `test_changed_prefix_runs_fresh`: the resubmission changes cell 1, and `log.txt` gets its second line.
  - `test_unknown_session_id_errors`, `test_session_profile_mismatch_errors`, `test_idle_session_reaped`. For the last one, pass a small `idle_seconds` and a fake clock.
  - `test_literate_agent_resubmission_resumes`: run `scripts/literate_agent.py`'s resubmit path with a scripted fake model, and assert the same single `log.txt` line.
- [ ] **Step 2: Run and see them fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** blq `test-all` and `test-mcp`. Expected: all pass.
- [ ] **Step 5: Commit** `feat(literate): retries resume on a matching prefix, in literate_agent and over MCP session_id (D10)`.

### Task 8: Documentation

**Files:**
- Modify: `docs/concepts/literate.md` (a new section, "Time limits and retries"), `docs/concepts/interpreters.md`
- Modify: the style-guide text that is generated from the dictionary (`scripts/gen_style_guide.py`). Regenerate, and do not hand-edit its output.
- Create: `docs/concepts/time-limits.md`, and link it from `mkdocs.yml`
- Test: the docs test that already checks literate fences (`tests/` docs test from literate stepy Task 6)

Content to state, measured where it says so:
- The limit, the `timeout` parameter and `max_timeout`, with the exact texts from Tasks 4–5.
- A timeout is not a refusal. Retry by resubmitting. Show the prefix rule with one example document, with the output measured on 2026-10-25 or later.
- Direct library use of `RestrictedRunner` or `CellSession` runs in-process, with no limit (spec D9).
- Linux only.
- The new vocabulary, and the four one-way changes with their hints.

- [ ] **Step 1: Write** the docs. Run each example document through `lackpyctl render --execute`, and paste the measured output.
- [ ] **Step 2: Run** blq `test-all` and `mkdocs build --strict`, the latter through `mcp__plugin_blq_blq_mcp__exec`. Expected: pass.
- [ ] **Step 3: Commit** `docs: time limits, retries and the Plan 3 vocabulary`.

---

## Gate coverage

| Gate | Task and test |
|---|---|
| 1 regex stops | 4 `test_regex_stops_at_limit`, 5 `test_run_program_timeout` |
| 2 FIFO stops | 4 `test_fifo_read_stops_at_limit` |
| 3 memory cap | 4 `test_memory_cap` |
| 4 literate retry, no second write | 7 `test_resume_*`, `test_literate_agent_resubmission_resumes` |
| 5 timeout names written file | 4 `test_trace_survives_kill`, 6 `test_timeout_names_written_file` |
| 6 no pickle in the channel | 3 `test_channel_imports_no_pickle` |
| 7 over the cap refuses | 5 `test_timeout_over_cap_refuses` |
| 8 surface test | 2 `test_return_surface` |
| 9 suite holds | every task's final step |
