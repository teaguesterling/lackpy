"""Literate sessions that a retry resumes (Plan 3 Task 7; spec D10).

A :class:`LiterateRun` is the parent's handle on one session worker
(:class:`~lackpy.run.process.SessionChannel`). Its :meth:`LiterateRun.submit`
applies the prefix rule to each full-document submission:

- It compares the new document's cells (from the literate parser) with the
  last submission's, and finds the first index ``k`` where they differ. The
  comparison is exact: cell type, source text, annotation arguments and
  options, and the frontmatter.
- ``k`` is no later than the first cell that did not complete in the session's
  last run (it was stopped at the limit): an identical resubmission resumes
  at the stopped cell (fix round 1, ruling A).
- A cell that runs no code (prose with no interpolation) takes no checkpoint
  of its own: the checkpoint at the start of a run of such cells serves each
  of them. So the resume point is the start of the segment that holds ``k``.
- If a checkpoint that can still run holds that point, the processes for that
  cell and every newer one are stopped, and the checkpoint resumes the new
  document there. The cells before it do not run again.
- Otherwise the worker is closed and the document runs fresh. So does the
  first submission, a submission with other params, and one after the session
  failed.

Retries are the caller's job (D7): ``submit`` resumes a resubmission, it never
retries one.

A :class:`SessionRegistry` holds the runs that MCP calls name by
``session_id``. A session belongs to one workspace and one profile. It ends on
an explicit close, after ``idle_seconds`` without a call, or when the registry
is full (the session idle longest goes first). Idle sessions are reaped when
the registry is next used.
"""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from ..interpreters.literate import runs_no_code
from ..interpreters.literate.parser import parse
from .base import ExecutionResult
from .paths import PathPolicy
from .process import _check_max_checkpoints


class SessionError(Exception):
    """An unknown or expired ``session_id``, or one used with another profile or workspace."""


def _shape(document: str) -> tuple[Any, list[tuple[Any, ...]]] | None:
    """What the prefix rule compares: the frontmatter and each cell. None on a parse error."""
    parsed = parse(document)
    if parsed.errors:
        return None
    fm = parsed.frontmatter
    cells = [(c.cell_type, c.content, tuple(sorted(c.annotation_args.items())),
              tuple(sorted(c.options.items())), c.truncated, runs_no_code(c))
             for c in parsed.cells]
    return (fm.echo, fm.output, fm.interpreter), cells


class LiterateRun:
    """The parent's handle on one literate session worker.

    Not for concurrent use by design, but safe: ``submit`` and ``close`` take
    one lock. ``last_used`` is the clock's time at the start and the end of the
    last call. ``busy`` is True while a caller holds the run (a registry lease,
    or a submit): a registry never closes a busy run. A closed run refuses
    ``submit``; it never runs fresh silently (fix round 1, minor 7).

    ``cell_timeout`` is the wall limit for each cell, and may change on each
    ``configure``. ``max_timeout`` (default ``cell_timeout``) is the CPU
    backstop the worker is spawned with: a later limit up to it keeps the
    session (fix round 1, ruling B).
    """

    def __init__(self, runner: Any, *, workspace: Path, profile_name: str,
                 path_policy: PathPolicy, cell_timeout: float,
                 rule_names: list[str] | None = None,
                 context_config: dict[str, Any] | None = None,
                 max_checkpoints: int = 32, session_id: str | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 max_timeout: float | None = None) -> None:
        _check_max_checkpoints(max_checkpoints)
        self.session_id = session_id or uuid.uuid4().hex
        self.workspace = Path(workspace)
        self.profile_name = profile_name
        self.max_checkpoints = max_checkpoints
        self._runner = runner
        self._clock = clock
        self._lock = threading.Lock()
        self._channel: Any = None
        self._last: tuple[Any, dict[str, Any]] | None = None  # (shape, params) last submitted
        self._settings: tuple[Any, ...] = ()
        self._policy = path_policy
        self._timeout = float(cell_timeout)
        self._max_timeout = float(cell_timeout)
        self._closed = False
        self.busy = False
        self.last_used = clock()
        self.configure(path_policy=path_policy, cell_timeout=cell_timeout,
                       rule_names=rule_names, context_config=context_config,
                       max_timeout=max_timeout)

    def configure(self, *, path_policy: PathPolicy, cell_timeout: float,
                  rule_names: list[str] | None = None,
                  context_config: dict[str, Any] | None = None,
                  max_timeout: float | None = None) -> None:
        """Set what the worker runs under.

        A change of policy, rules or config closes the worker, so the next
        submit is fresh: the checkpoints hold the old ones. A new
        ``cell_timeout`` keeps the worker while it is within the CPU backstop
        the worker was spawned with (``max_timeout``); above it, the worker is
        closed.
        """
        settings = (path_policy.to_data(), list(rule_names) if rule_names else None,
                    dict(context_config or {}))
        cap = max(float(cell_timeout), float(max_timeout or 0.0))
        with self._lock:
            channel = self._channel
            if settings != self._settings or (
                    channel is not None and float(cell_timeout) > channel.cpu_timeout):
                self._close_channel()
            self._settings = settings
            self._policy = path_policy
            self._timeout = float(cell_timeout)
            self._max_timeout = cap

    def release(self) -> None:
        """End a registry lease without a submit."""
        self.busy = False
        self.last_used = self._clock()

    def submit(self, document: str, params: dict[str, Any] | None = None) -> ExecutionResult:
        """Run ``document``, resuming from the checkpoint before its first changed cell."""
        with self._lock:
            if self._closed:
                self.busy = False
                return ExecutionResult(success=False,
                                       error=f"session {self.session_id!r} is closed")
            self.busy = True
            self.last_used = self._clock()
            try:
                return self._submit(document, dict(params or {}))
            finally:
                self.busy = False
                self.last_used = self._clock()

    def _submit(self, document: str, params: dict[str, Any]) -> ExecutionResult:
        shape = _shape(document)
        last, self._last = self._last, (shape, params)
        channel = self._channel
        if (channel is not None and last is not None and shape is not None
                and last[0] is not None and last[1] == params and last[0][0] == shape[0]):
            old, new = last[0][1], shape[1]
            k = next((i for i, (a, b) in enumerate(zip(old, new)) if a != b), min(len(old), len(new)))
            k = min([k, *channel.stopped])  # ruling A: no later than a stopped cell
            start = k
            while start > 0 and new[start - 1][-1]:  # back over cells that run no code
                start -= 1
            if channel.can_resume(start):
                channel.cell_timeout = self._timeout
                return channel.resume(start, document)
        return self._fresh(document, params)

    def _fresh(self, document: str, params: dict[str, Any]) -> ExecutionResult:
        self._close_channel()
        _, rule_names, config = self._settings
        self._channel = self._runner.open_session(
            path_policy=self._policy, workspace=self.workspace, cell_timeout=self._timeout,
            rule_names=rule_names, context_config=config, max_checkpoints=self.max_checkpoints,
            cpu_timeout=self._max_timeout)
        return self._channel.submit(document, params=params or None)

    def _close_channel(self) -> None:
        if self._channel is not None:
            self._channel.close()
            self._channel = None

    def close(self) -> None:
        """Stop every process of the session; later submits refuse. Idempotent."""
        with self._lock:
            self._closed = True
            self._close_channel()
            self._last = None


class SessionRegistry:
    """The live literate sessions that MCP calls name by ``session_id`` (D10).

    The defaults are starting values; the Task 7 report has the measured memory.
    """

    def __init__(self, *, max_sessions: int = 8, idle_seconds: float = 600,
                 max_checkpoints: int = 32,
                 clock: Callable[[], float] = time.monotonic) -> None:
        _check_max_checkpoints(max_checkpoints)
        self.max_sessions = max_sessions
        self.idle_seconds = idle_seconds
        self.max_checkpoints = max_checkpoints
        self._clock = clock
        self._lock = threading.Lock()
        self._runs: dict[str, LiterateRun] = {}

    def open(self, *, runner: Any, workspace: Path, profile_name: str,
             path_policy: PathPolicy, cell_timeout: float,
             rule_names: list[str] | None = None,
             context_config: dict[str, Any] | None = None,
             max_timeout: float | None = None) -> LiterateRun:
        """A new session, leased to the caller (its submit ends the lease).

        When the registry is full, the session idle longest is closed. A leased
        session is never closed: with every session leased, the registry holds
        one more than ``max_sessions`` until a lease ends.
        """
        run = LiterateRun(runner, workspace=workspace, profile_name=profile_name,
                          path_policy=path_policy, cell_timeout=cell_timeout,
                          rule_names=rule_names, context_config=context_config,
                          max_checkpoints=self.max_checkpoints, clock=self._clock,
                          max_timeout=max_timeout)
        run.busy = True
        with self._lock:
            ended = self._reap()
            while len(self._runs) >= self.max_sessions:
                idle = [r for r in self._runs.values() if not r.busy]
                if not idle:
                    break
                oldest = min(idle, key=lambda r: r.last_used)
                ended.append(self._runs.pop(oldest.session_id))
            self._runs[run.session_id] = run
        for r in ended:
            r.close()
        return run

    def get(self, session_id: str, *, workspace: Path, profile_name: str) -> LiterateRun:
        """The live session ``session_id``, leased to the caller (its submit, or
        ``release``, ends the lease); ``SessionError`` when it is unknown, expired
        or not this caller's."""
        with self._lock:
            ended = self._reap()
            run = self._runs.get(session_id)
            if run is not None:
                if run.profile_name != profile_name or run.workspace != Path(workspace):
                    run = None
                    error = (f"session {session_id!r} belongs to profile "
                             f"{self._runs[session_id].profile_name!r} in "
                             f"{self._runs[session_id].workspace}")
                else:
                    run.busy = True  # the lease: eviction and reaping skip it
                    run.last_used = self._clock()
            else:
                error = f"unknown or expired session_id {session_id!r}"
        for r in ended:
            r.close()
        if run is None:
            raise SessionError(error)
        return run

    def close(self, session_id: str) -> None:
        """End the session ``session_id``. An unknown id is ignored."""
        with self._lock:
            run = self._runs.pop(session_id, None)
        if run is not None:
            run.close()

    def __contains__(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._runs

    def close_all(self) -> None:
        with self._lock:
            runs = list(self._runs.values())
            self._runs.clear()
        for r in runs:
            r.close()

    def _reap(self) -> list[LiterateRun]:
        """Remove the sessions idle longer than ``idle_seconds``; the caller closes them."""
        now = self._clock()
        expired = [sid for sid, r in self._runs.items()
                   if not r.busy and now - r.last_used > self.idle_seconds]
        return [self._runs.pop(sid) for sid in expired]
