"""ProcessRunner: run a plain program in a worker process, under a time limit (Plan 3, Task 4).

A thread cannot stop a regex running in C or a read blocked in the OS (spec D1,
measured), so the limit is enforced by killing a process. The runner starts a
zygote (``lackpy.run.zygote``) lazily, as a new ``python -I`` process with a
scrubbed environment, and asks it to fork one worker for each run. The worker
validates and runs the program with ``RestrictedRunner`` in the workspace, and
streams its trace entries and stdout back as tagged JSON (``wire``). At the
deadline the parent SIGKILLs the worker, through a pidfd so a reused pid can
never be hit, and keeps what the worker had already sent.

The run's clock starts when the worker exists: the zygote's own start (about
0.3 s, once) is not charged to the program.
"""

from __future__ import annotations

import atexit
import math
import os
import select
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from . import wire
from .base import ExecutionResult
from .limits import ENV_KEEP, STDOUT_CAP
from .paths import PathPolicy
from .trace import START_SUFFIX, Trace, TraceEntry

__all__ = ["ProcessRunner", "STDOUT_CAP", "SessionChannel", "close_shared_runners",
           "shared_runner"]

STDOUT_MARK = "\n[lackpy: output cut at 10 MB]"
#: How long the parent keeps reading after the kill, for what the worker already sent.
DRAIN_SECONDS = 0.5
#: How long a spawn may take, the zygote's first start included.
SPAWN_WAIT = 30.0
#: The CPU backstop is this many seconds above the wall-clock limit (spec D5).
CPU_MARGIN = 2

_READ_TOOLS = ("Path.read_text", "Path.read_bytes")
_WRITE_TOOLS = ("Path.write_text", "Path.write_bytes")


class _SpawnError(Exception):
    pass


class _BadMessage(Exception):
    pass


class _HandoffStall(Exception):
    """A session process sent cell_end and then did not reach its next cell in time."""


def _traced_paths(entries: list[TraceEntry], tools: tuple[str, ...]) -> list[str]:
    """As ``service._traced_paths``: the paths of the successful ``tools`` calls, no repeats."""
    seen: dict[str, None] = {}
    for e in entries:
        if e.success and e.tool in tools and isinstance(e.args, list) and e.args:
            seen.setdefault(str(e.args[0]), None)
    return list(seen)


class _Run:
    """What the parent has heard from one worker."""

    def __init__(self, pid: int, pidfd: int | None) -> None:
        self.pid = pid
        self.pidfd = pidfd
        self.entries: list[TraceEntry] = []
        self.stdout: list[str] = []
        self.stdout_len = 0
        self.cut = False
        self.done: dict[str, Any] | None = None
        self.how: str | None = None
        self.ended = False  # exit event or EOF

    def take(self, msg: dict[str, Any] | None) -> None:
        if msg is None:
            self.ended = True
            return
        try:
            event = msg["event"]
            if event == "trace":
                entry = TraceEntry.from_data(msg["entry"])
                if not entry.tool.endswith(START_SUFFIX):  # a write's start: stream only
                    self.entries.append(entry)
            elif event == "stdout":
                self._stdout(msg["text"])
            elif event == "done":
                if not isinstance(msg["result"], dict):
                    raise _BadMessage("a done result must be a dict")
                self.done = msg["result"]
            elif event == "exit":
                self.how = str(msg["how"])
                self.ended = True
            elif event == "start":
                if msg["pid"] != self.pid:
                    raise _BadMessage(f"start names pid {msg['pid']!r}, not {self.pid}")
            else:
                raise _BadMessage(f"unknown event {event!r}")
        except (KeyError, TypeError) as e:
            raise _BadMessage(f"{type(e).__name__}: {e}") from None

    def _stdout(self, text: Any) -> None:
        if type(text) is not str:
            raise _BadMessage("stdout text must be str")
        if self.cut:
            return  # past the cap: read on to the result, keep nothing
        room = STDOUT_CAP - self.stdout_len
        if len(text) > room:
            text = text[:room]
            self.cut = True
        self.stdout.append(text)
        self.stdout_len += len(text)

    def trace(self) -> Trace:
        return Trace(entries=self.entries,
                     files_read=_traced_paths(self.entries, _READ_TOOLS),
                     files_modified=_traced_paths(self.entries, _WRITE_TOOLS))

    def stdout_text(self) -> str:
        return "".join(self.stdout) + (STDOUT_MARK if self.cut else "")

    def failed(self, error: str, error_phase: str | None = None) -> ExecutionResult:
        return ExecutionResult(success=False, error=error, error_phase=error_phase,
                               trace=self.trace(), stdout=self.stdout_text())


class ProcessRunner:
    """Run plain programs in a forked worker, each under its own time limit.

    Safe to call from several threads at once: each run has its own worker and
    socket; only the spawn request to the zygote is serialized.
    ``last_pid`` is the pid of the most recent worker (for tests).
    """

    #: Extra zygote arguments; tests set ``("--test-ops",)`` (ledger Ruling B).
    _zygote_flags: tuple[str, ...] = ()

    def __init__(self, *, memory_mb: int = 512, python: str = sys.executable) -> None:
        self.memory_mb = memory_mb
        self.python = python
        self.last_pid: int | None = None
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[bytes] | None = None
        self._control: socket.socket | None = None

    # --- the zygote ----------------------------------------------------------

    def _start_zygote(self) -> None:
        if not hasattr(os, "fork"):
            # Global constraint: never fall back to an in-process run silently.
            raise RuntimeError("the process runner needs os.fork (Linux)")
        parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        env = {k: os.environ[k] for k in ENV_KEEP if k in os.environ}
        try:
            self._proc = subprocess.Popen(
                [self.python, "-I", "-m", "lackpy.run.zygote", str(child.fileno()),
                 *self._zygote_flags],
                pass_fds=(child.fileno(),), stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, env=env, cwd="/", start_new_session=True)
        except BaseException:
            parent.close()
            raise
        finally:
            child.close()
        self._control = parent

    def _stop_zygote(self) -> None:
        if self._control is not None:
            self._control.close()  # EOF: the zygote kills its workers and exits
            self._control = None
        if self._proc is not None:
            try:
                self._proc.wait(2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(2)
            self._proc = None

    def _spawn(self, worker_end: socket.socket, cpu_seconds: int,
               ctl_end: socket.socket | None = None) -> int:
        """Ask the zygote for a worker on ``worker_end``; start or restart it as needed.

        With ``ctl_end`` the worker is a literate session root (``spawn_session``),
        which also gets the shared channel.
        """
        op = "spawn" if ctl_end is None else "spawn_session"
        request = wire.encode({"op": op, "memory_mb": self.memory_mb,
                               "cpu_seconds": cpu_seconds}).encode("ascii")
        fds = [worker_end.fileno()] + ([] if ctl_end is None else [ctl_end.fileno()])
        with self._lock:
            for _ in range(2):
                if self._proc is None or self._proc.poll() is not None:
                    self._stop_zygote()
                    self._start_zygote()
                assert self._control is not None
                try:
                    socket.send_fds(self._control, [request], fds)
                    self._control.settimeout(SPAWN_WAIT)
                    reply = self._control.recv(65536)
                except TimeoutError:
                    self._stop_zygote()
                    raise _SpawnError("the zygote did not answer") from None
                except OSError:
                    reply = b""
                if not reply:  # the zygote died: start a new one, once
                    self._stop_zygote()
                    continue
                msg = wire.decode(reply.decode("ascii"))
                if msg.get("event") == "spawned" and type(msg.get("pid")) is int:
                    return msg["pid"]
                raise _SpawnError(str(msg.get("error")))
            raise _SpawnError("the zygote ended twice")

    def close(self) -> None:
        """Stop the zygote, and with it every worker it has."""
        with self._lock:
            self._stop_zygote()

    # --- one run -------------------------------------------------------------

    def run(self, program: str, *, params: dict[str, Any] | None, path_policy: PathPolicy,
            workspace: Path, timeout: float, rules: list[str] | None = None) -> ExecutionResult:
        """Run ``program`` in a worker; never raises for anything the program does.

        The worker validates the program again (with ``rules``, by name), so this
        is safe for any caller. Failures that are not the program's own:

        - over ``timeout`` seconds: ``error_phase="timeout"``;
        - over ``memory_mb``: the ``memory:`` text, ``error_phase=None``;
        - the worker ended without a result: the crash text, with its signal or
          exit status.

        ``trace.files_read`` and ``files_modified`` are filled from the entries
        (as the service does), so they survive a kill. Stdout past ``STDOUT_CAP``
        characters is cut and marked.

        Raises ``ValueError`` when ``rules`` holds anything but names: a rule
        callable cannot cross to the worker, so the caller validates with it.
        """
        if rules and any(type(r) is not str for r in rules):
            raise ValueError("the process runner takes rule names, not callables")
        ours, theirs = socket.socketpair()
        with ours:
            try:
                try:
                    pid = self._spawn(theirs, math.ceil(timeout) + CPU_MARGIN)
                finally:
                    theirs.close()
            except (_SpawnError, OSError, wire.WireError) as e:
                return ExecutionResult(success=False,
                                       error=f"the run process could not be started ({e})")
            self.last_pid = pid
            try:
                pidfd: int | None = os.pidfd_open(pid)
            except ProcessLookupError:
                pidfd = None
            deadline = time.monotonic() + timeout
            run = _Run(pid, pidfd)
            try:
                try:
                    wire.send(ours, {"op": "run", "program": program, "params": params,
                                     "policy": path_policy.to_data(),
                                     "workspace": str(workspace), "rules": rules})
                except wire.WireError as e:
                    return run.failed(f"the run's params cannot be sent: {e}")
                except OSError:
                    pass  # the worker is gone; its exit event says how
                return self._collect(ours, run, deadline, timeout)
            finally:
                if run.done is None and not run.ended:
                    _kill(pidfd)
                if pidfd is not None:
                    os.close(pidfd)

    def _collect(self, sock: socket.socket, run: _Run, deadline: float,
                 timeout: float) -> ExecutionResult:
        reader = wire.Reader(sock)
        try:
            while run.done is None and not run.ended:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                run.take(reader.recv(remaining))
        except TimeoutError:
            return self._timed_out(reader, run, timeout)
        except (wire.WireError, _BadMessage, OSError) as e:
            return run.failed(f"the run process sent a message that is not valid ({e})")
        if run.done is None:
            how = run.how or "exit status unknown"
            return run.failed(f"the run process ended without a result ({how})")
        return self._result(run)

    def _timed_out(self, reader: wire.Reader, run: _Run, timeout: float) -> ExecutionResult:
        _kill(run.pidfd)
        end = time.monotonic() + DRAIN_SECONDS
        try:
            while not run.ended:
                remaining = end - time.monotonic()
                if remaining <= 0:
                    break
                msg = reader.recv(remaining)
                if msg is not None and msg.get("event") == "done":
                    continue  # past the deadline: the run timed out
                run.take(msg)
        except (TimeoutError, wire.WireError, _BadMessage, OSError):
            pass
        return run.failed(f"timeout: the run took longer than {timeout:g} s and was stopped",
                          error_phase="timeout")

    def _result(self, run: _Run) -> ExecutionResult:
        r = run.done
        assert r is not None
        if r.get("memory"):
            return run.failed(f"memory: the run used more than {self.memory_mb} MiB "
                              f"and was stopped")
        error, phase, variables = r.get("error"), r.get("error_phase"), r.get("variables")
        if (type(r.get("success")) is not bool or not isinstance(variables, dict)
                or not (error is None or type(error) is str)
                or not (phase is None or type(phase) is str)):
            return run.failed("the run process sent a result that is not valid")
        return ExecutionResult(success=r["success"], output=r.get("output"),
                               stdout=run.stdout_text(), error=error, trace=run.trace(),
                               variables=variables, error_phase=phase)

    # --- a literate session (Task 6) --------------------------------------

    def open_session(self, *, path_policy: PathPolicy, workspace: Path, cell_timeout: float,
                     rule_names: list[str] | None,
                     context_config: dict[str, Any]) -> "SessionChannel":
        """Start a literate session worker (``spawn_session``) and return its channel.

        The whole literate interpreter runs in the worker, with a checkpoint fork
        at the top of each cell (``session_worker``). Each cell runs under
        ``cell_timeout`` seconds. ``rule_names`` are resolved in the worker, which
        validates every cell with them (ledger Ruling D). ``context_config`` is the
        interpreter's ``context.config`` as data (the policy, the hook and the
        rules are added in the worker).

        Raises ``ValueError`` when ``rule_names`` holds anything but names, and
        ``RuntimeError`` when the session process cannot be started.
        """
        if rule_names and any(type(r) is not str for r in rule_names):
            raise ValueError("the process runner takes rule names, not callables")
        ours, theirs = socket.socketpair()
        ctl_ours, ctl_theirs = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        try:
            try:
                pid = self._spawn(theirs, math.ceil(cell_timeout) + CPU_MARGIN, ctl_theirs)
            finally:
                theirs.close()
                ctl_theirs.close()
            channel = SessionChannel(pid, ours, ctl_ours, path_policy=path_policy,
                                     workspace=workspace, cell_timeout=cell_timeout,
                                     rule_names=rule_names, context_config=context_config)
        except (_SpawnError, OSError, wire.WireError) as e:
            ours.close()
            ctl_ours.close()
            raise RuntimeError(f"the session process could not be started ({e})") from None
        self.last_pid = pid
        return channel

    def _worker_environ(self, timeout: float) -> dict[str, str] | None:
        """Test-only: a worker's ``os.environ``, or None when the zygote refuses the op."""
        ours, theirs = socket.socketpair()
        with ours:
            try:
                pid = self._spawn(theirs, math.ceil(timeout) + CPU_MARGIN)
            finally:
                theirs.close()
            reader = wire.Reader(ours)
            end = time.monotonic() + timeout
            first = reader.recv(timeout)
            if not first or first.get("pid") != pid:
                return None
            wire.send(ours, {"op": "environ"})
            msg = reader.recv(max(0.0, end - time.monotonic()))
            if msg is not None and msg.get("event") == "environ":
                return msg["environ"]
            return None


class _Proc:
    """One live process of a session, as the parent follows it."""

    def __init__(self, index: int | None, pid: int, sock: socket.socket) -> None:
        self.index = index  # the cell it runs; None for the root
        self.pid = pid
        self.sock = sock
        self.reader = wire.Reader(sock)
        self.cell_ended = False  # it sent cell_end; its cell_start comes on the shared channel
        self.wait_until = 0.0  # while cell_ended: the handoff bound, not the cell's deadline
        try:
            self.pidfd: int | None = os.pidfd_open(pid)
        except ProcessLookupError:
            self.pidfd = None

    def close(self) -> None:
        _kill(self.pidfd)
        if self.pidfd is not None:
            os.close(self.pidfd)
            self.pidfd = None
        self.sock.close()


class SessionChannel:
    """The parent's end of a literate session worker (Plan 3 Task 6; D6, D7).

    The parent follows the chain of live processes: the root, then one process
    for each cell that started. Each has its own socket (``session_worker``
    explains the protocol). ``send`` writes to the deepest process; ``events``
    reads its messages; ``kill`` stops one process. ``checkpoints`` lists the
    live ``(index, pid)`` of the cell processes, from the ``cell_start`` events.
    ``submit`` runs one document and returns its result; Task 7's ``LiterateRun``
    is built on the same methods.

    Not thread-safe: one caller at a time.

    The cell limit runs from ``cell_start`` to the cell's ``cell_end``. From
    ``cell_end`` to the next ``cell_start`` (the fork), and from the last
    ``cell_end`` to ``done`` (bookkeeping, the render, the result), the process
    runs host code only, so the cell is not charged for it: ``HANDOFF_SECONDS``
    bounds each such step instead, and a step over it ends the run as a failure
    of the session, never as a cell timeout.
    """

    #: The bound on one host step between cells (fix round 1). Measured on
    #: longbottom: the handoff (cell_end, socketpair, fork, cell_start) takes
    #: about 5 ms, so 10 s is about 2000 times that. It also covers the render
    #: and the result of a large document. A step over it is a stalled worker.
    HANDOFF_SECONDS = 10.0

    def __init__(self, pid: int, sock: socket.socket, ctl: socket.socket, *,
                 path_policy: PathPolicy, workspace: Path, cell_timeout: float,
                 rule_names: list[str] | None, context_config: dict[str, Any]) -> None:
        self.pid = pid
        self.cell_timeout = cell_timeout
        self._ctl = ctl
        self._stack = [_Proc(None, pid, sock)]
        self._submit_base = {"policy": path_policy.to_data(), "workspace": str(workspace),
                             "rules": rule_names, "config": context_config,
                             "cell_timeout": cell_timeout}
        first = self._stack[0].reader.recv(SPAWN_WAIT)
        if not first or first.get("event") != "start" or first.get("pid") != pid:
            root = self._stack[0]
            if root.pidfd is not None:  # the caller closes the socket; the pidfd is ours
                os.close(root.pidfd)
                root.pidfd = None
            raise _SpawnError(f"the session process did not start (it sent {first!r})")

    @property
    def checkpoints(self) -> list[tuple[int, int]]:
        return [(p.index, p.pid) for p in self._stack[1:] if p.index is not None]

    def send(self, msg: dict[str, Any]) -> None:
        """Write ``msg`` to the deepest live process (``WireError`` for non-data)."""
        wire.send(self._stack[-1].sock, msg)

    def kill(self, pid: int) -> None:
        """SIGKILL the process ``pid`` of this session (its descendants follow by PDEATHSIG)."""
        for p in self._stack:
            if p.pid == pid:
                _kill(p.pidfd)

    def events(self, deadline: float) -> Any:
        """The deepest process's messages until it ends; ``TimeoutError`` at ``deadline``.

        ``deadline`` is the running cell's; between cells (after ``cell_end``)
        each host step has ``HANDOFF_SECONDS`` instead, and a step over it raises
        ``_HandoffStall``.

        A ``cell_start`` makes the new cell's process the deepest one. A process
        that ends yields ``{"event": "ended", "pid", "how"}`` and the iteration stops.
        """
        while True:
            msg = self._next(deadline)
            yield msg
            if msg.get("event") == "ended":
                return

    def _next(self, deadline: float) -> dict[str, Any]:
        while True:
            cur = self._stack[-1]
            if not cur.cell_ended:
                # A message sent before the deadline is still read at the deadline.
                remaining = max(deadline - time.monotonic(), 0.001)
                msg = cur.reader.recv(remaining)
                if msg is None or msg.get("event") == "exit":
                    how = (msg or {}).get("how", "its socket closed")
                    return {"event": "ended", "pid": cur.pid, "how": how}
                if msg.get("event") == "cell_end":
                    cur.cell_ended = True
                    cur.wait_until = time.monotonic() + self.HANDOFF_SECONDS
                    continue
                return msg
            # Between cells: the cell's deadline is stopped. Wait for the next
            # cell_start on the shared channel, and for the process's own socket
            # too: a process that dies here (a failed fork, a failed send)
            # closes its socket, and the checkpoints above still hold the shared
            # channel, so it gives no EOF there; the last cell's process sends
            # done here. The channel wins when both are ready.
            remaining = cur.wait_until - time.monotonic()
            if remaining <= 0:
                raise _HandoffStall
            poller = select.poll()
            poller.register(self._ctl.fileno(), select.POLLIN)
            poller.register(cur.sock.fileno(), select.POLLIN)
            ready = {fd for fd, _ in poller.poll(math.ceil(remaining * 1000))}
            if not ready:
                raise _HandoffStall
            if self._ctl.fileno() not in ready:
                msg = cur.reader.recv(max(cur.wait_until - time.monotonic(), 0.001))
                if msg is None or msg.get("event") == "exit":
                    how = (msg or {}).get("how", "its socket closed")
                    return {"event": "ended", "pid": cur.pid, "how": how}
                event = msg.get("event")
                if event == "cell_end":  # another host step (cell_done, then a re-run's hook)
                    cur.wait_until = time.monotonic() + self.HANDOFF_SECONDS
                    continue
                if event in ("done", "trace"):
                    return msg
                raise _BadMessage(f"{event!r} between cells")
            self._ctl.settimeout(max(cur.wait_until - time.monotonic(), 0.001))
            data, fds, msg_flags, _ = socket.recv_fds(self._ctl, 65536, 4)
            if not data:
                for f in fds:
                    os.close(f)
                return {"event": "ended", "pid": cur.pid, "how": "the shared channel closed"}
            msg = wire.decode(data.decode("ascii"))
            if (isinstance(msg, dict) and msg.get("from") == cur.pid
                    and (len(fds) != 1 or msg_flags & socket.MSG_CTRUNC)):
                for f in fds:
                    os.close(f)
                raise _BadMessage("a cell_start came without its socket (is the service "
                                  "out of file descriptors?)")
            if (not isinstance(msg, dict) or msg.get("event") != "cell_start"
                    or msg.get("from") != cur.pid or len(fds) != 1
                    or type(msg.get("pid")) is not int or type(msg.get("index")) is not int):
                for f in fds:  # a packet from a process this session no longer follows
                    os.close(f)
                continue
            cur.cell_ended = False
            self._stack.append(_Proc(msg["index"], msg["pid"], socket.socket(fileno=fds[0])))
            return msg

    def submit(self, document: str, *, params: dict[str, Any] | None) -> ExecutionResult:
        """Run ``document`` in the session; never raises for anything the document does.

        Each cell runs under ``cell_timeout`` seconds (each ``cell_start`` re-arms
        the deadline; so does the start of the document and each timeout). At a
        deadline the parent SIGKILLs the deepest process, drains what it sent,
        and sends ``{"op": "timeout"}`` with the stopped cell's entries to the
        checkpoint above it; the document goes on (D7). With no checkpoint above
        (the root itself ran over), the run ends with the timeout text.
        """
        try:
            self.send({"op": "submit", "document": document, "params": params,
                       **self._submit_base})
        except wire.WireError as e:
            return ExecutionResult(success=False, error=f"the document's params cannot be sent: {e}")
        except OSError:
            pass  # the root is gone; its ended event says how
        entries: list[TraceEntry] = []  # the session's trace, start entries left out
        cell: list[TraceEntry] = []  # the deepest cell's entries, start entries included
        deadline = time.monotonic() + self.cell_timeout
        while True:
            try:
                msg = self._next(deadline)
                event = msg["event"]
                if event == "trace":
                    entry = TraceEntry.from_data(msg["entry"])
                    cell.append(entry)
                    if not entry.tool.endswith(START_SUFFIX):
                        entries.append(entry)
                elif event == "cell_start":
                    deadline = time.monotonic() + self.cell_timeout
                    cell = []
                elif event == "done":
                    return self._result(msg["result"], entries)
                elif event == "ended":
                    return self._failed(f"the session process ended without a result "
                                        f"({msg['how']})", entries)
                elif event not in ("start", "stdout"):
                    raise _BadMessage(f"unknown event {event!r}")
            except TimeoutError:
                if not self._timeout(cell, entries):
                    return self._failed(f"timeout: the document took longer than "
                                        f"{self.cell_timeout:g} s and was stopped",
                                        entries, error_phase="timeout")
                deadline = time.monotonic() + self.cell_timeout
                cell = []
            except _HandoffStall:
                self.close()
                return self._failed(f"the session process did not reach its next cell within "
                                    f"{self.HANDOFF_SECONDS:g} s and was stopped", entries)
            except (wire.WireError, _BadMessage, OSError, KeyError, TypeError) as e:
                self.close()
                return self._failed(f"the session process sent a message that is not valid ({e})",
                                    entries)

    def _timeout(self, cell: list[TraceEntry], entries: list[TraceEntry]) -> bool:
        """Stop the deepest process; tell its checkpoint. False when there is none."""
        dead = self._stack.pop()
        _kill(dead.pidfd)
        # Its socket: what it sent before the kill, then EOF (a partial last line
        # is dropped). The socket is closed after: nothing else ever uses it.
        end = time.monotonic() + DRAIN_SECONDS
        try:
            while True:
                remaining = end - time.monotonic()
                if remaining <= 0:
                    break
                msg = dead.reader.recv(remaining)
                if msg is None:
                    break
                if msg.get("event") == "trace":
                    entry = TraceEntry.from_data(msg["entry"])
                    cell.append(entry)
                    if not entry.tool.endswith(START_SUFFIX):
                        entries.append(entry)
        except (TimeoutError, wire.WireError, OSError, KeyError, TypeError):
            pass
        dead.close()
        if not self._stack:
            return False
        checkpoint = self._stack[-1]
        checkpoint.cell_ended = False
        try:
            self.send({"op": "timeout", "seconds": self.cell_timeout,
                       "entries": [e.to_data() for e in cell]})
        except (wire.WireError, OSError):
            return False
        return True

    def _failed(self, error: str, entries: list[TraceEntry],
                error_phase: str | None = None) -> ExecutionResult:
        return ExecutionResult(success=False, error=error, error_phase=error_phase,
                               trace=_trace(entries))

    def _result(self, r: Any, entries: list[TraceEntry]) -> ExecutionResult:
        if not isinstance(r, dict):
            return self._failed("the session process sent a result that is not valid", entries)
        error, phase, variables = r.get("error"), r.get("error_phase"), r.get("variables")
        if (type(r.get("success")) is not bool or not isinstance(variables, dict)
                or not (error is None or type(error) is str)
                or not (phase is None or type(phase) is str)
                or type(r.get("stdout")) is not str):
            return self._failed("the session process sent a result that is not valid", entries)
        return ExecutionResult(success=r["success"], output=r.get("output"), stdout=r["stdout"],
                               error=error, trace=_trace(entries), variables=variables,
                               error_phase=phase)

    def close(self) -> None:
        """Stop every process of the session, the deepest first."""
        while self._stack:
            self._stack.pop().close()
        self._ctl.close()


def _trace(entries: list[TraceEntry]) -> Trace:
    return Trace(entries=entries, files_read=_traced_paths(entries, _READ_TOOLS),
                 files_modified=_traced_paths(entries, _WRITE_TOOLS))


_shared: dict[int, ProcessRunner] = {}
_shared_lock = threading.Lock()


def shared_runner(memory_mb: int) -> ProcessRunner:
    """The process's one ``ProcessRunner`` for ``memory_mb`` (Task 5 fix round 1 ruling).

    Every ``LackpyService`` in a process shares it, so the zygote starts once.
    Runs share no state: the program, params, policy, workspace and rules
    cross on each run, and each run has its own forked worker.
    """
    with _shared_lock:
        runner = _shared.get(memory_mb)
        if runner is None:
            runner = _shared[memory_mb] = ProcessRunner(memory_mb=memory_mb)
        return runner


def close_shared_runners() -> None:
    """Stop every shared runner's zygote. A later ``shared_runner`` call starts a new one.

    Registered with ``atexit``; tests call it at the end of the session.
    """
    with _shared_lock:
        runners = list(_shared.values())
        _shared.clear()
    for runner in runners:
        runner.close()


atexit.register(close_shared_runners)


def _kill(pidfd: int | None) -> None:
    if pidfd is None:
        return
    try:
        signal.pidfd_send_signal(pidfd, signal.SIGKILL)
    except OSError:
        pass  # already gone
