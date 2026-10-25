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

import math
import os
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
from .trace import Trace, TraceEntry

__all__ = ["ProcessRunner", "STDOUT_CAP"]

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
                self.entries.append(TraceEntry.from_data(msg["entry"]))
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

    def _spawn(self, worker_end: socket.socket, cpu_seconds: int) -> int:
        """Ask the zygote for a worker on ``worker_end``; start or restart it as needed."""
        request = wire.encode({"op": "spawn", "memory_mb": self.memory_mb,
                               "cpu_seconds": cpu_seconds}).encode("ascii")
        with self._lock:
            for _ in range(2):
                if self._proc is None or self._proc.poll() is not None:
                    self._stop_zygote()
                    self._start_zygote()
                assert self._control is not None
                try:
                    socket.send_fds(self._control, [request], [worker_end.fileno()])
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
        """
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


def _kill(pidfd: int | None) -> None:
    if pidfd is None:
        return
    try:
        signal.pidfd_send_signal(pidfd, signal.SIGKILL)
    except OSError:
        pass  # already gone
