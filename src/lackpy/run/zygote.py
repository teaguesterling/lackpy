"""The zygote: a thread-free process that forks one worker for each run (Plan 3, Task 4; spec D2).

Run as ``python -I -m lackpy.run.zygote <control-fd> [--test-ops]`` by
:class:`lackpy.run.process.ProcessRunner`, never forked from the service (the
service may have threads). The zygote imports the runner, the namespace, the
policy and the validator once, scrubs its environment, and then waits on the
control socket (``SOCK_SEQPACKET``, one message per packet). It never starts
a thread.

Control protocol (parent to zygote):

- ``{"op": "spawn", "memory_mb", "cpu_seconds"}`` with one fd attached, the
  worker's end of a new socketpair. The limits ride on the spawn message because
  the CPU backstop depends on each run's timeout. The zygote forks and answers
  ``{"event": "spawned", "pid"}`` (or ``{"event": "error", "error"}``).
- ``{"op": "spawn_session", "memory_mb", "cpu_seconds"}`` with two fds
  attached: the session root's own stream socket, then the shared
  ``SOCK_SEQPACKET`` channel. The worker serves a literate session
  (``lackpy.run.session_worker``, Task 6); the answer is as for ``spawn``.
- EOF: the parent is gone. The zygote SIGKILLs every live worker, reaps it and
  exits. This, not ``PR_SET_PDEATHSIG``, ties the zygote to the service:
  PDEATHSIG fires when the parent *thread* exits, and the service starts the
  zygote from whatever thread calls ``run``.

The zygote keeps its copy of each worker's socket. When the worker ends, it
reaps it and sends ``{"event": "exit", "pid", "how"}`` on that socket, then
closes it, so the parent learns "signal N" or "exit status N" for a worker that
is not its own child.

Worker protocol (one run): the worker sends ``{"event": "start", "pid"}``,
reads ``{"op": "run", "program", "params", "policy", "workspace", "rules"}``,
and sends ``{"event": "trace", "entry"}`` and ``{"event": "stdout", "text"}``
as they happen, then ``{"event": "done", "result"}``. The test-only op
``{"op": "environ"}`` is served only with ``--test-ops`` (ledger Ruling B).
"""

from __future__ import annotations

import os
import select
import signal
import socket
import sys
from typing import Any

from lackpy.lang.validator import validate
from lackpy.run import wire
from lackpy.run.limits import ENV_KEEP, STDOUT_CAP, apply_child_limits
from lackpy.run import session_worker
# Warm: a session root and its checkpoint forks find the literate interpreter
# already imported (measured: it starts no thread).
import lackpy.interpreters.literate  # noqa: E402,F401
from lackpy.run.paths import PathPolicy
from lackpy.run.runner import RestrictedRunner
from lackpy.run.trace import TraceEntry
# The service's own rule names and failed-result text, so a worker's validation
# reads exactly as the service's. Measured: the import adds about 0.12 s to the
# zygote's start, once, and starts no thread.
from lackpy.service import resolve_rules, validation_error

#: A worker's exit status when its channel to the parent broke (ledger ruling).
EXIT_CHANNEL = 3
#: A worker's exit status after an error in the worker's own code.
EXIT_ERROR = 4
#: A worker's exit status for an op it does not serve.
EXIT_REFUSED = 5

#: Characters of stdout in one message: with ``ensure_ascii`` one character can
#: take 12 bytes, so a chunk stays well under the Reader's 16 MiB line limit.
STDOUT_CHUNK = 256 * 1024
#: How long a worker waits for its op after ``start``.
OP_WAIT = 60.0
#: A run that failed with SystemError, and whose peak virtual size came within this
#: margin of ``RLIMIT_AS``, is reported as the memory cap (a full deque raises SystemError, not MemoryError;
#: measured at 256 MiB: VmPeak 262104 kB for the deque, 245228 kB for a list).
#: The margin is at most an eighth of the cap: a worker starts near 47 MiB, so at
#: a small cap a fixed 32 MiB would call every failure "memory".
MEMORY_MARGIN = 32 * 1024 * 1024


def scrub_environ(env: Any = os.environ) -> None:
    """Keep only ``ENV_KEEP`` in ``env``; a variable that is not set stays unset."""
    for key in list(env):
        if key not in ENV_KEEP:
            del env[key]


# --- the worker -------------------------------------------------------------

class _Channel:
    """The worker's end of its socket. A failed send ends the worker: the parent
    sees the run end without a result (a crash), never a quietly short trace."""

    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.stdout_sent = 0

    def send(self, msg: dict[str, Any]) -> None:
        try:
            wire.send(self.sock, msg)
        except BaseException:
            os._exit(EXIT_CHANNEL)

    def trace(self, entry: TraceEntry) -> None:
        data = entry.to_data()
        for key in ("args", "result"):
            try:
                wire.encode(data[key])
            except wire.WireError:
                data[key] = _text(data[key])
        self.send({"event": "trace", "entry": data})

    def stdout(self, text: str) -> None:
        if self.stdout_sent > STDOUT_CAP:
            return  # the parent cuts at the cap; sending more is waste
        for i in range(0, len(text), STDOUT_CHUNK):
            self.send({"event": "stdout", "text": text[i:i + STDOUT_CHUNK]})
        self.stdout_sent += len(text)


def _text(value: Any) -> str:
    try:
        return repr(value)
    except BaseException:
        return f"<{type(value).__name__}>"


class _WorkerRunner(RestrictedRunner):
    """``RestrictedRunner`` with the worker's hooks on the trace and on ``print``.

    ``run`` is unchanged. ``_run`` receives the trace and the print function that
    ``run`` made; this sets the trace's listener, wraps ``print`` so each call's
    text streams out (and leaves the buffer empty), and notes the exception that
    ended the run, so a ``MemoryError`` can be told from the others.
    """

    def __init__(self, channel: _Channel) -> None:
        self._channel = channel
        self.failure: BaseException | None = None

    def _run(self, program, params, program_file, path_policy, trace, captured, print_fn):  # type: ignore[override]
        trace.listener = self._channel.trace

        def streaming_print(*args: Any, **kwargs: Any) -> None:
            try:
                print_fn(*args, **kwargs)
            finally:
                text = "".join(captured)
                captured.clear()
                if text:
                    self._channel.stdout(text)

        try:
            return super()._run(program, params, program_file, path_policy, trace,
                                captured, streaming_print)
        except BaseException as e:
            self.failure = e
            raise


def _failed(error: str) -> dict[str, Any]:
    return {"success": False, "output": None, "error": error, "variables": {},
            "error_phase": None, "memory": False}


def _near_memory_limit(memory_mb: int | None) -> bool:
    if memory_mb is None:
        return False
    try:
        with open("/proc/self/status", "rb") as f:
            for line in f:
                if line.startswith(b"VmPeak:"):
                    peak = int(line.split()[1]) * 1024
                    limit = memory_mb * 1024 * 1024
                    return peak >= limit - min(MEMORY_MARGIN, limit // 8)
    except (OSError, ValueError, IndexError):
        pass
    return False


def _run_one(channel: _Channel, msg: dict[str, Any], memory_mb: int | None) -> dict[str, Any]:
    try:
        os.chdir(msg["workspace"])
    except OSError as e:
        return _failed(f"cannot enter the workspace: {e.strerror}")
    program = msg["program"]
    try:
        rules = resolve_rules(msg.get("rules"))
    except ValueError as e:
        return _failed(str(e))
    # The runner alone can reach a frame (Task 2 review): validate on every route.
    validation = validate(program, allowed_names=set(), extra_rules=rules)
    if not validation.valid:
        return _failed(validation_error(validation))
    policy = PathPolicy.from_data(msg["policy"])
    runner = _WorkerRunner(channel)
    result = runner.run(program, params=msg.get("params"), path_policy=policy)
    # A MemoryError is always the cap. VmPeak is only consulted for a SystemError
    # (the deque at the cap): VmPeak stays high after a freed allocation, so any
    # other exception is reported as itself (fix round 1).
    memory = isinstance(runner.failure, MemoryError) or (
        isinstance(runner.failure, SystemError) and _near_memory_limit(memory_mb))
    return {"success": result.success, "output": result.output, "error": result.error,
            "variables": result.variables, "error_phase": result.error_phase,
            "memory": memory}


def _send_done(channel: _Channel, result: dict[str, Any]) -> None:
    try:
        line = wire.encode({"event": "done", "result": result})
        if len(line) >= wire.DEFAULT_MAX_LINE:
            raise wire.WireError(f"the result is larger than {wire.DEFAULT_MAX_LINE} bytes")
    except wire.WireError as e:
        line = wire.encode({"event": "done", "result": _failed(
            f"the run's result could not be sent: {e}")})
    except MemoryError:
        line = wire.encode({"event": "done", "result": dict(_failed(""), memory=True)})
    try:
        channel.sock.sendall((line + "\n").encode("ascii"))
    except BaseException:
        os._exit(EXIT_CHANNEL)


def serve(sock: socket.socket, *, test_ops: bool, memory_mb: int | None) -> int:
    """Serve one op on ``sock``; return the worker's exit status."""
    channel = _Channel(sock)
    channel.send({"event": "start", "pid": os.getpid()})
    try:
        msg = wire.Reader(sock).recv(OP_WAIT)
    except (TimeoutError, OSError, wire.WireError):
        return EXIT_CHANNEL
    if msg is None:
        return 0
    op = msg.get("op")
    if op == "environ" and test_ops:
        channel.send({"event": "environ", "environ": dict(os.environ)})
        return 0
    if op != "run":
        return EXIT_REFUSED
    try:
        result = _run_one(channel, msg, memory_mb)
    except MemoryError:
        result = dict(_failed(""), memory=True)
    _send_done(channel, result)
    return 0


def _child(run_fd: int, control: socket.socket, children: dict[int, tuple[int, socket.socket]],
           msg: dict[str, Any], test_ops: bool, zygote_pid: int,
           ctl_fd: int | None = None) -> None:
    """The forked worker. Never returns."""
    code = EXIT_ERROR
    try:
        # Hold no other channel: a worker must not write into the control
        # socket or into a concurrent run's socket.
        control.close()
        for pidfd, (_, other) in children.items():
            other.close()
            os.close(pidfd)
        sock = socket.socket(fileno=run_fd)
        memory_mb = msg.get("memory_mb")
        apply_child_limits(memory_mb=memory_mb, cpu_seconds=msg.get("cpu_seconds"))
        if os.getppid() != zygote_pid:
            os._exit(EXIT_ERROR)  # the zygote died before PDEATHSIG was set
        if ctl_fd is None:
            code = serve(sock, test_ops=test_ops, memory_mb=memory_mb)
        else:
            code = session_worker.serve_session(_Channel(sock), socket.socket(fileno=ctl_fd),
                                                test_ops=test_ops)
    except BaseException:
        pass
    finally:
        os._exit(code)


# --- the zygote -------------------------------------------------------------

def _how(status: int) -> str:
    code = os.waitstatus_to_exitcode(status)
    return f"signal {-code}" if code < 0 else f"exit status {code}"


def _reply(control: socket.socket, msg: dict[str, Any]) -> None:
    control.send(wire.encode(msg).encode("ascii"))


def main(argv: list[str]) -> int:
    if not argv or not argv[0].isdigit():
        print("usage: python -I -m lackpy.run.zygote <control-fd> [--test-ops]", file=sys.stderr)
        return 2
    test_ops = "--test-ops" in argv[1:]
    scrub_environ()
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the parent ends the zygote, by EOF
    control = socket.socket(fileno=int(argv[0]))
    zygote_pid = os.getpid()
    children: dict[int, tuple[int, socket.socket]] = {}  # pidfd -> (pid, run socket)
    poller = select.poll()
    poller.register(control.fileno(), select.POLLIN)

    while True:
        for fd, _ in poller.poll():
            if fd == control.fileno():
                try:
                    data, fds, _, _ = socket.recv_fds(control, 65536, 2)
                except OSError:
                    data, fds = b"", []
                if not data:
                    for f in fds:
                        os.close(f)
                    _shutdown(children)
                    return 0
                _spawn(control, data, fds, children, poller, test_ops, zygote_pid)
            elif fd in children:
                pid, sock = children.pop(fd)
                poller.unregister(fd)
                _, status = os.waitpid(pid, 0)
                os.close(fd)
                try:
                    sock.setblocking(False)  # a full buffer must never stall the zygote
                    sock.send((wire.encode({"event": "exit", "pid": pid,
                                            "how": _how(status)}) + "\n").encode("ascii"))
                except OSError:
                    pass
                sock.close()


def _spawn(control: socket.socket, data: bytes, fds: list[int],
           children: dict[int, tuple[int, socket.socket]], poller: Any,
           test_ops: bool, zygote_pid: int) -> None:
    try:
        msg = wire.decode(data.decode("ascii"))
    except (wire.WireError, UnicodeDecodeError):
        msg = None
    op = msg.get("op") if isinstance(msg, dict) else None
    want = {"spawn": 1, "spawn_session": 2}.get(op)  # type: ignore[arg-type]
    if want is None or len(fds) != want:
        for f in fds:
            os.close(f)
        _reply(control, {"event": "error",
                         "error": "expected a spawn op with one fd, or spawn_session with two"})
        return
    run_fd = fds[0]
    ctl_fd = fds[1] if want == 2 else None
    try:
        pid = os.fork()
    except OSError as e:
        for f in fds:
            os.close(f)
        _reply(control, {"event": "error", "error": f"fork failed: {e.strerror}"})
        return
    if pid == 0:
        _child(run_fd, control, children, msg, test_ops, zygote_pid, ctl_fd)
    if ctl_fd is not None:
        os.close(ctl_fd)  # the zygote keeps only the root's socket, for the exit event
    pidfd = os.pidfd_open(pid)
    children[pidfd] = (pid, socket.socket(fileno=run_fd))
    poller.register(pidfd, select.POLLIN)
    _reply(control, {"event": "spawned", "pid": pid})


def _shutdown(children: dict[int, tuple[int, socket.socket]]) -> None:
    for pidfd, (pid, sock) in children.items():
        try:
            signal.pidfd_send_signal(pidfd, signal.SIGKILL)
        except OSError:
            pass
    for pidfd, (pid, sock) in children.items():
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass
        os.close(pidfd)
        sock.close()
    children.clear()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
