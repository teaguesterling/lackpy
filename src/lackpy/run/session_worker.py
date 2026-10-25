"""The literate session worker: a checkpoint fork at the top of each cell (Plan 3, Task 6; spec D6, D7).

The zygote forks a session worker for ``{"op": "spawn_session"}``. The whole
literate interpreter runs in it (ledger Ruling D): the kernel's state (the
ledger, the versions, the dependency graph, the rendered parts, the holes) lives
in the same coroutine as the namespace, so a fork checkpoints all of it at once.

At the top of each cell, the kernel calls the hook that ``context.config
["checkpoint"]`` holds. The hook forks:

- The child returns None and runs the cell. It is now the deepest process.
- The parent becomes the checkpoint for that cell. It waits for the child.
  When the child is gone, it reads one instruction, forks a new child for it,
  and the new child returns the matching :class:`Resume` (Task 7). So the
  checkpoint still holds the state from before its cell, for the next retry.

A cell that does not time out goes on to the next cell's hook, so it is the
checkpoint for that cell (D6: "C1 becomes S1"). Every process stays alive while
the session is open, unless the parent drops the oldest checkpoints
(``max_checkpoints``): each one waits on its child, and the deepest one runs.
A prose cell takes no checkpoint of its own (Task 7): the hook runs at the
start of each segment, a run of prose cells and the cell after them.

Channels (the protocol decision for Task 6). Each process has its **own**
stream socket to the parent. The checkpoint makes a new socketpair for each
child, keeps neither end (the child keeps one, the parent gets the other), and
reads its instruction from its own socket. So:

- A cell killed in the middle of a message leaves a partial line only on its
  own socket, which the parent drains and closes. No later message shares it.
- An instruction can reach only its checkpoint: no other process holds that
  socket. A cell that is still dying can never read it.
- The shared channel (``SOCK_SEQPACKET``) carries only ``cell_start`` packets,
  each with the parent's end of the new socket attached. A packet is atomic, so
  a killed sender leaves no partial packet. A packet from a process the parent
  no longer follows is dropped, and its socket closed.

Messages, worker to parent, on a process's own socket: ``start`` (the root
only), ``trace`` (each entry as it happens, start entries included),
``cell_end`` (this process's output for its cell is complete; the
``cell_start`` for the next cell follows on the shared channel, or ``done`` on
this socket), ``done``. The parent stops the cell's deadline at ``cell_end``;
from there each host step has its own short bound.
On the shared channel: ``{"event": "cell_start", "index", "pid", "from"}``.

Messages, parent to worker: ``submit`` to the root; to a woken checkpoint,
``{"op": "timeout", "seconds", "entries"}`` (the stopped cell's streamed entries)
or ``{"op": "resume", "document"}`` (Task 7). At EOF a checkpoint exits, and
so does a waiting one when the zygote is gone (``_Hook`` has the lifetime rule).

No thread is ever started here: the hook refuses to fork when there is more
than one.
"""

from __future__ import annotations

import asyncio
import os
import select
import signal
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any

from . import wire
from .limits import PR_SET_PDEATHSIG, _prctl
from .trace import TraceEntry

#: Exit statuses, as the zygote's.
EXIT_CHANNEL = 3
EXIT_ERROR = 4
EXIT_REFUSED = 5

#: How long the root waits for ``submit`` after ``start``.
OP_WAIT = 60.0


@dataclass(frozen=True)
class Resume:
    """What a checkpoint's child left it to do.

    ``kind`` is ``"timeout"``: the kernel reifies the cell as timed out
    (``seconds`` was the limit; ``entries`` is what the stopped cell streamed)
    and goes to the next cell. Or ``"resume"``: the kernel parses ``document``
    and continues its loop at the same index over the new document's cells.
    """

    kind: str
    document: str | None = None
    seconds: float = 0.0
    entries: tuple[TraceEntry, ...] = ()


def _failed(error: str) -> dict[str, Any]:
    return {"success": False, "output": None, "stdout": "", "error": error,
            "variables": {}, "error_phase": None}


class _Hook:
    """``context.config["checkpoint"]``: fork, and return a Resume in the checkpoint.

    A checkpoint never goes on itself (Task 7). For each instruction it forks
    again, and the new child carries the instruction out, so the checkpoint
    still holds the state from before its cell for the next retry. A child
    forked for ``resume`` runs the cell itself: the kernel calls the hook again
    at the same index, and that call returns None without a fork.

    Lifetime (Task 7). A running process dies with its parent
    (``PR_SET_PDEATHSIG``, set again at each fork). A checkpoint clears its own
    PDEATHSIG while it waits: the parent may drop the oldest checkpoint
    (``max_checkpoints``), and the checkpoints after it must live on. A waiting
    checkpoint exits instead when its socket reaches EOF (the session is
    closed, or the service is gone) or when the zygote is gone.

    Each child gets a private epoll instance in place of the event loop's, which
    it shares with every ancestor at the fork. The child never uses the loop
    (the rule above ``checkpoint`` in the kernel), but its ``asyncio.run``
    teardown does: measured, ``loop.close()`` removed the ancestors' self-pipe
    registration from the shared instance.
    """

    def __init__(self, channel: Any, ctl: socket.socket, exit_at: int | None = None,
                 handoff_delay: float = 0.0) -> None:
        self.channel = channel  # zygote._Channel; ``.sock`` is this process's own socket
        self.ctl = ctl
        self.exit_at = exit_at  # test-only: exit after cell_end at this index
        self.handoff_delay = handoff_delay  # test-only: sleep after cell_end, before the fork
        self.resumed = False  # forked for a resume: the next call runs the cell itself
        self.epfd: int | None = None  # the event loop's epoll fd (found in the root)
        self.zygote: int | None = None  # a pidfd of the zygote

    def __call__(self, index: int) -> Resume | None:
        if self.resumed:
            self.resumed = False  # the parent already has this process's cell_start
            return None
        if threading.active_count() != 1:
            os._exit(EXIT_ERROR)  # a fork with threads is unsafe; never run on
        if self.epfd is None:  # the first call, in the root: its loop is running
            selector = getattr(asyncio.get_running_loop(), "_selector", None)
            self.epfd = selector.fileno() if selector is not None else -1
            try:
                self.zygote = os.pidfd_open(os.getppid())
            except OSError:
                os._exit(EXIT_ERROR)
        # Everything this process sent for its cell is on its socket before the
        # next cell_start reaches the shared channel.
        self.channel.send({"event": "cell_end", "index": index})
        if index == self.exit_at:
            os._exit(EXIT_ERROR)  # test-only (--test-ops): a death before cell_start
        if self.handoff_delay:
            time.sleep(self.handoff_delay)  # test-only (--test-ops): a slow handoff
        pid = self._fork(index)
        if pid is None:
            return None
        while True:
            self._wait(pid)
            resume = self._instruction()
            pid = self._fork(index)
            if pid is None:
                self.resumed = resume.kind == "resume"
                return resume

    def _fork(self, index: int) -> int | None:
        """Fork a child for cell ``index`` and hand its socket to the parent; None in the child."""
        ours, theirs = socket.socketpair()
        me = os.getpid()
        pid = os.fork()
        if pid == 0:
            try:
                # PDEATHSIG is cleared at fork: set it again, then check the
                # parent did not die before it was set.
                _prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
                if os.getppid() != me:
                    os._exit(EXIT_ERROR)
                ours.close()
                self.channel.sock.close()  # the checkpoint's socket is its own
                self.channel.sock = theirs
                if self.epfd is not None and self.epfd >= 0:
                    private = select.epoll()
                    os.dup2(private.fileno(), self.epfd)
                    private.close()
            except BaseException:
                os._exit(EXIT_ERROR)
            return None
        theirs.close()
        packet = wire.encode({"event": "cell_start", "index": index, "pid": pid,
                              "from": me}).encode("ascii")
        try:
            socket.send_fds(self.ctl, [packet], [ours.fileno()])
        except OSError:
            os.kill(pid, signal.SIGKILL)
            os._exit(EXIT_CHANNEL)
        finally:
            ours.close()
        return pid

    def _wait(self, pid: int) -> None:
        """Wait until the child ``pid`` is gone; exit when the session is."""
        try:
            _prctl(PR_SET_PDEATHSIG, 0)
            child = os.pidfd_open(pid)
        except OSError:
            os._exit(EXIT_ERROR)
        sock = self.channel.sock
        try:
            poller = select.poll()
            poller.register(child, select.POLLIN)
            poller.register(sock.fileno(), select.POLLIN)
            if self.zygote is not None:
                poller.register(self.zygote, select.POLLIN)
            while True:
                ready = {fd for fd, _ in poller.poll()}
                if child in ready:
                    break
                if self.zygote in ready:
                    os._exit(0)  # the zygote is gone: so is the session
                if sock.fileno() in ready:
                    try:
                        eof = sock.recv(1, socket.MSG_PEEK) == b""
                    except OSError:
                        eof = True
                    if eof:
                        os._exit(0)  # the session is closed
                    # An instruction: the parent stopped the child first, so
                    # read it once the child is gone.
                    poller.unregister(sock.fileno())
        finally:
            os.close(child)
        os.waitpid(pid, 0)

    def _instruction(self) -> Resume:
        try:
            msg = wire.Reader(self.channel.sock).recv(None)
        except (OSError, wire.WireError):
            os._exit(EXIT_CHANNEL)
        if msg is None:
            os._exit(0)  # the session is closed
        op = msg.get("op")
        try:
            if op == "timeout":
                return Resume("timeout", seconds=float(msg["seconds"]),
                              entries=tuple(TraceEntry.from_data(d) for d in msg["entries"]))
            if op == "resume" and isinstance(msg.get("document"), str):
                return Resume("resume", document=msg["document"])
        except (KeyError, TypeError, ValueError):
            pass
        os._exit(EXIT_REFUSED)


def _submit(channel: Any, ctl: socket.socket, msg: dict[str, Any], test_ops: bool) -> dict[str, Any]:
    # The service's own rule names and validation text, so a worker's
    # validation reads as the service's (imported here: the zygote has them).
    from ..interpreters.base import ExecutionContext
    from ..interpreters.literate import LiterateInterpreter
    from ..service import resolve_rules
    from .runner import _is_data
    from .paths import PathPolicy

    try:
        os.chdir(msg["workspace"])
    except OSError as e:
        return _failed(f"cannot enter the workspace: {e.strerror}")
    try:
        rules = resolve_rules(msg.get("rules"))
    except ValueError as e:
        return _failed(str(e))
    config = dict(msg.get("config") or {})
    block, partial = config.pop("test_block_writes", False), config.pop("test_partial_frame", False)
    exit_at = config.pop("test_exit_in_hook", None)
    if not test_ops or type(exit_at) is not int:
        exit_at = None
    # Test-only delays (--test-ops): between cell_end and the fork, and after
    # the document's last cell, before done.
    handoff_delay = config.pop("test_handoff_delay", 0.0) if test_ops else 0.0
    done_delay = config.pop("test_done_delay", 0.0) if test_ops else 0.0
    config.pop("test_handoff_delay", None)
    config.pop("test_done_delay", None)
    if test_ops and (block or partial):
        # Ruling C: block inside the write, so the parent kills the cell there.
        # With ``test_partial_frame``, first send half a message on the cell's
        # socket, as a cell killed in the middle of a send would leave it.
        from .paths import proxy

        def stall() -> None:
            if partial:
                channel.sock.sendall(b'{"event":"trace","entry":{"step":')
            time.sleep(3600)
        proxy._write_hook = stall
    config["path_policy"] = PathPolicy.from_data(msg["policy"])
    if rules:
        config["extra_rules"] = rules
    config["trace_listener"] = channel.trace
    config["checkpoint"] = _Hook(channel, ctl, exit_at, float(handoff_delay))
    # The end of a cell's run when no hook follows it (the last cell, a re-run):
    # what comes after is host code, not that cell's time (fix round 1).
    config["cell_done"] = lambda: channel.send({"event": "cell_end", "index": None})
    document = msg["document"]
    ctx = ExecutionContext(base_dir=str(msg["workspace"]), params=msg.get("params") or {},
                           config=config)
    interp = LiterateInterpreter()
    validation = interp.validate(document, ctx)
    if not validation.valid:
        return _failed(f"Validation failed: {'; '.join(validation.errors)}")
    result = asyncio.run(interp.execute(document, ctx))
    if done_delay:
        time.sleep(float(done_delay))
    meta = result.metadata
    variables = meta.get("variables", {})
    return {
        "success": result.success,
        # run_program's rule: the last expression's value, only for a run that
        # succeeded (review I1); the interpreter made it data-safe.
        "output": meta.get("value") if result.success else None,
        "stdout": str(result.output) if result.output else "",
        "error": result.error,
        "variables": {k: v for k, v in variables.items() if _is_data(v)},
        "error_phase": meta.get("error_phase"),
    }


def _send_done(channel: Any, result: dict[str, Any]) -> None:
    try:
        line = wire.encode({"event": "done", "result": result})
        if len(line) >= wire.DEFAULT_MAX_LINE:
            raise wire.WireError(f"the result is larger than {wire.DEFAULT_MAX_LINE} bytes")
    except wire.WireError as e:
        line = wire.encode({"event": "done", "result": _failed(
            f"the document's result could not be sent: {e}")})
    try:
        channel.sock.sendall((line + "\n").encode("ascii"))
    except BaseException:
        os._exit(EXIT_CHANNEL)


def serve_session(channel: Any, ctl: socket.socket, *, test_ops: bool) -> int:
    """Serve one session on ``channel`` (the root's own socket) and ``ctl``; return the exit status.

    Returns in the process that finished the document, after the parent closes
    its socket. A checkpoint never returns: it exits in the hook.
    """
    channel.send({"event": "start", "pid": os.getpid()})
    try:
        msg = wire.Reader(channel.sock).recv(OP_WAIT)
    except (TimeoutError, OSError, wire.WireError):
        return EXIT_CHANNEL
    if msg is None:
        return 0
    if msg.get("op") != "submit":
        return EXIT_REFUSED
    try:
        result = _submit(channel, ctl, msg, test_ops)
    except MemoryError:
        result = _failed("memory: the document's session process ran out of memory")
    _send_done(channel, result)
    # The newest checkpoint: wait for the next instruction (Task 7). At EOF, exit.
    try:
        wire.Reader(channel.sock).recv(None)
    except (OSError, wire.WireError):
        pass
    return 0
