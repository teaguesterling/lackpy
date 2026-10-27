"""OS limits for one run's worker process (Plan 3, Task 4; spec D5).

The worker calls :func:`apply_child_limits` right after the zygote forks it,
before it reads the program. ``RLIMIT_NPROC`` is never set: it was measured to
make ``os.fork()`` fail with EAGAIN, and an unprivileged process cannot raise
it again, while the literate checkpoints (Task 6) fork from a session process.
"""

from __future__ import annotations

import ctypes
import os
import resource
import signal

#: The only environment variables the zygote keeps (each one only when it is set).
ENV_KEEP = ("PATH", "HOME", "LANG", "TZ")

#: The stdout a run may return, in characters; the rest is cut.
STDOUT_CAP = 10_000_000

PR_SET_PDEATHSIG = 1
PR_SET_NO_NEW_PRIVS = 38

_libc = ctypes.CDLL(None, use_errno=True)


def _prctl(option: int, arg: int) -> None:
    if _libc.prctl(option, arg, 0, 0, 0) != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"prctl({option}): {os.strerror(err)}")


def apply_child_limits(*, memory_mb: int | None, cpu_seconds: int | None) -> None:
    """Set the worker's limits; raise ``OSError`` if one cannot be set.

    - ``PR_SET_NO_NEW_PRIVS``: no exec can gain privileges.
    - ``PR_SET_PDEATHSIG(SIGKILL)``: the worker dies with the zygote. The zygote
      is single-threaded, so "the parent thread exited" means the zygote exited.
    - ``RLIMIT_AS`` at ``memory_mb`` MiB, and ``RLIMIT_CPU`` at ``cpu_seconds``
      (soft; the hard limit is one second more), when they are not None.
    - ``RLIMIT_CORE`` at 0, so a crash never writes a core file into the
      workspace, which is the worker's working directory.
    """
    _prctl(PR_SET_NO_NEW_PRIVS, 1)
    _prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if memory_mb is not None:
        size = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (size, size))
    if cpu_seconds is not None:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
