"""pidfd_open and pidfd_send_signal, with or without the CPython bindings (final review C1).

python-build-standalone CPython (the builds ``uv`` installs; 3.13.13 and
3.11.15 measured) is built without ``os.pidfd_open`` and
``signal.pidfd_send_signal``, while the kernel has both syscalls (Linux 5.3).
These functions use the bindings when they exist, else the syscalls through
libc's ``syscall``. The bindings are looked up on each call, not at import.

A pid-based ``os.kill`` is never the fallback: a reused pid could be hit.
:func:`check` raises when neither path works, so the runner fails loud.

Syscall numbers: Linux gives every new syscall from 424 on the same number on
all architectures except alpha, so 434 and 424 hold on x86_64 and aarch64.
"""

from __future__ import annotations

import ctypes
import os
import signal

__all__ = ["check", "pidfd_open", "pidfd_send_signal"]

_NR_PIDFD_SEND_SIGNAL = 424
_NR_PIDFD_OPEN = 434

_libc = ctypes.CDLL(None, use_errno=True)
_syscall = _libc.syscall
_syscall.restype = ctypes.c_long


def _call(*args: int) -> int:
    ret = _syscall(*(ctypes.c_long(a) for a in args))
    if ret == -1:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))  # ESRCH becomes ProcessLookupError, as the binding's
    return int(ret)


def pidfd_open(pid: int) -> int:
    """A pidfd for ``pid``; ``OSError`` (``ProcessLookupError`` for no such pid) on failure."""
    binding = getattr(os, "pidfd_open", None)
    if binding is not None:
        return binding(pid)
    return _call(_NR_PIDFD_OPEN, pid, 0)


def pidfd_send_signal(fd: int, sig: int) -> None:
    """Send ``sig`` to the process of the pidfd ``fd``; ``OSError`` on failure."""
    binding = getattr(signal, "pidfd_send_signal", None)
    if binding is not None:
        binding(fd, sig)
        return
    _call(_NR_PIDFD_SEND_SIGNAL, fd, int(sig), 0, 0)  # siginfo NULL, flags 0


def check() -> None:
    """Raise ``RuntimeError`` naming pidfd when this process cannot open and signal a pidfd."""
    try:
        fd = pidfd_open(os.getpid())
    except OSError as e:
        raise RuntimeError(f"the process runner needs pidfd_open (Linux 5.3 or later): {e}") from None
    try:
        pidfd_send_signal(fd, 0)
    except OSError as e:
        raise RuntimeError(f"the process runner needs pidfd_send_signal "
                           f"(Linux 5.1 or later): {e}") from None
    finally:
        os.close(fd)
