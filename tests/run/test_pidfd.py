"""The pidfd shim (final review C1, gate 9).

python-build-standalone CPython (uv's 3.13.13 and 3.11.15, measured) has no
``os.pidfd_open`` and no ``signal.pidfd_send_signal``, while the kernel has
both syscalls. The shim uses the bindings when they exist, else the syscalls
through libc. A pid-based kill is never the fallback: a reused pid could be hit.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="pidfd is Linux only")


@pytest.fixture
def no_bindings(monkeypatch):
    """Hide the CPython bindings, as on python-build-standalone."""
    monkeypatch.delattr(os, "pidfd_open", raising=False)
    monkeypatch.delattr(signal, "pidfd_send_signal", raising=False)


def _is_pidfd(fd: int) -> bool:
    return os.readlink(f"/proc/self/fd/{fd}") == "anon_inode:[pidfd]"


def test_the_syscall_path_opens_and_signals_a_pidfd(no_bindings):
    from lackpy.run import pidfd

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        fd = pidfd.pidfd_open(child.pid)
        try:
            assert _is_pidfd(fd)
            pidfd.pidfd_send_signal(fd, 0)  # alive: no error
            pidfd.pidfd_send_signal(fd, signal.SIGKILL)
            assert child.wait(5) == -signal.SIGKILL
        finally:
            os.close(fd)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_the_syscall_path_raises_process_lookup_error_for_no_such_pid(no_bindings):
    """The errno maps as the binding's does, so the callers' catches keep their meaning."""
    from lackpy.run import pidfd

    with pytest.raises(ProcessLookupError):
        pidfd.pidfd_open(2 ** 22 + 12345)  # above pid_max's default ceiling


def test_the_syscall_path_signals_a_dead_process_as_an_oserror(no_bindings):
    from lackpy.run import pidfd

    child = subprocess.Popen([sys.executable, "-c", "pass"])
    fd = pidfd.pidfd_open(child.pid)
    try:
        child.wait(5)
        with pytest.raises(ProcessLookupError):
            pidfd.pidfd_send_signal(fd, signal.SIGKILL)
    finally:
        os.close(fd)


def test_the_check_passes_on_the_syscall_path(no_bindings):
    from lackpy.run import pidfd

    pidfd.check()  # no raise: the kernel has pidfd


def test_a_worker_with_no_pidfd_gets_no_program(monkeypatch, tmp_path):
    """An OSError other than ESRCH (here EMFILE) from pidfd_open is a failed run:
    a worker that cannot be killed never gets the program."""
    import errno

    from lackpy.run import process
    from lackpy.run.paths import PathPolicy

    def emfile(pid: int) -> int:
        raise OSError(errno.EMFILE, "Too many open files")

    runner = process.ProcessRunner()
    try:
        runner._start_zygote()  # the check runs with the real pidfd_open
        monkeypatch.setattr(process, "pidfd_open", emfile)
        marker = tmp_path / "ran"
        result = runner.run(f"from pathlib import Path\nPath({str(marker)!r}).write_text('x')",
                            params=None, path_policy=PathPolicy(tmp_path, []),
                            workspace=tmp_path, timeout=5)
    finally:
        runner.close()
    assert not result.success
    assert result.error.startswith("the run process could not be watched"), result.error
    assert not marker.exists()


def test_the_zygote_start_fails_loud_naming_pidfd(monkeypatch):
    """With neither the bindings nor the syscalls, the runner refuses to start."""
    from lackpy.run import pidfd
    from lackpy.run.process import ProcessRunner

    def broken(pid: int) -> int:
        raise OSError(38, "Function not implemented")

    monkeypatch.setattr(pidfd, "pidfd_open", broken)
    runner = ProcessRunner()
    try:
        with pytest.raises(RuntimeError, match="pidfd"):
            runner._start_zygote()
        assert runner._proc is None
    finally:
        runner.close()
