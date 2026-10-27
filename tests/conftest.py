"""Shared test fixtures for lackpy."""

import pytest


@pytest.fixture(autouse=True)
def _isolated_path_policy(monkeypatch, tmp_path_factory):
    """Keep every test away from the real ``~/.claude`` and ``/etc/claude-code``.

    ``LackpyService`` and ``PythonInterpreter`` call ``load_policy(workspace)`` on
    each run when no loader is given. That default reads the user's and the
    managed settings files. Here it reads an empty temporary home and no managed
    file instead. A test that passes its own loader is not affected.
    """
    from lackpy.run.paths import load_policy

    home = tmp_path_factory.mktemp("home")
    (home / ".claude").mkdir()

    def isolated(workspace, **_):
        return load_policy(workspace, home=home, managed=None)

    monkeypatch.setattr("lackpy.service.load_policy", isolated, raising=False)
    monkeypatch.setattr("lackpy.interpreters.python.load_policy", isolated, raising=False)


@pytest.fixture(autouse=True)
def _close_process_runners(monkeypatch):
    """Stop every run zygote a test started for itself (Plan 3 Task 5).

    Each ``ProcessRunner`` made during the test is closed at its teardown,
    except the process's shared runners (``shared_runner``, which every
    ``LackpyService`` uses): those live for the session and
    ``_no_zygote_left`` closes them at its end.
    """
    from lackpy.run import process
    from lackpy.run.process import ProcessRunner

    made: list[ProcessRunner] = []
    original = ProcessRunner.__init__

    def tracking(self, *args, **kwargs):
        original(self, *args, **kwargs)
        made.append(self)

    monkeypatch.setattr(ProcessRunner, "__init__", tracking)
    yield
    shared = {id(r) for r in process._shared.values()}
    for runner in made:
        if id(runner) not in shared:
            runner.close()


def _live_zygotes() -> list[int]:
    """Live (not zombie) ``lackpy.run.zygote`` processes below this pytest process."""
    import os
    from pathlib import Path

    parents: dict[int, int] = {}
    states: dict[int, str] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            stat = Path(f"/proc/{entry}/stat").read_text()
        except OSError:
            continue
        fields = stat[stat.rindex(")") + 2:].split()
        states[int(entry)], parents[int(entry)] = fields[0], int(fields[1])
    found: set[int] = set()
    frontier = {os.getpid()}
    while frontier:
        frontier = {p for p, pp in parents.items() if pp in frontier and p not in found}
        found |= frontier
    live = []
    for pid in sorted(found):
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            continue
        if b"lackpy.run.zygote" in cmdline and states.get(pid) != "Z":
            live.append(pid)
    return live


@pytest.fixture(autouse=True, scope="session")
def _no_zygote_left():
    """Fail the session when a run zygote is still alive at its end (Plan 3 Task 5).

    The shared runners are closed first: they are the only ones meant to
    outlive a test.
    """
    yield
    import sys
    from lackpy.run.process import close_shared_runners
    close_shared_runners()
    if sys.platform != "linux":
        return
    live = _live_zygotes()
    assert not live, f"run zygotes left alive at the end of the session: {live}"
