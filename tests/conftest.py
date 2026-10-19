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
