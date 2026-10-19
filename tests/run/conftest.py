"""Shared helper for the Task 6 equivalence and gate tests."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from lackpy.lang import validate
from lackpy.run.base import ExecutionResult
from lackpy.run.paths import load_policy
from lackpy.run.runner import RestrictedRunner

FIXTURES = Path(__file__).resolve().parents[2] / "scripts" / "eval" / "ste_usability" / "fixtures"


class Invalid(AssertionError):
    """The validator rejected a program the test expected to be valid."""


def run_in_lackpy(program: str, workspace: Path, home: Path, params: dict | None = None,
                  program_file: str | None = None) -> ExecutionResult:
    """Run ``program`` the way ``LackpyService.run_program`` will, minus the service.

    Validates with ``allowed=set()`` (no kit tools), then runs
    ``RestrictedRunner.run`` with ``path_policy=load_policy(workspace, home=home,
    managed=None)`` from ``workspace`` as the working directory. ``home`` must be
    a temporary directory so no real ``~/.claude`` settings leak in.

    TASK 7 SWITCHES THIS HELPER to ``LackpyService.run_program``: that method has
    no PathPolicy wiring until Task 7 (service.py has no ``load_policy`` today),
    so until then the invariant is measured at the runner, not the service
    (progress.md ruling, Task 6). This is the one place to change.
    """
    verdict = validate(program, allowed_names=set())
    if not verdict.valid:
        raise Invalid("; ".join(str(e) for e in verdict.errors))
    policy = load_policy(workspace, home=home, managed=None)
    old = os.getcwd()
    os.chdir(workspace)
    try:
        return RestrictedRunner().run(program, namespace={}, params=params,
                                      program_file=program_file, path_policy=policy)
    finally:
        os.chdir(old)


@pytest.fixture
def fixtures_copy(tmp_path):
    """A fresh copy of the eval fixtures, plus an empty temporary home."""
    def make(name: str = "ws") -> Path:
        dest = tmp_path / name
        shutil.copytree(FIXTURES, dest)
        return dest
    return make


@pytest.fixture
def tmp_home(tmp_path):
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    return home
