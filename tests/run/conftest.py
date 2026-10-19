"""Shared helper for the Task 6 equivalence and gate tests."""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path

import pytest

from lackpy.lang import validate
from lackpy.run.base import ExecutionResult
from lackpy.run.paths import load_policy
from lackpy.run.runner import RestrictedRunner
from lackpy.service import LackpyService

FIXTURES = Path(__file__).resolve().parents[2] / "scripts" / "eval" / "ste_usability" / "fixtures"


class Invalid(AssertionError):
    """The validator rejected a program the test expected to be valid."""


def run_in_lackpy(program: str, workspace: Path, home: Path, params: dict | None = None,
                  program_file: str | None = None) -> ExecutionResult:
    """Run ``program`` through ``LackpyService.run_program`` (parked item 5, Task 7).

    The service validates with ``allowed=set()`` (no kit tools), loads
    ``load_policy(workspace, home=home, managed=None)`` through its loader seam,
    and runs from ``workspace``. ``home`` must be a temporary directory so no real
    ``~/.claude`` settings leak in. The validator verdict is checked here first,
    so a rejected program raises ``Invalid`` rather than giving a failed result.

    ``program_file`` sets ``__file__``. The service has no such parameter, so a
    test that passes one runs the runner directly, with the same policy.
    """
    verdict = validate(program, allowed_names=set())
    if not verdict.valid:
        raise Invalid("; ".join(str(e) for e in verdict.errors))
    if program_file is not None:
        policy = load_policy(workspace, home=home, managed=None)
        old = os.getcwd()
        os.chdir(workspace)
        try:
            return RestrictedRunner().run(program, params=params,
                                          program_file=program_file, path_policy=policy)
        finally:
            os.chdir(old)
    service = LackpyService(
        workspace=workspace,
        path_policy_loader=lambda ws: load_policy(ws, home=home, managed=None))
    return asyncio.run(service.run_program(program, params=params))


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
