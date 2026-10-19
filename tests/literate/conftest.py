"""Shared fixtures for literate interpreter tests.

Every kernel here runs its cells through a ``CellSession`` (literate stepy
Task 2): there is no host-exec kernel to build. ``make_kernel()`` gives a
session with no path policy (``pathlib`` refuses); ``policy_kernel`` gives one
whose policy is a temporary workspace with an empty temporary home, so no
real ``~/.claude`` settings are read.
"""

from __future__ import annotations

import pytest

from lackpy.interpreters.literate.kernel.lightweight import LightweightKernel
from lackpy.run.cells import CellSession
from lackpy.run.paths import load_policy


def make_kernel(path_policy=None, **kwargs) -> LightweightKernel:
    """A kernel over a fresh ``CellSession`` with ``path_policy``."""
    return LightweightKernel(CellSession(path_policy=path_policy), **kwargs)


@pytest.fixture
def tmp_home(tmp_path):
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    return home


@pytest.fixture
def tmp_policy(tmp_path, tmp_home):
    """A ``PathPolicy`` for ``tmp_path / "ws"`` (created), with a temporary home."""
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    return load_policy(ws, home=tmp_home, managed=None)
