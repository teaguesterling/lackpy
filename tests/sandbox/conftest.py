"""Sandbox test configuration."""

from __future__ import annotations

import shutil

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "nsjail: requires nsjail binary")


def has_nsjail() -> bool:
    return shutil.which("nsjail") is not None


skip_no_nsjail = pytest.mark.skipif(
    not has_nsjail(),
    reason="nsjail binary not found",
)
