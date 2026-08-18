"""Interpreter tests run with the process cwd inside their own fixture directory.

fledgling >= 0.13 sandboxes the DuckDB connection to the project root — a
deliberate security fix, not a regression. These tests build fixtures under
``tmp_path``, which is outside any project root, so every query against them was
rejected:

    IO Error: Failed to initialize file processing:
      Failed to process pattern '/tmp/tmp.../sample.py'

That surfaced as 22 failures across test_plucker, test_ast_select and test_pss —
IO errors where the interpreter surfaced the message, and bare ``KeyError``s
where an empty result was indexed downstream. The same programs pass against a
file inside the repo, and pass against the fixture once cwd is the fixture
directory, which is what identifies the cause as the sandbox rather than the
interpreters.

Auto-used so a test that gains a ``tmp_path`` fixture later does not silently
rejoin the broken set.
"""

import os

import pytest


@pytest.fixture(autouse=True)
def _cwd_in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield
    # monkeypatch restores cwd; nothing else to undo.
    assert os.getcwd() is not None
