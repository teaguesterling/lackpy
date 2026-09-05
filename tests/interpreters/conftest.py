"""Interpreter tests run with the process cwd inside their own fixture directory.

fledgling >= 0.13 sandboxes the DuckDB connection to a project root and rejects
paths outside it. These tests build fixtures under ``tmp_path``, so every query
against them was rejected:

    IO Error: Failed to initialize file processing:
      Failed to process pattern '/tmp/tmp.../sample.py'

22 failures across test_plucker, test_ast_select and test_pss — and only
test_plucker surfaced that message. The other nine raised bare ``KeyError``s from
indexing an empty result downstream, giving no indication a sandbox was involved.
Deleting this fixture reproduces all 22, nine of them illegibly.

**The chdir is load-bearing, and it also hides something.** cwd == root is exactly
the configuration in which root-versus-cwd bugs are invisible: a relative glob
resolves correctly whatever the root handling does. A suite that only ever runs
this way can go green while a server serving a corpus it is *not* sitting in stays
broken — and that failure presents as "the interpreter is wrong" rather than "the
root is wrong", which is the expensive way round.

So the interpreters also honour ``ExecutionContext.base_dir`` — a field that
already documented itself as "directory the interpreter operates against" and was
being ignored — and ``test_base_dir.py`` exercises that path with cwd deliberately
elsewhere. That test is the one that would catch what this fixture masks; keep
them together.
"""

import pytest


@pytest.fixture(autouse=True)
def _cwd_in_tmp(tmp_path, monkeypatch):
    """Put the process cwd inside the fixture directory for this package.

    Autouse so a test that gains a ``tmp_path`` fixture later does not silently
    rejoin the broken set.
    """
    monkeypatch.chdir(tmp_path)
    yield
