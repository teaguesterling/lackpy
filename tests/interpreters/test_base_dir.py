"""``ExecutionContext.base_dir`` must work with the process cwd somewhere else.

This is the test the package conftest cannot be: it deliberately does NOT sit in
the corpus. fledgling >= 0.13 sandboxes the DuckDB connection to a project root,
and if the interpreter ignores ``base_dir`` that root falls back to the process
cwd — so reading a corpus you are not sitting in fails with

    IO Error: Failed to initialize file processing:
      Failed to process pattern '<path>'

naming neither the root nor the reason. That is the shape an MCP server hits,
because a server serves a corpus it is not chdir'd into.

The conftest's chdir makes cwd == root for every other test in this package,
which is convenient and which also makes this class of bug invisible by
construction. These tests are the counterweight.
"""

import os
from pathlib import Path

import pytest

from lackpy.interpreters.base import ExecutionContext, run_interpreter
from lackpy.interpreters.plucker import PluckerInterpreter

SOURCE = "def greet():\n    pass\n\n\ndef double(x):\n    return x * 2\n"


@pytest.fixture
def corpus(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("corpus")
    (d / "sample.py").write_text(SOURCE)
    return d


@pytest.fixture
def elsewhere(tmp_path_factory, monkeypatch) -> Path:
    """A cwd that is deliberately not the corpus."""
    d = tmp_path_factory.mktemp("elsewhere")
    monkeypatch.chdir(d)
    return d


@pytest.mark.asyncio
async def test_base_dir_is_honoured_from_another_cwd(corpus, elsewhere):
    assert Path(os.getcwd()) == elsewhere, "precondition: cwd is not the corpus"
    interp = PluckerInterpreter()
    ctx = ExecutionContext(base_dir=corpus)
    result = await run_interpreter(
        interp, f'source("{corpus / "sample.py"}").find(".fn").count()', ctx)
    assert result.success, result.error
    assert result.output == 2


@pytest.mark.asyncio
async def test_without_base_dir_the_sandbox_rejects(corpus, elsewhere):
    """The failure this guards against, asserted so the guard cannot rot.

    With no base_dir the root falls back to cwd, which does not contain the
    corpus, and the sandbox refuses. If this ever starts passing, either the
    sandbox changed or a default crept in — both worth knowing.
    """
    interp = PluckerInterpreter()
    ctx = ExecutionContext(base_dir=elsewhere)
    result = await run_interpreter(
        interp, f'source("{corpus / "sample.py"}").find(".fn").count()', ctx)
    assert not result.success
    assert "IO Error" in str(result.error) or "pattern" in str(result.error)
