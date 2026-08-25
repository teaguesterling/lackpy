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

# pluckit is an optional extra and CI installs only .[dev], so these skip there --
# matching tests/interpreters/test_plucker.py. Importing PluckerInterpreter is
# safe without it (the pluckit import lives inside the source() closure); running
# a program is not, and the failure is an unhelpful "requires pluckit" assertion
# rather than a skip.
pytest.importorskip("pluckit")

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


def _fledgling_installed() -> bool:
    """Whether the DuckDB extension that actually enforces the root is present.

    pluckit passes ``root=repo`` down, but the REJECTION is fledgling's. It is a
    DuckDB community extension, not a pip package, so no requirements file
    expresses it and `importlib.metadata` cannot see it -- checking for a
    `fledgling` python distribution reports "missing" on a machine where it is
    installed and working. Ask DuckDB.
    """
    try:
        import duckdb
        rows = duckdb.connect().execute(
            "select 1 from duckdb_extensions() "
            "where extension_name = 'fledgling' and installed"
        ).fetchall()
        return bool(rows)
    except Exception:
        return False


@pytest.mark.skipif(not _fledgling_installed(),
                    reason="rejection is enforced by the fledgling DuckDB extension")
@pytest.mark.asyncio
async def test_without_base_dir_the_sandbox_rejects(corpus, elsewhere):
    """The failure this guards against, asserted so the guard cannot rot.

    With a root that does not contain the corpus the sandbox refuses. Skipped
    where fledgling is absent: without it every read succeeds regardless of
    root, so an unguarded version of this test asserts a rejection that cannot
    happen and fails for a reason that has nothing to do with lackpy. That is
    exactly what it did.
    """
    interp = PluckerInterpreter()
    ctx = ExecutionContext(base_dir=elsewhere)
    result = await run_interpreter(
        interp, f'source("{corpus / "sample.py"}").find(".fn").count()', ctx)
    assert not result.success
    assert "IO Error" in str(result.error) or "pattern" in str(result.error)


# The two tests above go through PythonInterpreter.execute, which chdirs into
# context.base_dir before running anything -- so they pass even with the repo=
# forwarding deleted, and do not actually exercise it. These do: they build the
# kit directly, outside any chdir, which is the shape an MCP server has.


def test_repo_reaches_plucker_without_a_chdir(corpus, elsewhere):
    """The forwarding itself, with nothing else standing in for it.

    Asserts against ``_ctx.repo`` rather than behaviour on purpose: the
    behavioural difference needs the fledgling extension, and this must stay
    meaningful without it. ``Plucker`` accepts ``repo=`` but does not expose it
    as an attribute -- an earlier version of this test asserted ``plucker.repo``
    and simply raised AttributeError.
    """
    from lackpy.interpreters.plucker import _build_plucker_kit

    tools = _build_plucker_kit(None, [], base_dir=corpus)
    plucker = tools.callables["source"]((corpus / "sample.py").read_text())
    assert Path(plucker._ctx.repo).resolve() == corpus.resolve()


def test_relative_base_dir_survives_the_interpreters_chdir(tmp_path_factory,
                                                            monkeypatch):
    """A relative base_dir must resolve once, not once per layer.

    The chdir is reproduced explicitly because it is the whole mechanism:
    PythonInterpreter.execute does ``os.chdir(context.base_dir)`` BEFORE the
    program runs, so a relative path stringified inside the source() closure is
    re-resolved against the new cwd -- 'corpus' from parent/ becoming
    parent/corpus/corpus. Building the kit without that chdir passes whether or
    not the path was resolved, which is why the first version of this test
    proved nothing.

    ExecutionContext.base_dir is a plain Path with no normalization, and callers
    (ctl.py, StagedDslStrategy) do pass relative values.
    """
    import os

    from lackpy.interpreters.plucker import _build_plucker_kit

    parent = tmp_path_factory.mktemp("parent")
    (parent / "corpus").mkdir()
    (parent / "corpus" / "sample.py").write_text(SOURCE)
    monkeypatch.chdir(parent)

    tools = _build_plucker_kit(None, [], base_dir=Path("corpus"))
    monkeypatch.chdir(parent / "corpus")   # what the interpreter does next
    plucker = tools.callables["source"](SOURCE)
    assert Path(plucker._ctx.repo).resolve() == (parent / "corpus").resolve(), (
        "relative base_dir was re-resolved against the interpreter's new cwd"
    )
