"""Spec E1: a document stops at its first failure.

A failure is an exception, an unbound name, the effect-ceiling gate, a
timeout, or a prose expression that raises. Later cells do not run. The
run fails, and its error names the cell (index from 0, and its first line)
and the cause. A refusal ends the document, as before (D4).

The in-process interpreter runs these documents. A timeout needs the
session worker, so its tests are in ``test_checkpoint.py``.
"""

import pytest

from lackpy.interpreters.base import ExecutionContext
from lackpy.interpreters.literate import LiterateInterpreter
from lackpy.lang.grader import Grade
from lackpy.run.paths import load_policy


@pytest.fixture
def context(tmp_path, tmp_home):
    return ExecutionContext(base_dir=tmp_path, config={
        "path_policy": load_policy(tmp_path, home=tmp_home, managed=None)})


def lackpy_doc(*cells: str) -> str:
    """A literate document of ```lackpy fenced cells, one per argument."""
    return "\n\n".join(f"```lackpy\n{body}\n```" for body in cells)


def cause(result) -> str:
    """The error text after the ``cell {i} ({first line}): `` prefix."""
    return result.error.split("): ", 1)[1]


async def run(context, doc):
    return await LiterateInterpreter().execute(doc, context)


@pytest.mark.asyncio
async def test_a_raising_cell_stops_the_document(context):
    result = await run(context, lackpy_doc("a = 1", "1/0", 'print("later")'))
    assert not result.success
    assert "later" not in (result.output or "")
    assert result.error.startswith("cell 1 (1/0): ")
    assert "ZeroDivisionError" in result.error
    assert result.metadata["error_phase"] is None


@pytest.mark.asyncio
async def test_an_unbound_name_stops_the_document(context):
    result = await run(context, lackpy_doc("x = y + 1", 'print("later")'))
    assert not result.success
    assert result.error.startswith("cell 0 (x = y + 1): ")
    assert "y" in cause(result)
    assert "later" not in (result.output or "")


@pytest.mark.asyncio
async def test_a_gated_cell_stops_the_document(tmp_path):
    context = ExecutionContext(base_dir=tmp_path, config={"grade_ceiling": Grade(1, 1)})
    doc = "```lackpy @write(out.py)\nvalue = 1\n```\n\n" + lackpy_doc('print("later")')
    result = await run(context, doc)
    assert not result.success
    assert result.error.startswith("cell 0 (value = 1): ")
    assert "effect ceiling exceeded" in cause(result)
    assert "later" not in (result.output or "")
    assert not (tmp_path / "out.py").exists()


@pytest.mark.asyncio
async def test_a_raising_prose_expression_stops_the_document(context):
    doc = "The ratio is {abs(1/0)}.\n\n" + lackpy_doc('print("later")')
    result = await run(context, doc)
    assert not result.success
    assert result.error.startswith("cell 0 (The ratio is {abs(1/0)}.): ")
    assert "ZeroDivisionError" in cause(result)
    assert "later" not in (result.output or "")


@pytest.mark.asyncio
async def test_a_refusal_still_ends_the_document(tmp_path):
    # With no path policy, ``pathlib`` is refused (fail closed).
    context = ExecutionContext(base_dir=tmp_path)
    doc = lackpy_doc("before = 1", "from pathlib import Path\nt = 1", 'print("later")')
    result = await run(context, doc)
    assert not result.success
    assert result.metadata["error_phase"] == "refused"
    assert "not allowed in lackpy" in result.error
    assert "later" not in (result.output or "")


@pytest.mark.asyncio
async def test_cells_before_the_failure_keep_their_bindings(context):
    # ``c = 3`` comes after the failure: without the stop it would bind.
    result = await run(context, lackpy_doc("a = 1", "1/0", "c = 3"))
    assert not result.success
    assert result.metadata["variables"] == {"a": 1}


@pytest.mark.asyncio
async def test_a_rebind_in_the_failed_cell_is_not_kept(context):
    # The failed cell rebinds ``a`` and then raises: the result keeps a = 1.
    result = await run(context, lackpy_doc("a = 1", "a = 2\n1/0"))
    assert not result.success
    assert result.metadata["variables"] == {"a": 1}
