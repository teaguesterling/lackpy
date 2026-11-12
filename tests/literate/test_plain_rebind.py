"""A rebind re-runs nothing (spec E3).

Re-asserting a name versions it (an L1.3 ``superseded`` entry) and nothing
more: no earlier cell runs again, so a document gives python3's answer
(goal 4). Until spec E3 a heal re-ran the cells that read a name before it was
bound; the reactive re-run is retired.

These tests cover the batch/session path (``LiterateInterpreter._run_document``).
"""

import pytest

from lackpy.interpreters.base import ExecutionContext
from lackpy.interpreters.literate import LiterateInterpreter


@pytest.fixture
def context(tmp_path):
    return ExecutionContext(base_dir=tmp_path)


async def _run(context, doc):
    return await LiterateInterpreter().execute(doc, context)


class TestNoDependencyNoRerun:

    @pytest.mark.asyncio
    async def test_no_dependency_no_rerun(self, context):
        # Re-asserting a name nothing depends on re-runs nothing.
        result = await _run(
            context,
            "```lackpy\na = 1\n```\n\n"
            "```lackpy\nb = 50\n```\n\n"   # independent of a
            "```lackpy\na = 2\n```",
        )
        # b never recomputed; still its single original version.
        assert len(result.metadata["versions"].history("b")) == 1
        assert result.metadata["variables"]["b"] == 50


class TestNonDataBindingsAreTrackedByName:
    """Review I2: a binding to a non-data value (a def, a Path) supersedes by name."""

    @pytest.mark.asyncio
    async def test_data_then_a_def_supersedes_and_does_not_rerun(self, context):
        # A value-to-value rebind (data to a def) is versioned (superseded)
        # but re-runs nothing; y keeps python3's 4.
        result = await _run(
            context,
            "```lackpy\nf = 3\n```\n\n```lackpy\ny = f + 1\n```\n\n"
            "```lackpy\ndef f(n):\n    return n\n```\n")
        ledger = result.metadata["ledger"]
        sup = ledger.query(entry_type="superseded", name="f")
        assert sup and sup[-1].detail["cell_index"] == 2
        assert result.metadata["variables"]["y"] == 4


class TestPlainRebindDoesNotReexecute:
    """A plain rebind never re-runs earlier cells, so a document gives
    python3's answer (goal 4)."""

    CSV = ("```lackpy\nrows = 'name,n\\nann,3\\nbob,5\\ncat,7\\n'.splitlines()\n```\n\n"
           "```lackpy\nrows = rows[1:]\n```\n\n"
           "```lackpy\nrows = [r for r in rows if r]\n```\n\n"
           "There are {len(rows)} people.\n\n"
           "```lackpy\nlen(rows)\n```\n")

    @pytest.mark.asyncio
    async def test_the_reviewers_csv_document_gives_python3s_answer(self, context):
        # The final review measured output=2 and rows=['bob,5','cat,7'] (ann lost)
        # while the prose said 3; python3 on the joined cells gives 3.
        result = await _run(context, self.CSV)
        assert result.success, result.error
        assert result.metadata["value"] == 3
        assert result.metadata["variables"]["rows"] == ["ann,3", "bob,5", "cat,7"]
        assert "There are 3 people." in result.output

    @pytest.mark.asyncio
    async def test_two_slices_of_one_name(self, context):
        doc = ("```lackpy\nitems = ['a', 'b', 'c', 'd']\n```\n\n"
               "```lackpy\nitems = items[1:]\n```\n\n"
               "```lackpy\nitems = items[1:]\nitems\n```\n")
        result = await _run(context, doc)
        assert result.success, result.error
        assert result.metadata["value"] == ["c", "d"]

    @pytest.mark.asyncio
    async def test_an_earlier_reader_keeps_the_value_it_saw(self, context):
        doc = ("```lackpy\na = 1\n```\n\n```lackpy\nb = a + 1\n```\n\n"
               "```lackpy\na = 2\n```\n")
        result = await _run(context, doc)
        assert result.metadata["variables"]["b"] == 2  # python3: b = 2
