"""L1.3 acceptance: re-assertion versioning — new version + superseded ledger.

Ground rule 3 of the L1 design: **bindings are immutable versions**.
Re-asserting a name creates a NEW version and marks the prior version
superseded — with `supersedes` provenance in one ``superseded`` ledger entry
(``detail={"from_version": n, "to_version": n + 1, ...}``) — never a silent
in-place overwrite.  The flat exec-namespace dict remains the mutable
latest-wins surface Python ``exec`` reads; immutability is a property of the
RECORDED binding layer (:class:`BindingVersions` / the ledger), which is what
these tests pin.

A ``superseded`` entry is a visible transition, NOT a failure — it does
not make a round Left.

These tests cover the batch/session execution path
(``LiterateSession.step`` → ``LiterateInterpreter._run_document`` →
``LightweightKernel``), same as L1.0–L1.2.  The streaming path is untouched.
"""

import pytest

from lackpy.interpreters.base import ExecutionContext
from lackpy.interpreters.literate import LiterateInterpreter, LiterateSession
from lackpy.interpreters.literate.kernel import (
    SUPERSEDED,
    BindingVersions,
)
from lackpy.interpreters.literate.kernel.forgiveness import reified_failures


@pytest.fixture
def context(tmp_path):
    return ExecutionContext(base_dir=tmp_path)


class TestBindingVersions:
    """The history structure itself (unit level)."""

    def test_first_assertion_is_version_1_no_supersession(self):
        versions = BindingVersions()
        new, prior = versions.assert_binding("x", 1)
        assert new.version == 1 and new.value == 1 and new.superseded_by is None
        assert prior is None
        assert versions.current("x") is new

    def test_reassertion_supersedes_prior(self):
        versions = BindingVersions()
        versions.assert_binding("x", 1)
        new, prior = versions.assert_binding("x", 2)
        assert new.version == 2 and prior is not None
        assert prior.version == 1 and prior.superseded_by == 2 and prior.superseded
        # The history kept both, oldest first; current is the new one.
        v1, v2 = versions.history("x")
        assert (v1.value, v2.value) == (1, 2)
        assert v1.superseded_by == 2 and v2.superseded_by is None
        assert versions.current("x").value == 2
        assert versions.version_of("x", 1).value == 1


class TestReassertionVersioning:
    """L1.3 acceptance — the execution path versions re-assertions."""

    @pytest.mark.asyncio
    async def test_reassertion_creates_new_version_and_supersedes(self, context):
        session = LiterateSession(context)
        result = await session.step(
            "```lackpy\nx = 1\n```\n\n```lackpy\nx = 2\n```"
        )

        # Two immutable versions; v1 superseded by v2; latest-wins surface = 2.
        history = session.versions.history("x")
        assert [v.version for v in history] == [1, 2]
        assert history[0].value == 1 and history[0].superseded_by == 2
        assert history[1].value == 2 and history[1].superseded_by is None
        assert session._kernel.get_namespace()["x"] == 2

        # Exactly ONE `superseded` ledger entry, naming the transition.
        entries = session.ledger.query(entry_type=SUPERSEDED, name="x")
        assert len(entries) == 1
        assert entries[0].detail["from_version"] == 1
        assert entries[0].detail["to_version"] == 2
        assert entries[0].detail["prior"] == "value"

        # Re-assertion is LEGITIMATE: visible in the ledger, not a failure.
        assert result.ok
        assert not reified_failures(session.ledger.entries())

    @pytest.mark.asyncio
    async def test_version_history_queryable(self, context):
        # The binding layer did NOT lose v1: prior versions are recoverable
        # from the history structure AND from the ledger's supersede entries.
        session = LiterateSession(context)
        await session.step("```lackpy\nx = 'first'\n```")
        await session.step("```lackpy\nx = 'second'\n```")

        assert session.versions.version_of("x", 1).value == "first"
        assert session.versions.current("x").value == "second"
        assert "x" in session.versions and session.versions.names() >= {"x"}

        # Ledger-side recoverability: the transition chain is walkable.
        entries = session.ledger.query(entry_type=SUPERSEDED, name="x")
        assert [
            (e.detail["from_version"], e.detail["to_version"]) for e in entries
        ] == [(1, 2)]

    @pytest.mark.asyncio
    async def test_batch_path_versions_and_ledgers(self, context):
        # The batch execute() runs the same instrumented path; the history is
        # surfaced in metadata alongside the ledger.
        result = await LiterateInterpreter().execute(
            "```lackpy\nx = 1\n```\n\n```lackpy\nx = 2\n```", context
        )
        assert result.success  # supersession is not a failure
        versions = result.metadata["versions"]
        assert [v.value for v in versions.history("x")] == [1, 2]
        entries = result.metadata["ledger"].query(entry_type=SUPERSEDED, name="x")
        assert len(entries) == 1
        assert entries[0].detail == {
            "from_version": 1,
            "to_version": 2,
            "prior": "value",
            "cell_index": 1,
            "cell_type": "code",
        }
