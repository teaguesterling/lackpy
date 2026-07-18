"""L1.3 acceptance: re-assertion versioning — new version + superseded ledger.

Ground rule 3 of the L1 design: **bindings are immutable versions**.
Re-asserting a name creates a NEW version and marks the prior version
superseded — with `supersedes` provenance in one ``superseded`` ledger entry
(``detail={"from_version": n, "to_version": n + 1, ...}``) — never a silent
in-place overwrite.  The flat exec-namespace dict remains the mutable
latest-wins surface Python ``exec`` reads; immutability is a property of the
RECORDED binding layer (:class:`BindingVersions` / the ledger), which is what
these tests pin.

Both re-assertion and write-once-reactive styles are legitimate: a
``superseded`` entry is a visible transition, NOT a reified failure — it does
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
    ErrorValue,
    Hole,
)
from lackpy.interpreters.literate.kernel.forgiveness import round_is_left


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
        assert not round_is_left(session.ledger.entries())

    @pytest.mark.asyncio
    async def test_hole_then_definition_is_versioned_supersede(self, context):
        # L1.1 composition: "redefinition fills a hole" is a versioned
        # transition — the hole is v1 (superseded), the real value v2.
        session = LiterateSession(context)
        assert not (await session.step("```lackpy\ny = x + 1\n```")).ok
        fixed = await session.step("```lackpy\nx = 41\ny = x + 1\n```")
        assert fixed.ok, fixed.errors

        x_history = session.versions.history("x")
        assert len(x_history) == 2
        assert isinstance(x_history[0].value, Hole)      # v1: the hole
        assert x_history[0].superseded_by == 2
        assert x_history[1].value == 41                  # v2: the real value

        entries = session.ledger.query(entry_type=SUPERSEDED, name="x")
        assert len(entries) == 1
        assert entries[0].detail["from_version"] == 1
        assert entries[0].detail["to_version"] == 2
        assert entries[0].detail["prior"] == "hole"

        # The chained hole on y is filled the same way (v1 hole -> v2 value).
        y_history = session.versions.history("y")
        assert isinstance(y_history[0].value, Hole)
        assert y_history[-1].value == 42
        assert session.ledger.query(entry_type=SUPERSEDED, name="y")

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
    async def test_chained_hole_supersedes_real_value(self, context):
        # A blocked cell re-binding an existing REAL name to a chained hole is
        # a re-assertion too — versioned + ledgered, never a silent overwrite.
        session = LiterateSession(context)
        assert (await session.step("```lackpy\nx = 1\n```")).ok
        await session.step("```lackpy\nx = missing + 1\n```")

        history = session.versions.history("x")
        assert history[0].value == 1 and history[0].superseded_by == 2
        assert isinstance(history[1].value, Hole)
        entries = session.ledger.query(entry_type=SUPERSEDED, name="x")
        assert len(entries) == 1 and entries[0].detail["prior"] == "value"

    @pytest.mark.asyncio
    async def test_error_value_reassertion_and_partial_binding_versioned(
        self, context
    ):
        # L1.2 composition: (a) a partial binding completed before the raise
        # is a real re-assertion (the failed cell reports no delta, so the
        # runner observes it directly); (b) a name reified as an ErrorValue
        # then later defined supersedes with prior == "error".
        session = LiterateSession(context)
        assert (await session.step("```lackpy\nx = 1\n```")).ok
        await session.step(
            "```lackpy\nx = 2\nboom = 1 / 0\n```"
        )
        history = session.versions.history("x")
        assert [v.value for v in history] == [1, 2]
        assert session.ledger.query(entry_type=SUPERSEDED, name="x")
        assert isinstance(session.versions.current("boom").value, ErrorValue)

        fixed = await session.step("```lackpy\nboom = 'defused'\n```")
        assert fixed.ok
        entries = session.ledger.query(entry_type=SUPERSEDED, name="boom")
        assert len(entries) == 1 and entries[0].detail["prior"] == "error"
        assert session.versions.current("boom").value == "defused"

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
