"""L1.0 acceptance tests — the queryable forgiveness Ledger (foundation).

Covers the four L1.0 acceptance criteria:
  1. driver events that previously landed in the in-memory proto-ledger
     (``driver._log``) now flow through the queryable Ledger — faithfully;
  2. the ledger is queryable three ways (session_id / name / entry_type);
  3. the row shape mirrors AIDR's ``_aidr_ledger`` contract exactly;
  4. regression — everything that read ``driver.execution_log`` (including
     the pause/resume path) still works after the migration.
"""

import dataclasses

import pytest

from lackpy.interpreters.literate.kernel.driver import (
    CellExecutionEvent,
    StreamingDriver,
)
from lackpy.interpreters.literate.kernel.formats import render_markdown, to_notebook
from lackpy.interpreters.literate.kernel.ledger import (
    AIDR_LEDGER_COLUMNS,
    Ledger,
    LedgerEntry,
)
from lackpy.interpreters.literate.kernel.lightweight import LightweightKernel
from lackpy.interpreters.literate.kernel.recovery import NoRecoveryHandler
from lackpy.interpreters.literate.parser import Frontmatter


@pytest.fixture
def kernel():
    return LightweightKernel()


@pytest.fixture
def driver(kernel):
    return StreamingDriver(kernel=kernel, recovery=NoRecoveryHandler())


def _counter_clock():
    """Deterministic injectable clock: 1.0, 2.0, 3.0, ..."""
    state = {"t": 0.0}

    def clock() -> float:
        state["t"] += 1.0
        return state["t"]

    return clock


class TestLedgerRecordsExistingEvents:
    """Acceptance 1: the driver's proto-ledger events are now Ledger rows."""

    @pytest.mark.asyncio
    async def test_ledger_records_existing_events(self, driver):
        # Exercise the statuses driver._log recorded before the migration:
        # executed (code + prose), continue_requested, and pending (a COMPLETE
        # cell parsed in the same feed but queued behind the pause; bare
        # trailing prose never completes into a cell while paused).
        text = (
            "```lackpy @hidden\nx = 1\n```\n\n"
            "Value: {x}\n\n"
            "```lackpy @continue\n```\n\n"
            "```lackpy\nz = 3\n```\n"
        )
        await driver.feed(text)
        await driver.flush()

        events = driver.execution_log
        rows = driver.ledger.entries()

        # Same events, now queryable: one row per event, in order, with the
        # event's status carried verbatim as entry_type (no invented types).
        assert len(rows) == len(events) > 0
        assert [r.entry_type for r in rows] == [e.status for e in events]
        assert [r.detail["cell_index"] for r in rows] == [e.cell_index for e in events]
        assert [r.detail["cell_type"] for r in rows] == [
            e.cell.cell_type for e in events
        ]
        statuses = {r.entry_type for r in rows}
        assert {"executed", "continue_requested", "pending"} <= statuses

        # entry_id is monotonic; every row is stamped with the session.
        assert [r.entry_id for r in rows] == sorted(r.entry_id for r in rows)
        assert len({r.entry_id for r in rows}) == len(rows)
        assert all(r.session_id == driver.ledger.session_id for r in rows)

    @pytest.mark.asyncio
    async def test_failure_events_are_ledgered_with_error_detail(self, driver):
        await driver.feed("```lackpy\ny = 1/0\n```\n")
        await driver.flush()
        aborted = driver.ledger.query(entry_type="aborted")
        assert len(aborted) == 1
        assert aborted[0].detail["error"]
        assert aborted[0].detail["error_phase"] in ("static", "runtime")

    @pytest.mark.asyncio
    async def test_injected_ledger_is_used(self, kernel):
        ledger = Ledger(session_id="sess-inject", clock=_counter_clock())
        driver = StreamingDriver(
            kernel=kernel, recovery=NoRecoveryHandler(), ledger=ledger
        )
        await driver.feed("Hello\n")
        await driver.flush()
        assert driver.ledger is ledger
        assert len(ledger) == 1
        assert ledger.entries()[0].session_id == "sess-inject"
        assert ledger.entries()[0].created_at == 1.0


class TestLedgerQueryableThreeWays:
    """Acceptance 2: query by session, by name, by entry_type."""

    def test_ledger_queryable_three_ways(self):
        ledger = Ledger(session_id="sess-a", clock=_counter_clock())
        ledger.record("executed", name="x", detail={"cell_index": 0})
        ledger.record("executed", name="y", detail={"cell_index": 1})
        ledger.record("aborted", name="x", detail={"cell_index": 2})

        # by session
        assert len(ledger.query(session_id="sess-a")) == 3
        assert ledger.query(session_id="sess-other") == []

        # by name
        by_name = ledger.query(name="x")
        assert [e.entry_id for e in by_name] == [0, 2]

        # by entry_type
        by_type = ledger.query(entry_type="executed")
        assert [e.entry_id for e in by_type] == [0, 1]

        # criteria AND together (mirrors A1's ledger(session_id=, name=, entry_type=))
        both = ledger.query(session_id="sess-a", name="x", entry_type="aborted")
        assert [e.entry_id for e in both] == [2]

        # no criteria == everything
        assert ledger.query() == ledger.entries()

    @pytest.mark.asyncio
    async def test_driver_events_queryable_by_entry_type(self, driver):
        text = "```lackpy @hidden\na = 1\n```\n\n```lackpy\nb = 2\n```\n"
        await driver.feed(text)
        await driver.flush()
        executed = driver.ledger.query(entry_type="executed")
        assert len(executed) == 2
        assert driver.ledger.query(entry_type="aborted") == []


class TestLedgerRowShapeMirrorsAidr:
    """Acceptance 3: the row shape IS the _aidr_ledger contract."""

    def test_ledger_row_shape_mirrors_aidr(self):
        expected = (
            "entry_id",
            "session_id",
            "document_id",
            "entry_type",
            "name",
            "detail",
            "created_at",
        )
        # The published contract and the dataclass agree, exactly and in order.
        assert AIDR_LEDGER_COLUMNS == expected
        assert (
            tuple(f.name for f in dataclasses.fields(LedgerEntry)) == expected
        )

        # A recorded row serializes to exactly those keys — future AIDR
        # persistence is mechanical.
        ledger = Ledger(session_id="s", document_id="doc-1", clock=_counter_clock())
        entry = ledger.record("executed", name="x", detail={"k": "v"})
        row = dataclasses.asdict(entry)
        assert set(row) == set(expected)
        assert row["entry_id"] == 0
        assert row["session_id"] == "s"
        assert row["document_id"] == "doc-1"
        assert row["entry_type"] == "executed"
        assert row["name"] == "x"
        assert row["detail"] == {"k": "v"}
        assert row["created_at"] == 1.0

    def test_rows_are_append_only(self):
        ledger = Ledger(session_id="s")
        entry = ledger.record("executed")
        with pytest.raises(dataclasses.FrozenInstanceError):
            entry.entry_type = "mutated"  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_driver_row_detail_is_plain_data(self, driver):
        """The rows themselves (sans payload) carry only serializable data."""
        import json

        await driver.feed("```lackpy @hidden\nx = 1\n```\n")
        await driver.flush()
        for entry in driver.ledger.entries():
            json.dumps(dataclasses.asdict(entry))  # must not raise


class TestExecutionLogRegression:
    """Acceptance 4: nothing that used driver._log broke."""

    @pytest.mark.asyncio
    async def test_pause_resume_path_still_reads_log(self, driver):
        # Round 1: pause at @continue; trailing content becomes pending.
        text1 = (
            "```lackpy @hidden\nx = 1\n```\n\n"
            "```lackpy @continue\n```\n\n"
            "After pause\n"
        )
        await driver.feed(text1)
        await driver.flush()
        statuses1 = [e.status for e in driver.execution_log]
        assert "continue_requested" in statuses1
        assert driver.generation == 0

        # Resume and run round 2 against the SAME ledger-backed log.
        driver.resume()
        await driver.feed("```lackpy\ny = x + 1\n```\n")
        await driver.flush()
        assert driver.generation == 1

        events = driver.execution_log
        assert [e.generation for e in events] == sorted(e.generation for e in events)
        assert events[-1].status == "executed"
        # The ledger saw both generations too.
        assert {r.detail["generation"] for r in driver.ledger.entries()} == {0, 1}

    @pytest.mark.asyncio
    async def test_formats_still_consume_execution_log(self, driver):
        await driver.feed("Intro prose\n\n```lackpy\nprint('hi')\n```\n")
        await driver.flush()
        log = driver.execution_log
        assert all(isinstance(e, CellExecutionEvent) for e in log)

        nb = to_notebook(log, Frontmatter())
        assert len(nb["cells"]) == len(log)

        md = render_markdown(log, Frontmatter())
        assert "Intro prose" in md
        assert "print('hi')" in md

    @pytest.mark.asyncio
    async def test_execution_log_returns_fresh_list(self, driver):
        await driver.feed("Hello\n")
        await driver.flush()
        log = driver.execution_log
        log.clear()
        assert len(driver.execution_log) == 1
