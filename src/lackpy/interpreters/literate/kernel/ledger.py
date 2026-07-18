"""Queryable forgiveness ledger — the literate kernel's event ledger (L1.0).

Forgiveness is a LEDGER, not tolerance: every notable execution event is
recorded as first-class, queryable state — nothing silent.  This module is the
FOUNDATION of the L1 forgiveness work: it replaces the driver's in-memory
proto-ledger (``StreamingDriver._log``, a bare ``list[CellExecutionEvent]``
that could only be dumped wholesale) with an append-only store whose rows can
be queried by session, by name, and by entry type.

Row-shape contract (the AIDR mirror)
------------------------------------
:class:`LedgerEntry` deliberately mirrors AIDR's ``_aidr_ledger`` table —
``entry_id`` (monotonic), ``session_id``, ``document_id``, ``entry_type``,
``name``, ``detail``, ``created_at`` — so the deferred cross-project
persistence (writing this ledger into AIDR/DuckDB) is a *serialize*, not a
redesign.  :data:`AIDR_LEDGER_COLUMNS` states the contract; a conformance test
pins it.  This module does NOT import aidr and does NOT persist anything:
in-repo, in-memory, queryable.

Entry types
-----------
L1.0 records only the event kinds that already occur today — the streaming
driver's cell statuses (``executed`` / ``pending`` / ``continue_requested`` /
``recovered`` / ``skipped`` / ``aborted``).  The forgiveness semantics'
entry types (``hole_opened``, ``error_reified``, ``superseded``,
``pending_deferred``, …) are added by their own sub-PRs (L1.1+); nothing here
constrains ``entry_type`` to a closed set.

Payloads
--------
``record(..., payload=obj)`` optionally attaches an arbitrary in-memory object
to an entry.  Payloads are an IN-REPO convenience only — they are NOT part of
the row contract and are dropped at any serialization boundary.  The driver
uses them to keep the rich :class:`~.driver.CellExecutionEvent` objects that
``formats.to_notebook`` / ``formats.render_markdown`` consume, so the ledger
is the single store and ``execution_log`` becomes a derived view.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Iterator

#: The ``_aidr_ledger`` column contract that :class:`LedgerEntry` mirrors.
#: Future AIDR persistence serializes these fields mechanically; renaming or
#: reshaping any of them is a cross-project contract change.
AIDR_LEDGER_COLUMNS: tuple[str, ...] = (
    "entry_id",
    "session_id",
    "document_id",
    "entry_type",
    "name",
    "detail",
    "created_at",
)


@dataclass(frozen=True)
class LedgerEntry:
    """One append-only ledger row.  Field set == :data:`AIDR_LEDGER_COLUMNS`."""

    entry_id: int
    session_id: str
    document_id: str | None
    entry_type: str
    name: str | None
    detail: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0


# Sanity: the dataclass and the published contract cannot drift.
assert tuple(f.name for f in fields(LedgerEntry)) == AIDR_LEDGER_COLUMNS


class Ledger:
    """Append-only, queryable, in-memory event ledger.

    Args:
        session_id: Identity stamped on every row.  Auto-generated when not
            given (each driver run is its own session until the literate
            session wiring lands).
        document_id: Default ``document_id`` for rows (nullable; the streaming
            driver has no document identity today, so it stays ``None`` there).
        clock: Injectable time source for ``created_at`` (defaults to
            :func:`time.time`).  Inject a counter in tests for determinism.
    """

    def __init__(
        self,
        session_id: str | None = None,
        document_id: str | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._session_id = session_id or uuid.uuid4().hex[:12]
        self._document_id = document_id
        self._clock = clock or time.time
        self._entries: list[LedgerEntry] = []
        self._payloads: dict[int, Any] = {}
        self._next_id = 0

    @property
    def session_id(self) -> str:
        return self._session_id

    def record(
        self,
        entry_type: str,
        *,
        name: str | None = None,
        detail: dict[str, Any] | None = None,
        document_id: str | None = None,
        payload: Any = None,
    ) -> LedgerEntry:
        """Append one entry and return it.  ``entry_id`` is monotonic.

        ``payload`` attaches an in-memory object (see module docstring); it is
        not part of the row and is retrievable via :meth:`payload_of` /
        :meth:`payloads`.
        """
        entry = LedgerEntry(
            entry_id=self._next_id,
            session_id=self._session_id,
            document_id=document_id if document_id is not None else self._document_id,
            entry_type=entry_type,
            name=name,
            detail=dict(detail) if detail else {},
            created_at=self._clock(),
        )
        self._next_id += 1
        self._entries.append(entry)
        if payload is not None:
            self._payloads[entry.entry_id] = payload
        return entry

    # -- query surface ----------------------------------------------------

    def entries(self) -> list[LedgerEntry]:
        """All entries in append order (a copy; the ledger stays append-only)."""
        return list(self._entries)

    def query(
        self,
        *,
        session_id: str | None = None,
        name: str | None = None,
        entry_type: str | None = None,
    ) -> list[LedgerEntry]:
        """Filter entries; all given criteria must match (AND semantics).

        Mirrors AIDR's ``ledger(session_id=, name=, entry_type=)`` query
        surface.  ``query()`` with no criteria == :meth:`entries`.
        """
        out = self._entries
        if session_id is not None:
            out = [e for e in out if e.session_id == session_id]
        if name is not None:
            out = [e for e in out if e.name == name]
        if entry_type is not None:
            out = [e for e in out if e.entry_type == entry_type]
        return list(out)

    # -- payload side-channel (in-repo only, not part of the row contract) --

    def payload_of(self, entry_id: int) -> Any:
        """The payload attached to ``entry_id``, or ``None``."""
        return self._payloads.get(entry_id)

    def payloads(self) -> list[Any]:
        """All attached payloads, in entry order (entries without one skipped)."""
        return [
            self._payloads[e.entry_id]
            for e in self._entries
            if e.entry_id in self._payloads
        ]

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[LedgerEntry]:
        return iter(self._entries)
