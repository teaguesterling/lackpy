"""Failure ledger entry types and their wording (spec E1, E3).

A failure during batch/session execution stops the document at its cell. It is
ledgered, and it binds nothing. The entry types below are the ledger's record
of it:

* ``hole_opened`` — the cell reads a name no earlier cell bound (an unbound
  name). The type name is kept: it is the stored ledger vocabulary.
* ``error_reified`` — the cell raised.
* ``source_unavailable`` — the effect ceiling gate refused the cell.
* ``timed_out`` — the cell ran over its time limit and was stopped.

The per-round Either the session's callers rely on is a view derived from the
ledger: :func:`reified_failures` is empty for a Right round and holds the
failure for a Left round.

Until spec E3 a failure bound a typed value (``Hole``, ``ErrorValue``,
``Unavailable``) and the run went on. Those values and the binding layer are
retired.

This module is consumed by the batch/session path
(``LiterateInterpreter._run_document``); the streaming driver's recovery
mechanism (model-driven retry) is a different layer and is untouched.
"""

from __future__ import annotations

from typing import Iterable

from .ledger import LedgerEntry

#: Ledger entry type for an unbound name: the cell reads a name no earlier cell
#: bound, so the document stops at that cell.
HOLE_OPENED = "hole_opened"

#: Ledger entry type for a cell that raised: the failure is recorded and the
#: document stops at that cell.
ERROR_REIFIED = "error_reified"

#: Ledger entry type for a cell the effect ceiling gate refuses to run (L1.5).
#: Deliberately NOT named "pending" (design conflict #4): the streaming
#: driver's ``pending`` status means "not executed due to @continue pause" — a
#: different concept on a different path.
SOURCE_UNAVAILABLE = "source_unavailable"

#: Ledger entry type for a cell that ran over its time limit and was stopped
#: (Plan 3 D7). The document stops at that cell. A timeout is not a refusal:
#: the session stays open, and a retry may succeed.
TIMED_OUT = "timed_out"

#: The entry types that make a round Left under the aggregate view.
FORGIVENESS_ENTRY_TYPES: frozenset[str] = frozenset(
    {HOLE_OPENED, ERROR_REIFIED, SOURCE_UNAVAILABLE, TIMED_OUT}
)


def reified_failures(entries: Iterable[LedgerEntry]) -> list[LedgerEntry]:
    """The failure entries within ``entries`` (ledger order kept)."""
    return [e for e in entries if e.entry_type in FORGIVENESS_ENTRY_TYPES]


def describe_failure(entry: LedgerEntry) -> str:
    """One human/model-readable line per failure entry (the ``[kernel]`` note)."""
    if entry.entry_type == HOLE_OPENED:
        return f"the name '{entry.name}' is not bound"
    if entry.entry_type == ERROR_REIFIED:
        return f"error: {entry.detail.get('error', 'unknown error')}"
    if entry.entry_type == SOURCE_UNAVAILABLE:
        return f"unavailable: {entry.detail.get('reason', 'source unavailable')}"
    if entry.entry_type == TIMED_OUT:
        return str(entry.detail.get("note", "timeout"))
    return f"{entry.entry_type}: {entry.name or ''}".rstrip(": ")
