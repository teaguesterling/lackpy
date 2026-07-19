"""Typed holes + error values + unavailable sources — the L1 binding-layer
forgiveness values.

The two-layer forgiveness contract (design conflict #2, DECIDED option (c)):

* **Binding layer (the default mechanism).**  A failure during batch/session
  execution is REIFIED AS A BOUND VALUE and ledgered; execution continues and
  nothing rolls back.  An unknown name binds a :class:`Hole`
  (``hole_opened``); a runtime error binds an :class:`ErrorValue`
  (``error_reified``); a name whose SOURCE the kernel refuses to run right
  now — a cell over the effect ceiling gate — binds an :class:`Unavailable`
  (``source_unavailable``).  A cell that *references* any forgiving value
  cannot produce a real value, so its own bindings become chained holes.
  Chaining is detected **statically** — a referenced name bound to a
  forgiving value blocks the cell before it executes — so a forgiving value
  never flows into arithmetic and never has to survive ``+``/``:.2f``/etc.
* **Aggregate layer (the preserved signal).**  The per-round Either/Left the
  session's callers rely on SURVIVES as a view derived from the ledger:
  :func:`round_is_left` — a round containing any reified failure still
  reports Left to callers who ask.  It is no longer a control-flow abort.

Reprs use the display module's ``⟨…⟩`` convention: like the truncated display
form, a forgiveness value's repr is a *display artifact*, deliberately not
valid Python — it can never round-trip back in as a real binding (the L2
refusal law).

This module is consumed by the batch/session path
(``LiterateInterpreter._run_document``); the streaming driver's recovery
mechanism (model-driven retry) is a different layer and is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .ledger import LedgerEntry

#: Ledger entry type for a typed hole (L1.1): an unknown name bound as a
#: :class:`Hole`, or a binding blocked by an upstream forgiving value.
HOLE_OPENED = "hole_opened"

#: Ledger entry type for a reified error (L1.2): a cell failure recorded as a
#: value/ledger entry instead of an abort.
ERROR_REIFIED = "error_reified"

#: Ledger entry type for an unavailable source (L1.5): a cell the effect
#: ceiling gate refuses to run binds its names as :class:`Unavailable` values
#: instead of refusing the whole document.  Deliberately NOT named "pending"
#: (design conflict #4): the streaming driver's ``pending`` status means "not
#: executed due to @continue pause" — a different concept on a different path.
SOURCE_UNAVAILABLE = "source_unavailable"

#: The entry types that make a round Left under the aggregate view.
FORGIVENESS_ENTRY_TYPES: frozenset[str] = frozenset(
    {HOLE_OPENED, ERROR_REIFIED, SOURCE_UNAVAILABLE}
)


@dataclass(frozen=True)
class Hole:
    """A typed hole ⟨name: unbound⟩ — the reified form of an unknown name.

    ``blocked_by`` is empty for an *origin* hole (the name was referenced
    before any definition existed) and names the upstream causes for a
    *chained* hole (the cell that would have bound this name referenced a
    hole or error value, so it could not run).
    """

    name: str
    blocked_by: tuple[str, ...] = ()

    def __repr__(self) -> str:
        if self.blocked_by:
            return f"⟨{self.name}: blocked by {', '.join(self.blocked_by)}⟩"
        return f"⟨{self.name}: unbound⟩"


@dataclass(frozen=True)
class ErrorValue:
    """A runtime failure reified as a value, carrying the exception info.

    Bound to the names a failed cell declared but did not successfully
    (re)bind this pass — including a failed RE-binding of a DOCUMENT-ASSERTED
    name (one with version history), whose prior value/hole is superseded
    (never silently kept stale).  Names the cell *did* bind before the error
    keep their real values (completed work is never thrown away — the failure
    itself is what gets recorded), and INJECTED names (params / tools /
    environment, no document-asserted version) always keep their provided
    values — a failed re-bind never clobbers what the document never owned.
    """

    name: str
    error: str  # "ExceptionType: message", as the kernel reports it
    error_phase: str = "runtime"

    def __repr__(self) -> str:
        return f"⟨{self.name}: error {self.error}⟩"


@dataclass(frozen=True)
class Unavailable:
    """An unavailable source reified as a value (L1.5).

    Bound to the names a cell would have bound when the kernel REFUSES to run
    the cell's required effect right now — concretely, the effect ceiling
    gate (:func:`..effects.exceeds_ceiling`): the cell's statically-classified
    effects exceed the profile's grade ceiling, so its source is unavailable
    under the current ceiling.  Not unknown (that's a :class:`Hole`), not a
    runtime failure (that's an :class:`ErrorValue`) — the source exists and
    would run under a raised ceiling / different profile.  A later
    within-ceiling assertion of the name supersedes this value through the
    normal L1.3 versioning path.

    ``reason`` carries the gate's human-readable refusal (the
    "effect ceiling exceeded: ..." line).
    """

    name: str
    reason: str

    def __repr__(self) -> str:
        return f"⟨{self.name}: unavailable — {self.reason}⟩"


def is_forgiving(value: Any) -> bool:
    """True for the binding-layer forgiveness values (holes, error values,
    unavailable sources)."""
    return isinstance(value, (Hole, ErrorValue, Unavailable))


def reified_failures(entries: Iterable[LedgerEntry]) -> list[LedgerEntry]:
    """The reified-failure entries within ``entries`` (ledger order kept)."""
    return [e for e in entries if e.entry_type in FORGIVENESS_ENTRY_TYPES]


def round_is_left(entries: Iterable[LedgerEntry]) -> bool:
    """THE aggregate Either view: a round is Left iff its ledger slice
    contains any reified failure.  Derived state, not control flow — the
    round's bindings (real values, holes, error values) all survive."""
    return bool(reified_failures(entries))


def describe_failure(entry: LedgerEntry) -> str:
    """One human/model-readable line per reified failure, for Left errors."""
    if entry.entry_type == HOLE_OPENED:
        blocked = entry.detail.get("blocked_by")
        if blocked:
            who = f"'{entry.name}'" if entry.name else "cell"
            return f"hole: {who} blocked by {', '.join(blocked)}"
        return f"hole: '{entry.name}' unbound"
    if entry.entry_type == ERROR_REIFIED:
        err = entry.detail.get("error", "unknown error")
        if entry.name:
            return f"error: {err} ('{entry.name}' bound as error value)"
        return f"error: {err}"
    if entry.entry_type == SOURCE_UNAVAILABLE:
        reason = entry.detail.get("reason", "source unavailable")
        if entry.name:
            return f"unavailable: '{entry.name}' — {reason}"
        return f"unavailable: {reason}"
    return f"{entry.entry_type}: {entry.name or ''}".rstrip(": ")
