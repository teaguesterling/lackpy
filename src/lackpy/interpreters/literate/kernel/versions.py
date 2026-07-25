"""Binding version history — re-assertion as versioned supersession (L1.3).

Ground rule 3 of the L1 design: **bindings are immutable versions**.  When a
cell binds a name that already has a recorded binding, that is a
RE-ASSERTION — not a silent in-place overwrite.  The kernel records a NEW
version, marks the prior version superseded, and writes one ``superseded``
ledger entry naming the transition.  Both re-assertion and write-once-reactive
styles are legitimate; the kernel supports both; the ledger makes the choice
visible.

Where the immutability lives
----------------------------
The flat execution-namespace dict REMAINS the "current/latest" surface that
Python ``exec`` reads and writes — latest-wins, mutable, exactly as before.
Making that dict immutable is impossible (``exec`` mutates it) and is not the
contract.  Immutability is a property of the RECORDED BINDING LAYER: this
module's :class:`BindingVersions` keeps a per-name append-only history of
:class:`BindingVersion` records, so prior versions are never lost and every
supersession is ledgered.  The namespace is a *view* (the latest version of
every live name); the history is the *record*.

The AIDR mirror
---------------
:class:`BindingVersion` mirrors AIDR's ``_aidr_bindings`` model — a 1-based
``version`` plus ``superseded_by`` (the version that replaced this one, or
``None`` while current) — and the ``superseded`` ledger entry's ``detail``
carries the transition (``{"from_version": n, "to_version": n + 1}``), so the
deferred AIDR persistence of both the ledger and the binding history is a
serialize, not a redesign.

Scope
-----
The history covers names asserted BY DOCUMENT CELLS on the batch/session path
(including holes and error values bound by the forgiveness runner — a hole
filled by a later definition is a versioned transition like any other).
Pre-existing environment names (tools, params, injected callables) have no
version 0: the first *document* assertion of such a name is version 1 with no
supersession, because the binding layer never asserted the environment value.
A ``superseded`` entry records a legitimate transition — it is NOT a reified
failure and does not make a round Left (see ``forgiveness.round_is_left``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Callable

#: Ledger entry type for a re-assertion (L1.3): a prior binding version was
#: superseded by a new one.  ``detail`` carries ``from_version``/``to_version``
#: (and the prior value's kind); it is a legitimate transition, not a failure.
SUPERSEDED = "superseded"


@dataclass(frozen=True)
class BindingVersion:
    """One immutable version of a binding.  Mirrors ``_aidr_bindings``:
    1-based ``version``; ``superseded_by`` names the replacing version, or is
    ``None`` while this version is current.

    ``created_at`` is the assert-time wall clock (float epoch, same source as
    the ledger's ``created_at`` — :func:`time.time`), stamped by
    :meth:`BindingVersions.assert_binding`.  A directly-constructed version
    defaults to ``0.0`` (no assertion time known)."""

    name: str
    version: int
    value: Any
    superseded_by: int | None = None
    created_at: float = 0.0

    @property
    def superseded(self) -> bool:
        return self.superseded_by is not None

    @property
    def kind(self) -> str:
        """The binding-layer kind of this version's value —
        ``"value"`` / ``"hole"`` / ``"error"`` / ``"unavailable"`` (via
        :func:`value_kind`).  NOTE 'superseded' is a *lifecycle* state
        (``superseded_by`` is set), not a value kind."""
        return value_kind(self.value)

    def to_dict(
        self,
        *,
        session_id: str | None = None,
        document_id: str | None = None,
    ) -> dict[str, Any]:
        """Serialized form carrying what AIDR's ``_aidr_bindings`` columns
        need: ``name`` / ``version`` / ``kind`` / ``value_json`` /
        ``superseded_by`` / ``created_at`` plus the identity
        (``session_id`` / ``document_id``, stamped by the caller — a bare
        ``BindingVersion`` has no identity; :meth:`BindingVersions.serialize`
        stamps its container's).

        ``value_json`` is the DECISION-1 best-effort serialization (see
        :func:`.persistence.serialize_value`): forgiveness values via their
        ``to_dict``; JSON-able values as-is; anything else as an inspectable
        ``__nonserializable__`` marker.  NEVER raises."""
        from .persistence import serialize_value

        return {
            "name": self.name,
            "version": self.version,
            "kind": self.kind,
            "value_json": serialize_value(self.value),
            "superseded_by": self.superseded_by,
            "created_at": self.created_at,
            "session_id": session_id,
            "document_id": document_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BindingVersion":
        """Reconstruct from :meth:`to_dict` output.  Forgiveness values and
        JSON-able values round-trip exactly; a non-serializable value comes
        back as its ``__nonserializable__`` marker dict (inspectable, not the
        real object — expected and honest).  The identity fields are the
        container's, not the version's, and are dropped here."""
        from .persistence import deserialize_value

        return cls(
            name=data["name"],
            version=data["version"],
            value=deserialize_value(data["value_json"]),
            superseded_by=data.get("superseded_by"),
            created_at=data.get("created_at", 0.0),
        )


def value_kind(value: Any) -> str:
    """``"hole"`` / ``"error"`` / ``"unavailable"`` / ``"value"`` — the
    binding-layer kind of a version's value, for the ``superseded`` entry's
    detail."""
    # Local import: forgiveness imports nothing from here, but keep the
    # modules decoupled at import time regardless of future direction.
    from .forgiveness import ErrorValue, Hole, Unavailable

    if isinstance(value, Hole):
        return "hole"
    if isinstance(value, ErrorValue):
        return "error"
    if isinstance(value, Unavailable):
        return "unavailable"
    return "value"


class BindingVersions:
    """Append-only per-name version history for the binding layer.

    One instance is threaded across a session's rounds (like the L1.0
    ledger); the batch path creates one per run.  :meth:`assert_binding` is
    the single write point: it records the new version and, when a prior
    version existed, marks it superseded and returns it so the caller can
    write the ``superseded`` ledger entry (the ledger write stays with the
    runner, which knows the cell context).

    Args:
        session_id: Optional identity carried by this history, stamped onto
            serialized versions by :meth:`serialize` (mirrors the ledger's
            identity; ``None`` — the default, today's behavior — serializes
            as null identity).
        document_id: Optional document identity, same treatment.
        clock: Injectable time source for each version's ``created_at``
            (defaults to :func:`time.time`, the SAME source the ledger uses).
            Inject a counter in tests for determinism.
    """

    def __init__(
        self,
        session_id: str | None = None,
        document_id: str | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._history: dict[str, list[BindingVersion]] = {}
        self._session_id = session_id
        self._document_id = document_id
        self._clock = clock or time.time

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def document_id(self) -> str | None:
        return self._document_id

    def assert_binding(
        self, name: str, value: Any
    ) -> tuple[BindingVersion, BindingVersion | None]:
        """Record one assertion of ``name`` = ``value``.

        Returns ``(new_version, superseded_prior)`` — ``superseded_prior`` is
        the just-superseded prior version (already marked with
        ``superseded_by``), or ``None`` for a first assertion.  The new
        version is stamped with assert-time ``created_at``; supersession does
        NOT alter the prior's ``created_at``.
        """
        versions = self._history.setdefault(name, [])
        new = BindingVersion(
            name=name,
            version=len(versions) + 1,
            value=value,
            created_at=self._clock(),
        )
        prior: BindingVersion | None = None
        if versions:
            prior = replace(versions[-1], superseded_by=new.version)
            versions[-1] = prior
        versions.append(new)
        return new, prior

    def serialize(self, version: BindingVersion) -> dict[str, Any]:
        """``version.to_dict()`` stamped with THIS history's identity —
        the serialized-form entry point persistence backends receive."""
        return version.to_dict(
            session_id=self._session_id, document_id=self._document_id
        )

    # -- query surface ----------------------------------------------------

    def history(self, name: str) -> list[BindingVersion]:
        """All versions of ``name``, oldest first (a copy; append-only)."""
        return list(self._history.get(name, ()))

    def current(self, name: str) -> BindingVersion | None:
        """The latest (non-superseded) version of ``name``, or ``None``."""
        versions = self._history.get(name)
        return versions[-1] if versions else None

    def version_of(self, name: str, version: int) -> BindingVersion | None:
        """A specific prior version of ``name`` (1-based), or ``None``."""
        versions = self._history.get(name, [])
        if 1 <= version <= len(versions):
            return versions[version - 1]
        return None

    def names(self) -> set[str]:
        """Every name the binding layer has recorded at least once."""
        return set(self._history)

    def __contains__(self, name: str) -> bool:
        return name in self._history

    def __len__(self) -> int:
        return len(self._history)
