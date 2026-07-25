"""Persistence-backend protocol + best-effort value serialization (Option B
Stage 1 — kernel persistence-readiness).

This module makes the kernel's two record stores — the L1.0 :class:`~.ledger.Ledger`
and the L1.3 :class:`~.versions.BindingVersions` — *persistence-ready* without
persisting anything itself and without lackpy importing ANY other project.
The intended consumer is a plugin (living in neither lackpy nor the
persistence target) that implements :class:`PersistenceBackend` and hands the
serialized dicts to wherever they should land (e.g. AIDR's ``_aidr_ledger`` /
``_aidr_bindings``).  lackpy owns the protocol; the plugin owns the wiring;
the target owns the ingestion.  No cross-repo dependency in any direction.

The no-op-when-unplugged guarantee
----------------------------------
When no backend is configured, nothing in this module runs: the kernel uses
the plain :class:`~.ledger.Ledger` / :class:`~.versions.BindingVersions` and
behaves byte-identically to a build without this module.  With a backend, the
:class:`PersistentLedger` / :class:`PersistentBindingVersions` subclasses call
the backend strictly AFTER the normal in-memory operation at the two choke
points (``record`` / ``assert_binding``) — persistence is an appended
side-effect, never a semantic change.  Backend exceptions are NOT swallowed
(a plugin that wants fire-and-forget wraps its own try/except); backend
*serialization* never raises (see :func:`serialize_value`).

The value wall (DECISION 1: best-effort JSON + repr-fallback marker)
--------------------------------------------------------------------
A binding's value is an arbitrary Python object.  Forgiveness values
(:class:`~.forgiveness.Hole` / :class:`~.forgiveness.ErrorValue` /
:class:`~.forgiveness.Unavailable`) serialize exactly via their ``to_dict``
and round-trip.  JSON-able values are stored in their JSON-normalized form
(``json.loads(json.dumps(v))`` — tuples become lists, non-string dict keys
become strings — so what is stored is exactly what survives a real JSON
boundary) and round-trip.  Everything else becomes an *inspectable marker*::

    {"__repr__": repr(value), "__nonserializable__": True, "type": type(value).__name__}

which deserializes to itself — legible, honest, never reconstructed to the
real object, and NEVER raises (a persistence layer must not crash the
kernel).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Protocol, runtime_checkable

from .forgiveness import FORGIVENESS_KINDS, forgiveness_from_dict, is_forgiving
from .ledger import AIDR_LEDGER_COLUMNS, Ledger, LedgerEntry
from .versions import BindingVersion, BindingVersions

__all__ = [
    "PersistenceBackend",
    "PersistentLedger",
    "PersistentBindingVersions",
    "ledger_entry_to_dict",
    "serialize_value",
    "deserialize_value",
]


def serialize_value(value: Any) -> Any:
    """Best-effort serialization of a binding value (DECISION 1, option (a)).

    * Forgiveness value → its exact ``to_dict`` (round-trips).
    * JSON-able value → its JSON-normalized form (round-trips; tuples/dict-key
      coercion normalized up front so the stored form IS the JSON form).
    * Anything else → the ``__nonserializable__`` repr marker.  Never raises.
    """
    if is_forgiving(value):
        return value.to_dict()
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError, RecursionError):
        pass
    try:
        rendered = repr(value)
    except Exception:  # a hostile __repr__ must not crash the kernel either
        rendered = f"<unreprable {type(value).__name__}>"
    return {
        "__repr__": rendered,
        "__nonserializable__": True,
        "type": type(value).__name__,
    }


def deserialize_value(value: Any) -> Any:
    """Inverse of :func:`serialize_value`, to the extent honesty allows.

    Forgiveness dicts reconstruct to the real Hole/ErrorValue/Unavailable;
    JSON-able values pass through unchanged; a ``__nonserializable__`` marker
    stays a marker dict (inspectable — deliberately NOT reconstructed)."""
    if isinstance(value, dict) and value.get("__kind__") in FORGIVENESS_KINDS:
        return forgiveness_from_dict(value)
    return value


def ledger_entry_to_dict(entry: LedgerEntry) -> dict[str, Any]:
    """Mechanical serialization of one ledger row — exactly the
    :data:`~.ledger.AIDR_LEDGER_COLUMNS` fields (the pinned ``_aidr_ledger``
    mirror).  Payloads are an in-memory side channel and are dropped at this
    boundary, per the ledger's contract."""
    data = asdict(entry)
    assert tuple(data) == AIDR_LEDGER_COLUMNS
    return data


@runtime_checkable
class PersistenceBackend(Protocol):
    """The plugin seam: what a persistence backend must implement.

    Both methods receive plain dicts (JSON-ready except for whatever a
    non-JSON-able ``detail`` payload a caller recorded — kernel-recorded
    details are JSON-able).  Calls arrive synchronously, in event order,
    AFTER the in-memory operation has completed.
    """

    def persist_ledger_entry(self, entry: dict[str, Any]) -> None:
        """One appended ledger row (:func:`ledger_entry_to_dict` form).
        Rows are append-only: each call is a new row, never an update."""
        ...

    def persist_binding(self, binding: dict[str, Any]) -> None:
        """One binding version (:meth:`~.versions.BindingVersions.serialize`
        form).  NOT append-only: when an assertion supersedes a prior
        version, the prior is re-sent with ``superseded_by`` now set — treat
        ``(session_id, document_id, name, version)`` as an upsert key."""
        ...


class PersistentLedger(Ledger):
    """A :class:`~.ledger.Ledger` that ALSO hands each recorded row to a
    backend.  In-memory behavior is byte-identical to the base class — the
    backend call is appended strictly after the normal append."""

    def __init__(self, backend: PersistenceBackend, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._backend = backend

    def record(self, entry_type: str, **kwargs: Any) -> LedgerEntry:
        entry = super().record(entry_type, **kwargs)
        self._backend.persist_ledger_entry(ledger_entry_to_dict(entry))
        return entry


class PersistentBindingVersions(BindingVersions):
    """A :class:`~.versions.BindingVersions` that ALSO hands each asserted
    version to a backend.  In-memory behavior is byte-identical to the base
    class.  On a supersession the backend receives TWO calls: the updated
    prior (``superseded_by`` set) and the new current version."""

    def __init__(self, backend: PersistenceBackend, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._backend = backend

    def assert_binding(
        self, name: str, value: Any
    ) -> tuple[BindingVersion, BindingVersion | None]:
        new, prior = super().assert_binding(name, value)
        if prior is not None:
            self._backend.persist_binding(self.serialize(prior))
        self._backend.persist_binding(self.serialize(new))
        return new, prior
