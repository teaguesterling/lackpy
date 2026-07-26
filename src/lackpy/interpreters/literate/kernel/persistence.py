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
A binding's value is an arbitrary Python object.  :func:`serialize_value`
wraps EVERY value in a self-describing envelope so the deserializer never has
to sniff user data for reserved keys::

    {"v": "fjson",  "data": <forgiveness to_dict>}   # Hole/ErrorValue/Unavailable
    {"v": "json",   "data": <JSON-normalized value>} # any JSON-able value, verbatim
    {"v": "marker", "data": {"__repr__": ..., "__nonserializable__": True,
                             "type": ...}}           # everything else

Forgiveness values (:class:`~.forgiveness.Hole` /
:class:`~.forgiveness.ErrorValue` / :class:`~.forgiveness.Unavailable`)
serialize exactly via their ``to_dict`` and round-trip.  JSON-able values are
stored in their JSON-normalized form (``json.loads(json.dumps(v))`` — tuples
become lists, non-string dict keys become strings — so what is stored is
exactly what survives a real JSON boundary) and round-trip VERBATIM: a user
dict that happens to contain ``__kind__`` (or any other reserved-looking key)
is just ``data`` under ``"v": "json"`` and is never interpreted as a
forgiveness value.  Everything else becomes an *inspectable marker* (repr
bounded at :data:`MAX_MARKER_REPR` chars) which deserializes to the marker
dict — legible, honest, never reconstructed to the real object.
:func:`serialize_value` NEVER raises (a persistence layer must not crash the
kernel); :func:`deserialize_value` dispatches ONLY on the envelope's ``"v"``
tag.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Protocol, runtime_checkable

from .forgiveness import forgiveness_from_dict, is_forgiving
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


#: Upper bound on the ``__repr__`` string stored in a non-serializable
#: marker.  A hostile/huge ``__repr__`` must not produce an unbounded wire
#: payload; anything longer is truncated with an explicit indicator.
MAX_MARKER_REPR = 2000


def serialize_value(value: Any) -> dict[str, Any]:
    """Best-effort serialization of a binding value (DECISION 1, option (a)),
    wrapped in a self-describing envelope ``{"v": <tag>, "data": ...}``.

    * Forgiveness value → ``{"v": "fjson", "data": <its exact to_dict>}``
      (round-trips).
    * JSON-able value → ``{"v": "json", "data": <JSON-normalized form>}``
      (round-trips verbatim; tuples/dict-key coercion normalized up front so
      the stored form IS the JSON form).
    * Anything else → ``{"v": "marker", "data": <__nonserializable__ repr
      marker>}``, with the repr bounded at :data:`MAX_MARKER_REPR` chars.

    The envelope is always the OUTER wrapper and user data is always nested
    under ``data``, so a user value can never collide with the format's
    discriminator — no reserved-key sniffing on user data, by construction.
    Never raises.
    """
    if is_forgiving(value):
        return {"v": "fjson", "data": value.to_dict()}
    try:
        return {"v": "json", "data": json.loads(json.dumps(value))}
    except (TypeError, ValueError, RecursionError):
        pass
    try:
        rendered = repr(value)
    except Exception:  # a hostile __repr__ must not crash the kernel either
        rendered = f"<unreprable {type(value).__name__}>"
    if len(rendered) > MAX_MARKER_REPR:
        rendered = (
            rendered[:MAX_MARKER_REPR]
            + f"…[truncated {len(rendered) - MAX_MARKER_REPR} chars]"
        )
    return {
        "v": "marker",
        "data": {
            "__repr__": rendered,
            "__nonserializable__": True,
            "type": type(value).__name__,
        },
    }


def deserialize_value(value: Any) -> Any:
    """Inverse of :func:`serialize_value`, to the extent honesty allows.

    Dispatches ONLY on the envelope's ``"v"`` tag — never on the shape of the
    nested ``data``.  ``"fjson"`` reconstructs the real
    Hole/ErrorValue/Unavailable; ``"json"`` returns ``data`` verbatim (even
    if it is a dict containing ``__kind__`` — user data is never sniffed);
    ``"marker"`` returns the ``__nonserializable__`` marker dict (inspectable
    — deliberately NOT reconstructed).  Anything that is not a
    :func:`serialize_value` envelope raises ``ValueError``."""
    if isinstance(value, dict) and "v" in value and "data" in value:
        tag = value["v"]
        if tag == "fjson":
            return forgiveness_from_dict(value["data"])
        if tag in ("json", "marker"):
            return value["data"]
    raise ValueError(
        f"not a serialize_value envelope: {type(value).__name__} "
        f"(expected {{'v': 'fjson'|'json'|'marker', 'data': ...}})"
    )


def ledger_entry_to_dict(entry: LedgerEntry) -> dict[str, Any]:
    """Mechanical serialization of one ledger row — exactly the
    :data:`~.ledger.AIDR_LEDGER_COLUMNS` fields (the pinned ``_aidr_ledger``
    mirror).  The ``payload`` side channel (``record(..., payload=obj)``, an
    in-memory-only attachment that is not a :class:`LedgerEntry` field) is
    dropped at this boundary, per the ledger's contract; ``detail`` IS
    included in the output and must be JSON-able to survive persistence
    (kernel-recorded details are; a caller-recorded non-JSON-able ``detail``
    is the caller's responsibility)."""
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
