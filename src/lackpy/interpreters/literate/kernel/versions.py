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

from dataclasses import dataclass, replace
from typing import Any

#: Ledger entry type for a re-assertion (L1.3): a prior binding version was
#: superseded by a new one.  ``detail`` carries ``from_version``/``to_version``
#: (and the prior value's kind); it is a legitimate transition, not a failure.
SUPERSEDED = "superseded"


@dataclass(frozen=True)
class BindingVersion:
    """One immutable version of a binding.  Mirrors ``_aidr_bindings``:
    1-based ``version``; ``superseded_by`` names the replacing version, or is
    ``None`` while this version is current."""

    name: str
    version: int
    value: Any
    superseded_by: int | None = None

    @property
    def superseded(self) -> bool:
        return self.superseded_by is not None


def value_kind(value: Any) -> str:
    """``"hole"`` / ``"error"`` / ``"value"`` — the binding-layer kind of a
    version's value, for the ``superseded`` entry's detail."""
    # Local import: forgiveness imports nothing from here, but keep the
    # modules decoupled at import time regardless of future direction.
    from .forgiveness import ErrorValue, Hole

    if isinstance(value, Hole):
        return "hole"
    if isinstance(value, ErrorValue):
        return "error"
    return "value"


class BindingVersions:
    """Append-only per-name version history for the binding layer.

    One instance is threaded across a session's rounds (like the L1.0
    ledger); the batch path creates one per run.  :meth:`assert_binding` is
    the single write point: it records the new version and, when a prior
    version existed, marks it superseded and returns it so the caller can
    write the ``superseded`` ledger entry (the ledger write stays with the
    runner, which knows the cell context).
    """

    def __init__(self) -> None:
        self._history: dict[str, list[BindingVersion]] = {}

    def assert_binding(
        self, name: str, value: Any
    ) -> tuple[BindingVersion, BindingVersion | None]:
        """Record one assertion of ``name`` = ``value``.

        Returns ``(new_version, superseded_prior)`` — ``superseded_prior`` is
        the just-superseded prior version (already marked with
        ``superseded_by``), or ``None`` for a first assertion.
        """
        versions = self._history.setdefault(name, [])
        new = BindingVersion(name=name, version=len(versions) + 1, value=value)
        prior: BindingVersion | None = None
        if versions:
            prior = replace(versions[-1], superseded_by=new.version)
            versions[-1] = prior
        versions.append(new)
        return new, prior

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
