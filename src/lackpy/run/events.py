"""The emit/event seam for profile execution.

A profile (see :mod:`lackpy.profiles`) is **lifecycle-decoupled**: the same profile is
run by different *drivers* — one-shot, incremental/literate, conversation, eventful,
scheduled, parallel (see RFC 0002 increment 5, §3.6). Several of those drivers need the
run to **emit** results *as it goes* — a rendered cell, a partial result, a tool-call
record, a notification — not only return a final value.

This module defines that channel as a tiny, driver-agnostic seam:

- ``ExecutionEvent`` — one emitted unit (kind + data + optional index).
- ``Emit`` — a callable a driver provides; the runtime calls it per event.

It is intentionally minimal and **not yet wired** into ``RestrictedRunner`` /
``ExecutionResult``. It exists so the profile/execution contract is emit-capable from
the start (a one-shot driver simply passes no emitter and reads the return value;
literate/conversation/eventful drivers pass one and stream events). Wiring lands with
the drivers, not in the additive profile-model phase.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ExecutionEvent:
    """One unit emitted during a profile run.

    ``kind`` is an open string (e.g. ``"cell"``, ``"tool_call"``, ``"partial"``,
    ``"notify"``, ``"final"``) so new drivers can introduce event kinds without
    changing this type. ``index`` orders events within a session (e.g. cell index).
    """

    kind: str
    data: Any = None
    index: int | None = None


#: A driver-provided sink. The runtime calls it once per ``ExecutionEvent``. ``None``
#: means "no emission" — the one-shot path, where the return value is the whole result.
Emit = Callable[[ExecutionEvent], None]
