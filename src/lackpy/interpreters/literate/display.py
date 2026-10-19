"""Bounded rendering of interpolated values in prose (L6).

Prose interpolation used to render the FULL str() of arbitrarily large values:
a 500-element list printed all 500 elements into the document. The kernel now
formats each ``{expr}`` value on the HOST side (design D3): it evaluates the
expression with ``CellSession.eval_expr`` and passes the value to the
display built here, which truncates large collections to a head + count +
tail form::

    [1, 2, 3, … n=500 … 497, 498, 499]

The value is data (``runner._is_data``) or a ``repr`` text, never a live
object, so nothing here can run document code.

Contract:

  - Values at or under the threshold render EXACTLY as before (``str(value)``)
    -- small values are untouched.
  - Only collections (list/tuple/set/frozenset/dict) are truncated, by element
    count. Strings are prose payload, never truncated here (bounding them is a
    separate decision).
  - The truncated form is a DISPLAY ARTIFACT, not data: the ``…`` ellipsis
    (U+2026) is not valid Python, so the form cannot round-trip -- pasting it
    back into a cell as a binding fails at the syntax level rather than
    silently re-materializing a partial value.
  - The threshold is configurable per run via
    ``ExecutionContext.config["display_threshold"]``; a kernel built without
    one uses the default, so the streaming path is covered too.
"""

from __future__ import annotations

from typing import Any, Callable

#: Collections with more elements than this render truncated. Chosen so a
#: screenful of scalars still prints whole while "I interpolated the whole
#: dataset" cannot flood the document.
DEFAULT_DISPLAY_THRESHOLD = 20

#: How many leading/trailing elements the truncated form shows.
_HEAD = 3
_TAIL = 3

_ELLIPSIS = "…"  # … — deliberately not valid Python syntax


def truncated_display(
    value: Any,
    threshold: int = DEFAULT_DISPLAY_THRESHOLD,
    *,
    head: int = _HEAD,
    tail: int = _TAIL,
) -> str:
    """Render ``value`` for prose interpolation, truncating large collections.

    Non-collections and collections with ``len(value) <= threshold`` render as
    ``str(value)`` -- byte-identical to the pre-guard f-string behavior.
    """
    if isinstance(value, (list, tuple)):
        n = len(value)
        if n <= threshold:
            return str(value)
        open_b, close_b = ("[", "]") if isinstance(value, list) else ("(", ")")
        body = _elide([repr(v) for v in value[:head]],
                      [repr(v) for v in value[-tail:]], n)
        return open_b + body + close_b

    if isinstance(value, dict):
        n = len(value)
        if n <= threshold:
            return str(value)
        entries = list(value.items())
        body = _elide([f"{k!r}: {v!r}" for k, v in entries[:head]],
                      [f"{k!r}: {v!r}" for k, v in entries[-tail:]], n)
        return "{" + body + "}"

    if isinstance(value, (set, frozenset)):
        n = len(value)
        if n <= threshold:
            return str(value)
        elems = [repr(v) for v in value]  # iteration order: a display artifact
        return "{" + _elide(elems[:head], elems[-tail:], n) + "}"

    return str(value)


def _elide(head_items: list[str], tail_items: list[str], n: int) -> str:
    """Join as ``a, b, c, … n=N … x, y, z`` (the ellipsis segment joins the
    tail with a space, not a comma, so it reads as an annotation rather than
    another element)."""
    return (
        ", ".join(head_items)
        + f", {_ELLIPSIS} n={n} {_ELLIPSIS} "
        + ", ".join(tail_items)
    )


def make_display(threshold: int = DEFAULT_DISPLAY_THRESHOLD) -> Callable[[Any], str]:
    """Build the host-side display for a kernel, binding the (possibly
    context-configured) threshold."""

    def _display(value: Any) -> str:
        return truncated_display(value, threshold)

    return _display
