"""Derive a lackpy security Grade(w, d) from MCP tool annotations (RFC 0002 §6).

MCP annotations are advisory booleans. We map them conservatively: any relevant
hint absent ⇒ Grade(3, 3). External coupling raises w; destructiveness /
non-idempotence raises d. Config overrides always win (applied in the source).
"""

from __future__ import annotations

from typing import Any

from ...lang.grader import Grade

CONSERVATIVE = Grade(w=3, d=3)


def grade_from_annotations(ann: Any) -> Grade:
    """Map a ToolAnnotations-like object (or None) to a Grade."""
    if ann is None:
        return CONSERVATIVE
    read_only = getattr(ann, "readOnlyHint", None)
    open_world = getattr(ann, "openWorldHint", None)
    idempotent = getattr(ann, "idempotentHint", None)
    destructive = getattr(ann, "destructiveHint", None)

    if read_only is True:
        if open_world is False:
            return Grade(w=1, d=0)   # local read, no effects
        if open_world is True:
            return Grade(w=2, d=1)   # network read couples to external systems
        return CONSERVATIVE          # open-world unknown
    if read_only is False:
        if idempotent is True and destructive is False:
            return Grade(w=3, d=2)   # scoped write, safe to retry
        return CONSERVATIVE          # non-idempotent / destructive / unknown
    return CONSERVATIVE              # read-only unknown
