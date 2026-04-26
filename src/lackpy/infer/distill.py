"""Doc refinement callbacks for Kibitzer's doc context pipeline.

Kibitzer retrieves documentation sections, then refines them through
callbacks we provide here. The select callback picks sections relevant
to a specific failure mode; the present callback is a no-op for now
(raw sections are already concise enough for small models).
"""

from __future__ import annotations

from typing import Any

_FAILURE_RELEVANT_SECTIONS: dict[str, list[str]] = {
    "stdlib_leak": ["Signature", "Parameters", "Notes"],
    "implement_not_orchestrate": ["Signature", "Examples"],
    "key_hallucination": ["Signature", "Returns", "Notes"],
    "path_prefix": ["Notes", "Parameters"],
    "wrong_output": ["Signature", "Returns", "Examples"],
}


def _select_sections(
    sections: list[Any],
    context: dict[str, Any],
) -> list[Any]:
    """Pick doc sections relevant to the failure mode and tool."""
    failure_mode = context.get("failure_mode", "")
    relevant_titles = _FAILURE_RELEVANT_SECTIONS.get(failure_mode)

    if not relevant_titles:
        return sections[:2]

    matched = [s for s in sections if s.title in relevant_titles]
    return matched if matched else sections[:2]


def build_doc_refinement() -> Any:
    """Build a DocRefinement with lackpy's domain-specific select callback."""
    from kibitzer import DocRefinement
    return DocRefinement(select=_select_sections)
