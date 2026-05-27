"""KibitzerPolicySource: coaching, hints, and doc context from Kibitzer."""

from __future__ import annotations

from typing import Any

from ..types import PolicyContext, PolicyResult


class KibitzerPolicySource:
    """Adds prompt hints, doc context, and coaching from a Kibitzer session.

    Never modifies allowed_tools or denied_tools — Kibitzer is a
    coaching layer, not a policy authority.
    """

    name = "kibitzer"
    priority = 50

    def __init__(self, session: Any) -> None:
        self._session = session

    def resolve(self, current: PolicyResult, context: PolicyContext) -> PolicyResult:
        hints: list[str] = []
        docs = list(current.docs)

        history = context.get("history")
        if history and history.current:
            prog = history.current
            if not prog.valid and prog.errors:
                correction = self._session.get_correction_hints(
                    errors=prog.errors,
                    model=context.get("model"),
                    attempt=len(history.programs),
                )
                if correction.hints:
                    hints.extend(correction.hints)
                if correction.doc_context:
                    docs.append(correction.doc_context)

        ns_desc = current.namespace_desc
        # Optional namespace-description coaching hook. The current KibitzerSession does not
        # expose has_coaching()/apply_coaching() (the active coaching path is the correction
        # hints + suggestions above), so guard rather than assume the API — calling missing
        # methods here previously crashed every kibitzer-backed delegation. Use it if a
        # session provides it; skip gracefully otherwise.
        _has = getattr(self._session, "has_coaching", None)
        _apply = getattr(self._session, "apply_coaching", None)
        if ns_desc and callable(_has) and callable(_apply) and _has():
            ns_desc = _apply(ns_desc)

        return current.replace(
            namespace_desc=ns_desc,
            prompt_hints=current.prompt_hints + tuple(hints),
            docs=tuple(docs),
            resolved=False,
        )
