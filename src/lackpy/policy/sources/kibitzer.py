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
        if ns_desc and self._session.has_coaching():
            ns_desc = self._session.apply_coaching(ns_desc)

        return current.replace(
            namespace_desc=ns_desc,
            prompt_hints=current.prompt_hints + tuple(hints),
            docs=tuple(docs),
            resolved=False,
        )
