"""KitPolicySource: baseline policy from resolved kit."""

from __future__ import annotations

from typing import Any

from ..types import PolicyContext, PolicyResult


class KitPolicySource:
    """Translates a ResolvedKit into the initial PolicyResult.

    Always present, lowest priority. Establishes the baseline
    allowed_tools, grade, and namespace_desc.
    """

    name = "kit"
    priority = 0

    def __init__(self, toolbox: Any) -> None:
        self._toolbox = toolbox

    def resolve(self, current: PolicyResult, context: PolicyContext) -> PolicyResult:
        kit = context["kit"]
        return PolicyResult(
            allowed_tools=frozenset(kit.tools.keys()),
            grade=kit.grade,
            namespace_desc=self._toolbox.format_description(list(kit.tools.keys())),
            resolved=False,
        )
