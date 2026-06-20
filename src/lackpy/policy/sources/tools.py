"""ToolsPolicySource: baseline policy from resolved tools."""

from __future__ import annotations

from typing import Any

from ..types import PolicyContext, PolicyResult


class ToolsPolicySource:
    """Translates a ResolvedTools into the initial PolicyResult.

    Always present, lowest priority. Establishes the baseline
    allowed_tools, grade, and namespace_desc.
    """

    name = "tools"
    priority = 0

    def __init__(self, toolbox: Any) -> None:
        self._toolbox = toolbox

    def resolve(self, current: PolicyResult, context: PolicyContext) -> PolicyResult:
        tools = context["tools"]
        return PolicyResult(
            allowed_tools=frozenset(tools.tools.keys()),
            grade=tools.grade,
            namespace_desc=self._toolbox.format_description(list(tools.tools.keys())),
            resolved=False,
        )
