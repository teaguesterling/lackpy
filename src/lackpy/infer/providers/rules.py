"""Tier 1: Rule-based inference — deterministic keyword-to-program mapping."""

from __future__ import annotations

import re


class RulesProvider:
    @property
    def name(self) -> str:
        return "rules"

    def available(self) -> bool:
        return True

    async def generate(self, intent: str, namespace_desc: str,
                       config: dict | None = None, error_feedback: list[str] | None = None) -> str | None:
        lower = intent.lower().strip()
        original = intent.strip()

        m = re.match(r"read (?:the )?file (.+)", lower)
        if m and "read(" in namespace_desc:
            path = re.match(r"read (?:the )?file (.+)", original, re.IGNORECASE).group(1).strip().strip("'\"")
            return f"content = read('{path}')\ncontent"

        m = re.match(r"find (?:the )?definitions? (?:of |for )?(.+)", lower)
        if m and "find_definitions(" in namespace_desc:
            name = re.match(r"find (?:the )?definitions? (?:of |for )?(.+)", original, re.IGNORECASE).group(1).strip().strip("'\"")
            return f"results = find_definitions('{name}')\nresults"

        m = re.match(r"find (?:all )?(?:callers?|usages?|references?) (?:of |for )?(.+)", lower)
        if m and "find_callers(" in namespace_desc:
            name = re.match(r"find (?:all )?(?:callers?|usages?|references?) (?:of |for )?(.+)", original, re.IGNORECASE).group(1).strip().strip("'\"")
            return f"results = find_callers('{name}')\nresults"

        m = re.match(r"(?:find|list) all (\w+) files", lower)
        if m and "glob(" in namespace_desc:
            ext = m.group(1).strip()
            return f"files = glob('**/*.{ext}')\nfiles"

        m = re.match(r"glob (.+)", lower)
        if m and "glob(" in namespace_desc:
            pattern = re.match(r"glob (.+)", original, re.IGNORECASE).group(1).strip().strip("'\"")
            return f"files = glob('{pattern}')\nfiles"

        return None
