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

        # Match "read file <path>" only when path is a single token (no spaces)
        # or quoted. Multi-word intents like "read file X and do Y" go to the model.
        m = re.match(r"read (?:the )?file ['\"]?(\S+)['\"]?\s*$", lower)
        if m and "read_file(" in namespace_desc:
            path = re.match(r"read (?:the )?file ['\"]?(\S+)['\"]?\s*$", original, re.IGNORECASE).group(1)
            return f"content = read_file('{path}')\ncontent"

        m = re.match(r"find (?:the )?definitions? (?:of |for )?(.+)", lower)
        if m and "find_definitions(" in namespace_desc:
            name = re.match(r"find (?:the )?definitions? (?:of |for )?(.+)", original, re.IGNORECASE).group(1).strip().strip("'\"")
            return f"results = find_definitions('{name}')\nresults"

        m = re.match(r"find (?:all )?(?:callers?|usages?|references?) (?:of |for )?(.+)", lower)
        if m and "find_callers(" in namespace_desc:
            name = re.match(r"find (?:all )?(?:callers?|usages?|references?) (?:of |for )?(.+)", original, re.IGNORECASE).group(1).strip().strip("'\"")
            return f"results = find_callers('{name}')\nresults"

        m = re.match(r"(?:find|list) all (\w+) files\s*$", lower)
        if m and "find_files(" in namespace_desc:
            # Map common language names to file extensions
            ext_map = {"python": "py", "javascript": "js", "typescript": "ts",
                       "rust": "rs", "ruby": "rb", "yaml": "yml"}
            ext = m.group(1).strip()
            ext = ext_map.get(ext, ext)
            return f"files = find_files('**/*.{ext}')\nfiles"

        m = re.match(r"glob (.+)", lower)
        if m and "find_files(" in namespace_desc:
            pattern = re.match(r"glob (.+)", original, re.IGNORECASE).group(1).strip().strip("'\"")
            return f"files = find_files('{pattern}')\nfiles"

        return None
