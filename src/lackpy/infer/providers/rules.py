"""Tier 1: Rule-based inference — deterministic keyword-to-program mapping.

Each rule emits a lackpy program that reads files through ``pathlib.Path``
(stepy100). A rule whose only form was a kit call (find definitions, find
callers) is gone: it has no Path form, so those intents go to the model.
"""

from __future__ import annotations

import re

_HEADER = "from pathlib import Path\n\n"

# Common language names to file extensions.
_EXT = {"python": "py", "javascript": "js", "typescript": "ts",
        "rust": "rs", "ruby": "rb", "yaml": "yml"}


def _read(path: str) -> str:
    # end="" prints the file exactly, as cat does.
    return _HEADER + f"text = Path({path!r}).read_text()\nprint(text, end=\"\")\n"


def _glob(pattern: str) -> str:
    return _HEADER + (f"paths = Path(\".\").glob({pattern!r})\n"
                      "for p in sorted(paths):\n"
                      "    print(p)\n")


class RulesProvider:
    @property
    def name(self) -> str:
        return "rules"

    def available(self) -> bool:
        return True

    async def generate(self, intent: str, namespace_desc: str,
                       config: dict | None = None, error_feedback: list[str] | None = None,
                       system_prompt_override: str | None = None,
                       interpreter: object | None = None) -> str | None:
        original = intent.strip()

        # Match "read file <path>" only when path is a single token (no spaces)
        # or quoted. Multi-word intents like "read file X and do Y" go to the model.
        m = re.match(r"read (?:the )?file ['\"]?(\S+?)['\"]?\s*$", original, re.IGNORECASE)
        if m:
            return _read(m.group(1))

        m = re.match(r"(?:find|list) all (\w+) files\s*$", original, re.IGNORECASE)
        if m:
            ext = m.group(1).lower()
            return _glob(f"**/*.{_EXT.get(ext, ext)}")

        m = re.match(r"glob (.+)", original, re.IGNORECASE)
        if m:
            return _glob(m.group(1).strip().strip("'\""))

        return None
