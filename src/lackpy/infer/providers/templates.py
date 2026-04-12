"""Tier 0: Template-based inference — pattern match against stored templates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Template:
    name: str
    pattern: str
    program: str
    success_count: int = 0
    fail_count: int = 0
    _compiled: re.Pattern | None = None

    @property
    def regex(self) -> re.Pattern:
        if self._compiled is None:
            regex_str = self.pattern
            for match in re.finditer(r"\{(\w+)\}", self.pattern):
                placeholder = match.group(0)
                group_name = match.group(1)
                regex_str = regex_str.replace(placeholder, f"(?P<{group_name}>.+)")
            self._compiled = re.compile(regex_str, re.IGNORECASE)
        return self._compiled

    def match(self, intent: str) -> dict[str, str] | None:
        m = self.regex.search(intent)
        return m.groupdict() if m else None

    def instantiate(self, captures: dict[str, str]) -> str:
        result = self.program
        for key, value in captures.items():
            result = result.replace(f"{{{key}}}", value)
        return result


class TemplatesProvider:
    def __init__(self, templates_dir: Path) -> None:
        self._dir = templates_dir
        self._templates: list[Template] | None = None

    @property
    def name(self) -> str:
        return "templates"

    def available(self) -> bool:
        return self._dir.exists() and any(self._dir.glob("*.tmpl"))

    def _load(self) -> list[Template]:
        if self._templates is not None:
            return self._templates
        self._templates = []
        if not self._dir.exists():
            return self._templates
        for path in sorted(self._dir.glob("*.tmpl")):
            template = _parse_template_file(path)
            if template:
                self._templates.append(template)
        return self._templates

    async def generate(self, intent: str, namespace_desc: str,
                       config: dict | None = None, error_feedback: list[str] | None = None,
                       system_prompt_override: str | None = None,
                       interpreter: object | None = None) -> str | None:
        for template in self._load():
            captures = template.match(intent)
            if captures is not None:
                return template.instantiate(captures)
        return None


def _parse_template_file(path: Path) -> Template | None:
    text = path.read_text()
    lines = text.strip().split("\n")
    if not lines or lines[0].strip() != "---":
        return None
    frontmatter: dict[str, str] = {}
    body_start = 1
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            body_start = i + 1
            break
        if ":" in line:
            key, _, value = line.partition(":")
            frontmatter[key.strip()] = value.strip().strip('"').strip("'")
    program = "\n".join(lines[body_start:]).strip()
    name = frontmatter.get("name", path.stem)
    pattern = frontmatter.get("pattern", "")
    if not pattern:
        return None
    return Template(
        name=name, pattern=pattern, program=program,
        success_count=int(frontmatter.get("success_count", 0)),
        fail_count=int(frontmatter.get("fail_count", 0)),
    )
