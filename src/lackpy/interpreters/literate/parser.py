"""Parse literate markdown documents into cell sequences.

Uses markdown-it-py for proper fence detection, then extracts prose
from raw source text between fences. Frontmatter is parsed separately
since it's not part of CommonMark.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from markdown_it import MarkdownIt

CellType = Literal[
    "prose", "code", "hidden", "gather", "continue",
    "read", "write", "diff", "scratch",
]

_ANNOTATION_RE = re.compile(
    r"^lackpy"
    r"(?:\s+@(\w+)(?:\(([^)]*)\))?)?"
    r"((?:\s+\w+=\S+)*)"
    r"\s*$"
)

_ANNOTATION_TYPES: set[str] = {
    "hidden", "gather", "continue", "read", "write", "diff", "scratch",
}

_BODY_ANNOTATION_RE = re.compile(r"^\s*@(\w+)(?:\(.*\))?\s*$")

_PATH_ANNOTATIONS: set[str] = {"read", "write", "diff"}

_md = MarkdownIt("commonmark")


@dataclass
class Cell:
    cell_type: CellType
    content: str
    annotation_args: dict[str, str] = field(default_factory=dict)
    line_number: int = 0
    options: dict[str, str] = field(default_factory=dict)


@dataclass
class Frontmatter:
    echo: str = "true"
    output: str = "auto"
    interpreter: str = "python"


@dataclass
class ParseResult:
    frontmatter: Frontmatter
    cells: list[Cell]
    errors: list[str] = field(default_factory=list)


def _strip_frontmatter(text: str) -> tuple[Frontmatter, str, int]:
    """Strip YAML frontmatter, returning (frontmatter, remaining_text, lines_consumed)."""
    lines = text.split("\n")
    if not lines or lines[0].rstrip() != "---":
        return Frontmatter(), text, 0

    for i in range(1, len(lines)):
        if lines[i].rstrip() == "---":
            fm = _parse_frontmatter_block(lines[1:i])
            remaining = "\n".join(lines[i + 1:])
            return fm, remaining, i + 1

    return Frontmatter(), text, 0


def _parse_frontmatter_block(lines: list[str]) -> Frontmatter:
    fm = Frontmatter()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if key == "echo":
                fm.echo = value
            elif key == "output":
                fm.output = value
            elif key == "interpreter":
                fm.interpreter = value
    return fm


def _parse_info_string(info: str) -> tuple[CellType, dict[str, str], dict[str, str], list[str]]:
    """Parse a fence info string into (cell_type, annotation_args, options, errors)."""
    errors: list[str] = []
    m = _ANNOTATION_RE.match(info)
    if not m:
        return "code", {}, {}, errors

    annotation = m.group(1)
    annotation_arg = m.group(2)
    options_str = m.group(3) or ""

    options: dict[str, str] = {}
    for pair in options_str.split():
        if "=" in pair:
            key, _, value = pair.partition("=")
            options[key.strip()] = value.strip()

    if not annotation:
        return "code", {}, options, errors

    if annotation not in _ANNOTATION_TYPES:
        errors.append(f"unknown annotation @{annotation}")
        return "code", {}, options, errors

    annotation_args: dict[str, str] = {}
    if annotation in _PATH_ANNOTATIONS and annotation_arg:
        annotation_args["path"] = annotation_arg

    return annotation, annotation_args, options, errors


def _extract_path_from_body(content: str) -> tuple[str, str]:
    """For @read/@write/@diff without path in parens, try the first line as path."""
    lines = content.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped:
            remaining = "\n".join(lines[i + 1:])
            return stripped, remaining
    return "", content


def parse(document: str) -> ParseResult:
    """Parse a literate markdown document into a cell sequence."""
    frontmatter, body, fm_lines = _strip_frontmatter(document)
    source_lines = body.split("\n")

    tokens = _md.parse(body)

    cells: list[Cell] = []
    errors: list[str] = []

    fence_regions: list[tuple[int, int, str, str]] = []
    for token in tokens:
        if token.type == "fence" and token.map is not None:
            start, end = token.map
            fence_regions.append((start, end, token.info, token.content))

    prev_end = 0
    for fence_start, fence_end, info, content in fence_regions:
        if fence_start > prev_end:
            prose_text = "\n".join(source_lines[prev_end:fence_start])
            if prose_text.strip():
                cells.append(Cell(
                    cell_type="prose",
                    content=prose_text,
                    line_number=fm_lines + prev_end + 1,
                ))

        if not info.startswith("lackpy"):
            prose_text = "\n".join(source_lines[fence_start:fence_end])
            if prose_text.strip():
                cells.append(Cell(
                    cell_type="prose",
                    content=prose_text,
                    line_number=fm_lines + fence_start + 1,
                ))
            prev_end = fence_end
            continue

        cell_type, annotation_args, options, info_errors = _parse_info_string(info)
        for e in info_errors:
            errors.append(f"Line {fm_lines + fence_start + 1}: {e}")

        content_stripped = content.rstrip("\n")

        if cell_type == "code" and content_stripped:
            first_line = content_stripped.split("\n", 1)[0]
            body_m = _BODY_ANNOTATION_RE.match(first_line)
            if body_m and body_m.group(1) in _ANNOTATION_TYPES:
                ann = body_m.group(1)
                errors.append(
                    f"Line {fm_lines + fence_start + 1}: "
                    f"@{ann} found inside code body — "
                    f"annotations go on the fence line: ```lackpy @{ann}"
                )

        if cell_type in _PATH_ANNOTATIONS and "path" not in annotation_args:
            path, content_stripped = _extract_path_from_body(content_stripped)
            if path:
                annotation_args["path"] = path
            else:
                errors.append(
                    f"Line {fm_lines + fence_start + 1}: @{cell_type} requires a path"
                )

        cells.append(Cell(
            cell_type=cell_type,
            content=content_stripped,
            annotation_args=annotation_args,
            line_number=fm_lines + fence_start + 1,
            options=options,
        ))
        prev_end = fence_end

    if prev_end < len(source_lines):
        prose_text = "\n".join(source_lines[prev_end:])
        if prose_text.strip():
            cells.append(Cell(
                cell_type="prose",
                content=prose_text,
                line_number=fm_lines + prev_end + 1,
            ))

    return ParseResult(frontmatter=frontmatter, cells=cells, errors=errors)
