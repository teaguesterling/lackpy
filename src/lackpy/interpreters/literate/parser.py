"""Parse literate markdown documents into cell sequences.

Splits a markdown document with ```lackpy fenced code blocks into an
ordered sequence of Cell objects — prose cells (the text between blocks)
and code cells (the block contents, with parsed annotations).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

CellType = Literal[
    "prose", "code", "hidden", "gather", "continue",
    "read", "write", "diff", "scratch",
]

_FENCE_OPEN = re.compile(
    r"^```lackpy"
    r"(?:\s+@(\w+)(?:\(([^)]*)\))?)?"  # optional @annotation(args)
    r"((?:\s+\w+=\S+)*)"               # optional key=value pairs
    r"\s*$"
)
_FENCE_CLOSE = re.compile(r"^```\s*$")

_ANNOTATION_TYPES: set[str] = {
    "hidden", "gather", "continue", "read", "write", "diff", "scratch",
}

_PATH_ANNOTATIONS: set[str] = {"read", "write", "diff"}


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


def _parse_frontmatter(lines: list[str]) -> tuple[Frontmatter, int]:
    """Extract YAML frontmatter if present. Returns (frontmatter, first_content_line)."""
    if not lines or lines[0].rstrip() != "---":
        return Frontmatter(), 0

    end = None
    for i in range(1, len(lines)):
        if lines[i].rstrip() == "---":
            end = i
            break

    if end is None:
        return Frontmatter(), 0

    fm = Frontmatter()
    for line in lines[1:end]:
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

    return fm, end + 1


def _parse_options(options_str: str) -> dict[str, str]:
    """Parse key=value pairs from fence info string."""
    result: dict[str, str] = {}
    for pair in options_str.split():
        if "=" in pair:
            key, _, value = pair.partition("=")
            result[key.strip()] = value.strip()
    return result


def parse(document: str) -> ParseResult:
    """Parse a literate markdown document into a cell sequence."""
    lines = document.split("\n")
    frontmatter, start_line = _parse_frontmatter(lines)

    cells: list[Cell] = []
    errors: list[str] = []
    prose_lines: list[str] = []
    prose_start = start_line
    in_code = False
    code_lines: list[str] = []
    code_cell_type: CellType = "code"
    code_annotation_args: dict[str, str] = {}
    code_options: dict[str, str] = {}
    code_start = 0

    def flush_prose() -> None:
        nonlocal prose_lines, prose_start
        if prose_lines:
            text = "\n".join(prose_lines)
            if text.strip():
                cells.append(Cell(
                    cell_type="prose",
                    content=text,
                    line_number=prose_start + 1,
                ))
            prose_lines = []

    for i in range(start_line, len(lines)):
        line = lines[i]

        if in_code:
            if _FENCE_CLOSE.match(line):
                cells.append(Cell(
                    cell_type=code_cell_type,
                    content="\n".join(code_lines),
                    annotation_args=code_annotation_args,
                    line_number=code_start + 1,
                    options=code_options,
                ))
                code_lines = []
                in_code = False
                prose_start = i + 1
                prose_lines = []
            else:
                code_lines.append(line)
            continue

        m = _FENCE_OPEN.match(line)
        if m:
            flush_prose()
            annotation = m.group(1)
            annotation_arg = m.group(2)
            options_str = m.group(3) or ""

            code_cell_type = "code"
            code_annotation_args = {}
            code_options = _parse_options(options_str)
            code_start = i

            if annotation:
                if annotation not in _ANNOTATION_TYPES:
                    errors.append(
                        f"Line {i + 1}: unknown annotation @{annotation}"
                    )
                    code_cell_type = "code"
                else:
                    code_cell_type = annotation
                    if annotation in _PATH_ANNOTATIONS:
                        if annotation_arg:
                            code_annotation_args["path"] = annotation_arg
                        else:
                            errors.append(
                                f"Line {i + 1}: @{annotation} requires a path argument"
                            )

            in_code = True
            code_lines = []
        else:
            prose_lines.append(line)

    if in_code:
        errors.append(f"Line {code_start + 1}: unclosed code block")
        cells.append(Cell(
            cell_type=code_cell_type,
            content="\n".join(code_lines),
            annotation_args=code_annotation_args,
            line_number=code_start + 1,
            options=code_options,
        ))

    flush_prose()

    return ParseResult(frontmatter=frontmatter, cells=cells, errors=errors)
