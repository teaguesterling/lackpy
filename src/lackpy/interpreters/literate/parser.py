"""Parse literate markdown documents into cell sequences.

Entry point for the batch path: parse(document) → ParseResult with
a list of Cell objects. Uses markdown-it-py for proper fence detection,
then extracts prose from raw source text between fences. Frontmatter
is parsed separately since --- is <hr> in CommonMark.

The streaming path uses kernel.StreamingCellParser instead, which
detects fences incrementally in partial model output.

Cell types are determined by the fence info string:
  ```lackpy           → code
  ```lackpy @hidden   → hidden
  ```lackpy @read(p)  → read (with path)
  (text between fences) → prose
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from markdown_it import MarkdownIt

from .annotations import strip_kernel_blocks

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

# --- <compute> tags ----------------------------------------------------------
# The tag form is the documented authoring syntax; fences remain accepted. Both
# normalise to one internal representation here, so every downstream stage --
# annotations, static analysis, effects, rendering -- has a single code path.
#
# The tag exists because a fence cannot carry a payload that itself contains a
# fence: the inner ``` closes the outer block. Measured against a local 30B code
# model: a write block whose body held a ```python sample was truncated 5/5
# times in fence form and 0/5 in tag form.
_COMPUTE_OPEN_RE = re.compile(r"^[ \t]*<compute([^>]*)>[ \t]*$", re.MULTILINE)
_COMPUTE_CLOSE = "</compute>"


def _attrs_to_info(attrs: str) -> str:
    """``hidden`` -> ``lackpy @hidden``; ``write="p"`` -> ``lackpy @write(p)``."""
    attrs = attrs.strip()
    if not attrs:
        return "lackpy"
    m = re.match(r'(\w+)\s*=\s*["\']?([^"\']*)["\']?\s*$', attrs)
    if m:
        return f"lackpy @{m.group(1)}({m.group(2)})"
    return f"lackpy @{attrs.split()[0]}"


def _pick_fence(body: str) -> str:
    """A fence longer than any backtick run in the body, so it cannot be closed early."""
    longest = max((len(run) for run in re.findall(r"`+", body)), default=0)
    return "`" * max(3, longest + 1)


def normalize_compute_tags(document: str, *, close_unterminated: bool = True) -> str:
    """Rewrite `<compute …>` blocks into equivalent fenced blocks.

    The single place that knows the tag syntax. Both parsers call it -- the
    batch parser on the whole document, the streaming parser on its buffer --
    so the delimiter is implemented once and neither can drift from the other.

    A body containing fences is wrapped in a *longer* fence, which is how
    CommonMark nests fenced content -- that is what preserves the payload the
    plain three-backtick form would truncate.

    ``close_unterminated`` decides what an unclosed trailing tag means:

    - ``True`` (batch): honour it as a complete block. A document cut at a pause
      marker mid-stream ends exactly this way.
    - ``False`` (streaming): leave it verbatim, untranslated. The outer fence
      length depends on the WHOLE body -- a ``` arriving in a later chunk needs a
      longer fence -- so converting early would freeze a wrong delimiter. Left
      raw, it matches no fence opener, and the incremental scanner holds it until
      the closing tag arrives, which is the behaviour an incomplete block needs.
    """
    if "<compute" not in document:
        return document

    out: list[str] = []
    pos = 0
    for m in _COMPUTE_OPEN_RE.finditer(document):
        if m.start() < pos:  # inside a body already consumed
            continue
        body_start = m.end() + 1 if document[m.end():m.end() + 1] == "\n" else m.end()
        close = document.find(_COMPUTE_CLOSE, body_start)
        if close == -1 and not close_unterminated:
            break  # incomplete: leave this tag and everything after it raw
        out.append(document[pos:m.start()])
        body = document[body_start:] if close == -1 else document[body_start:close]
        body = body.strip("\n")
        fence = _pick_fence(body)
        out.append(f"{fence}{_attrs_to_info(m.group(1))}\n{body}\n{fence}")
        pos = len(document) if close == -1 else close + len(_COMPUTE_CLOSE)
    out.append(document[pos:])
    return "".join(out)


@dataclass
class Cell:
    cell_type: CellType
    content: str
    annotation_args: dict[str, str] = field(default_factory=dict)
    line_number: int = 0
    options: dict[str, str] = field(default_factory=dict)
    truncated: bool = False


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
    body = normalize_compute_tags(body)
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
            # Strip the L2 annotation channel: [kernel]…[/kernel] spans are
            # kernel-generated notes, inert on reparse — never prose cells.
            prose_text = strip_kernel_blocks("\n".join(source_lines[prev_end:fence_start]))
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
        prose_text = strip_kernel_blocks("\n".join(source_lines[prev_end:]))
        if prose_text.strip():
            cells.append(Cell(
                cell_type="prose",
                content=prose_text,
                line_number=fm_lines + prev_end + 1,
            ))

    return ParseResult(frontmatter=frontmatter, cells=cells, errors=errors)


_FENCE_OPEN_RE = re.compile(r"^(`{3,})lackpy(.*)$")


def _info_to_attrs(info: str) -> str:
    """``@hidden`` -> ``hidden``; ``@write(p)`` -> ``write="p"``."""
    info = info.strip()
    if not info:
        return ""
    m = re.match(r"@(\w+)(?:\(([^)]*)\))?\s*$", info)
    if not m:
        return ""
    name, arg = m.group(1), m.group(2)
    if name in _PATH_ANNOTATIONS and arg:
        return f' {name}="{arg}"'
    return f" {name}"


def to_compute_tags(document: str) -> str:
    """Render lackpy fences back as `<compute>` tags — the inverse of normalisation.

    Round-trip artifacts must be spelled the way the writer spelled them. A
    document authored in tags that returns as fences teaches the writer, mid
    conversation, that fences are the syntax — and it resumes in the form that
    truncates any payload containing a fence.
    """
    lines = document.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        m = _FENCE_OPEN_RE.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        ticks, info = m.group(1), m.group(2)
        close_re = re.compile(r"^`{%d,}\s*$" % len(ticks))
        body: list[str] = []
        j = i + 1
        while j < len(lines) and not close_re.match(lines[j]):
            body.append(lines[j])
            j += 1
        out.append(f"<compute{_info_to_attrs(info)}>")
        out.extend(body)
        out.append(_COMPUTE_CLOSE)
        i = j + 1 if j < len(lines) else j
    return "\n".join(out)


def to_markdown(document: str) -> str:
    """A literate document as PORTABLE markdown: fences, no kernel channel.

    For anything that renders markdown but not lackpy -- a README, a docs site,
    a viewer -- where ```lackpy fences highlight and a <compute> tag is raw
    HTML. Pure text: nothing is executed, so rendering a document is safe even
    though running one writes files and shells out.

    A block whose body contains a fence keeps its longer outer fence, so the
    payload survives as valid CommonMark.
    """
    return strip_kernel_blocks(normalize_compute_tags(document)).strip()
