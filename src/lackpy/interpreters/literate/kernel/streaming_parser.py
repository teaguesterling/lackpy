"""Streaming cell parser — detects fence boundaries in partial model output.

Used by StreamingDriver to parse-as-you-generate: the model's response
streams in via feed(), and complete Cell objects are yielded as soon as
their closing fence arrives. Prose before a fence is held until the
fence completes, so partial fences don't emit incorrect prose cells.

Not used by the batch path (LiterateInterpreter), which uses parser.parse()
on the complete document instead.
"""

from __future__ import annotations

import re

from ..annotations import strip_kernel_blocks
from ..parser import (
    Cell, Frontmatter, _parse_info_string, _PATH_ANNOTATIONS,
    _extract_path_from_body, _BODY_ANNOTATION_RE, _ANNOTATION_TYPES,
)


def _prose_cell(text: str) -> Cell | None:
    """Build a prose cell with the L2 annotation channel stripped.

    ``[kernel]…[/kernel]`` spans are kernel-generated notes, inert on reparse —
    they must never become prose. Returns ``None`` when nothing authored remains.
    """
    stripped = strip_kernel_blocks(text).strip()
    if not stripped:
        return None
    return Cell(cell_type="prose", content=stripped)

_FENCE_OPEN = re.compile(r"^```(\S.*)?$", re.MULTILINE)
_FENCE_CLOSE = re.compile(r"^```\s*$", re.MULTILINE)


class StreamingCellParser:
    """Parse LLM output incrementally, yielding complete Cell objects.

    Prose preceding a lackpy fence is held in the buffer until the fence's
    closing ``` is received, then both prose and code cells are yielded
    together.  This avoids emitting prose from partial fence blocks.
    """

    def __init__(self) -> None:
        self._buffer: str = ""
        self._frontmatter: Frontmatter = Frontmatter()
        self._frontmatter_parsed: bool = False

    @property
    def frontmatter(self) -> Frontmatter:
        return self._frontmatter

    def feed(self, chunk: str) -> list[Cell]:
        self._buffer += chunk.replace("\r", "")
        if not self._frontmatter_parsed:
            self._try_parse_frontmatter()
        return self._extract_cells()

    def flush(self) -> list[Cell]:
        """Drain whatever remains in the buffer, treating unclosed fences as complete."""
        cells: list[Cell] = []
        buf = self._buffer

        # Check if we're mid-fence: scan for open without matching close.
        open_match = _FENCE_OPEN.search(buf)
        if open_match is not None:
            info = open_match.group(1) or ""
            if info.startswith("lackpy"):
                # Prose before the fence.
                prose_before = buf[:open_match.start()].rstrip("\n")
                prose_cell = _prose_cell(prose_before)
                if prose_cell is not None:
                    cells.append(prose_cell)
                # Content after the fence open line (skip the newline).
                after_open = open_match.end()
                if after_open < len(buf) and buf[after_open] == "\n":
                    after_open += 1
                content = buf[after_open:].rstrip("\n")
                cell = self._make_fence_cell(info, content)
                cell.truncated = True
                cells.append(cell)
                self._buffer = ""
                return cells

        # No mid-fence: treat everything as prose.
        prose_cell = _prose_cell(buf)
        if prose_cell is not None:
            cells.append(prose_cell)
        self._buffer = ""
        return cells

    def reset(self) -> None:
        self._buffer = ""
        self._frontmatter = Frontmatter()
        self._frontmatter_parsed = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _try_parse_frontmatter(self) -> None:
        if not self._buffer.startswith("---"):
            self._frontmatter_parsed = True
            return
        lines = self._buffer.split("\n")
        if len(lines) < 2:
            return
        for i in range(1, len(lines)):
            if lines[i].rstrip() == "---":
                self._frontmatter = self._parse_fm_lines(lines[1:i])
                self._buffer = "\n".join(lines[i + 1:])
                self._frontmatter_parsed = True
                return

    def _parse_fm_lines(self, lines: list[str]) -> Frontmatter:
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

    def _extract_cells(self) -> list[Cell]:
        """Scan the buffer and yield all complete fence segments.

        A "complete segment" is either:
          - prose up to a lackpy fence that has a matching close, OR
          - a non-lackpy fence (open + content + close) treated as prose.

        Prose that sits before an unclosed (or not-yet-opened) lackpy fence
        is NOT emitted — it stays in the buffer so it can be emitted
        together with the fence once the close arrives.
        """
        cells: list[Cell] = []
        pos = 0  # current read position in self._buffer

        while pos < len(self._buffer):
            # Find the next fence opener from pos.
            open_match = _FENCE_OPEN.search(self._buffer, pos)
            if open_match is None:
                # No fence opener at all — nothing to emit yet.
                break

            # The fence open line must end with a newline (not be a partial line).
            open_end = open_match.end()
            if open_end == len(self._buffer) and not self._buffer.endswith("\n"):
                # Partial fence line — wait for more data.
                break

            info = open_match.group(1) or ""

            if not info.startswith("lackpy"):
                # Non-lackpy fence: find its close.
                close_match = _FENCE_CLOSE.search(self._buffer, open_end + 1)
                if close_match is None:
                    # Incomplete non-lackpy fence — stop processing.
                    break
                # The entire non-lackpy fence (including any prose before it)
                # becomes a prose cell.  Emit prose-before first, then the fence.
                prose_before = self._buffer[pos:open_match.start()].rstrip("\n")
                prose_cell = _prose_cell(prose_before)
                if prose_cell is not None:
                    cells.append(prose_cell)
                fence_text = self._buffer[open_match.start():close_match.end()]
                cells.append(Cell(cell_type="prose", content=fence_text))
                pos = close_match.end()
                if pos < len(self._buffer) and self._buffer[pos] == "\n":
                    pos += 1
                continue

            # It IS a lackpy fence — find its closing ```.
            # Content starts after the open line's newline.
            content_start = open_end
            if content_start < len(self._buffer) and self._buffer[content_start] == "\n":
                content_start += 1

            close_match = _FENCE_CLOSE.search(self._buffer, content_start)
            if close_match is None:
                # Fence not yet closed — stop; don't emit prose before it.
                break

            # Emit any prose before the fence open.
            prose_before = self._buffer[pos:open_match.start()].rstrip("\n")
            prose_cell = _prose_cell(prose_before)
            if prose_cell is not None:
                cells.append(prose_cell)

            content = self._buffer[content_start:close_match.start()].rstrip("\n")
            cells.append(self._make_fence_cell(info, content))

            pos = close_match.end()
            if pos < len(self._buffer) and self._buffer[pos] == "\n":
                pos += 1

        # Discard the consumed portion of the buffer.
        self._buffer = self._buffer[pos:]
        return cells

    def _make_fence_cell(self, info: str, content: str) -> Cell:
        cell_type, annotation_args, options, _errors = _parse_info_string(info)
        if cell_type in _PATH_ANNOTATIONS and "path" not in annotation_args:
            path, content = _extract_path_from_body(content)
            if path:
                annotation_args["path"] = path

        # Auto-correct misplaced annotations (e.g. @hidden on first line of body)
        if cell_type == "code" and content:
            first_line = content.split("\n", 1)[0]
            body_m = _BODY_ANNOTATION_RE.match(first_line)
            if body_m and body_m.group(1) in _ANNOTATION_TYPES:
                cell_type = body_m.group(1)
                content = content.split("\n", 1)[1] if "\n" in content else ""

        return Cell(
            cell_type=cell_type,
            content=content,
            annotation_args=annotation_args,
            options=options,
        )
