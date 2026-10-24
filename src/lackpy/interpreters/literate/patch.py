"""Apply a unified diff to text, in pure Python (literate ``@diff``, design D6).

Host-side only. The kernel reads the file through the session's ``Path``
stand-in, calls :func:`apply_patch`, and writes the result through the same
stand-in, so the policy checks both. No function here is ever in a cell's
namespace. (Moved from the deleted kit module ``tools.py``.)
"""

from __future__ import annotations

import re


def apply_patch(text: str, diff_text: str) -> str:
    """``text`` with the unified diff ``diff_text`` applied.

    Every context line and every removed (``-``) line must match the line of
    ``text`` it stands for. On a mismatch, ``ValueError`` names the line, what
    the file has and what the diff expects; nothing is returned, so the caller
    writes nothing (ruling: a diff applied to the wrong text corrupts a file).

    When the diff does not apply but its reverse does, the file already holds
    the diff's result (a resubmitted document ran it before): the
    ``ValueError`` then says "diff already applied" (ruling after C1).
    """
    lines = text.splitlines(keepends=True)
    try:
        return "".join(_apply_unified_diff(lines, diff_text))
    except ValueError as forward:
        try:
            _apply_unified_diff(lines, _reversed(diff_text))
        except ValueError:
            raise forward from None
        raise ValueError(
            "diff already applied: the file already holds the result of this diff; "
            "nothing was written") from None


_HEADER_SIDES = re.compile(r"^@@ -(\S+) \+(\S+) @@")


def _reversed(diff_text: str) -> str:
    """``diff_text`` with its two sides swapped: ``+`` lines become ``-`` lines,
    and the header's old and new ranges change places."""
    out = []
    for line in diff_text.splitlines():
        m = _HEADER_SIDES.match(line)
        if m:
            line = f"@@ -{m.group(2)} +{m.group(1)} @@{line[m.end():]}"
        elif line.startswith("+") and not line.startswith("+++ "):
            line = "-" + line[1:]
        elif line.startswith("-") and not line.startswith("--- "):
            line = "+" + line[1:]
        out.append(line)
    return "\n".join(out)


def _apply_unified_diff(original: list[str], diff_text: str) -> list[str]:
    """Apply a unified diff to a list of lines, hunk by hunk, verifying as it goes."""
    result = list(original)
    offset = 0  # lines added minus lines removed by the earlier hunks

    for old_start, hunk_lines in _parse_hunks(diff_text):
        cursor = max(old_start - 1, 0) + offset
        old_line = max(old_start, 1)  # the line number in the original file
        for op, text in hunk_lines:
            if op in (" ", "-"):
                _expect(result, cursor, text, old_line, len(original))
                old_line += 1
                if op == " ":
                    cursor += 1
                else:
                    result.pop(cursor)
                    offset -= 1
            elif op == "+":
                result.insert(cursor, text + "\n")
                cursor += 1
                offset += 1

    return result


def _expect(lines: list[str], index: int, text: str, line: int, total: int) -> None:
    """Raise ``ValueError`` unless ``lines[index]`` is ``text`` (line ``line`` of the file)."""
    if index >= len(lines):
        raise ValueError(
            f"the diff does not apply: line {line} is past the end of the file "
            f"({total} lines); the diff expects {text!r}")
    found = lines[index].rstrip("\n")
    if found != text:
        raise ValueError(
            f"the diff does not apply: line {line} is {found!r}, the diff expects {text!r}")


def _parse_hunks(diff_text: str):
    """Yield (old_start, hunk_lines) from a unified diff; ``ValueError`` on a bad hunk.

    hunk_lines is a list of (op, text) tuples where op is ' ', '-', or '+'.

    The header counts (``@@ -a,b +c,d @@``; a missing count is 1) bound each
    hunk (final review I2). Inside a hunk an empty line is an empty context
    line, as GNU patch reads it: models, editors and markdown strip the space
    of a context line for an empty file line. A hunk that ends before its
    counts, a line in a hunk that is not a diff line, and a diff line after a
    hunk's counts are used up each raise, so a diff never applies in part.
    Text outside the hunks (``---``/``+++`` headers, blank lines) is skipped.
    At the end of the diff text, missing lines that can only be context
    (equal old and new counts left) are taken as empty context lines: a
    markdown fence strips trailing empty lines. They are still verified
    against the file.
    """
    lines = diff_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith("@@"):
            if line.startswith((" ", "+", "-")) and not line.startswith(("--- ", "+++ ")) \
                    and _after_a_hunk(lines, i):
                raise ValueError(
                    f"the diff does not apply: {line!r} is past the end of the hunk "
                    "its header counts describe")
            i += 1
            continue
        old_start, old_left, new_left = _hunk_header(line)
        header = line
        hunk_lines: list[tuple[str, str]] = []
        i += 1
        while (old_left or new_left) and i < len(lines) and not lines[i].startswith("@@"):
            dl = lines[i]
            i += 1
            if dl.startswith("\\"):
                continue  # "\ No newline at end of file"
            op, text = (dl[:1], dl[1:]) if dl else (" ", "")
            if op not in (" ", "-", "+"):
                raise ValueError(
                    f"the diff does not apply: {dl!r} in the hunk {header!r} is not a "
                    "context (' '), removed ('-') or added ('+') line")
            if op in (" ", "-"):
                old_left -= 1
            if op in (" ", "+"):
                new_left -= 1
            if old_left < 0 or new_left < 0:
                raise ValueError(
                    f"the diff does not apply: the hunk {header!r} has more lines than "
                    "its header counts")
            hunk_lines.append((op, text))
        if old_left or new_left:
            if i >= len(lines) and old_left == new_left:
                hunk_lines.extend([(" ", "")] * old_left)  # stripped empty context lines
            else:
                raise ValueError(
                    f"the diff does not apply: the hunk {header!r} ends before the "
                    f"lines its header counts ({old_left} old, {new_left} new missing)")
        yield (old_start, hunk_lines)


def _hunk_header(line: str) -> tuple[int, int, int]:
    """``(old_start, old_count, new_count)`` of an ``@@ -a,b +c,d @@`` line."""
    parts = line.split()
    try:
        old, new = parts[1], parts[2]
        if not (old.startswith("-") and new.startswith("+")):
            raise ValueError
        old_start, _, old_count = old[1:].partition(",")
        _, _, new_count = new[1:].partition(",")
        return int(old_start), int(old_count or 1), int(new_count or 1)
    except (IndexError, ValueError):
        raise ValueError(f"the diff does not apply: {line!r} is not a hunk header "
                         "such as '@@ -1,3 +1,3 @@'") from None


def _after_a_hunk(lines: list[str], index: int) -> bool:
    """Whether a hunk header comes before ``lines[index]``."""
    return any(line.startswith("@@") for line in lines[:index])
