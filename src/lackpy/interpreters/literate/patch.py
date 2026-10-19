"""Apply a unified diff to text, in pure Python (literate ``@diff``, design D6).

Host-side only. The kernel reads the file through the session's ``Path``
stand-in, calls :func:`apply_patch`, and writes the result through the same
stand-in, so the policy checks both. No function here is ever in a cell's
namespace. (Moved from the deleted kit module ``tools.py``.)
"""

from __future__ import annotations


def apply_patch(text: str, diff_text: str) -> str:
    """``text`` with the unified diff ``diff_text`` applied.

    Every context line and every removed (``-``) line must match the line of
    ``text`` it stands for. On a mismatch, ``ValueError`` names the line, what
    the file has and what the diff expects; nothing is returned, so the caller
    writes nothing (ruling: a diff applied to the wrong text corrupts a file).
    """
    return "".join(_apply_unified_diff(text.splitlines(keepends=True), diff_text))


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
    """Yield (old_start, hunk_lines) from unified diff.

    hunk_lines is a list of (op, text) tuples where op is ' ', '-', or '+'.
    """
    lines = diff_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("@@"):
            parts = line.split()
            old_range = parts[1]
            old_start = int(old_range.split(",")[0].lstrip("-"))

            hunk_lines: list[tuple[str, str]] = []
            i += 1
            while i < len(lines) and not lines[i].startswith("@@"):
                dl = lines[i]
                if dl.startswith("-"):
                    hunk_lines.append(("-", dl[1:]))
                elif dl.startswith("+"):
                    hunk_lines.append(("+", dl[1:]))
                elif dl.startswith(" "):
                    hunk_lines.append((" ", dl[1:]))
                elif dl.startswith("\\"):
                    i += 1
                    continue
                else:
                    break
                i += 1

            yield (old_start, hunk_lines)
        else:
            i += 1
