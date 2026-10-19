"""Apply a unified diff to text, in pure Python (literate ``@diff``, design D6).

Host-side only. The kernel reads the file through the session's ``Path``
stand-in, calls :func:`apply_patch`, and writes the result through the same
stand-in, so the policy checks both. No function here is ever in a cell's
namespace. (Moved from the deleted kit module ``tools.py``.)
"""

from __future__ import annotations


def apply_patch(text: str, diff_text: str) -> str:
    """``text`` with the unified diff ``diff_text`` applied.

    Raises ``ValueError`` when a context line does not match.
    """
    return "".join(_apply_unified_diff(text.splitlines(keepends=True), diff_text))


def _apply_unified_diff(original: list[str], diff_text: str) -> list[str]:
    """Apply a unified diff to a list of lines.

    Processes each hunk sequentially: walks the old-file cursor through
    context and removal lines, inserting additions at the correct position.
    """
    result = list(original)
    offset = 0

    for old_start, hunk_lines in _parse_hunks(diff_text):
        pos = old_start - 1 + offset
        cursor = pos
        added = 0
        removed = 0

        for op, text in hunk_lines:
            if op == " ":
                if 0 <= cursor < len(result) and result[cursor].rstrip("\n") != text:
                    raise ValueError(
                        f"Context mismatch at line {cursor + 1}: "
                        f"expected {text!r}, got {result[cursor].rstrip(chr(10))!r}"
                    )
                cursor += 1
            elif op == "-":
                if 0 <= cursor < len(result):
                    result.pop(cursor)
                removed += 1
            elif op == "+":
                result.insert(cursor, text + "\n")
                cursor += 1
                added += 1

        offset += added - removed

    return result


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
