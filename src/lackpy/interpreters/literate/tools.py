"""Kit tools injected into the literate execution namespace.

These are plain Python functions (not lackpy ToolSpec registrations)
that get added directly to the execution namespace. They provide the
file I/O and shell primitives that @read, @write, @diff blocks compile
down to, plus search and test-running utilities.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _resolve(path: str, base: Path | None) -> Path:
    p = Path(path)
    if base and not p.is_absolute():
        return base / p
    return p


def read_file(path: str, *, _base: Path | None = None) -> str:
    """Read and return the contents of a file."""
    return _resolve(path, _base).read_text()


def write_file(path: str, content: str, *, _base: Path | None = None) -> None:
    """Write content to a file, creating parent directories as needed."""
    p = _resolve(path, _base)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


def apply_diff(path: str, diff_text: str, *, _base: Path | None = None) -> str:
    """Apply a unified diff to a file and return the result.

    Uses Python's difflib to parse and apply the patch rather than
    shelling out to `patch`, so it works without external tools.
    """
    target = _resolve(path, _base)
    if not target.exists():
        raise FileNotFoundError(f"Cannot apply diff: {path} does not exist")

    original_lines = target.read_text().splitlines(keepends=True)
    patched_lines = _apply_unified_diff(original_lines, diff_text)
    result = "".join(patched_lines)
    target.write_text(result)
    return result


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


def search_content(pattern: str, path: str = ".", *, _base: Path | None = None) -> str:
    """Grep-like search for a pattern in files under path."""
    try:
        result = subprocess.run(
            ["grep", "-rn", "--include=*.py", pattern, str(_resolve(path, _base))],
            capture_output=True, text=True, timeout=30,
        )
        return result.stdout or "(no matches)"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return "(search failed)"


def run_command(cmd: str, *, _base: Path | None = None) -> str:
    """Run a shell command and return combined stdout+stderr.

    Intentionally uses shell=True — nsjail provides the security boundary.
    """
    try:
        result = subprocess.run(
            cmd, shell=True,  # noqa: S602
            capture_output=True, text=True, timeout=60,
            cwd=str(_base) if _base else None,
        )
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        return output.strip()
    except subprocess.TimeoutExpired:
        return "(command timed out after 60s)"


def run_tests(path: str = ".", *, _base: Path | None = None) -> str:
    """Run pytest on a path and return the output."""
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", str(_resolve(path, _base)), "-v", "--tb=short"],
            capture_output=True, text=True, timeout=120,
            cwd=str(_base) if _base else None,
        )
        return result.stdout + (result.stderr or "")
    except subprocess.TimeoutExpired:
        return "(tests timed out after 120s)"


def make_tool_namespace(base_dir: str | Path | None = None) -> dict:
    """Create a namespace dict with all literate tools.

    When base_dir is provided, tool functions resolve relative paths
    against it instead of relying on os.chdir().
    """
    from functools import partial
    base = Path(base_dir) if base_dir else None
    return {
        "read_file": partial(read_file, _base=base),
        "write_file": partial(write_file, _base=base),
        "apply_diff": partial(apply_diff, _base=base),
        "search_content": partial(search_content, _base=base),
        "run_command": partial(run_command, _base=base),
        "run_tests": partial(run_tests, _base=base),
    }
