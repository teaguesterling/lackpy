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


def read_file(path: str) -> str:
    """Read and return the contents of a file."""
    return Path(path).read_text()


def write_file(path: str, content: str) -> None:
    """Write content to a file, creating parent directories as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


def apply_diff(path: str, diff_text: str) -> str:
    """Apply a unified diff to a file and return the result.

    Uses Python's difflib to parse and apply the patch rather than
    shelling out to `patch`, so it works without external tools.
    """
    target = Path(path)
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
                else:
                    break
                i += 1

            yield (old_start, hunk_lines)
        else:
            i += 1


def search_content(pattern: str, path: str = ".") -> str:
    """Grep-like search for a pattern in files under path."""
    try:
        result = subprocess.run(
            ["grep", "-rn", "--include=*.py", pattern, path],
            capture_output=True, text=True, timeout=30,
        )
        return result.stdout or "(no matches)"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return "(search failed)"


def run_command(cmd: str) -> str:
    """Run a shell command and return combined stdout+stderr.

    Intentionally uses shell=True — nsjail provides the security boundary.
    """
    try:
        result = subprocess.run(
            cmd, shell=True,  # noqa: S602
            capture_output=True, text=True, timeout=60,
        )
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        return output.strip()
    except subprocess.TimeoutExpired:
        return "(command timed out after 60s)"


def run_tests(path: str = ".") -> str:
    """Run pytest on a path and return the output."""
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", path, "-v", "--tb=short"],
            capture_output=True, text=True, timeout=120,
        )
        return result.stdout + (result.stderr or "")
    except subprocess.TimeoutExpired:
        return "(tests timed out after 120s)"


def make_tool_namespace(base_dir: str | Path | None = None) -> dict:
    """Create a namespace dict with all literate tools.

    The tools use relative paths resolved against cwd at call time.
    The interpreter's execute() method handles chdir to base_dir.
    """
    return {
        "read_file": read_file,
        "write_file": write_file,
        "apply_diff": apply_diff,
        "search_content": search_content,
        "run_command": run_command,
        "run_tests": run_tests,
    }
