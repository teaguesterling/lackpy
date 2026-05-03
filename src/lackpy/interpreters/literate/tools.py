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
    """Parse and apply a unified diff to a list of lines."""
    result = list(original)
    offset = 0

    for hunk_start, hunk_old_start, hunk_old_count, removals, additions in _parse_hunks(diff_text):
        pos = hunk_old_start - 1 + offset

        for line_offset in sorted(removals, reverse=True):
            idx = pos + line_offset
            if 0 <= idx < len(result):
                result.pop(idx)

        insert_pos = pos + (min(removals) if removals else 0)
        for i, line in enumerate(additions):
            result.insert(insert_pos + i, line)

        offset += len(additions) - len(removals)

    return result


def _parse_hunks(diff_text: str):
    """Yield (hunk_index, old_start, old_count, removal_offsets, addition_lines) from unified diff."""
    lines = diff_text.splitlines()
    hunk_idx = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("@@"):
            parts = line.split()
            old_range = parts[1]  # -start,count
            old_start = int(old_range.split(",")[0].lstrip("-"))
            old_count_str = old_range.split(",")[1] if "," in old_range else "1"
            old_count = int(old_count_str)

            removals: list[int] = []
            additions: list[str] = []
            context_offset = 0
            i += 1
            while i < len(lines) and not lines[i].startswith("@@"):
                dl = lines[i]
                if dl.startswith("-"):
                    removals.append(context_offset)
                    context_offset += 1
                elif dl.startswith("+"):
                    additions.append(dl[1:] + "\n")
                elif dl.startswith(" "):
                    context_offset += 1
                else:
                    break
                i += 1

            yield (hunk_idx, old_start, old_count, removals, additions)
            hunk_idx += 1
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
    """Run a shell command and return combined stdout+stderr."""
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
    """Create a namespace dict with all literate tools, rooted at base_dir."""
    prev = os.getcwd()
    if base_dir:
        os.chdir(base_dir)

    ns = {
        "read_file": read_file,
        "write_file": write_file,
        "apply_diff": apply_diff,
        "search_content": search_content,
        "run_command": run_command,
        "run_tests": run_tests,
    }

    if base_dir:
        os.chdir(prev)

    return ns
