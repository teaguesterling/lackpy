"""Built-in tool implementations.

These are referenced as data by the shipped ``default_tools.toml`` (via the python
provider's ``module``/``function`` resolution), not registered by hard-coded
``ToolSpec`` lists. Paths are relative to the execution cwd (the runner chdirs into
the workspace before execution).
"""

from __future__ import annotations

from pathlib import Path


def read_file(path: str) -> str:
    return Path(path).read_text()


def find_files(pattern: str) -> list[str]:
    return sorted(str(p) for p in Path(".").glob(pattern))


def write_file(path: str, content: str) -> bool:
    Path(path).write_text(content)
    return True


def edit_file(path: str, old_str: str, new_str: str) -> bool:
    p = Path(path)
    text = p.read_text()
    if old_str not in text:
        return False
    p.write_text(text.replace(old_str, new_str, 1))
    return True
