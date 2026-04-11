"""Harness-local eval kit — builtin file tools + grep-based code-intel helpers.

This module does NOT live in `src/lackpy/kit/providers/` on purpose: it is
a research-only kit used by the prompt evaluation harness. Production
lackpy users build their own kits (pluckit, fledgling, etc.) — this kit
exists so the python-interpreter corpus can orchestrate `find_def` and
`find_refs` tool calls without depending on an external code-intel layer.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from lackpy.kit.registry import ResolvedKit
from lackpy.kit.toolbox import ArgSpec, ToolSpec
from lackpy.lang.grader import compute_grade


def build_eval_kit(base_dir: Path) -> ResolvedKit:
    """Construct the eval kit rooted at `base_dir`.

    The kit exposes four tools:
      - read_file(path: str) -> str
      - find_files(pattern: str) -> list[str]
      - find_def(name: str) -> list[dict]   (grep for def/class <name>)
      - find_refs(name: str) -> list[dict]  (grep for <name>()  call sites)

    find_def and find_refs are closed over `base_dir` so the corpus
    intents can reference symbols by name without worrying about cwd.
    """
    base_dir = Path(base_dir).resolve()

    def _read(path: str) -> str:
        p = Path(path)
        p = p if p.is_absolute() else (base_dir / p)
        p = p.resolve()
        if not p.is_relative_to(base_dir):
            raise PermissionError(
                f"Path {path!r} escapes the eval kit's base_dir"
            )
        return p.read_text()

    def _glob(pattern: str) -> list[str]:
        return sorted(str(p.relative_to(base_dir)) for p in base_dir.glob(pattern))

    def _find_def(name: str) -> list[dict]:
        """Return rows for every `def <name>(` or `class <name>(` site."""
        pattern = re.compile(rf"^\s*(?:async\s+)?(?:def|class)\s+{re.escape(name)}\b")
        rows: list[dict] = []
        for pyfile in sorted(base_dir.rglob("*.py")):
            for i, line in enumerate(pyfile.read_text().splitlines(), start=1):
                if pattern.search(line):
                    rows.append({
                        "file": str(pyfile.relative_to(base_dir)),
                        "line": i,
                        "text": line.strip(),
                    })
        return rows

    def _find_refs(name: str) -> list[dict]:
        """Return rows for every `<name>(` call site (excluding its own def/class line)."""
        call_re = re.compile(rf"\b{re.escape(name)}\s*\(")
        def_re = re.compile(rf"^\s*(?:async\s+)?(?:def|class)\s+{re.escape(name)}\s*\(")
        rows: list[dict] = []
        for pyfile in sorted(base_dir.rglob("*.py")):
            for i, line in enumerate(pyfile.read_text().splitlines(), start=1):
                if def_re.search(line):
                    continue
                if call_re.search(line):
                    rows.append({
                        "file": str(pyfile.relative_to(base_dir)),
                        "line": i,
                        "text": line.strip(),
                    })
        return rows

    tools: dict[str, ToolSpec] = {
        "read_file": ToolSpec(
            name="read_file", provider="eval",
            description="Read a file under the toybox base_dir; returns its text.",
            args=[ArgSpec(name="path", type="str", description="Relative or absolute path")],
            returns="str", grade_w=1, effects_ceiling=1,
        ),
        "find_files": ToolSpec(
            name="find_files", provider="eval",
            description="Glob files under the toybox base_dir.",
            args=[ArgSpec(name="pattern", type="str", description="Glob pattern, e.g. '**/*.py'")],
            returns="list[str]", grade_w=1, effects_ceiling=1,
        ),
        "find_def": ToolSpec(
            name="find_def", provider="eval",
            description="Find where a function or class named `name` is defined. Returns a list of {file, line, text} dicts.",
            args=[ArgSpec(name="name", type="str", description="Symbol name to look up")],
            returns="list[dict]", grade_w=1, effects_ceiling=1,
        ),
        "find_refs": ToolSpec(
            name="find_refs", provider="eval",
            description="Find call sites for `name`. Returns a list of {file, line, text} dicts for every `name(` occurrence.",
            args=[ArgSpec(name="name", type="str", description="Symbol name to look up")],
            returns="list[dict]", grade_w=1, effects_ceiling=1,
        ),
    }
    callables: dict[str, Any] = {
        "read_file": _read,
        "find_files": _glob,
        "find_def": _find_def,
        "find_refs": _find_refs,
    }

    grade_input = {
        n: {"grade_w": s.grade_w, "effects_ceiling": s.effects_ceiling}
        for n, s in tools.items()
    }
    grade = compute_grade(grade_input)

    description_lines = [
        "Available tools (call by name):",
        "  read_file(path: str) -> str — read a file",
        "  find_files(pattern: str) -> list[str] — glob files",
        "  find_def(name: str) -> list[dict] — find definitions (function or class)",
        "  find_refs(name: str) -> list[dict] — find call sites",
    ]
    return ResolvedKit(
        tools=tools,
        callables=callables,
        grade=grade,
        description="\n".join(description_lines),
    )
