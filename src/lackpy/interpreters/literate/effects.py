"""Effect helpers for literate cells (literate stepy Task 3, design D6/D7).

Two questions, two helpers:

* **Before a cell runs** — :func:`classify_effects` grades the cell's compiled
  source from the ``Path`` members it names, in terms of lackpy's
  :class:`~lackpy.lang.grader.Grade` lattice (``w``: 0 pure, 1 pinhole read,
  3 scoped write). The ceiling gate in the interpreter uses it to reify a cell
  over the profile's ceiling as ``Unavailable`` instead of running it (L1.5).
  This is a cooperative planner, not the boundary: a write it cannot see (a
  helper defined in an earlier cell) is refused by the policy, and a read-only
  profile's policy is :func:`lackpy.run.paths.read_only_policy` (D7).
* **After a cell ran** — :func:`path_writes` names the ``Path`` writes in the
  cell's slice of the session trace. Strict reactive mode withholds a re-run of
  a cell that wrote (D7). The ``Path`` stand-in is a cell's only route to the
  filesystem, so the trace sees every write.

The kit grade map (``tool_effects.toml``) and the audit-hook observer are gone
with the kit tools (D7).
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass

from ...lang.grader import Grade
from ...run.trace import TraceEntry

#: ``Path`` members that write, and the trace tool names they record.
PATH_WRITE_MEMBERS = frozenset({"write_text", "write_bytes", "mkdir"})
PATH_WRITE_TOOLS = frozenset(f"Path.{m}" for m in PATH_WRITE_MEMBERS)
#: ``Path`` members that read the filesystem.
PATH_READ_MEMBERS = frozenset({"read_text", "read_bytes", "exists", "is_file", "is_dir",
                               "glob", "rglob", "iterdir"})

_PURE = Grade(0, 0)
_READ = Grade(1, 1)
_WRITE = Grade(3, 3)


@dataclass(frozen=True)
class CellEffects:
    """A cell's statically-determined effect grade."""

    grade: Grade


def classify_effects(compiled_source: str) -> CellEffects:
    """Grade one compiled cell without running it.

    Every attribute named ``write_text``, ``write_bytes`` or ``mkdir`` — called
    or not, so an alias counts — makes the cell a scoped write; a ``Path`` read
    member makes it a pinhole read. A cell that does not parse is graded a
    write (conservative); the kernel reports the syntax error.
    """
    try:
        tree = ast.parse(compiled_source)
    except SyntaxError:
        return CellEffects(_WRITE)
    grade = _PURE
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if node.attr in PATH_WRITE_MEMBERS:
                return CellEffects(_WRITE)
            if node.attr in PATH_READ_MEMBERS:
                grade = _READ
    return CellEffects(grade)


def classify_cell(cell_type: str, compiled_source: str) -> CellEffects:
    """Grade one cell: a ``@diff`` is a write by its type (its compiled source
    is a comment: the kernel applies it on the host side); any other cell by
    :func:`classify_effects`."""
    if cell_type == "diff":
        return CellEffects(_WRITE)
    return classify_effects(compiled_source)


def path_writes(entries: Iterable[TraceEntry]) -> tuple[str, ...]:
    """The ``Path`` write tools in ``entries``, each once, sorted.

    A failed write counts too: it may have changed the file part way.
    """
    return tuple(sorted({e.tool for e in entries if e.tool in PATH_WRITE_TOOLS}))


def exceeds_ceiling(effects: CellEffects, ceiling: Grade) -> str | None:
    """Return a human-readable reason if ``effects`` exceeds ``ceiling``, else None.

    Both grade axes are compared, world-coupling (``w``) first."""
    if effects.grade.w > ceiling.w:
        return f"world-coupling w={effects.grade.w} exceeds ceiling w={ceiling.w}"
    if effects.grade.d > ceiling.d:
        return f"effects-depth d={effects.grade.d} exceeds ceiling d={ceiling.d}"
    return None


def as_grade(value: object) -> Grade:
    """Coerce a ceiling spec into a :class:`Grade`.

    Accepts a ``Grade`` (returned as-is), a ``(w, d)`` pair, or a single int
    (applied to both axes), so a config/TOML-sourced ``grade_ceiling`` works."""
    if isinstance(value, Grade):
        return value
    if isinstance(value, bool):  # bool is an int subclass; reject the ambiguity
        raise TypeError(f"grade_ceiling must not be a bool; got {value!r}")
    if isinstance(value, int):
        return Grade(value, value)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return Grade(int(value[0]), int(value[1]))
    raise TypeError(
        f"grade_ceiling must be a Grade, (w, d) pair, or int; got {value!r}"
    )
