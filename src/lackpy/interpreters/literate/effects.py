"""Static effect classification for literate cells.

Maps a *compiled* cell (the Python source the kernel will ``exec``) to an
:class:`CellEffects` profile -- *without running it*. The profile says, in terms
of lackpy's existing :class:`~lackpy.lang.grader.Grade` lattice
(``w``: 0=pure, 1=pinhole read, 2=scoped exec, 3=scoped write):

  - what grade the cell needs (so the step can refuse cells over a profile's
    effect ceiling *before* a single effect happens),
  - which file paths it writes/reads with a *statically known literal* (so the
    step can journal exactly those files and roll them back on failure), and
  - whether it reaches for effect surface this pass can't bound (``import``,
    ``open``, raw exec) or a dynamic path it can't journal -- i.e. the parts that
    must be *sandboxed* rather than *transacted*.

This is the foundation the effect-aware step consumes four ways: the ceiling
gate, the file journal, the sandbox decision, and the dry-run manifest shown to
the model/policy before commit.

**Grades are config-driven, not hardcoded.** Each literate tool's grade + effect
kind + path argument is declared in ``tool_effects.toml`` (a ToolSpec-shaped
table) and loaded into :data:`LITERATE_TOOL_EFFECTS`. Callers may inject a
different map via ``classify_effects(..., tool_effects=)`` -- the seam through
which the resolved toolbox will supply grades once the literate tools are
registered as real graded specs (today they bypass the toolbox as plain
functions; unifying them is the remaining follow-up).

**Scope -- read this.** This is a *cooperative planner*, NOT a security boundary.
Because literate cells run un-restricted Python (the prompt permits ``import os``
etc.), no name-based AST pass can soundly enumerate every effect path. The
classifier classifies the *declared* surface (the literate tools) precisely and
flags a curated set of raw escape hatches as ``unanalyzable``; a determined cell
can still alias around it. Soundness against adversarial code comes from the
restricted-AST whitelist (the ``python`` interpreter's validator) and/or nsjail --
not from this module. ``unanalyzable`` is exactly the "I can't bound this; the
step must sandbox it and must not promise rollback" signal.
"""

from __future__ import annotations

import ast
import tomllib
from dataclasses import dataclass
from importlib.resources import files
from typing import Mapping

from ...lang.grader import Grade


@dataclass(frozen=True)
class ToolEffect:
    """Effect descriptor for one literate tool (one row of tool_effects.toml).

    Attributes:
        grade: The tool's world-coupling/effects grade.
        kind: ``"read"`` | ``"write"`` | ``"exec"`` -- selects the enforcement
            mechanism (read=run, write=journal, exec=sandbox).
        path_arg: Name of the parameter carrying a file path, or None.
        path_index: Positional index of that parameter, or None.
    """

    grade: Grade
    kind: str
    path_arg: str | None = None
    path_index: int | None = None


def _load_tool_effects() -> dict[str, ToolEffect]:
    """Load the literate tool effect table from the shipped TOML data."""
    raw = (files(__package__) / "tool_effects.toml").read_text(encoding="utf-8")
    table: dict[str, ToolEffect] = {}
    for t in tomllib.loads(raw).get("tools", []):
        table[t["name"]] = ToolEffect(
            grade=Grade(w=t.get("grade_w", 3), d=t.get("effects_ceiling", 3)),
            kind=t["kind"],
            path_arg=t.get("path_arg"),
            path_index=t.get("path_index"),
        )
    return table


# The literate tool namespace's effect grades -- the single, config-driven source
# of truth (tool_effects.toml). Injectable per-call via classify_effects().
LITERATE_TOOL_EFFECTS: dict[str, ToolEffect] = _load_tool_effects()

# Raw effect primitives that defeat name-based classification. Calling any of
# these (or importing anything) means we can no longer bound the cell's effects.
_ESCAPE_HATCH_CALLS: frozenset[str] = frozenset(
    {"open", "eval", "exec", "compile", "__import__", "globals", "locals",
     "getattr", "setattr", "delattr", "vars", "input"}
)

_PURE = Grade(0, 0)
_CONSERVATIVE = Grade(3, 3)


@dataclass(frozen=True)
class CellEffects:
    """A literate cell's statically-determined effect profile.

    Attributes:
        grade: Aggregate world-coupling/effects grade (max over the cell's
            recognized effectful calls; forced to ``Grade(3, 3)`` when
            ``unanalyzable``).
        writes: File paths the cell writes via a *literal* argument -- the exact
            set the step can journal for rollback.
        reads: File paths the cell reads via a literal argument (informational;
            reads need no rollback).
        exec_calls: Names of exec-kind tools the cell invokes (``run_command``,
            ``run_tests``). Non-empty => a sandbox is required.
        dynamic_paths: A path-bearing tool was called with a non-literal path, so
            its target can't be journaled statically.
        unanalyzable: The cell reaches raw effect surface (``import`` / ``open`` /
            raw exec) this pass can't bound. The step must sandbox it and must
            not promise transactional rollback.
    """

    grade: Grade
    writes: frozenset[str]
    reads: frozenset[str]
    exec_calls: frozenset[str]
    dynamic_paths: bool
    unanalyzable: bool

    @property
    def needs_sandbox(self) -> bool:
        """Whether running this cell safely requires a sandbox boundary."""
        return self.unanalyzable or bool(self.exec_calls)

    @property
    def transactional(self) -> bool:
        """Whether the cell's filesystem effects can be journaled and rolled back.

        True when every effect is either a pure/read op (nothing to undo) or a
        write to a statically-known path. Exec, dynamic paths, or raw effect
        surface make the cell non-transactional -- its effects can't be cleanly
        reversed at the ``@continue`` commit boundary.
        """
        return not (self.unanalyzable or self.dynamic_paths or self.exec_calls)


def classify_effects(
    compiled_source: str,
    *,
    tool_effects: Mapping[str, ToolEffect] | None = None,
) -> CellEffects:
    """Classify the effects of one compiled cell without executing it.

    Args:
        compiled_source: The Python source the kernel will ``exec`` for this
            cell (the output of ``compiler._COMPILERS[...]``). Annotated cells
            (``@read``/``@write``/``@diff``) compile to literal-path tool calls,
            so they classify through the same path as raw code cells.
        tool_effects: Name -> :class:`ToolEffect` map defining which calls are
            graded effects. Defaults to :data:`LITERATE_TOOL_EFFECTS` (loaded
            from ``tool_effects.toml``); inject a toolbox-derived map to override.

    Returns:
        A :class:`CellEffects`. A cell that doesn't parse is treated as
        ``unanalyzable`` (the kernel's own static check reports the syntax error
        separately; here we just refuse to vouch for its effects).
    """
    effects_map = tool_effects if tool_effects is not None else LITERATE_TOOL_EFFECTS
    try:
        tree = ast.parse(compiled_source)
    except SyntaxError:
        return CellEffects(
            grade=_CONSERVATIVE, writes=frozenset(), reads=frozenset(),
            exec_calls=frozenset(), dynamic_paths=True, unanalyzable=True,
        )

    grade = _PURE
    writes: set[str] = set()
    reads: set[str] = set()
    exec_calls: set[str] = set()
    dynamic_paths = False
    unanalyzable = False

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            unanalyzable = True
            continue
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            # Attribute/subscript calls (p.write_text(), os.remove()) can hide
            # effects we can't attribute to a known tool. We don't flag every
            # method call as unanalyzable (str.join etc. are everywhere); the
            # curated escape-hatch set below covers the dangerous bare names.
            continue

        name = node.func.id
        if name in _ESCAPE_HATCH_CALLS:
            unanalyzable = True
            continue

        te = effects_map.get(name)
        if te is None:
            continue

        grade = _max_grade(grade, te.grade)
        if te.kind == "exec":
            exec_calls.add(name)
        elif te.kind in ("read", "write"):
            literal = _literal_path(node, te.path_arg, te.path_index)
            if literal is None:
                dynamic_paths = True
            elif te.kind == "write":
                writes.add(literal)
            else:
                reads.add(literal)

    if unanalyzable:
        grade = _max_grade(grade, _CONSERVATIVE)

    return CellEffects(
        grade=grade,
        writes=frozenset(writes),
        reads=frozenset(reads),
        exec_calls=frozenset(exec_calls),
        dynamic_paths=dynamic_paths,
        unanalyzable=unanalyzable,
    )


def combine(effects: list[CellEffects]) -> CellEffects:
    """Aggregate a segment's per-cell effects into one profile for the manifest.

    Grade is the max; path sets union; the boolean flags OR together. Describes
    everything that happens between two ``@continue`` commit points -- the ceiling
    gate and the dry-run manifest both consume this."""
    grade = _PURE
    writes: set[str] = set()
    reads: set[str] = set()
    exec_calls: set[str] = set()
    dynamic_paths = False
    unanalyzable = False
    for e in effects:
        grade = _max_grade(grade, e.grade)
        writes |= e.writes
        reads |= e.reads
        exec_calls |= e.exec_calls
        dynamic_paths = dynamic_paths or e.dynamic_paths
        unanalyzable = unanalyzable or e.unanalyzable
    return CellEffects(
        grade=grade, writes=frozenset(writes), reads=frozenset(reads),
        exec_calls=frozenset(exec_calls), dynamic_paths=dynamic_paths,
        unanalyzable=unanalyzable,
    )


def exceeds_ceiling(effects: CellEffects, ceiling: Grade) -> str | None:
    """Return a human-readable reason if ``effects`` exceeds ``ceiling``, else None.

    The gate compares both grade axes -- world-coupling (``w``) and effects-depth
    (``d``). The literate step calls this on a segment's combined effects to refuse
    a document whose aggregate effects exceed the profile's allowed grade *before*
    any cell runs. ``w`` is reported first because it is the coupling axis that
    decides enforcement (read/exec/write)."""
    if effects.grade.w > ceiling.w:
        return f"world-coupling w={effects.grade.w} exceeds ceiling w={ceiling.w}"
    if effects.grade.d > ceiling.d:
        return f"effects-depth d={effects.grade.d} exceeds ceiling d={ceiling.d}"
    return None


def _max_grade(a: Grade, b: Grade) -> Grade:
    return Grade(w=max(a.w, b.w), d=max(a.d, b.d))


def _literal_path(node: ast.Call, pname: str | None, idx: int | None) -> str | None:
    """Extract a string-literal path argument, or None if absent/non-literal."""
    if pname is not None:
        for kw in node.keywords:  # keyword form: f(path="x")
            if kw.arg == pname:
                return _const_str(kw.value)
    if idx is not None and idx < len(node.args):  # positional form: f("x")
        return _const_str(node.args[idx])
    return None


def _const_str(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None
