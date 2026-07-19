"""Dirty-subgraph re-execution — the minimal derived dependency graph (L1.4).

The last forgiveness semantic (sem 3): when a name is RE-ASSERTED (L1.3 already
detects this and versions it), the cells that transitively DEPEND on that name
are stale and must be RE-EXECUTED — and ONLY those, not the whole document, not
independent cells.

DECIDED APPROACH (design-lackpy-L1.4, option (a)): a *minimal derived
dependency graph*, NOT a plan IR.  Cells stay Python source strings and still
``exec`` directly (ground rule 5's "IR core" is explicitly DEFERRED as debt).
The graph is a derived side-structure built by REUSING the per-cell def/ref sets
:mod:`static_analysis` already computes (``collect_names`` /
``collect_bindings``) — no new IR, no surface→IR compiler.

Edge rule (last-writer-wins, consistent with L1.3's latest-wins namespace)
--------------------------------------------------------------------------
A cell that REFERENCES name ``X`` depends on the cell that LAST BOUND ``X``.
:meth:`DependencyGraph.dirty_closure` turns a set of re-asserted names into the
ordered set of dependent cells to re-run.

Staleness is FORWARD and INDEX-GATED (the correctness core)
-----------------------------------------------------------
This module runs on the batch/session path (``_run_document``), where a
document's cells execute LINEARLY top-to-bottom exactly once, then this pass
re-runs the stale dependents.  Two rules make "only the stale dependents" exact:

* **Direct seed is index-gated.**  A re-asserted name ``N``'s current (latest)
  binder sits at some cell index ``cb``.  A reader of ``N`` at index ``< cb``
  used a now-superseded version → STALE → dirty.  A reader at index ``>= cb``
  already used the current value → NOT stale (this is what keeps a same-round
  ``x = 41`` / ``y = x + 1`` from spuriously re-running ``y`` — ``y`` already
  saw ``41``).
* **Propagation is forward-only.**  Re-running a dirty cell changes the names it
  binds; every reader of those names at a *higher* index becomes dirty in turn
  (the transitive subgraph).  Forward-only (``i > d``) both matches
  define-before-use ordering (a name used before it is bound is a hole, not a
  dependency) AND is the cycle-breaker: indices strictly increase, so the
  worklist cannot loop — the cascade always terminates.

The graph is DERIVED/ADVISORY, so the dependency edges are approximate: static
def/ref only.  Dynamic names, ``import *``, and aliasing are NOT tracked; nor is
in-memory mutation (``lst.append(x)``).  Callers must bound re-execution to
effect-free cells (see ``_run_document``'s purity gate) so the approximation can
never replay a world effect.
"""

from __future__ import annotations

from collections import defaultdict

#: Ledger entry type for a dirtied cell (L1.4): a cell marked stale by an
#: upstream re-assertion.  ``detail`` carries ``cell_index``, ``triggered_by``
#: (the re-asserted upstream names), and ``reexecuted`` (whether the cell was
#: actually re-run — False when it was withheld because re-execution would
#: replay a world effect).  Nothing silent: every dirtied cell is ledgered even
#: when it is not re-run.
DIRTY = "dirty"


class DependencyGraph:
    """Derived per-document dependency graph over source-string cells (L1.4).

    Built incrementally as ``_run_document`` walks the cells: each cell
    contributes its statically-collected REFERENCE and BINDING name sets
    (:func:`static_analysis.collect_names` / ``collect_bindings``).  No IR, no
    execution state — just the def/ref sides needed to answer "which cells
    transitively depend on these re-asserted names, in re-exec order?".

    One instance is created per ``_run_document`` call (per document / per
    session round).  The graph is NOT threaded across rounds: a stale cell must
    be a cell that exists in the current document to be re-runnable.
    """

    def __init__(self) -> None:
        self._refs: dict[int, frozenset[str]] = {}
        self._binds: dict[int, frozenset[str]] = {}
        self._order: list[int] = []

    def add_cell(self, index: int, refs: set[str], binds: set[str]) -> None:
        """Record one cell's static reference/binding sets.

        Called for EVERY cell — executed, reified (hole/error/unavailable), or
        skipped — so the closure is complete even when a cell never produced a
        runtime namespace delta.  Cells that don't compile/parse contribute
        empty sets (they cannot participate in name-flow dependencies).
        """
        self._refs[index] = frozenset(refs)
        self._binds[index] = frozenset(binds)
        self._order.append(index)

    def _current_binders(self) -> dict[str, int]:
        """``name -> highest cell index that binds it`` (last-writer-wins)."""
        binders: dict[str, int] = {}
        for i in self._order:  # ascending document order
            for name in self._binds[i]:
                binders[name] = i
        return binders

    def dirty_closure(
        self, reasserted_names: set[str]
    ) -> list[tuple[int, set[str]]]:
        """Cells made stale by re-asserting ``reasserted_names``, in re-exec order.

        Returns ``[(cell_index, triggering_names), ...]`` sorted by ascending
        cell index (a valid topological / dependency order for a
        define-before-use document).  The seed is index-gated (readers strictly
        BEFORE a re-asserted name's current binder) and propagation is
        forward-only (readers at a HIGHER index than a just-dirtied binder) — so
        an independent cell, and a dependent that already saw the new value, are
        both correctly excluded, and the cascade always terminates.
        """
        binders = self._current_binders()
        triggers: dict[int, set[str]] = defaultdict(set)
        dirty: set[int] = set()

        # Direct seed: a reader BEFORE the current binder used a stale version.
        for name in reasserted_names:
            binder = binders.get(name)
            if binder is None:
                continue
            for i in self._order:
                if i < binder and name in self._refs[i]:
                    dirty.add(i)
                    triggers[i].add(name)

        # Forward propagation: re-running a dirty cell changes the names it
        # binds; higher-index readers of those names become dirty in turn.
        worklist = sorted(dirty)
        while worklist:
            d = worklist.pop(0)
            for name in self._binds[d]:
                for i in self._order:
                    if i > d and name in self._refs[i] and i not in dirty:
                        dirty.add(i)
                        triggers[i].add(name)
                        worklist.append(i)

        return [(i, triggers[i]) for i in sorted(dirty)]
