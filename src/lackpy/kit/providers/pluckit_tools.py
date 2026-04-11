"""Pluckit tool specifications for the mock provider.

Registers all pluckit API operations as mock tools in the lackpy toolbox.
Used for testing composition patterns and generating training data
before pluckit is implemented.
"""

from __future__ import annotations

from ..toolbox import ArgSpec, ToolSpec

PLUCKIT_TOOLS: list[ToolSpec] = [
    # --- Entry points ---
    ToolSpec(
        name="select_code", provider="mock",
        description="Select AST nodes with CSS selectors",
        args=[ArgSpec(name="selector", type="str", description="CSS selector over AST nodes")],
        returns="list[dict]", grade_w=0, effects_ceiling=0,
    ),
    ToolSpec(
        name="source", provider="mock",
        description="Create a source from file glob patterns",
        args=[ArgSpec(name="glob", type="str", description="File glob pattern")],
        returns="list[str]", grade_w=0, effects_ceiling=0,
    ),

    # --- Query operations ---
    ToolSpec(
        name="find", provider="mock",
        description="Narrow selection to descendants matching a selector",
        args=[ArgSpec(name="selection", type="Any"), ArgSpec(name="selector", type="str", description="CSS selector")],
        returns="list[dict]", grade_w=0, effects_ceiling=0,
    ),
    ToolSpec(
        name="filter_code", provider="mock",
        description="Filter selection by a predicate",
        args=[ArgSpec(name="selection", type="Any"), ArgSpec(name="predicate", type="str", description="Filter condition")],
        returns="list[dict]", grade_w=0, effects_ceiling=0,
    ),
    ToolSpec(
        name="callers", provider="mock",
        description="Functions that call the selected functions",
        args=[ArgSpec(name="selection", type="Any")],
        returns="list[dict]", grade_w=0, effects_ceiling=0,
    ),
    ToolSpec(
        name="callees", provider="mock",
        description="Functions called by the selected functions",
        args=[ArgSpec(name="selection", type="Any")],
        returns="list[dict]", grade_w=0, effects_ceiling=0,
    ),
    ToolSpec(
        name="references", provider="mock",
        description="All references to the selected names (imports, annotations, calls)",
        args=[ArgSpec(name="selection", type="Any")],
        returns="list[dict]", grade_w=0, effects_ceiling=0,
    ),
    ToolSpec(
        name="dependents", provider="mock",
        description="Everything that would break if this selection changed (transitive callers)",
        args=[ArgSpec(name="selection", type="Any")],
        returns="list[dict]", grade_w=0, effects_ceiling=0,
    ),
    ToolSpec(
        name="dependencies", provider="mock",
        description="Everything this selection depends on (transitive callees + imports)",
        args=[ArgSpec(name="selection", type="Any")],
        returns="list[dict]", grade_w=0, effects_ceiling=0,
    ),
    ToolSpec(
        name="reachable", provider="mock",
        description="All nodes reachable in the call graph up to max_depth",
        args=[ArgSpec(name="selection", type="Any"), ArgSpec(name="max_depth", type="int", description="Maximum traversal depth")],
        returns="list[dict]", grade_w=0, effects_ceiling=0,
    ),
    ToolSpec(
        name="call_chain", provider="mock",
        description="Linear call chain from this function in execution order",
        args=[ArgSpec(name="selection", type="Any")],
        returns="list[dict]", grade_w=0, effects_ceiling=0,
    ),
    ToolSpec(
        name="similar", provider="mock",
        description="Find structurally similar nodes (AST similarity, not text)",
        args=[ArgSpec(name="selection", type="Any"), ArgSpec(name="threshold", type="float", description="Minimum similarity 0.0-1.0")],
        returns="list[dict]", grade_w=0, effects_ceiling=0,
    ),
    ToolSpec(
        name="common_pattern", provider="mock",
        description="Compute the shared AST skeleton across all nodes in the selection",
        args=[ArgSpec(name="selection", type="Any")],
        returns="dict", grade_w=0, effects_ceiling=0,
    ),
    ToolSpec(
        name="refs", provider="mock",
        description="Name references within this selection",
        args=[ArgSpec(name="selection", type="Any"), ArgSpec(name="name", type="str", description="Optional name to filter by")],
        returns="list[dict]", grade_w=0, effects_ceiling=0,
    ),
    ToolSpec(
        name="defs", provider="mock",
        description="Name definitions within this selection",
        args=[ArgSpec(name="selection", type="Any"), ArgSpec(name="name", type="str", description="Optional name to filter by")],
        returns="list[dict]", grade_w=0, effects_ceiling=0,
    ),
    ToolSpec(
        name="unused_params", provider="mock",
        description="Find parameters not referenced within the function body",
        args=[ArgSpec(name="selection", type="Any")],
        returns="list[dict]", grade_w=0, effects_ceiling=0,
    ),
    ToolSpec(
        name="shadows", provider="mock",
        description="Find variables that shadow an outer scope variable",
        args=[ArgSpec(name="selection", type="Any")],
        returns="list[dict]", grade_w=0, effects_ceiling=0,
    ),

    # --- Reading ---
    ToolSpec(
        name="text", provider="mock",
        description="Return the source text of each node",
        args=[ArgSpec(name="selection", type="Any")],
        returns="str", grade_w=0, effects_ceiling=0,
    ),
    ToolSpec(
        name="attr", provider="mock",
        description="Return a node attribute (name, line, file, end_line)",
        args=[ArgSpec(name="selection", type="Any"), ArgSpec(name="name", type="str", description="Attribute name")],
        returns="Any", grade_w=0, effects_ceiling=0,
    ),
    ToolSpec(
        name="count", provider="mock",
        description="Return the number of nodes in the selection",
        args=[ArgSpec(name="selection", type="Any")],
        returns="int", grade_w=0, effects_ceiling=0,
    ),
    ToolSpec(
        name="names", provider="mock",
        description="Return the names of all nodes in the selection",
        args=[ArgSpec(name="selection", type="Any")],
        returns="list[str]", grade_w=0, effects_ceiling=0,
    ),
    ToolSpec(
        name="complexity", provider="mock",
        description="Return cyclomatic complexity of the selection",
        args=[ArgSpec(name="selection", type="Any")],
        returns="int", grade_w=0, effects_ceiling=0,
    ),
    ToolSpec(
        name="interface", provider="mock",
        description="Compute read/write interface from scope analysis: {reads, writes, calls}",
        args=[ArgSpec(name="selection", type="Any")],
        returns="dict", grade_w=0, effects_ceiling=0,
    ),
    ToolSpec(
        name="params", provider="mock",
        description="Return the parameters of a function",
        args=[ArgSpec(name="selection", type="Any")],
        returns="list[dict]", grade_w=0, effects_ceiling=0,
    ),
    ToolSpec(
        name="body", provider="mock",
        description="Return the body of a function, class, loop, or conditional",
        args=[ArgSpec(name="selection", type="Any")],
        returns="str", grade_w=0, effects_ceiling=0,
    ),

    # --- Structural mutations ---
    ToolSpec(
        name="addParam", provider="mock",
        description="Add a parameter to function signatures",
        args=[ArgSpec(name="selection", type="Any"), ArgSpec(name="spec", type="str", description="Parameter spec, e.g. 'timeout: int = 30'")],
        returns="list[dict]", grade_w=3, effects_ceiling=3,
    ),
    ToolSpec(
        name="removeParam", provider="mock",
        description="Remove a parameter by name",
        args=[ArgSpec(name="selection", type="Any"), ArgSpec(name="name", type="str")],
        returns="list[dict]", grade_w=3, effects_ceiling=3,
    ),
    ToolSpec(
        name="retype", provider="mock",
        description="Change a parameter's type annotation",
        args=[ArgSpec(name="selection", type="Any"), ArgSpec(name="new_type", type="str")],
        returns="list[dict]", grade_w=3, effects_ceiling=3,
    ),
    ToolSpec(
        name="rename", provider="mock",
        description="Rename a definition and all its references (scope-aware)",
        args=[ArgSpec(name="selection", type="Any"), ArgSpec(name="new_name", type="str")],
        returns="list[dict]", grade_w=3, effects_ceiling=3,
    ),
    ToolSpec(
        name="prepend", provider="mock",
        description="Insert code at the top of a body",
        args=[ArgSpec(name="selection", type="Any"), ArgSpec(name="code", type="str")],
        returns="list[dict]", grade_w=3, effects_ceiling=3,
    ),
    ToolSpec(
        name="append", provider="mock",
        description="Insert code at the bottom of a body",
        args=[ArgSpec(name="selection", type="Any"), ArgSpec(name="code", type="str")],
        returns="list[dict]", grade_w=3, effects_ceiling=3,
    ),
    ToolSpec(
        name="wrap", provider="mock",
        description="Wrap the selection in surrounding code",
        args=[ArgSpec(name="selection", type="Any"), ArgSpec(name="before", type="str"), ArgSpec(name="after", type="str")],
        returns="list[dict]", grade_w=3, effects_ceiling=3,
    ),
    ToolSpec(
        name="unwrap", provider="mock",
        description="Remove wrapping construct, dedent contents",
        args=[ArgSpec(name="selection", type="Any")],
        returns="list[dict]", grade_w=3, effects_ceiling=3,
    ),
    ToolSpec(
        name="replaceWith", provider="mock",
        description="Replace the selection's source text",
        args=[ArgSpec(name="selection", type="Any"), ArgSpec(name="code", type="str")],
        returns="list[dict]", grade_w=3, effects_ceiling=3,
    ),
    ToolSpec(
        name="remove", provider="mock",
        description="Remove the selection from the source",
        args=[ArgSpec(name="selection", type="Any")],
        returns="list[dict]", grade_w=3, effects_ceiling=3,
    ),
    ToolSpec(
        name="move_to", provider="mock",
        description="Move selection to a different file, update imports and references",
        args=[ArgSpec(name="selection", type="Any"), ArgSpec(name="path", type="str")],
        returns="list[dict]", grade_w=3, effects_ceiling=3,
    ),
    ToolSpec(
        name="extract", provider="mock",
        description="Extract selection into a new function, auto-detect parameters from scope",
        args=[ArgSpec(name="selection", type="Any"), ArgSpec(name="name", type="str")],
        returns="list[dict]", grade_w=3, effects_ceiling=3,
    ),
    ToolSpec(
        name="inline", provider="mock",
        description="Replace call sites with the function body",
        args=[ArgSpec(name="selection", type="Any")],
        returns="list[dict]", grade_w=3, effects_ceiling=3,
    ),
    ToolSpec(
        name="refactor", provider="mock",
        description="Extract common pattern into a named function, replace all instances",
        args=[ArgSpec(name="selection", type="Any"), ArgSpec(name="name", type="str")],
        returns="list[dict]", grade_w=3, effects_ceiling=3,
    ),

    # --- Delegates ---
    ToolSpec(
        name="guard", provider="mock",
        description="Add context-aware error handling",
        args=[ArgSpec(name="selection", type="Any"), ArgSpec(name="exception_type", type="str"), ArgSpec(name="strategy", type="str")],
        returns="list[dict]", grade_w=3, effects_ceiling=3,
    ),
    ToolSpec(
        name="black", provider="mock",
        description="Format selection with black",
        args=[ArgSpec(name="selection", type="Any")],
        returns="list[dict]", grade_w=1, effects_ceiling=1,
    ),
    ToolSpec(
        name="ruff_fix", provider="mock",
        description="Auto-fix selection with ruff",
        args=[ArgSpec(name="selection", type="Any")],
        returns="list[dict]", grade_w=1, effects_ceiling=1,
    ),
    ToolSpec(
        name="isort", provider="mock",
        description="Sort imports with isort",
        args=[ArgSpec(name="selection", type="Any")],
        returns="list[dict]", grade_w=1, effects_ceiling=1,
    ),
    ToolSpec(
        name="test", provider="mock",
        description="Run the selection in isolation via sandbox",
        args=[ArgSpec(name="selection", type="Any"), ArgSpec(name="inputs", type="dict", description="Optional test inputs")],
        returns="dict", grade_w=2, effects_ceiling=2,
    ),
    ToolSpec(
        name="isolate", provider="mock",
        description="Extract a block into an independently runnable form with auto-detected interface",
        args=[ArgSpec(name="selection", type="Any")],
        returns="dict", grade_w=0, effects_ceiling=0,
    ),
    ToolSpec(
        name="save", provider="mock",
        description="Commit staged mutations via jetsam. Auto-generates message from intent.",
        args=[ArgSpec(name="selection", type="Any"), ArgSpec(name="message", type="str", description="Optional commit message")],
        returns="dict", grade_w=3, effects_ceiling=3,
    ),

    # --- History ---
    ToolSpec(
        name="history", provider="mock",
        description="Every version of this selection, indexed by commit",
        args=[ArgSpec(name="selection", type="Any")],
        returns="list[dict]", grade_w=0, effects_ceiling=0,
    ),
    ToolSpec(
        name="at", provider="mock",
        description="The selection at a point in time (date, ref, or named marker)",
        args=[ArgSpec(name="selection", type="Any"), ArgSpec(name="ref", type="str", description="Time reference e.g. 'last_green_build'")],
        returns="list[dict]", grade_w=0, effects_ceiling=0,
    ),
    ToolSpec(
        name="diff", provider="mock",
        description="Structural diff between two versions of a selection",
        args=[ArgSpec(name="selection", type="Any"), ArgSpec(name="other", type="Any")],
        returns="dict", grade_w=0, effects_ceiling=0,
    ),
    ToolSpec(
        name="blame", provider="mock",
        description="Per-AST-node attribution (structural blame, not per-line)",
        args=[ArgSpec(name="selection", type="Any")],
        returns="list[dict]", grade_w=0, effects_ceiling=0,
    ),
    ToolSpec(
        name="authors", provider="mock",
        description="Who has modified this selection",
        args=[ArgSpec(name="selection", type="Any")],
        returns="list[str]", grade_w=0, effects_ceiling=0,
    ),
    ToolSpec(
        name="filmstrip", provider="mock",
        description="Visual evolution: each version as a snapshot",
        args=[ArgSpec(name="selection", type="Any")],
        returns="list[dict]", grade_w=0, effects_ceiling=0,
    ),
    ToolSpec(
        name="when", provider="mock",
        description="When did this structural property first appear?",
        args=[ArgSpec(name="selection", type="Any"), ArgSpec(name="selector", type="str")],
        returns="dict", grade_w=0, effects_ceiling=0,
    ),
    ToolSpec(
        name="co_changes", provider="mock",
        description="Find code that always changes together (shotgun surgery detection)",
        args=[ArgSpec(name="selection", type="Any"), ArgSpec(name="threshold", type="float")],
        returns="list[dict]", grade_w=0, effects_ceiling=0,
    ),

    # --- Behavior ---
    ToolSpec(
        name="coverage", provider="mock",
        description="Branch coverage for this selection",
        args=[ArgSpec(name="selection", type="Any")],
        returns="float", grade_w=0, effects_ceiling=0,
    ),
    ToolSpec(
        name="failures", provider="mock",
        description="Executions where this selection's code failed",
        args=[ArgSpec(name="selection", type="Any")],
        returns="list[dict]", grade_w=0, effects_ceiling=0,
    ),
    ToolSpec(
        name="timing", provider="mock",
        description="Execution time distribution (mean, p50, p95, trend)",
        args=[ArgSpec(name="selection", type="Any")],
        returns="dict", grade_w=0, effects_ceiling=0,
    ),
    ToolSpec(
        name="inputs", provider="mock",
        description="Observed input values from traced executions",
        args=[ArgSpec(name="selection", type="Any")],
        returns="list[dict]", grade_w=0, effects_ceiling=0,
    ),
    ToolSpec(
        name="outputs", provider="mock",
        description="Observed output values from traced executions",
        args=[ArgSpec(name="selection", type="Any")],
        returns="list[Any]", grade_w=0, effects_ceiling=0,
    ),

    # --- View ---
    ToolSpec(
        name="impact", provider="mock",
        description="Blast radius: selection + callers + tests + coverage annotations",
        args=[ArgSpec(name="selection", type="Any")],
        returns="dict", grade_w=0, effects_ceiling=0,
    ),
    ToolSpec(
        name="compare", provider="mock",
        description="Structural comparison across all nodes in the selection",
        args=[ArgSpec(name="selection", type="Any")],
        returns="dict", grade_w=0, effects_ceiling=0,
    ),

    # --- Metadata ---
    ToolSpec(
        name="intent", provider="mock",
        description="Attach intent metadata for tracing and commit messages",
        args=[ArgSpec(name="selection", type="Any"), ArgSpec(name="description", type="str")],
        returns="list[dict]", grade_w=0, effects_ceiling=0,
    ),
    ToolSpec(
        name="preview", provider="mock",
        description="Show what the chain would produce without applying it",
        args=[ArgSpec(name="selection", type="Any")],
        returns="str", grade_w=0, effects_ceiling=0,
    ),
    ToolSpec(
        name="explain", provider="mock",
        description="Show the query plan: which operations are queries, mutations, delegations",
        args=[ArgSpec(name="selection", type="Any")],
        returns="str", grade_w=0, effects_ceiling=0,
    ),
    ToolSpec(
        name="dry_run", provider="mock",
        description="Execute the chain, show the diff, but don't write to disk",
        args=[ArgSpec(name="selection", type="Any")],
        returns="dict", grade_w=0, effects_ceiling=0,
    ),
]
