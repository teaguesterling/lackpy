"""ast-select interpreter corpus: bare-selector view composition."""

from __future__ import annotations

from typing import Any

from .intents import GateResult, Intent


def _ast_select_gate(program: str) -> GateResult:
    """Structural gate for bare selectors.

    Accepts any non-empty single-line string that contains no brace
    characters. Deliberately loose about the selector syntax itself;
    we do not want to penalize correct selectors for a pluckit-grammar
    mismatch at gate time — the execution stage will catch those.
    """
    if not program or not program.strip():
        return GateResult(passed=False, errors=["empty program"])
    stripped = program.strip()
    if "{" in stripped or "}" in stripped:
        return GateResult(
            passed=False,
            errors=["bare selector must not contain declaration braces"],
        )
    lines = [ln for ln in stripped.splitlines() if ln.strip()]
    if len(lines) > 1:
        return GateResult(
            passed=False,
            errors=[f"expected a single-line selector, got {len(lines)} lines"],
        )
    # Must start with a class selector (pluckit uses .fn, .cls, .call, etc.)
    if not stripped.lstrip().startswith((".", "#", "*", "[")):
        return GateResult(
            passed=False,
            errors=["selector should start with . # * or ["],
        )
    return GateResult(passed=True)


def _markdown_contains(substrs: list[str]):
    """Execution assertion: the rendered markdown mentions each substring."""
    def check(output: Any) -> bool:
        if not isinstance(output, str):
            return False
        return all(s in output for s in substrs)
    return check


def _markdown_nonempty(output: Any) -> bool:
    return isinstance(output, str) and len(output.strip()) > 0


def _markdown_count_at_least(n: int):
    def check(output: Any) -> bool:
        if not isinstance(output, str):
            return False
        # Count per-match H2 headings: the renderer uses "## " for each match
        return output.count("\n## ") + (1 if output.startswith("## ") else 0) >= n
    return check


AST_SELECT_INTENTS: list[Intent] = [
    # Core (8)
    Intent(
        id="as.core.01",
        interpreter="ast-select",
        difficulty="core",
        text="Show every function defined in the codebase as a view.",
        return_shape="markdown",
        structural_gate=_ast_select_gate,
        exec_assertion=_markdown_count_at_least(5),
        notes="Simplest `.fn` selector; the toybox has many fns so a match count >=5 is a generous lower bound.",
    ),
    Intent(
        id="as.core.02",
        interpreter="ast-select",
        difficulty="core",
        text="Show the class named User as a view.",
        return_shape="markdown",
        structural_gate=_ast_select_gate,
        exec_assertion=_markdown_contains(["User"]),
        notes=".cls#User or .cls[name='User'].",
    ),
    Intent(
        id="as.core.03",
        interpreter="ast-select",
        difficulty="core",
        text="Show the function named validate_token as a view.",
        return_shape="markdown",
        structural_gate=_ast_select_gate,
        exec_assertion=_markdown_contains(["validate_token"]),
        notes=".fn#validate_token.",
    ),
    Intent(
        id="as.core.04",
        interpreter="ast-select",
        difficulty="core",
        text="Show every function whose name starts with 'test_' as a view.",
        return_shape="markdown",
        structural_gate=_ast_select_gate,
        exec_assertion=_markdown_contains(["test_"]),
        notes='.fn[name^="test_"]',
    ),
    Intent(
        id="as.core.05",
        interpreter="ast-select",
        difficulty="core",
        text="Show every function decorated with @deprecated as a view.",
        return_shape="markdown",
        structural_gate=_ast_select_gate,
        exec_assertion=_markdown_nonempty,
        notes="Pluckit-grammar-contingent (decorator match). The assertion accepts any non-empty output because models may use variant syntax.",
    ),
    Intent(
        id="as.core.06",
        interpreter="ast-select",
        difficulty="core",
        text="Show every async function in the codebase as a view.",
        return_shape="markdown",
        structural_gate=_ast_select_gate,
        exec_assertion=_markdown_nonempty,
        notes="The toybox has no async functions by design; an empty match set renders to '' which fails _markdown_nonempty. Accept any non-empty output as a sign the selector was understood; a true zero-match successful run is indistinguishable from an execution failure. Revisit if async functions land in toybox v2.",
    ),
    Intent(
        id="as.core.07",
        interpreter="ast-select",
        difficulty="core",
        text="Show every function decorated with @route as a view.",
        return_shape="markdown",
        structural_gate=_ast_select_gate,
        exec_assertion=_markdown_nonempty,
        notes="Pluckit-grammar-contingent. Should match 4 routes in app.py.",
    ),
    Intent(
        id="as.core.08",
        interpreter="ast-select",
        difficulty="core",
        text="Show every function whose name starts with an underscore as a view.",
        return_shape="markdown",
        structural_gate=_ast_select_gate,
        exec_assertion=_markdown_contains(["_"]),
        notes='.fn[name^="_"]',
    ),

    # Stretch (6)
    Intent(
        id="as.stretch.01",
        interpreter="ast-select",
        difficulty="stretch",
        text="Show every private method (name starting with underscore) of the class User as a view.",
        return_shape="markdown",
        structural_gate=_ast_select_gate,
        exec_assertion=_markdown_contains(["_internal_state"]),
        notes='.cls#User .fn[name^="_"] — nested selector.',
    ),
    Intent(
        id="as.stretch.02",
        interpreter="ast-select",
        difficulty="stretch",
        text="Show every function that contains a call to execute_sql as a view.",
        return_shape="markdown",
        structural_gate=_ast_select_gate,
        exec_assertion=_markdown_nonempty,
        notes=".fn:has(.call#execute_sql) — pluckit :has support needed.",
    ),
    Intent(
        id="as.stretch.03",
        interpreter="ast-select",
        difficulty="stretch",
        text="Show every function whose name does not start with 'test_' as a view.",
        return_shape="markdown",
        structural_gate=_ast_select_gate,
        exec_assertion=_markdown_nonempty,
        notes='.fn:not([name^="test_"]) — pluckit :not with attribute selector.',
    ),
    Intent(
        id="as.stretch.04",
        interpreter="ast-select",
        difficulty="stretch",
        text="Show every route handler (a function with @route) that also calls execute_sql.",
        return_shape="markdown",
        structural_gate=_ast_select_gate,
        exec_assertion=_markdown_nonempty,
        notes="Combined :has + :has or :has + attribute — hardest stretch item for ast-select; may need to be dropped if pluckit cannot chain two :has clauses on the same node.",
    ),
    Intent(
        id="as.stretch.05",
        interpreter="ast-select",
        difficulty="stretch",
        text="Show every class that contains an __init__ method as a view.",
        return_shape="markdown",
        structural_gate=_ast_select_gate,
        exec_assertion=_markdown_contains(["User"]),
        notes=".cls:has(.fn#__init__). User and Session both have __init__.",
    ),
    Intent(
        id="as.stretch.06",
        interpreter="ast-select",
        difficulty="stretch",
        text="Show every function that contains a raise statement as a view.",
        return_shape="markdown",
        structural_gate=_ast_select_gate,
        exec_assertion=_markdown_nonempty,
        notes=".fn:has(.raise) — depends on pluckit exposing raise nodes.",
    ),
]
