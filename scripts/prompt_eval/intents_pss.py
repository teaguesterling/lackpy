"""pss interpreter corpus: multi-rule selector sheets."""

from __future__ import annotations

from typing import Any

from .intents import GateResult, Intent


def _pss_gate(program: str) -> GateResult:
    """Structural gate for selector sheets.

    Rules: non-empty, balanced braces, and must look like a selector
    sheet (contains a CSS-style selector token like .fn, .cls, .call).
    Rejects Python code that happens to have balanced braces.
    """
    if not program or not program.strip():
        return GateResult(passed=False, errors=["empty program"])
    stripped = program.strip()
    open_count = stripped.count("{")
    close_count = stripped.count("}")
    if open_count != close_count:
        return GateResult(
            passed=False,
            errors=[f"unbalanced braces: {open_count} opening, {close_count} closing"],
        )
    # Must contain at least one selector-like token. CSS selectors in
    # pluckit start with . (e.g. .fn, .cls, .call). Without this check,
    # Python code with balanced braces passes the gate.
    import re
    has_selector = bool(re.search(r'\.(fn|cls|call|decorator|raise)\b', stripped))
    if not has_selector:
        # Also accept bare # selectors (e.g. #validate_token)
        has_selector = bool(re.search(r'#\w+', stripped))
    if not has_selector:
        return GateResult(
            passed=False,
            errors=["program does not contain a CSS-style selector (.fn, .cls, etc.)"],
        )
    return GateResult(passed=True)


def _markdown_contains(substrs: list[str]):
    def check(output: Any) -> bool:
        if not isinstance(output, str):
            return False
        return all(s in output for s in substrs)
    return check


def _markdown_nonempty(output: Any) -> bool:
    return isinstance(output, str) and len(output.strip()) > 0


PSS_INTENTS: list[Intent] = [
    # Core (8) — 1-2 rule sheets targeting known-safe symbols
    Intent(
        id="pss.core.01",
        interpreter="pss",
        difficulty="core",
        text="Create a selector sheet that shows the validate_token function with its full body.",
        return_shape="markdown",
        structural_gate=_pss_gate,
        exec_assertion=_markdown_contains(["validate_token"]),
        notes="1-rule: .fn#validate_token { show: body; }",
    ),
    Intent(
        id="pss.core.02",
        interpreter="pss",
        difficulty="core",
        text="Create a selector sheet that shows an outline of the class User.",
        return_shape="markdown",
        structural_gate=_pss_gate,
        exec_assertion=_markdown_contains(["User"]),
        notes="1-rule: .cls#User { show: outline; }",
    ),
    Intent(
        id="pss.core.03",
        interpreter="pss",
        difficulty="core",
        text="Create a selector sheet that shows the function named login with its body.",
        return_shape="markdown",
        structural_gate=_pss_gate,
        exec_assertion=_markdown_contains(["login"]),
        notes="1-rule: .fn#login { show: body; } — known symbol in app.py. Replaces plan's @route-filter intent.",
    ),
    Intent(
        id="pss.core.04",
        interpreter="pss",
        difficulty="core",
        text="Create a selector sheet that shows every function whose name starts with 'test_' as a signature.",
        return_shape="markdown",
        structural_gate=_pss_gate,
        exec_assertion=_markdown_contains(["test_"]),
        notes='1-rule: .fn[name^="test_"] { show: signature; }',
    ),
    Intent(
        id="pss.core.05",
        interpreter="pss",
        difficulty="core",
        text="Create a selector sheet with two rules: show validate_token with its body, and show hash_password with its body.",
        return_shape="markdown",
        structural_gate=_pss_gate,
        exec_assertion=_markdown_contains(["validate_token", "hash_password"]),
        notes="2-rule sheet.",
    ),
    Intent(
        id="pss.core.06",
        interpreter="pss",
        difficulty="core",
        text="Create a selector sheet with two rules: show the User class body, and show the Session class outline.",
        return_shape="markdown",
        structural_gate=_pss_gate,
        exec_assertion=_markdown_contains(["User", "Session"]),
        notes="2-rule sheet with mixed show modes.",
    ),
    Intent(
        id="pss.core.07",
        interpreter="pss",
        difficulty="core",
        text="Create a selector sheet that shows the hash_password function with its body.",
        return_shape="markdown",
        structural_gate=_pss_gate,
        exec_assertion=_markdown_contains(["hash_password"]),
        notes="1-rule: .fn#hash_password { show: body; }. Replaces plan's async intent (toybox has no async).",
    ),
    Intent(
        id="pss.core.08",
        interpreter="pss",
        difficulty="core",
        text="Create a selector sheet that shows every class in the codebase with its outline.",
        return_shape="markdown",
        structural_gate=_pss_gate,
        exec_assertion=_markdown_contains(["User"]),
        notes="1-rule: .cls { show: outline; } — toybox has User/Session/AuditLog plus exception classes. Replaces plan's @deprecated intent.",
    ),

    # Stretch (6) — 2-3 rule curated views
    Intent(
        id="pss.stretch.01",
        interpreter="pss",
        difficulty="stretch",
        text="Create a security review sheet with three rules: every function that contains an execute_sql call with its body, every function whose name starts with 'validate_' as a signature, and every route handler as a signature.",
        return_shape="markdown",
        structural_gate=_pss_gate,
        exec_assertion=_markdown_contains(["validate_token"]),
        notes="3-rule sheet; the hardest structural challenge.",
    ),
    Intent(
        id="pss.stretch.02",
        interpreter="pss",
        difficulty="stretch",
        text="Create a documentation sheet for the auth module with two rules: show every function with its body and every class with its outline.",
        return_shape="markdown",
        structural_gate=_pss_gate,
        exec_assertion=_markdown_contains(["validate_token"]),
        notes="2-rule mixed show modes.",
    ),
    Intent(
        id="pss.stretch.03",
        interpreter="pss",
        difficulty="stretch",
        text="Create a test-surface sheet with two rules: every test function as a signature and every class whose name appears in a test file as an outline.",
        return_shape="markdown",
        structural_gate=_pss_gate,
        exec_assertion=_markdown_contains(["test_"]),
        notes="2-rule. The second rule is contextually hard — models will likely default to 'show all classes'.",
    ),
    Intent(
        id="pss.stretch.04",
        interpreter="pss",
        difficulty="stretch",
        text="Create a sheet with two rules: public functions (name does not start with underscore) as signatures and private functions (name starts with underscore) as bodies.",
        return_shape="markdown",
        structural_gate=_pss_gate,
        exec_assertion=_markdown_nonempty,
        notes="2-rule with :not and attribute selector.",
    ),
    Intent(
        id="pss.stretch.05",
        interpreter="pss",
        difficulty="stretch",
        text="Create a selector sheet that shows every function defined in the file api_v2.py with its body.",
        return_shape="markdown",
        structural_gate=_pss_gate,
        exec_assertion=_markdown_contains(["get_user"]),
        notes="File-scoped rule — models may need to use a file filter or path predicate.",
    ),
    Intent(
        id="pss.stretch.06",
        interpreter="pss",
        difficulty="stretch",
        text="Create a deprecated-code review sheet with two rules: every @deprecated function's body and every function that calls a deprecated function as a signature.",
        return_shape="markdown",
        structural_gate=_pss_gate,
        exec_assertion=_markdown_nonempty,
        notes="Relationship-aware; may exceed pluckit's pss grammar. If unsupported, drop and file an issue.",
    ),
]
