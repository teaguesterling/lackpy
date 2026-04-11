"""plucker interpreter corpus: fluent chain expressions."""

from __future__ import annotations

import ast
from typing import Any

from .intents import GateResult, Intent


def _plucker_gate(program: str) -> GateResult:
    """Structural gate for plucker chains.

    Requires the program to parse as a single expression whose AST
    shape is `source(...).method(...)...terminal(...)`. A bare
    `source(...)` with no chain fails — the interpreter needs a
    terminal operation to produce a result.
    """
    if not program or not program.strip():
        return GateResult(passed=False, errors=["empty program"])
    try:
        tree = ast.parse(program.strip(), mode="eval")
    except SyntaxError as e:
        return GateResult(passed=False, errors=[f"parse error: {e}"])
    node = tree.body

    # Walk from the outermost expression down to the innermost call
    # and verify it's `source(...)`. The chain must include at least
    # one attribute access somewhere between the outer expr and source().
    has_chain = False
    current = node
    while True:
        if isinstance(current, ast.Call):
            if isinstance(current.func, ast.Attribute):
                has_chain = True
                current = current.func.value
                continue
            if isinstance(current.func, ast.Name):
                if current.func.id != "source":
                    return GateResult(
                        passed=False,
                        errors=[f"chain must start with source(...), got {current.func.id}(...)"],
                    )
                break
            return GateResult(passed=False, errors=["chain root is not a simple Name call"])
        if isinstance(current, ast.Attribute):
            has_chain = True
            current = current.value
            continue
        return GateResult(passed=False, errors=[f"unexpected node in chain: {type(current).__name__}"])

    if not has_chain:
        return GateResult(passed=False, errors=["bare source() call has no chain or terminal"])
    return GateResult(passed=True)


def _is_int_at_least(n: int):
    def check(x: Any) -> bool:
        return isinstance(x, int) and not isinstance(x, bool) and x >= n
    return check


def _is_list_containing_any(substrs: list[str]):
    def check(x: Any) -> bool:
        if not isinstance(x, list):
            return False
        joined = " ".join(str(item) for item in x)
        return any(s in joined for s in substrs)
    return check


def _is_markdown_containing(substrs: list[str]):
    def check(x: Any) -> bool:
        if not isinstance(x, str):
            return False
        return all(s in x for s in substrs)
    return check


PLUCKER_INTENTS: list[Intent] = [
    # Core (8)
    Intent(
        id="pl.core.01",
        interpreter="plucker",
        difficulty="core",
        text="Return the count of every function in the codebase as an integer.",
        return_shape="int",
        structural_gate=_plucker_gate,
        exec_assertion=_is_int_at_least(5),
        notes="source().find('.fn').count()",
    ),
    Intent(
        id="pl.core.02",
        interpreter="plucker",
        difficulty="core",
        text="Return the names of every class in the codebase as a list of strings.",
        return_shape="list[str]",
        structural_gate=_plucker_gate,
        exec_assertion=_is_list_containing_any(["User", "Session"]),
        notes="source().find('.cls').names()",
    ),
    Intent(
        id="pl.core.03",
        interpreter="plucker",
        difficulty="core",
        text="Return the count of the function named validate_token as an integer.",
        return_shape="int",
        structural_gate=_plucker_gate,
        exec_assertion=_is_int_at_least(1),
        notes="source().find('.fn#validate_token').count() — expected 1.",
    ),
    Intent(
        id="pl.core.04",
        interpreter="plucker",
        difficulty="core",
        text="Return the count of every class in the codebase as an integer.",
        return_shape="int",
        structural_gate=_plucker_gate,
        exec_assertion=_is_int_at_least(3),
        notes="source().find('.cls').count(). Toybox has User, Session, AuditLog, plus exception classes. Replaces plan's async intent (toybox has no async functions).",
    ),
    Intent(
        id="pl.core.05",
        interpreter="plucker",
        difficulty="core",
        text="Return a markdown view of the class named User.",
        return_shape="markdown",
        structural_gate=_plucker_gate,
        exec_assertion=_is_markdown_containing(["User"]),
        notes="source().find('.cls#User').view()",
    ),
    Intent(
        id="pl.core.06",
        interpreter="plucker",
        difficulty="core",
        text="Return the count of every function whose name starts with 'test_' as an integer.",
        return_shape="int",
        structural_gate=_plucker_gate,
        exec_assertion=_is_int_at_least(4),
        notes="Toybox has 4 test_ functions: test_login_flow, test_user_list, test_hash_password, test_user_create.",
    ),
    Intent(
        id="pl.core.07",
        interpreter="plucker",
        difficulty="core",
        text="Return the names of every method defined inside the class User as a list of strings.",
        return_shape="list[str]",
        structural_gate=_plucker_gate,
        exec_assertion=_is_list_containing_any(["__init__", "add_role", "_internal_state"]),
        notes="source().find('.cls#User .fn').names()",
    ),
    Intent(
        id="pl.core.08",
        interpreter="plucker",
        difficulty="core",
        text="Return a markdown view of the function named hash_password.",
        return_shape="markdown",
        structural_gate=_plucker_gate,
        exec_assertion=_is_markdown_containing(["hash_password"]),
        notes="source().find('.fn#hash_password').view(). Replaces plan's @deprecated intent (grammar-contingent).",
    ),

    # Stretch (6)
    Intent(
        id="pl.stretch.01",
        interpreter="plucker",
        difficulty="stretch",
        text="Return the names of every caller of the function validate_token as a list of strings.",
        return_shape="list[str]",
        structural_gate=_plucker_gate,
        exec_assertion=lambda x: isinstance(x, list),
        notes="source().find('.fn#validate_token').callers().names() — depends on pluckit Selection.callers() being reachable.",
    ),
    Intent(
        id="pl.stretch.02",
        interpreter="plucker",
        difficulty="stretch",
        text="Return the names of every function whose cyclomatic complexity is greater than 5 as a list of strings.",
        return_shape="list[str]",
        structural_gate=_plucker_gate,
        exec_assertion=lambda x: isinstance(x, list),
        notes=".filter(...) predicate; may not be supported in current pluckit.",
    ),
    Intent(
        id="pl.stretch.03",
        interpreter="plucker",
        difficulty="stretch",
        text="Return a markdown view of every route handler (function decorated with @route).",
        return_shape="markdown",
        structural_gate=_plucker_gate,
        exec_assertion=lambda x: isinstance(x, str),
        notes="Decorator filter + .view() terminal.",
    ),
    Intent(
        id="pl.stretch.04",
        interpreter="plucker",
        difficulty="stretch",
        text="Return the names of every function that contains a call to execute_sql as a list of strings.",
        return_shape="list[str]",
        structural_gate=_plucker_gate,
        exec_assertion=lambda x: isinstance(x, list),
        notes=".fn:has(.call#execute_sql) + .names().",
    ),
    Intent(
        id="pl.stretch.05",
        interpreter="plucker",
        difficulty="stretch",
        text="Return a markdown view of every function defined in the file api_v2.py.",
        return_shape="markdown",
        structural_gate=_plucker_gate,
        exec_assertion=_is_markdown_containing(["get_user"]),
        notes="File-scoped find.",
    ),
    Intent(
        id="pl.stretch.06",
        interpreter="plucker",
        difficulty="stretch",
        text="Return a tuple where the first element is the count of test functions and the second element is the count of non-test functions.",
        return_shape="tuple[int, int]",
        structural_gate=_plucker_gate,
        exec_assertion=lambda x: isinstance(x, tuple) and len(x) == 2 and all(isinstance(n, int) for n in x),
        notes="Tests whether the model composes two chains with a literal tuple. Hardest stretch item.",
    ),
]
