"""Mock tool provider — returns plausible shaped data without real execution.

Used for testing tool composition with APIs that aren't implemented yet.
Each mock tool returns structurally correct data matching its declared return type.
"""

from __future__ import annotations

from typing import Any, Callable

from ..toolbox import ToolSpec

# Reusable fake data for consistent mock responses
_MOCK_FILES = [
    "src/auth/tokens.py", "src/auth/permissions.py",
    "src/api/routes.py", "src/db/queries.py",
    "src/models/user.py", "src/middleware/logging.py",
    "tests/test_auth.py", "tests/test_api.py",
]

_MOCK_FUNCTIONS = [
    {"name": "validate_token", "file": "src/auth/tokens.py", "line": 42, "type": "function", "params": ["token: str", "strict: bool = False"]},
    {"name": "check_permissions", "file": "src/auth/permissions.py", "line": 15, "type": "function", "params": ["user_id: int", "resource: str"]},
    {"name": "handle_request", "file": "src/api/routes.py", "line": 8, "type": "function", "params": ["request: Request"]},
    {"name": "get_user_roles", "file": "src/db/queries.py", "line": 23, "type": "function", "params": ["user_id: int"]},
    {"name": "process_data", "file": "src/models/user.py", "line": 55, "type": "function", "params": ["items: list", "threshold: float"]},
    {"name": "AuthService", "file": "src/auth/tokens.py", "line": 1, "type": "class", "params": []},
]

_MOCK_SOURCE = '''def validate_token(token: str, strict: bool = False) -> bool:
    """Validate an authentication token."""
    if not token:
        return False
    if token.startswith("expired_"):
        return None
    return check_permissions(token, "read")
'''


class MockProvider:
    """Provider that returns plausible mock data for testing composition patterns."""

    @property
    def name(self) -> str:
        return "mock"

    def available(self) -> bool:
        return True

    def resolve(self, tool_spec: ToolSpec) -> Callable[..., Any]:
        implementations = {
            # Selection operations
            "select": _mock_select,
            "source": _mock_source_fn,

            # Query operations
            "find": _mock_find,
            "filter": _mock_filter,
            "callers": _mock_callers,
            "callees": _mock_callees,
            "references": _mock_references,
            "dependents": _mock_dependents,
            "dependencies": _mock_dependencies,
            "reachable": _mock_reachable,
            "call_chain": _mock_call_chain,
            "similar": _mock_similar,
            "common_pattern": _mock_common_pattern,
            "refs": _mock_refs,
            "defs": _mock_defs,
            "unused_params": _mock_unused_params,
            "shadows": _mock_shadows,

            # Reading
            "text": _mock_text,
            "attr": _mock_attr,
            "count": _mock_count,
            "names": _mock_names,
            "complexity": _mock_complexity,
            "interface": _mock_interface,
            "params": _mock_params,
            "body": _mock_body,

            # Mutations (return selection-shaped data)
            "addParam": _mock_mutate,
            "removeParam": _mock_mutate,
            "retype": _mock_mutate,
            "rename": _mock_mutate,
            "prepend": _mock_mutate,
            "append": _mock_mutate,
            "wrap": _mock_mutate,
            "unwrap": _mock_mutate,
            "replaceWith": _mock_mutate,
            "remove": _mock_mutate,
            "move_to": _mock_mutate,
            "extract": _mock_mutate,
            "inline": _mock_mutate,
            "refactor": _mock_mutate,
            "guard": _mock_mutate,

            # Delegates (return result-shaped data)
            "black": _mock_mutate,
            "ruff_fix": _mock_mutate,
            "isort": _mock_mutate,
            "format": _mock_mutate,
            "test": _mock_test,
            "isolate": _mock_isolate,
            "fuzz": _mock_fuzz,
            "benchmark": _mock_benchmark,
            "trace": _mock_trace,
            "save": _mock_save,

            # History
            "history": _mock_history,
            "at": _mock_at,
            "diff": _mock_diff,
            "blame": _mock_blame,
            "authors": _mock_authors,
            "filmstrip": _mock_filmstrip,
            "when": _mock_when,
            "co_changes": _mock_co_changes,

            # Behavior
            "coverage": _mock_coverage,
            "failures": _mock_failures,
            "timing": _mock_timing,
            "inputs": _mock_inputs,
            "outputs": _mock_outputs,
            "runs": _mock_runs,

            # View
            "impact": _mock_impact,
            "compare": _mock_compare,

            # Metadata
            "intent": _mock_intent,
            "preview": _mock_preview,
            "explain": _mock_explain,
            "dry_run": _mock_dry_run,
        }
        fn = implementations.get(tool_spec.name)
        if fn is None:
            raise KeyError(f"No mock implementation for '{tool_spec.name}'")
        return fn


# --- Entry / Query mocks ---

def _mock_select(selector: str = "") -> list[dict]:
    """Return mock AST nodes matching selector."""
    return [f for f in _MOCK_FUNCTIONS if _selector_matches(selector, f)]


def _mock_source_fn(glob: str = "") -> list[str]:
    return [f for f in _MOCK_FILES if _glob_matches(glob, f)]


def _mock_find(selection: Any = None, selector: str = "") -> list[dict]:
    return _MOCK_FUNCTIONS[:3]


def _mock_filter(selection: Any = None, predicate: str = "") -> list[dict]:
    if isinstance(selection, list):
        return selection[:max(1, len(selection) - 1)]
    return _MOCK_FUNCTIONS[:2]


def _mock_callers(selection: Any = None) -> list[dict]:
    return [
        {"name": "handle_request", "file": "src/api/routes.py", "line": 12, "type": "function"},
        {"name": "refresh_token", "file": "src/auth/tokens.py", "line": 89, "type": "function"},
    ]


def _mock_callees(selection: Any = None) -> list[dict]:
    return [
        {"name": "check_permissions", "file": "src/auth/permissions.py", "line": 15, "type": "function"},
        {"name": "get_user_roles", "file": "src/db/queries.py", "line": 23, "type": "function"},
    ]


def _mock_references(selection: Any = None) -> list[dict]:
    return _mock_callers() + [{"name": "validate_token", "file": "src/auth/__init__.py", "line": 3, "type": "import"}]


def _mock_dependents(selection: Any = None) -> list[dict]:
    return _mock_callers()


def _mock_dependencies(selection: Any = None) -> list[dict]:
    return _mock_callees()


def _mock_reachable(selection: Any = None, max_depth: int = 3) -> list[dict]:
    return _MOCK_FUNCTIONS[:min(max_depth + 1, len(_MOCK_FUNCTIONS))]


def _mock_call_chain(selection: Any = None) -> list[dict]:
    return [
        {"name": "handle_request", "file": "src/api/routes.py", "line": 8, "step": 1},
        {"name": "validate_token", "file": "src/auth/tokens.py", "line": 42, "step": 2},
        {"name": "check_permissions", "file": "src/auth/permissions.py", "line": 15, "step": 3},
        {"name": "get_user_roles", "file": "src/db/queries.py", "line": 23, "step": 4},
    ]


def _mock_similar(selection: Any = None, threshold: float = 0.7) -> list[dict]:
    return [
        {"name": "validate_session", "file": "src/auth/sessions.py", "line": 10, "similarity": 0.82},
        {"name": "validate_api_key", "file": "src/auth/api_keys.py", "line": 5, "similarity": 0.76},
    ]


def _mock_common_pattern(selection: Any = None) -> dict:
    return {"skeleton": "parse credential → check expiry → check permissions → return", "variations": 3}


def _mock_refs(selection: Any = None, name: str = "") -> list[dict]:
    return [
        {"name": name or "token", "file": "src/auth/tokens.py", "line": 44, "role": "reference"},
        {"name": name or "token", "file": "src/auth/tokens.py", "line": 47, "role": "reference"},
    ]


def _mock_defs(selection: Any = None, name: str = "") -> list[dict]:
    return [{"name": name or "token", "file": "src/auth/tokens.py", "line": 42, "role": "definition"}]


def _mock_unused_params(selection: Any = None) -> list[dict]:
    return [{"name": "strict", "file": "src/auth/tokens.py", "line": 42, "function": "validate_token"}]


def _mock_shadows(selection: Any = None) -> list[dict]:
    return [{"name": "result", "outer_line": 10, "inner_line": 25, "file": "src/models/user.py"}]


# --- Reading mocks ---

def _mock_text(selection: Any = None) -> str:
    return _MOCK_SOURCE


def _mock_attr(selection: Any = None, name: str = "name") -> Any:
    attrs = {"name": "validate_token", "line": 42, "file": "src/auth/tokens.py", "end_line": 49}
    return attrs.get(name, None)


def _mock_count(selection: Any = None) -> int:
    if isinstance(selection, list):
        return len(selection)
    return 4


def _mock_names(selection: Any = None) -> list[str]:
    return ["validate_token", "check_permissions", "handle_request", "get_user_roles"]


def _mock_complexity(selection: Any = None) -> int:
    return 7


def _mock_interface(selection: Any = None) -> dict:
    return {"reads": ["token", "config"], "writes": ["result"], "calls": ["check_permissions"]}


def _mock_params(selection: Any = None) -> list[dict]:
    return [{"name": "token", "type": "str"}, {"name": "strict", "type": "bool", "default": "False"}]


def _mock_body(selection: Any = None) -> str:
    return '    if not token:\n        return False\n    return check_permissions(token, "read")'


# --- Mutation mocks ---

def _mock_mutate(*args: Any, **kwargs: Any) -> list[dict]:
    """All mutations return the selection (enabling chaining)."""
    return _MOCK_FUNCTIONS[:3]


# --- Behavior mocks ---

def _mock_test(selection: Any = None, inputs: dict | None = None) -> dict:
    return {"passed": True, "output": "ok", "duration_ms": 45.2, "coverage": 0.85}


def _mock_isolate(selection: Any = None) -> dict:
    return {
        "interface": {"reads": ["items", "threshold"], "writes": ["filtered"], "calls": ["classify"]},
        "wrapped_code": "def _isolated(items, threshold, classify):\n    ...",
        "runnable": True,
    }


def _mock_fuzz(selection: Any = None, n: int = 100) -> list[dict]:
    return [{"passed": True, "input": {"x": i}} for i in range(min(n, 5))]


def _mock_benchmark(selection: Any = None, n: int = 1000) -> dict:
    return {"mean_ms": 2.3, "p50_ms": 1.8, "p95_ms": 5.1, "p99_ms": 12.4, "iterations": n}


def _mock_trace(selection: Any = None, inputs: dict | None = None) -> list[dict]:
    return [
        {"step": 1, "call": "check_permissions", "args": {"token": "abc", "resource": "read"}, "result": True},
        {"step": 2, "call": "get_user_roles", "args": {"user_id": 42}, "result": ["admin"]},
    ]


def _mock_save(selection: Any = None, message: str = "") -> dict:
    return {"committed": True, "message": message or "auto-generated commit", "files_changed": 3}


# --- History mocks ---

def _mock_history(selection: Any = None) -> list[dict]:
    return [
        {"sha": "abc123", "date": "2025-03-01", "author": "dev1", "complexity": 4},
        {"sha": "def456", "date": "2025-06-15", "author": "dev2", "complexity": 7},
        {"sha": "ghi789", "date": "2025-09-01", "author": "dev1", "complexity": 12},
    ]


def _mock_at(selection: Any = None, ref: str = "HEAD") -> list[dict]:
    return _MOCK_FUNCTIONS[:3]


def _mock_diff(selection: Any = None, other: Any = None) -> dict:
    return {"added": [".if:has(.call#check_grace_period)"], "removed": [], "changed": [".ret:last"]}


def _mock_blame(selection: Any = None) -> list[dict]:
    return [
        {"line": 42, "author": "dev1", "sha": "abc123", "date": "2025-03-01"},
        {"line": 45, "author": "dev2", "sha": "def456", "date": "2025-06-15"},
    ]


def _mock_authors(selection: Any = None) -> list[str]:
    return ["dev1", "dev2"]


def _mock_filmstrip(selection: Any = None) -> list[dict]:
    return _mock_history()


def _mock_when(selection: Any = None, selector: str = "") -> dict:
    return {"sha": "def456", "date": "2025-06-15", "author": "dev2"}


def _mock_co_changes(selection: Any = None, threshold: float = 0.8) -> list[dict]:
    return [{"pair": ["src/auth/tokens.py", "src/auth/permissions.py"], "co_change_rate": 0.85}]


# --- Behavior data mocks ---

def _mock_coverage(selection: Any = None) -> float:
    return 0.72


def _mock_failures(selection: Any = None) -> list[dict]:
    return [
        {"sha": "def456", "date": "2025-06-16", "error": "AssertionError: expected True", "test": "test_validate_expired"},
    ]


def _mock_timing(selection: Any = None) -> dict:
    return {"mean_ms": 3.2, "p50_ms": 2.1, "p95_ms": 8.4, "trend": "stable"}


def _mock_inputs(selection: Any = None) -> list[dict]:
    return [{"token": "valid_abc", "strict": False}, {"token": "expired_xyz", "strict": True}]


def _mock_outputs(selection: Any = None) -> list:
    return [True, None, False]


def _mock_runs(selection: Any = None) -> list[dict]:
    return [{"date": "2025-09-01", "passed": True}, {"date": "2025-09-02", "passed": False}]


# --- View mocks ---

def _mock_impact(selection: Any = None) -> dict:
    return {
        "target": "validate_token",
        "direct_callers": 12,
        "indirect_callers": 3,
        "covering_tests": 8,
        "low_coverage_callers": ["handle_admin_request", "batch_validate"],
    }


def _mock_compare(selection: Any = None) -> dict:
    return {
        "functions": ["validate_token", "validate_session", "validate_api_key"],
        "shared_pattern": "parse → check expiry → check permissions → return",
        "differences": {"validate_token": "checks signature", "validate_session": "checks IP binding"},
    }


# --- Metadata mocks ---

def _mock_intent(selection: Any = None, description: str = "") -> list[dict]:
    return _MOCK_FUNCTIONS[:3]


def _mock_preview(selection: Any = None) -> str:
    return "--- a/src/auth/tokens.py\n+++ b/src/auth/tokens.py\n@@ -42,1 +42,1 @@\n-def validate_token(token: str):\n+def validate_token(token: str, timeout: int = 30):"


def _mock_explain(selection: Any = None) -> str:
    return "1. QUERY: select .fn:exported (DuckDB)\n2. MUTATE: addParam (renderer + splice)\n3. DELEGATE: black (subprocess)\n4. DELEGATE: test (blq sandbox)"


def _mock_dry_run(selection: Any = None) -> dict:
    return {"files_changed": 3, "additions": 47, "deletions": 0, "diff": _mock_preview()}


# --- Helpers ---

def _selector_matches(selector: str, func: dict) -> bool:
    """Very rough mock selector matching."""
    if not selector:
        return True
    name = func.get("name", "")
    typ = func.get("type", "")
    if f"#{name}" in selector:
        return True
    if ".fn" in selector and typ == "function":
        return True
    if ".cls" in selector and typ == "class":
        return True
    return True  # default: return everything for mock purposes


def _glob_matches(pattern: str, path: str) -> bool:
    """Very rough mock glob matching."""
    if not pattern:
        return True
    if "**" in pattern:
        ext = pattern.split(".")[-1] if "." in pattern else ""
        return path.endswith(f".{ext}") if ext else True
    return pattern in path
