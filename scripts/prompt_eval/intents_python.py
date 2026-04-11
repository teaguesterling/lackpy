"""Python interpreter corpus: 8 core + 6 stretch delegation tasks.

Every intent returns a usable artifact for an orchestrator (list, dict,
int, str) rather than making a side-effectful change. Assertions check
the content of the returned value against the known toybox properties.

All programs run under the eval kit (read_file, find_files, find_def,
find_refs) against the toybox base_dir set by the harness.
"""

from __future__ import annotations

from typing import Any

from lackpy.lang.validator import validate

from .intents import GateResult, Intent


# ── Shared structural gate ────────────────────────────────────────────

_ALLOWED_KIT_NAMES = {"read_file", "find_files", "find_def", "find_refs"}


def _python_gate(program: str) -> GateResult:
    """Gate a generated python program via lackpy's validator.

    The gate passes when the program parses, contains only allowed AST
    nodes, only calls names that are in the eval kit or ALLOWED_BUILTINS,
    and satisfies lackpy's dunder/name restrictions.
    """
    result = validate(program, allowed_names=_ALLOWED_KIT_NAMES)
    return GateResult(passed=result.valid, errors=list(result.errors))


# ── Assertion helpers ─────────────────────────────────────────────────

def _contains_all(substrs: list[str]):
    def check(x: Any) -> bool:
        if not isinstance(x, str):
            return False
        return all(s in x for s in substrs)
    return check


def _is_list_with_any(substrs: list[str]):
    def check(x: Any) -> bool:
        if not isinstance(x, list):
            return False
        joined = " ".join(str(item) for item in x)
        return any(s in joined for s in substrs)
    return check


def _is_list_with_all(substrs: list[str]):
    def check(x: Any) -> bool:
        if not isinstance(x, list):
            return False
        joined = " ".join(str(item) for item in x)
        return all(s in joined for s in substrs)
    return check


def _is_int_at_least(n: int):
    def check(x: Any) -> bool:
        return isinstance(x, int) and not isinstance(x, bool) and x >= n
    return check


def _is_dict_with_keys(keys: list[str]):
    def check(x: Any) -> bool:
        if not isinstance(x, dict):
            return False
        return all(k in x for k in keys)
    return check


def _is_dict_or_list_referencing(substrs: list[str]):
    """Flexible check: value is a dict/list whose str-repr mentions each substr."""
    def check(x: Any) -> bool:
        if not isinstance(x, (dict, list, tuple, set)):
            return False
        s = str(x)
        return all(sub in s for sub in substrs)
    return check


# ── Corpus ─────────────────────────────────────────────────────────────

PYTHON_INTENTS: list[Intent] = [
    # Core (8) — single- and small-composition lookups
    Intent(
        id="py.core.01",
        interpreter="python",
        difficulty="core",
        text="Find the definition of validate_token. Return a dict with two keys: 'file' (the path) and 'body' (the full text of the file it is defined in).",
        return_shape="dict",
        structural_gate=_python_gate,
        exec_assertion=_is_dict_or_list_referencing(["auth.py", "def validate_token"]),
        notes="Single find_def + read_file composition.",
    ),
    Intent(
        id="py.core.02",
        interpreter="python",
        difficulty="core",
        text="Find all callers of execute_sql in the codebase and return them as a list of caller filenames.",
        return_shape="list[str]",
        structural_gate=_python_gate,
        exec_assertion=_is_list_with_any(["app.py", "api_v1.py", "api_v2.py", "auth.py"]),
        notes="Single find_refs call; return shape is the file column of the result.",
    ),
    Intent(
        id="py.core.03",
        interpreter="python",
        difficulty="core",
        text="Find every file that defines a function named hash_password and return the list of file paths.",
        return_shape="list[str]",
        structural_gate=_python_gate,
        exec_assertion=_is_list_with_any(["auth.py"]),
        notes="find_def + extract file paths.",
    ),
    Intent(
        id="py.core.04",
        interpreter="python",
        difficulty="core",
        text="Read app.py from the toybox and return its full contents as a string.",
        return_shape="str",
        structural_gate=_python_gate,
        exec_assertion=_contains_all(["@route", "login", "execute_sql"]),
        notes="Simplest possible read_file call.",
    ),
    Intent(
        id="py.core.05",
        interpreter="python",
        difficulty="core",
        text="Find the definition of the class User. Return a dict with 'file' and 'body' keys where body is the full text of the file.",
        return_shape="dict",
        structural_gate=_python_gate,
        exec_assertion=_is_dict_or_list_referencing(["models.py", "class User"]),
        notes="find_def targeting a class.",
    ),
    Intent(
        id="py.core.06",
        interpreter="python",
        difficulty="core",
        text="Find all callers of validate_token and return a list of the file paths where each caller lives.",
        return_shape="list[str]",
        structural_gate=_python_gate,
        exec_assertion=_is_list_with_any(["app.py", "auth.py"]),
        notes="find_refs for a single symbol.",
    ),
    Intent(
        id="py.core.07",
        interpreter="python",
        difficulty="core",
        text="Find every test file under the tests directory and return them as a list of file paths.",
        return_shape="list[str]",
        structural_gate=_python_gate,
        exec_assertion=_is_list_with_all(["tests/test_app.py", "tests/test_auth.py", "tests/test_models.py"]),
        notes="find_files with a glob.",
    ),
    Intent(
        id="py.core.08",
        interpreter="python",
        difficulty="core",
        text="Find the definition of DatabaseError and return the contents of the file it is defined in.",
        return_shape="str",
        structural_gate=_python_gate,
        exec_assertion=_contains_all(["class DatabaseError", "errors.py"]),
        notes="find_def + read_file chaining. Assertion also checks the filename — models often prepend a header.",
    ),

    # Stretch (6) — multi-call compositions
    Intent(
        id="py.stretch.01",
        interpreter="python",
        difficulty="stretch",
        text="For every caller of execute_sql, return a list of (file_path, caller_line_text) pairs.",
        return_shape="list[tuple|list]",
        structural_gate=_python_gate,
        exec_assertion=_is_list_with_any(["app.py", "api_v1.py", "api_v2.py", "auth.py"]),
        notes="Iterate over find_refs result and extract fields.",
    ),
    Intent(
        id="py.stretch.02",
        interpreter="python",
        difficulty="stretch",
        text="Find every definition of hash_password across the codebase and return the count.",
        return_shape="int",
        structural_gate=_python_gate,
        exec_assertion=_is_int_at_least(1),
        notes="find_def + len() builtin. Assertion accepts >=1 to tolerate string-matching width.",
    ),
    Intent(
        id="py.stretch.03",
        interpreter="python",
        difficulty="stretch",
        text="Find every test file under tests/ and return a dict mapping filename to the file's contents.",
        return_shape="dict[str, str]",
        structural_gate=_python_gate,
        exec_assertion=lambda x: isinstance(x, dict) and len(x) >= 3 and all(isinstance(v, str) and len(v) > 0 for v in x.values()),
        notes="find_files + iterate + read_file + build dict.",
    ),
    Intent(
        id="py.stretch.04",
        interpreter="python",
        difficulty="stretch",
        text="Find all callers of validate_token and return the unique set of file paths where they live, as a list.",
        return_shape="list[str]",
        structural_gate=_python_gate,
        exec_assertion=lambda x: isinstance(x, (list, set, tuple)) and any("app.py" in str(p) or "auth.py" in str(p) for p in x),
        notes="find_refs + de-duplicate via set().",
    ),
    Intent(
        id="py.stretch.05",
        interpreter="python",
        difficulty="stretch",
        text="Find the definition of the class User and all of its callers. Return a dict with two keys: 'definition' (the file path of the class) and 'callers' (a list of file paths where it is used).",
        return_shape="dict",
        structural_gate=_python_gate,
        exec_assertion=_is_dict_with_keys(["definition", "callers"]),
        notes="Combines find_def + find_refs.",
    ),
    Intent(
        id="py.stretch.06",
        interpreter="python",
        difficulty="stretch",
        text="Find every function whose name starts with 'test_' and return a list of the file paths where each is defined.",
        return_shape="list[str]",
        structural_gate=_python_gate,
        exec_assertion=_is_list_with_any(["test_app.py", "test_auth.py", "test_models.py"]),
        notes="Requires scanning test files — may go via find_files('tests/test_*.py') or via find_def calls. Both paths are acceptable.",
    ),
]
