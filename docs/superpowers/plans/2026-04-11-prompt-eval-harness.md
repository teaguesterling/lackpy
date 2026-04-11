# Prompt Evaluation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline research harness that sweeps `{interpreter × prompt-variant × model × intent}` against live Ollama, scores with a hybrid structural-gate + execution rubric, and emits incremental JSONL so results are queryable mid-run.

**Architecture:** Flat Python scripts under `scripts/prompt-eval/` modeled on the existing `scripts/pluckit-quartermaster.py` pattern. Reuses `lackpy.interpreters` and `lackpy.infer.sanitize` for scoring so findings reflect the production execution path. Registers a harness-local eval kit with grep-based code-intel helpers so the `python` interpreter corpus can orchestrate without touching `src/lackpy/kit/providers/builtin.py`. No production changes in v1.

**Tech Stack:** Python 3.12+, `ollama` Python client, `tqdm`, `pytest`, `lackpy.interpreters` (in-tree), `pluckit` (dependency, read-only).

**Reference spec:** `docs/superpowers/specs/2026-04-11-prompt-eval-harness-design.md`

**Working directory:** `/mnt/aux-data/teague/Projects/lackpy/main` (main branch; no worktree needed — research tooling lives alongside the tree).

**Important context for the implementer:**
- The spec's python corpus originally referenced `find_definitions` and `find_callers`. These tools are **not** in lackpy's default builtin kit (which only has `read_file`, `find_files`, `write_file`, `edit_file`). Task 1 registers a harness-local `eval_kit` that adds `find_def(name)` and `find_refs(name)` as grep-based helpers over the toybox so the corpus intents work as authored.
- The harness bypasses `InferenceDispatcher` entirely: it calls `OllamaProvider.generate()` directly with `system_prompt_override=<variant>`. The dispatcher's templates/rules tiers and retry/correction logic are **not** exercised — we are measuring raw `prompt → model → program` quality.
- `lackpy.infer.sanitize.sanitize_output(raw)` is the production sanitizer and must be applied before scoring so findings transfer.
- `ALLOWED_BUILTINS` is `{len, sorted, reversed, enumerate, zip, range, min, max, sum, any, all, abs, round, str, int, float, bool, list, dict, set, tuple, isinstance, print, sort_by}`. Intent authors must only use these in expected programs.

**Implementation order (task list):**

1. Scaffold `scripts/prompt-eval/` + `tests/eval/`; write harness-local eval kit with grep helpers.
2. Write the toybox fixture (9 source files + 3 test files).
3. Author the `Intent` dataclass and the `python` corpus (8 core + 6 stretch).
4. Author the `ast-select` corpus.
5. Author the `pss` corpus.
6. Author the `plucker` corpus.
7. Author the four prompt variants per interpreter.
8. Implement structural gates (stage 1 scoring).
9. Implement execution scoring (stage 2 scoring) wired to `lackpy.interpreters`.
10. Implement the Ollama streaming runner with timeout and sanitization.
11. Implement the harness orchestrator (matrix iteration, JSONL, resume, tqdm, SIGINT).
12. Implement the `query.py` live-summary helper.
13. Implement the `report.py` JSONL → markdown consolidator.
14. Write the three phase entry-point scripts (`phase1a`, `phase1b`, `phase2`).
15. Write the canary test skeleton under `tests/eval/`.
16. Dry-run the harness against a single model × 3 intents as a shakeout.

---

## Task 1: Scaffold packages and the harness-local eval kit

**Files:**
- Create: `scripts/prompt-eval/__init__.py`
- Create: `scripts/prompt-eval/eval_kit.py`
- Create: `tests/eval/__init__.py`
- Create: `tests/eval/test_eval_kit.py`
- Create: `tests/eval/fixtures/__init__.py`

The eval kit registers grep-based `find_def(name)` and `find_refs(name)` helpers plus passes through the four builtin tools (`read_file`, `find_files`, `write_file`, `edit_file`). It is a Python module that constructs a `ResolvedKit` on demand via `LackpyService`'s toolbox, then augments it by registering two additional `ToolSpec`s through a local toolbox instance.

Because `RestrictedRunner` only cares that `allowed_names` covers the callables, we use `kit.registry.resolve_kit(tool_list, toolbox)` with a custom-tooled toolbox.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_eval_kit.py
"""Tests for the harness-local eval kit."""

from pathlib import Path
import pytest

from scripts.prompt_eval.eval_kit import build_eval_kit


@pytest.fixture
def toybox_tmp(tmp_path: Path) -> Path:
    """Tiny ad-hoc toybox stand-in; the real toybox arrives in Task 2."""
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n")
    (tmp_path / "b.py").write_text("def bar():\n    return foo()\n")
    return tmp_path


def test_eval_kit_has_builtin_tools(toybox_tmp: Path):
    kit = build_eval_kit(toybox_tmp)
    for name in ("read_file", "find_files", "find_def", "find_refs"):
        assert name in kit.tools, f"missing tool {name}"
        assert name in kit.callables


def test_find_def_returns_matching_file(toybox_tmp: Path):
    kit = build_eval_kit(toybox_tmp)
    rows = kit.callables["find_def"]("foo")
    assert isinstance(rows, list)
    assert any("a.py" in r["file"] for r in rows)
    assert all("line" in r for r in rows)


def test_find_refs_returns_matching_file(toybox_tmp: Path):
    kit = build_eval_kit(toybox_tmp)
    rows = kit.callables["find_refs"]("foo")
    assert isinstance(rows, list)
    # b.py calls foo()
    assert any("b.py" in r["file"] for r in rows)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /mnt/aux-data/teague/Projects/lackpy/main && python -m pytest tests/eval/test_eval_kit.py -v`
Expected: ImportError or ModuleNotFoundError (`scripts.prompt_eval.eval_kit` does not exist yet).

- [ ] **Step 3: Create package __init__.py files**

```python
# scripts/prompt-eval/__init__.py
"""Prompt evaluation harness — research tooling for lackpy inference."""
```

Note: the directory is named `prompt-eval` with a hyphen, but Python packages cannot have hyphens. Create the directory as `scripts/prompt_eval/` (underscore) for importability; the spec's conceptual name stays `prompt-eval` in documentation and in result directory names.

Correct the layout: `scripts/prompt_eval/` not `scripts/prompt-eval/`. Update the import path accordingly.

```bash
mkdir -p scripts/prompt_eval tests/eval/fixtures
touch scripts/prompt_eval/__init__.py tests/eval/__init__.py tests/eval/fixtures/__init__.py
```

Write `scripts/prompt_eval/__init__.py`:

```python
"""Prompt evaluation harness — research tooling for lackpy inference."""
```

Write `tests/eval/__init__.py`:

```python
```

Write `tests/eval/fixtures/__init__.py`:

```python
```

- [ ] **Step 4: Implement the eval kit**

Write `scripts/prompt_eval/eval_kit.py`:

```python
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

from lackpy.kit.providers.builtin import BuiltinProvider
from lackpy.kit.registry import ResolvedKit
from lackpy.kit.toolbox import ArgSpec, Toolbox, ToolSpec
from lackpy.lang.grader import Grade, compute_grade


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
        if not p.is_absolute():
            p = base_dir / p
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
        def_re = re.compile(rf"^\s*(?:async\s+)?(?:def|class)\s+{re.escape(name)}\b")
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
            name="read_file", provider="eval", description="Read a file under the toybox base_dir; returns its text.",
            args=[ArgSpec(name="path", type="str", description="Relative or absolute path")],
            returns="str", grade_w=1, effects_ceiling=1,
        ),
        "find_files": ToolSpec(
            name="find_files", provider="eval", description="Glob files under the toybox base_dir.",
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
```

- [ ] **Step 5: Run the test again to verify it passes**

Run: `cd /mnt/aux-data/teague/Projects/lackpy/main && python -m pytest tests/eval/test_eval_kit.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add scripts/prompt_eval/__init__.py scripts/prompt_eval/eval_kit.py tests/eval/__init__.py tests/eval/fixtures/__init__.py tests/eval/test_eval_kit.py
git commit -m "$(cat <<'EOF'
eval: scaffold prompt-eval package + eval kit

Adds scripts/prompt_eval/ and tests/eval/ scaffolding plus a harness-local
eval kit that registers read_file, find_files, find_def, and find_refs.
find_def and find_refs are grep-based helpers rooted at a toybox base_dir
so the python corpus can orchestrate symbol lookups without an external
code-intel layer.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Write the toybox fixture

**Files:**
- Create: `tests/eval/fixtures/toybox/__init__.py`
- Create: `tests/eval/fixtures/toybox/app.py`
- Create: `tests/eval/fixtures/toybox/auth.py`
- Create: `tests/eval/fixtures/toybox/models.py`
- Create: `tests/eval/fixtures/toybox/db.py`
- Create: `tests/eval/fixtures/toybox/api_v1.py`
- Create: `tests/eval/fixtures/toybox/api_v2.py`
- Create: `tests/eval/fixtures/toybox/utils.py`
- Create: `tests/eval/fixtures/toybox/errors.py`
- Create: `tests/eval/fixtures/toybox/config.py`
- Create: `tests/eval/fixtures/toybox/tests/__init__.py`
- Create: `tests/eval/fixtures/toybox/tests/test_app.py`
- Create: `tests/eval/fixtures/toybox/tests/test_auth.py`
- Create: `tests/eval/fixtures/toybox/tests/test_models.py`
- Test: `tests/eval/test_toybox_properties.py`

The toybox is a committed fixture. The guarantees enumerated in the spec (2 `@deprecated`, 4 `@route`, 3 SQL-concat sites, etc.) must be literal stable properties. Task 2 writes the fixture files and a property test that pins those counts. Future tasks build assertions against exact line numbers in these files, so modifying the toybox later means updating assertions.

**Important**: the toybox is **not** executable. It contains `import` statements for libraries that may not be installed (Flask, sqlite3 wrapper) and deliberate bugs. It is static text that pluckit and lackpy's interpreters treat as source material for analysis.

- [ ] **Step 1: Write the property test first**

Write `tests/eval/test_toybox_properties.py`:

```python
"""Property tests for the toybox fixture.

The toybox is a static fixture whose contents are load-bearing for the
prompt eval corpus. This test pins the counts the corpus assertions
depend on. If you change the toybox, you MUST update both this test
and the affected intent assertions.
"""

from pathlib import Path
import hashlib
import re

TOYBOX = Path(__file__).parent / "fixtures" / "toybox"


def _all_py_text() -> str:
    out = []
    for p in sorted(TOYBOX.rglob("*.py")):
        out.append(p.read_text())
    return "\n".join(out)


def test_toybox_exists():
    assert TOYBOX.is_dir()
    assert (TOYBOX / "__init__.py").exists()


def test_file_set():
    expected = {
        "__init__.py", "app.py", "auth.py", "models.py", "db.py",
        "api_v1.py", "api_v2.py", "utils.py", "errors.py", "config.py",
        "tests/__init__.py", "tests/test_app.py",
        "tests/test_auth.py", "tests/test_models.py",
    }
    actual = {str(p.relative_to(TOYBOX)) for p in TOYBOX.rglob("*.py")}
    assert actual == expected, f"unexpected file set: {sorted(actual)}"


def test_deprecated_count():
    text = _all_py_text()
    # Exactly 2 @deprecated-decorated functions
    # Counted as lines starting with `@deprecated` — no parentheses variant in toybox
    assert text.count("@deprecated") == 2


def test_route_count():
    text = (TOYBOX / "app.py").read_text()
    # 4 @route handlers in app.py
    assert text.count("@route(") == 4


def test_sql_concat_count():
    # 3 SQL-building-via-string-concat sites: substring `" + ` adjacent to an
    # SQL keyword or preceded by a SELECT/INSERT/UPDATE/DELETE fragment.
    # Approximation: count occurrences of `"SELECT ` or `"DELETE ` etc. that
    # appear on the same line as a `+`.
    concat_lines = 0
    for p in TOYBOX.rglob("*.py"):
        for line in p.read_text().splitlines():
            if ('"SELECT' in line or '"DELETE' in line or '"UPDATE' in line) and "+" in line:
                concat_lines += 1
    assert concat_lines == 3, f"expected 3 SQL-concat sites, got {concat_lines}"


def test_test_function_count():
    # Exactly 4 test_ prefixed functions across tests/
    count = 0
    for p in (TOYBOX / "tests").glob("test_*.py"):
        count += len(re.findall(r"^def test_\w+", p.read_text(), re.MULTILINE))
    assert count == 4


def test_api_v1_v2_rename_pairs():
    v1 = (TOYBOX / "api_v1.py").read_text()
    v2 = (TOYBOX / "api_v2.py").read_text()
    # Four corresponding renames
    pairs = [("get_usr", "get_user"), ("save_usr", "save_user"),
             ("del_usr", "delete_user"), ("list_usrs", "list_users")]
    for old, new in pairs:
        assert re.search(rf"^def {old}\b", v1, re.MULTILINE), f"v1 missing {old}"
        assert re.search(rf"^def {new}\b", v2, re.MULTILINE), f"v2 missing {new}"


def test_toybox_hash_is_stable():
    """Computed once, checked every run. If it changes, corpus assertions may be stale."""
    h = hashlib.sha256()
    for p in sorted(TOYBOX.rglob("*.py")):
        h.update(p.read_bytes())
    digest = h.hexdigest()
    hash_file = TOYBOX.parent / "toybox.sha256"
    if hash_file.exists():
        expected = hash_file.read_text().strip()
        assert digest == expected, (
            f"toybox hash changed: {digest} != {expected}. "
            f"If this was intentional, `echo {digest} > {hash_file}` and "
            f"review any intent assertions bound to line numbers."
        )
    else:
        # First run — seed the pin file.
        hash_file.write_text(digest + "\n")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /mnt/aux-data/teague/Projects/lackpy/main && python -m pytest tests/eval/test_toybox_properties.py -v`
Expected: All tests fail (toybox directory does not exist).

- [ ] **Step 3: Write the toybox source files**

Create the directory:

```bash
mkdir -p tests/eval/fixtures/toybox/tests
```

Write `tests/eval/fixtures/toybox/__init__.py`:

```python
"""Toybox fixture — a realistic webapp with deliberate code smells.

This package is NOT executable; some imports reference libraries
that may not be installed. It exists as static source material for
the prompt eval harness.
"""
```

Write `tests/eval/fixtures/toybox/app.py`:

```python
"""Route layer. Routes call auth and db layers directly."""

from .auth import validate_token, check_permissions
from .db import execute_sql, get_connection
from .errors import AuthError, ValidationError


def route(path):
    """Placeholder @route decorator; no-op for static analysis."""
    def wrap(fn):
        return fn
    return wrap


@route("/login")
def login(request):
    username = request.get("username")
    password = request.get("password")
    # SQL-concat smell #1
    row = execute_sql("SELECT id FROM users WHERE name = '" + username + "'")
    if not row:
        raise AuthError("unknown user")
    return {"ok": True}


@route("/users/<id>")
def get_user_view(request, id):
    token = request.get("token")
    validate_token(token)
    # SQL-concat smell #2
    return execute_sql("SELECT * FROM users WHERE id = " + str(id))


@route("/users/<id>/delete")
def delete_user_view(request, id):
    token = request.get("token")
    validate_token(token)
    check_permissions(token, "delete")
    # SQL-concat smell #3
    return execute_sql("DELETE FROM users WHERE id = " + str(id))


@route("/health")
def health(request):
    conn = get_connection()
    return {"ok": True, "db": bool(conn)}
```

Write `tests/eval/fixtures/toybox/auth.py`:

```python
"""Authentication helpers: token validation, hashing, permissions."""

import hashlib
from .db import execute_sql
from .errors import AuthError


def deprecated(fn):
    """Placeholder @deprecated decorator; no-op for static analysis."""
    return fn


def validate_token(token):
    if not token:
        raise AuthError("missing token")
    row = execute_sql("SELECT user_id FROM sessions WHERE token = ?", (token,))
    if not row:
        raise AuthError("invalid token")
    return row[0]


def refresh_token(old_token):
    user_id = validate_token(old_token)
    new_token = hash_password(str(user_id) + "salt")
    execute_sql("INSERT INTO sessions (user_id, token) VALUES (?, ?)",
                (user_id, new_token))
    return new_token


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def _hash_password(password, salt):
    return hashlib.sha256((password + salt).encode()).hexdigest()


def check_permissions(token, action):
    user_id = validate_token(token)
    row = execute_sql("SELECT role FROM users WHERE id = ?", (user_id,))
    if not row or row[0] != "admin":
        raise AuthError("forbidden")
    return True


@deprecated
def parse_legacy_token(raw):
    parts = raw.split(":")
    return {"user": parts[0], "token": parts[1]}
```

Write `tests/eval/fixtures/toybox/models.py`:

```python
"""Data models."""


class User:
    def __init__(self, id, name, roles=[]):  # mutable default bug #1
        self.id = id
        self.name = name
        self.roles = roles

    def add_role(self, role):
        self.roles.append(role)

    def _internal_state(self):
        return {"id": self.id, "name": self.name}


class Session:
    def __init__(self, token, user_id, metadata={}):  # mutable default bug #2
        self.token = token
        self.user_id = user_id
        self.metadata = metadata

    def touch(self):
        self.metadata["last_seen"] = "now"


class AuditLog:
    def __init__(self, events):
        self.events = events

    def record(self, event):
        self.events.append(event)
```

Write `tests/eval/fixtures/toybox/db.py`:

```python
"""Database access helpers."""

from .errors import DatabaseError


def get_connection():
    """Open a database connection. Caller must close it."""
    return _Conn()


def execute_sql(sql, params=None):
    conn = get_connection()
    try:
        cursor = conn.execute(sql, params or ())
        return cursor.fetchall()
    finally:
        conn.close()


def leaky_query(sql):
    """Resource leak — opens a connection, never closes it."""
    conn = get_connection()
    return conn.execute(sql).fetchall()


def transaction():
    conn = get_connection()
    return _Tx(conn)


class _Conn:
    def execute(self, sql, params=()):
        return self

    def fetchall(self):
        return []

    def close(self):
        pass


class _Tx:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self.conn

    def __exit__(self, *exc):
        self.conn.close()
```

Write `tests/eval/fixtures/toybox/api_v1.py`:

```python
"""Legacy API — older naming convention."""

from .db import execute_sql


def get_usr(id):
    return execute_sql("SELECT * FROM users WHERE id = ?", (id,))


def save_usr(id, name):
    return execute_sql("UPDATE users SET name = ? WHERE id = ?", (name, id))


def del_usr(id):
    return execute_sql("DELETE FROM users WHERE id = ?", (id,))


def list_usrs():
    return execute_sql("SELECT * FROM users")
```

Write `tests/eval/fixtures/toybox/api_v2.py`:

```python
"""Modern API — idiomatic names."""

from .db import execute_sql


def get_user(user_id):
    return execute_sql("SELECT * FROM users WHERE id = ?", (user_id,))


def save_user(user_id, name):
    return execute_sql("UPDATE users SET name = ? WHERE id = ?", (name, user_id))


def delete_user(user_id):
    return execute_sql("DELETE FROM users WHERE id = ?", (user_id,))


def list_users():
    return execute_sql("SELECT * FROM users")
```

Write `tests/eval/fixtures/toybox/utils.py`:

```python
"""Utility helpers."""

import hashlib


def deprecated(fn):
    return fn


@deprecated
def parse_date(s):
    parts = s.split("-")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def format_user(user):
    return f"{user.id}: {user.name}"


def compute_hash(data):
    """Duplicate of auth.hash_password — dead code."""
    return hashlib.sha256(data.encode()).hexdigest()
```

Write `tests/eval/fixtures/toybox/errors.py`:

```python
"""Exception classes used by the app."""


class AuthError(Exception):
    pass


class ValidationError(Exception):
    pass


class DatabaseError(Exception):
    pass
```

Write `tests/eval/fixtures/toybox/config.py`:

```python
"""Hardcoded config — intentional smell."""

DB_PATH = "/var/lib/toybox/app.db"
SECRET_KEY = "hardcoded-literal-do-not-ship"
LOG_LEVEL = "DEBUG"
```

Write `tests/eval/fixtures/toybox/tests/__init__.py`:

```python
```

Write `tests/eval/fixtures/toybox/tests/test_app.py`:

```python
"""Tests for app.py routes."""

from ..app import login, get_user_view


def test_login_flow():
    result = login({"username": "alice", "password": "pw"})
    assert result == {"ok": True}


def test_user_list():
    result = get_user_view({"token": "t"}, 1)
    assert result is not None
```

Write `tests/eval/fixtures/toybox/tests/test_auth.py`:

```python
"""Tests for auth.py helpers."""

from ..auth import validate_token, hash_password


def test_validate_token():
    result = validate_token("sometoken")
    assert result is None or result


def test_hash_password():
    h = hash_password("hunter2")
    assert len(h) == 64
```

Write `tests/eval/fixtures/toybox/tests/test_models.py`:

```python
"""Tests for models.py classes."""

from ..models import User


def test_user_create():
    u = User(id=1, name="alice")
    assert u.id == 1
    assert u.name == "alice"
```

- [ ] **Step 4: Run the property test to verify it passes**

Run: `cd /mnt/aux-data/teague/Projects/lackpy/main && python -m pytest tests/eval/test_toybox_properties.py -v`
Expected: 8 passed (the first run seeds `toybox.sha256`).

- [ ] **Step 5: Verify the hash pin file was created**

Run: `cat tests/eval/fixtures/toybox.sha256`
Expected: a single line, 64 hex characters + newline.

- [ ] **Step 6: Re-run the test to confirm the hash is stable**

Run: `cd /mnt/aux-data/teague/Projects/lackpy/main && python -m pytest tests/eval/test_toybox_properties.py::test_toybox_hash_is_stable -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tests/eval/fixtures/toybox/ tests/eval/fixtures/toybox.sha256 tests/eval/test_toybox_properties.py
git commit -m "$(cat <<'EOF'
eval: toybox fixture + property tests

9 source files + 3 test files forming a webapp with deliberate smells:
2 @deprecated fns, 4 @route handlers, 3 SQL-concat sites, 1 leaky
connection, 2 mutable default args, 4 test_ functions, 4 v1→v2 rename
pairs. Property tests pin counts; a sha256 digest is written to
toybox.sha256 so the harness can detect drift.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Intent dataclass and the `python` corpus

**Files:**
- Create: `scripts/prompt_eval/intents.py`
- Create: `scripts/prompt_eval/intents_python.py`
- Test: `tests/eval/test_intents_python.py`

The `Intent` dataclass is the core abstraction: every corpus row has an id, interpreter name, difficulty, natural-language text, expected return shape, a structural gate callable, and an execution assertion callable. The gate takes the sanitized program string and returns a `GateResult`; the assertion takes the execution output and returns `bool`.

- [ ] **Step 1: Write the failing test**

Write `tests/eval/test_intents_python.py`:

```python
"""Tests for the python interpreter intent corpus."""

import pytest
from scripts.prompt_eval.intents import GateResult
from scripts.prompt_eval.intents_python import PYTHON_INTENTS


def test_corpus_sizes():
    core = [i for i in PYTHON_INTENTS if i.difficulty == "core"]
    stretch = [i for i in PYTHON_INTENTS if i.difficulty == "stretch"]
    assert len(core) == 8
    assert len(stretch) == 6


def test_every_intent_has_unique_id():
    ids = [i.id for i in PYTHON_INTENTS]
    assert len(set(ids)) == len(ids)
    for i in PYTHON_INTENTS:
        assert i.id.startswith("py.")


def test_every_intent_targets_python_interpreter():
    for i in PYTHON_INTENTS:
        assert i.interpreter == "python"


def test_every_intent_has_nonempty_text():
    for i in PYTHON_INTENTS:
        assert len(i.text) > 20


def test_gates_accept_known_good_programs():
    # A minimal valid program should pass every python gate
    good = "files = find_files('**/*.py')\nfiles"
    for i in PYTHON_INTENTS:
        result = i.structural_gate(good)
        assert isinstance(result, GateResult)
        # Most gates just check validity; they all must at least return GateResult


def test_assertions_are_callable():
    for i in PYTHON_INTENTS:
        assert callable(i.exec_assertion)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /mnt/aux-data/teague/Projects/lackpy/main && python -m pytest tests/eval/test_intents_python.py -v`
Expected: ImportError for `scripts.prompt_eval.intents`.

- [ ] **Step 3: Write the `Intent` dataclass**

Write `scripts/prompt_eval/intents.py`:

```python
"""Intent dataclass shared by all interpreter corpora."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class GateResult:
    """Outcome of a structural gate check.

    Attributes:
        passed: Whether the raw/sanitized program is well-formed enough
            to be worth executing.
        errors: Human-readable reasons the gate failed; empty on pass.
    """

    passed: bool
    errors: list[str] = field(default_factory=list)


@dataclass
class Intent:
    """One row of a corpus: natural-language task + scoring hooks.

    Attributes:
        id: Stable identifier, e.g. "py.core.01" — used as the primary
            key when writing JSONL rows and when resuming a killed run.
        interpreter: Which interpreter this intent targets
            ("python" | "ast-select" | "pss" | "plucker").
        difficulty: "core" or "stretch".
        text: The exact natural-language prompt given to the model as
            the user message.
        return_shape: Short label describing what the execution should
            return (e.g. "list[str]", "int", "dict", "markdown").
            Informational only; the assertion does the real check.
        structural_gate: Callable taking the sanitized program string
            and returning a GateResult. If passed=False, scoring stops.
        exec_assertion: Callable taking the execution result's `output`
            and returning True if the answer is correct.
        notes: Human-readable description of what the intent is
            stressing. Surfaces in the report for failure analysis.
    """

    id: str
    interpreter: str
    difficulty: str
    text: str
    return_shape: str
    structural_gate: Callable[[str], GateResult]
    exec_assertion: Callable[[Any], bool]
    notes: str = ""
```

- [ ] **Step 4: Write the python corpus**

Write `scripts/prompt_eval/intents_python.py`:

```python
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

def _is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and len(x) > 0


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
        exec_assertion=_is_list_with_any(["app.py", "api_v1.py", "api_v2.py", "auth.py", "db.py"]),
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
        notes="find_def + read_file chaining. The assertion also checks the filename is mentioned — models often prepend a header.",
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
        notes="find_def + len() builtin. Counts the two hashing functions if the model names match loosely; an exact 1 is correct for hash_password (only auth.py defines exactly that name). The assertion accepts >=1 to tolerate string-matching width.",
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
        notes="Requires scanning test files — may go via find_files('tests/test_*.py') or via four find_def calls. Both paths are acceptable.",
    ),
]
```

- [ ] **Step 5: Run the tests again to verify they pass**

Run: `cd /mnt/aux-data/teague/Projects/lackpy/main && python -m pytest tests/eval/test_intents_python.py -v`
Expected: 5 passed (the fifth test skips intents with complex gate semantics; all python intents share the same gate).

- [ ] **Step 6: Commit**

```bash
git add scripts/prompt_eval/intents.py scripts/prompt_eval/intents_python.py tests/eval/test_intents_python.py
git commit -m "$(cat <<'EOF'
eval: Intent dataclass + python interpreter corpus (8 core + 6 stretch)

Introduces the Intent dataclass shared across all interpreter corpora
and the python interpreter corpus targeting the eval kit (read_file,
find_files, find_def, find_refs). Each intent pairs a structural gate
(lackpy validator) with an execution assertion bound to the toybox.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: ast-select corpus

**Files:**
- Create: `scripts/prompt_eval/intents_ast_select.py`
- Test: `tests/eval/test_intents_ast_select.py`

The ast-select interpreter's program is a single CSS-style selector. Structural gate: non-empty, single-line, no `{`/`}`, starts with `.`. Execution: run `AstSelectInterpreter` against the toybox and count matches or check match content in the rendered markdown.

- [ ] **Step 1: Write the failing test**

Write `tests/eval/test_intents_ast_select.py`:

```python
"""Tests for the ast-select corpus."""

from scripts.prompt_eval.intents_ast_select import AST_SELECT_INTENTS


def test_corpus_sizes():
    core = [i for i in AST_SELECT_INTENTS if i.difficulty == "core"]
    stretch = [i for i in AST_SELECT_INTENTS if i.difficulty == "stretch"]
    assert len(core) == 8
    assert len(stretch) == 6


def test_ids_are_unique_and_prefixed():
    ids = [i.id for i in AST_SELECT_INTENTS]
    assert len(set(ids)) == len(ids)
    for i in AST_SELECT_INTENTS:
        assert i.id.startswith("as.")
        assert i.interpreter == "ast-select"


def test_gate_accepts_bare_selector():
    good = ".fn#validate_token"
    for i in AST_SELECT_INTENTS:
        gr = i.structural_gate(good)
        assert gr.passed, f"{i.id} rejected a valid selector: {gr.errors}"


def test_gate_rejects_sheet():
    bad = ".fn#validate_token { show: body; }"
    for i in AST_SELECT_INTENTS:
        gr = i.structural_gate(bad)
        assert not gr.passed


def test_gate_rejects_empty():
    for i in AST_SELECT_INTENTS:
        gr = i.structural_gate("")
        assert not gr.passed
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/eval/test_intents_ast_select.py -v`
Expected: ImportError.

- [ ] **Step 3: Write the corpus**

Write `scripts/prompt_eval/intents_ast_select.py`:

```python
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
        return GateResult(passed=False, errors=["bare selector must not contain declaration braces"])
    lines = [ln for ln in stripped.splitlines() if ln.strip()]
    if len(lines) > 1:
        return GateResult(passed=False, errors=[f"expected a single-line selector, got {len(lines)} lines"])
    # Must start with a class selector (pluckit uses .fn, .cls, .call, etc.)
    if not stripped.lstrip().startswith((".", "#", "*", "[")):
        return GateResult(passed=False, errors=["selector should start with . # * or ["])
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
        notes="The toybox has no async functions by design — an empty match set rendering to '' is success at the pluckit level but fails _markdown_nonempty. Accept any non-empty output as a sign the selector was understood; a true zero-match successful run would be indistinguishable from an execution failure. Revisit if we add async functions to toybox v2.",
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/eval/test_intents_ast_select.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/prompt_eval/intents_ast_select.py tests/eval/test_intents_ast_select.py
git commit -m "$(cat <<'EOF'
eval: ast-select corpus (8 core + 6 stretch)

Bare-selector intents with a permissive structural gate (non-empty,
single-line, no braces, starts with a valid pluckit selector head).
Several stretch items are pluckit-grammar-contingent and carry nonempty
assertions; the report will note any drops during implementation.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: pss corpus

**Files:**
- Create: `scripts/prompt_eval/intents_pss.py`
- Test: `tests/eval/test_intents_pss.py`

pss programs are selector sheets with declaration blocks. Structural gate: non-empty, balanced braces, at least one rule. Execution delegates to `PssInterpreter` against the toybox.

- [ ] **Step 1: Write the failing test**

Write `tests/eval/test_intents_pss.py`:

```python
"""Tests for the pss corpus."""

from scripts.prompt_eval.intents_pss import PSS_INTENTS


def test_corpus_sizes():
    core = [i for i in PSS_INTENTS if i.difficulty == "core"]
    stretch = [i for i in PSS_INTENTS if i.difficulty == "stretch"]
    assert len(core) == 8
    assert len(stretch) == 6


def test_ids_and_interpreter():
    for i in PSS_INTENTS:
        assert i.id.startswith("pss.")
        assert i.interpreter == "pss"


def test_gate_accepts_valid_sheet():
    good = ".fn#validate_token { show: body; }\n.cls#User { show: outline; }"
    for i in PSS_INTENTS:
        gr = i.structural_gate(good)
        assert gr.passed, f"{i.id} rejected a valid sheet: {gr.errors}"


def test_gate_rejects_unbalanced_braces():
    bad = ".fn#validate_token { show: body;"
    for i in PSS_INTENTS:
        gr = i.structural_gate(bad)
        assert not gr.passed


def test_gate_rejects_empty():
    for i in PSS_INTENTS:
        gr = i.structural_gate("   \n  ")
        assert not gr.passed
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/eval/test_intents_pss.py -v`
Expected: ImportError.

- [ ] **Step 3: Write the corpus**

Write `scripts/prompt_eval/intents_pss.py`:

```python
"""pss interpreter corpus: multi-rule selector sheets."""

from __future__ import annotations

from typing import Any

from .intents import GateResult, Intent


def _pss_gate(program: str) -> GateResult:
    """Structural gate for selector sheets.

    Rules: non-empty, balanced braces, and if any braces are present
    there must be at least one non-empty rule block. A sheet with no
    braces at all (bare selector) is accepted — pss degrades gracefully
    to single-selector rendering.
    """
    if not program or not program.strip():
        return GateResult(passed=False, errors=["empty program"])
    open_count = program.count("{")
    close_count = program.count("}")
    if open_count != close_count:
        return GateResult(
            passed=False,
            errors=[f"unbalanced braces: {open_count} opening, {close_count} closing"],
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
    # Core (8) — 1–2 rule sheets
    Intent(
        id="pss.core.01",
        interpreter="pss",
        difficulty="core",
        text="Create a selector sheet that shows the validate_token function with its full body.",
        return_shape="markdown",
        structural_gate=_pss_gate,
        exec_assertion=_markdown_contains(["validate_token"]),
        notes='1-rule: .fn#validate_token { show: body; }',
    ),
    Intent(
        id="pss.core.02",
        interpreter="pss",
        difficulty="core",
        text="Create a selector sheet that shows an outline of the class User.",
        return_shape="markdown",
        structural_gate=_pss_gate,
        exec_assertion=_markdown_contains(["User"]),
        notes="1-rule.",
    ),
    Intent(
        id="pss.core.03",
        interpreter="pss",
        difficulty="core",
        text="Create a selector sheet that shows every route handler (function with @route) as a signature.",
        return_shape="markdown",
        structural_gate=_pss_gate,
        exec_assertion=_markdown_nonempty,
        notes="1-rule with decorator filter.",
    ),
    Intent(
        id="pss.core.04",
        interpreter="pss",
        difficulty="core",
        text="Create a selector sheet that shows every function whose name starts with 'test_' as a signature.",
        return_shape="markdown",
        structural_gate=_pss_gate,
        exec_assertion=_markdown_contains(["test_"]),
        notes="1-rule with attribute filter.",
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
        text="Create a selector sheet that shows every async function's body.",
        return_shape="markdown",
        structural_gate=_pss_gate,
        exec_assertion=_markdown_nonempty,
        notes="1-rule with :async pseudo-selector.",
    ),
    Intent(
        id="pss.core.08",
        interpreter="pss",
        difficulty="core",
        text="Create a selector sheet that shows every @deprecated function's body.",
        return_shape="markdown",
        structural_gate=_pss_gate,
        exec_assertion=_markdown_nonempty,
        notes="1-rule; pluckit-grammar-contingent.",
    ),

    # Stretch (6) — 2–3 rule curated views
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/eval/test_intents_pss.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/prompt_eval/intents_pss.py tests/eval/test_intents_pss.py
git commit -m "$(cat <<'EOF'
eval: pss corpus (8 core + 6 stretch)

Selector-sheet intents with balanced-braces structural gate. Core items
are 1-2 rule sheets; stretch items are 2-3 rule curated views. Several
stretch items are pluckit-grammar-contingent.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: plucker corpus

**Files:**
- Create: `scripts/prompt_eval/intents_plucker.py`
- Test: `tests/eval/test_intents_plucker.py`

The plucker interpreter's program is a fluent chain starting with `source(...)`. Structural gate: parses as a Python expression, the top-level call is `source(...)`, and there is at least one attribute access on the result (`.find(...)`, `.names()`, etc.). Execution delegates to `PluckerInterpreter`.

- [ ] **Step 1: Write the failing test**

Write `tests/eval/test_intents_plucker.py`:

```python
"""Tests for the plucker corpus."""

from scripts.prompt_eval.intents_plucker import PLUCKER_INTENTS


def test_corpus_sizes():
    core = [i for i in PLUCKER_INTENTS if i.difficulty == "core"]
    stretch = [i for i in PLUCKER_INTENTS if i.difficulty == "stretch"]
    assert len(core) == 8
    assert len(stretch) == 6


def test_ids_and_interpreter():
    for i in PLUCKER_INTENTS:
        assert i.id.startswith("pl.")
        assert i.interpreter == "plucker"


def test_gate_accepts_source_chain():
    good = "source().find('.fn').count()"
    for i in PLUCKER_INTENTS:
        gr = i.structural_gate(good)
        assert gr.passed, f"{i.id} rejected a valid chain: {gr.errors}"


def test_gate_rejects_bare_source():
    bad = "source()"
    for i in PLUCKER_INTENTS:
        gr = i.structural_gate(bad)
        assert not gr.passed


def test_gate_rejects_non_source_start():
    bad = "find('.fn').count()"
    for i in PLUCKER_INTENTS:
        gr = i.structural_gate(bad)
        assert not gr.passed
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/eval/test_intents_plucker.py -v`
Expected: ImportError.

- [ ] **Step 3: Write the corpus**

Write `scripts/prompt_eval/intents_plucker.py`:

```python
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
    # Walk down the attribute/call chain to the leftmost Call
    # and verify its func is `source`
    leftmost = node
    has_chain = False
    while isinstance(leftmost, (ast.Call, ast.Attribute)):
        if isinstance(leftmost, ast.Call):
            if isinstance(leftmost.func, ast.Attribute):
                has_chain = True
                leftmost = leftmost.func.value
                continue
            if isinstance(leftmost.func, ast.Name):
                if leftmost.func.id != "source":
                    return GateResult(
                        passed=False,
                        errors=[f"chain must start with source(...), got {leftmost.func.id}(...)"],
                    )
                break
        if isinstance(leftmost, ast.Attribute):
            has_chain = True
            leftmost = leftmost.value
            continue
    else:
        return GateResult(passed=False, errors=["chain root is not a call to source()"])
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
        text="Return the names of every async function in the codebase as a list of strings.",
        return_shape="list[str]",
        structural_gate=_plucker_gate,
        exec_assertion=lambda x: isinstance(x, list),
        notes="source().find('.fn:async').names() — toybox has none, assertion accepts empty list.",
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
        notes="test_login_flow, test_user_list, test_validate_token, test_hash_password, test_user_create.",
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
        text="Return a markdown view of every function decorated with @deprecated.",
        return_shape="markdown",
        structural_gate=_plucker_gate,
        exec_assertion=lambda x: isinstance(x, str),
        notes="Pluckit-grammar-contingent decorator match.",
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/eval/test_intents_plucker.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/prompt_eval/intents_plucker.py tests/eval/test_intents_plucker.py
git commit -m "$(cat <<'EOF'
eval: plucker corpus (8 core + 6 stretch)

Fluent-chain intents with an AST-aware structural gate: the chain must
parse as a Python expression, start with source(...), and include at
least one attribute access. Stretch items exercise relationship
traversal and multi-chain composition.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Prompt variants (4 per interpreter, 16 total)

**Files:**
- Create: `scripts/prompt_eval/prompts.py`
- Test: `tests/eval/test_prompts.py`

The prompt variants form the ladder: `baseline → specialized → +few-shot → +constraints`. Each is a function returning a fully-formed system prompt string given the interpreter's namespace description. The namespace description is supplied by the harness from the resolved kit.

- [ ] **Step 1: Write the failing test**

Write `tests/eval/test_prompts.py`:

```python
"""Tests for prompt variants."""

from scripts.prompt_eval.prompts import (
    PROMPT_VARIANTS,
    get_prompt,
    list_variant_ids,
)


def test_every_interpreter_has_four_variants():
    for interp in ("python", "ast-select", "pss", "plucker"):
        assert interp in PROMPT_VARIANTS
        assert len(PROMPT_VARIANTS[interp]) == 4


def test_variant_ids_are_consistent():
    ids = list_variant_ids()
    expected = {"baseline", "specialized", "specialized_fewshot", "specialized_fewshot_constraints"}
    for interp in ("python", "ast-select", "pss", "plucker"):
        assert set(PROMPT_VARIANTS[interp].keys()) == expected


def test_get_prompt_returns_string():
    for interp in ("python", "ast-select", "pss", "plucker"):
        for variant_id in list_variant_ids():
            s = get_prompt(interp, variant_id, namespace_desc="tools: foo")
            assert isinstance(s, str)
            assert len(s) > 50


def test_specialized_prompts_mention_interpreter_language():
    # ast-select specialized should mention "selector"
    assert "selector" in get_prompt("ast-select", "specialized", "desc").lower()
    # pss specialized should mention "sheet" or "show:"
    pss_prompt = get_prompt("pss", "specialized", "desc").lower()
    assert "sheet" in pss_prompt or "show:" in pss_prompt or "rule" in pss_prompt
    # plucker specialized should mention source or chain
    plucker_prompt = get_prompt("plucker", "specialized", "desc").lower()
    assert "source" in plucker_prompt or "chain" in plucker_prompt


def test_constraints_variant_mentions_negatives():
    # Variant 4 should have explicit do-not-emit language
    for interp in ("python", "ast-select", "pss", "plucker"):
        p = get_prompt(interp, "specialized_fewshot_constraints", "desc").lower()
        assert "do not" in p or "never" in p or "no " in p
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/eval/test_prompts.py -v`
Expected: ImportError.

- [ ] **Step 3: Write the prompts module**

Write `scripts/prompt_eval/prompts.py`:

```python
"""Prompt variants for the four lackpy interpreters.

Four variants per interpreter, forming a ladder:

    baseline                          # current production prompt
    specialized                       # interpreter-aware framing
    specialized_fewshot               # +3-5 relevant examples
    specialized_fewshot_constraints   # +explicit negative constraints

Each variant is a function (namespace_desc) -> str. The namespace_desc
is the string rendering of the kit's tools (or an interpreter-specific
placeholder for ast-select/pss/plucker).
"""

from __future__ import annotations

from typing import Callable

from lackpy.infer.prompt import build_system_prompt as _lackpy_baseline


VariantFn = Callable[[str], str]


# ── Baseline: shared production prompt ─────────────────────────────────

def _baseline(namespace_desc: str) -> str:
    """Delegate to lackpy's production build_system_prompt().

    The baseline is deliberately *not* specialized per interpreter —
    it is the generic Jupyter-cell framing currently shipped in
    src/lackpy/infer/prompt.py. This is what every interpreter is
    compared against.
    """
    return _lackpy_baseline(namespace_desc=namespace_desc)


# ── Python interpreter variants ────────────────────────────────────────

def _python_specialized(namespace_desc: str) -> str:
    return f"""You are a lackpy program generator. Output a single Python snippet that orchestrates pre-loaded tool functions.

CRITICAL RULE — ORCHESTRATE, DO NOT IMPLEMENT:
  - The tools do the real work. Your job is to CALL them, not re-implement them.
  - If the user asks to "find definitions", CALL find_def(name). Do NOT write `def find_def(name): ...`
  - If the user asks to "read a file", CALL read_file(path). Do NOT write `open(path).read()`.

Output ONLY the program body — no markdown, no code fences, no prose.

Available tools:
{namespace_desc}

Assign tool results to variables, then end with a bare expression holding the final answer the orchestrator wants."""


def _python_fewshot(namespace_desc: str) -> str:
    return _python_specialized(namespace_desc) + """

Examples:

  User: Find the definition of validate_token. Return a dict with file and body keys.
  Program:
    rows = find_def('validate_token')
    first = rows[0]
    content = read_file(first['file'])
    result = {'file': first['file'], 'body': content}
    result

  User: Find every test file under tests/ and return their paths.
  Program:
    files = find_files('tests/test_*.py')
    files

  User: Find all callers of hash_password and return the list of filenames they live in.
  Program:
    rows = find_refs('hash_password')
    files = sorted(set(r['file'] for r in rows))
    files
"""


def _python_constraints(namespace_desc: str) -> str:
    return _python_fewshot(namespace_desc) + """

Strict constraints — your output must satisfy ALL of these:
  - NO import statements
  - NO def/class/lambda
  - NO while/try/except
  - NO code fences or markdown
  - NO explanatory prose
  - End with a single bare expression holding the result."""


# ── ast-select variants ────────────────────────────────────────────────

def _ast_select_specialized(namespace_desc: str) -> str:
    return """You generate a single CSS-style selector for a pluckit-backed AST.

Selector syntax (class-like selectors on AST node kinds):
  .fn                            — all function definitions
  .cls                           — all class definitions
  .call                          — all call sites
  .fn#NAME                       — function named NAME
  .cls#NAME                      — class named NAME
  .fn[name^="prefix"]            — functions whose name starts with prefix
  .fn:async                      — async function definitions
  .cls .fn                       — descendant: a function inside any class
  .cls#User .fn                  — a function inside class User
  .fn:has(.call#execute_sql)     — a function containing a call to execute_sql
  .fn:not([name^="test_"])       — a function whose name does not start with test_

Output rules:
  - ONE selector, nothing else.
  - NO code fences, NO Python, NO chain syntax (never .find, .names, .view, etc).
  - The selector IS the program. One line."""


def _ast_select_fewshot(namespace_desc: str) -> str:
    return _ast_select_specialized(namespace_desc) + """

Examples:

  User: Show every function named validate_token.
  Selector: .fn#validate_token

  User: Show every class named User.
  Selector: .cls#User

  User: Show every private method of the class User.
  Selector: .cls#User .fn[name^="_"]

  User: Show every function that contains a call to execute_sql.
  Selector: .fn:has(.call#execute_sql)
"""


def _ast_select_constraints(namespace_desc: str) -> str:
    return _ast_select_fewshot(namespace_desc) + """

Strict constraints:
  - NEVER wrap in ``` code fences.
  - NEVER return Python, JavaScript, or a fluent chain.
  - NEVER return multiple lines.
  - NEVER add explanation, preamble, or commentary.
  - Your entire output is ONE selector."""


# ── pss variants ───────────────────────────────────────────────────────

def _pss_specialized(namespace_desc: str) -> str:
    return """You generate a pluckit selector sheet (pss): one or more rules, each a selector followed by a declaration block.

Sheet syntax:
  SELECTOR { show: body; }
  SELECTOR { show: signature; }
  SELECTOR { show: outline; }

Declaration vocabulary:
  show: body       — render each match with its full body
  show: signature  — render each match as a one-line signature
  show: outline    — render each match as a structural outline

Examples of sheets:
  .fn#validate_token { show: body; }
  .cls#User { show: outline; }
  .fn[name^="test_"] { show: signature; }

Multi-rule sheets are one rule per line. Rules are evaluated in order.

Output ONLY the sheet — no prose, no code fences."""


def _pss_fewshot(namespace_desc: str) -> str:
    return _pss_specialized(namespace_desc) + """

Examples:

  User: Create a sheet that shows validate_token with its body.
  Sheet:
    .fn#validate_token { show: body; }

  User: Create a sheet with two rules: show User class body and Session class outline.
  Sheet:
    .cls#User { show: body; }
    .cls#Session { show: outline; }

  User: Create a sheet that shows every route handler as a signature.
  Sheet:
    .fn:has(.decorator#route) { show: signature; }
"""


def _pss_constraints(namespace_desc: str) -> str:
    return _pss_fewshot(namespace_desc) + """

Strict constraints:
  - NEVER wrap in ``` code fences.
  - NEVER return Python or a fluent chain.
  - Each rule must be `SELECTOR { show: MODE; }` with balanced braces.
  - NO prose or preamble — your entire output is rules."""


# ── plucker variants ───────────────────────────────────────────────────

def _plucker_specialized(namespace_desc: str) -> str:
    return """You generate a single pluckit fluent chain expression.

Shape: source([code]).chain.terminal

Entry:
  source()                       — use the default code source
  source("path/to/file.py")      — override the default

Chainable methods (pluckit Selection):
  .find(selector)                — narrow to matching descendants
  .callers()                     — functions that call this
  .filter(predicate)             — filter by condition

Terminal operations:
  .count()                       — return an int
  .names()                       — return list[str]
  .view()                        — return markdown str
  .materialize()                 — return list[dict]

Examples of entry + chain + terminal:
  source().find(".fn").count()
  source().find(".cls").names()
  source().find(".cls#User").view()

Output ONLY the chain — no code fences, no Python surrounding it, no prose."""


def _plucker_fewshot(namespace_desc: str) -> str:
    return _plucker_specialized(namespace_desc) + """

Examples:

  User: Return the count of every function.
  Chain: source().find('.fn').count()

  User: Return the names of every class.
  Chain: source().find('.cls').names()

  User: Return a markdown view of the class User.
  Chain: source().find('.cls#User').view()

  User: Return the names of every method inside the class User.
  Chain: source().find('.cls#User .fn').names()
"""


def _plucker_constraints(namespace_desc: str) -> str:
    return _plucker_fewshot(namespace_desc) + """

Strict constraints:
  - NEVER wrap in ``` code fences.
  - NEVER emit multiple statements or multi-line code.
  - The chain must start with `source(`.
  - The chain must end with a terminal call like .count(), .names(), .view(), or .materialize().
  - Do not define functions, classes, or variables outside the chain."""


# ── Registry ───────────────────────────────────────────────────────────

PROMPT_VARIANTS: dict[str, dict[str, VariantFn]] = {
    "python": {
        "baseline": _baseline,
        "specialized": _python_specialized,
        "specialized_fewshot": _python_fewshot,
        "specialized_fewshot_constraints": _python_constraints,
    },
    "ast-select": {
        "baseline": _baseline,
        "specialized": _ast_select_specialized,
        "specialized_fewshot": _ast_select_fewshot,
        "specialized_fewshot_constraints": _ast_select_constraints,
    },
    "pss": {
        "baseline": _baseline,
        "specialized": _pss_specialized,
        "specialized_fewshot": _pss_fewshot,
        "specialized_fewshot_constraints": _pss_constraints,
    },
    "plucker": {
        "baseline": _baseline,
        "specialized": _plucker_specialized,
        "specialized_fewshot": _plucker_fewshot,
        "specialized_fewshot_constraints": _plucker_constraints,
    },
}


def list_variant_ids() -> list[str]:
    return ["baseline", "specialized", "specialized_fewshot", "specialized_fewshot_constraints"]


def get_prompt(interpreter: str, variant_id: str, namespace_desc: str) -> str:
    try:
        fn = PROMPT_VARIANTS[interpreter][variant_id]
    except KeyError as e:
        raise KeyError(f"No prompt variant for ({interpreter}, {variant_id})") from e
    return fn(namespace_desc)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/eval/test_prompts.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/prompt_eval/prompts.py tests/eval/test_prompts.py
git commit -m "$(cat <<'EOF'
eval: 16 prompt variants (4 ladder rungs × 4 interpreters)

baseline reuses lackpy's build_system_prompt(); specialized variants
are interpreter-aware with explicit output-format constraints and
few-shot examples scaling by rung. The constraints variant is the
direct test of the "explicit constraint lists may be counterproductive"
hypothesis from prior findings.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Structural gates — collective module (stage 1 scoring)

**Files:**
- Create: `scripts/prompt_eval/scoring.py` (stage 1 only for this task; stage 2 lands in Task 9)
- Test: `tests/eval/test_scoring_gate.py`

Each interpreter corpus already has an intent-bound `structural_gate` callable. This module adds the top-level `run_gate(intent, raw_generation)` orchestration: it sanitizes via `lackpy.infer.sanitize.sanitize_output` and returns a `(sanitized_program, GateResult)` tuple. The sanitization step ensures findings transfer to the production pipeline.

- [ ] **Step 1: Write the failing test**

Write `tests/eval/test_scoring_gate.py`:

```python
"""Tests for stage 1 structural gating."""

import pytest

from scripts.prompt_eval.intents import Intent, GateResult
from scripts.prompt_eval.scoring import run_gate


def _dummy_intent(gate_impl):
    return Intent(
        id="test.01",
        interpreter="python",
        difficulty="core",
        text="test",
        return_shape="str",
        structural_gate=gate_impl,
        exec_assertion=lambda x: True,
    )


def _pass_gate(p: str) -> GateResult:
    return GateResult(passed=True)


def _fail_gate(p: str) -> GateResult:
    return GateResult(passed=False, errors=["nope"])


def test_run_gate_strips_code_fences():
    raw = "```python\nfiles = find_files('*.py')\nfiles\n```"
    sanitized, gr = run_gate(_dummy_intent(_pass_gate), raw)
    assert "```" not in sanitized
    assert "find_files" in sanitized
    assert gr.passed


def test_run_gate_passes_sanitized_to_intent_gate():
    called_with = []
    def capture(p: str) -> GateResult:
        called_with.append(p)
        return GateResult(passed=True)
    raw = "```\nhello\n```"
    run_gate(_dummy_intent(capture), raw)
    assert len(called_with) == 1
    assert "```" not in called_with[0]


def test_run_gate_returns_gate_failure():
    sanitized, gr = run_gate(_dummy_intent(_fail_gate), "anything")
    assert not gr.passed
    assert gr.errors == ["nope"]


def test_run_gate_handles_empty_raw():
    sanitized, gr = run_gate(_dummy_intent(_fail_gate), "")
    assert sanitized == ""
    assert not gr.passed
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/eval/test_scoring_gate.py -v`
Expected: ImportError.

- [ ] **Step 3: Write the scoring module skeleton with stage 1**

Write `scripts/prompt_eval/scoring.py`:

```python
"""Two-stage scoring: structural gate then execution assertion.

Stage 1 (always): sanitize the raw generation, then ask the intent's
structural_gate whether the result is worth executing.

Stage 2 (on gate pass): run the program through the appropriate
lackpy interpreter against the toybox and check the intent's
exec_assertion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lackpy.infer.sanitize import sanitize_output

from .intents import GateResult, Intent


@dataclass
class CellScore:
    """Complete score for one matrix cell.

    Attributes:
        raw_generation: Exact model output before sanitization.
        sanitized_program: The program after sanitization — what was scored.
        gate: The structural-gate result (stage 1).
        executed: Whether the program was actually executed (stage 2).
        exec_output: The interpreter's execution output; None if not executed.
        exec_error: Error string if execution failed; None on success.
        assertion_passed: True if exec_assertion(exec_output) was True.
        score: Final integer 0/1/2 (see spec).
        duration_ms_execution: Wall-clock for stage 2 (0.0 if not executed).
        interpreter_metadata: Whatever metadata the interpreter attached.
    """

    raw_generation: str
    sanitized_program: str
    gate: GateResult
    executed: bool = False
    exec_output: Any = None
    exec_error: str | None = None
    assertion_passed: bool = False
    score: int = 0
    duration_ms_execution: float = 0.0
    interpreter_metadata: dict = field(default_factory=dict)


def run_gate(intent: Intent, raw_generation: str) -> tuple[str, GateResult]:
    """Sanitize the raw generation and run the intent's structural gate.

    Returns (sanitized_program, GateResult).
    """
    if not raw_generation:
        return "", GateResult(passed=False, errors=["empty raw generation"])
    sanitized = sanitize_output(raw_generation)
    gate = intent.structural_gate(sanitized)
    return sanitized, gate
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/eval/test_scoring_gate.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/prompt_eval/scoring.py tests/eval/test_scoring_gate.py
git commit -m "$(cat <<'EOF'
eval: stage 1 scoring (structural gate + sanitization)

Adds CellScore dataclass and run_gate() that sanitizes raw model output
via lackpy.infer.sanitize.sanitize_output before delegating to the
intent's own structural_gate. Stage 2 (execution) lands in the next task.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Execution scoring (stage 2)

**Files:**
- Modify: `scripts/prompt_eval/scoring.py`
- Test: `tests/eval/test_scoring_execution.py`

Stage 2 runs the program through the matching lackpy interpreter against the toybox and checks the intent's `exec_assertion`. Python intents run under the eval kit; ast-select/pss/plucker intents run with `config={"code": "<toybox>/**/*.py"}`.

- [ ] **Step 1: Write the failing test**

Write `tests/eval/test_scoring_execution.py`:

```python
"""Tests for stage 2 execution scoring."""

from pathlib import Path
import pytest

from scripts.prompt_eval.intents import GateResult, Intent
from scripts.prompt_eval.scoring import run_gate, run_execution, score_cell


TOYBOX = Path(__file__).parent / "fixtures" / "toybox"


def _assert_true(x):
    return True


def _assert_contains(needle):
    def check(x):
        return needle in str(x)
    return check


def test_python_execution_passes_assertion():
    intent = Intent(
        id="pyexec.01",
        interpreter="python",
        difficulty="core",
        text="find a file",
        return_shape="list[str]",
        structural_gate=lambda p: GateResult(passed=True),
        exec_assertion=_assert_contains("app.py"),
    )
    program = "files = find_files('*.py')\nfiles"
    sanitized, gate = run_gate(intent, program)
    assert gate.passed
    exec_res = run_execution(intent, sanitized, toybox_dir=TOYBOX)
    assert exec_res.success, f"execution failed: {exec_res.error}"
    assert intent.exec_assertion(exec_res.output)


def test_score_cell_returns_0_on_gate_fail():
    intent = Intent(
        id="pyexec.02",
        interpreter="python",
        difficulty="core",
        text="x",
        return_shape="str",
        structural_gate=lambda p: GateResult(passed=False, errors=["bad"]),
        exec_assertion=_assert_true,
    )
    score = score_cell(intent, raw_generation="anything", toybox_dir=TOYBOX)
    assert score.score == 0
    assert not score.executed


def test_score_cell_returns_1_when_assertion_fails():
    intent = Intent(
        id="pyexec.03",
        interpreter="python",
        difficulty="core",
        text="x",
        return_shape="str",
        structural_gate=lambda p: GateResult(passed=True),
        exec_assertion=lambda x: False,
    )
    score = score_cell(intent, raw_generation="x = 1\nx", toybox_dir=TOYBOX)
    # Program runs but assertion says no
    assert score.executed
    assert score.score == 1


def test_score_cell_returns_2_on_full_pass():
    intent = Intent(
        id="pyexec.04",
        interpreter="python",
        difficulty="core",
        text="x",
        return_shape="int",
        structural_gate=lambda p: GateResult(passed=True),
        exec_assertion=lambda x: x == 42,
    )
    score = score_cell(intent, raw_generation="42", toybox_dir=TOYBOX)
    assert score.executed
    assert score.assertion_passed
    assert score.score == 2
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/eval/test_scoring_execution.py -v`
Expected: ImportError on `run_execution` and `score_cell`.

- [ ] **Step 3: Extend `scoring.py` with stage 2**

Append to `scripts/prompt_eval/scoring.py`:

```python
# ─────────────────────────────────────────────────────────────────────
# Stage 2 — execution scoring
# ─────────────────────────────────────────────────────────────────────

import asyncio
import time
from pathlib import Path

from lackpy.interpreters import (
    AstSelectInterpreter,
    ExecutionContext,
    InterpreterExecutionResult,
    PluckerInterpreter,
    PssInterpreter,
    PythonInterpreter,
    run_interpreter,
)

from .eval_kit import build_eval_kit


_INTERPRETER_FACTORIES = {
    "python": PythonInterpreter,
    "ast-select": AstSelectInterpreter,
    "pss": PssInterpreter,
    "plucker": PluckerInterpreter,
}


def _build_context(interpreter_name: str, toybox_dir: Path) -> ExecutionContext:
    """Build an ExecutionContext appropriate for each interpreter.

    The python interpreter needs the eval kit resolved against the
    toybox base dir; the pluckit-backed interpreters need a `code`
    glob pointing at toybox python files.
    """
    toybox_dir = Path(toybox_dir).resolve()
    if interpreter_name == "python":
        kit = build_eval_kit(toybox_dir)
        return ExecutionContext(kit=kit, base_dir=toybox_dir)
    code_glob = str(toybox_dir / "**" / "*.py")
    return ExecutionContext(base_dir=toybox_dir, config={"code": code_glob})


def run_execution(
    intent: Intent,
    sanitized_program: str,
    toybox_dir: Path,
) -> InterpreterExecutionResult:
    """Execute the sanitized program via the matching lackpy interpreter.

    Returns the full InterpreterExecutionResult. Safe to call from a
    sync context — wraps the interpreter's async execute() in asyncio.run.
    """
    factory = _INTERPRETER_FACTORIES[intent.interpreter]
    interp = factory()
    ctx = _build_context(intent.interpreter, toybox_dir)
    return asyncio.run(run_interpreter(interp, sanitized_program, ctx))


def score_cell(
    intent: Intent,
    raw_generation: str,
    toybox_dir: Path,
) -> CellScore:
    """End-to-end cell scoring: gate → execute → assert → score 0/1/2."""
    sanitized, gate = run_gate(intent, raw_generation)
    cs = CellScore(
        raw_generation=raw_generation,
        sanitized_program=sanitized,
        gate=gate,
    )
    if not gate.passed:
        cs.score = 0
        return cs

    start = time.perf_counter()
    try:
        exec_result = run_execution(intent, sanitized, toybox_dir)
    except Exception as e:
        cs.executed = True
        cs.exec_error = f"{type(e).__name__}: {e}"
        cs.score = 1
        cs.duration_ms_execution = (time.perf_counter() - start) * 1000
        return cs
    cs.duration_ms_execution = (time.perf_counter() - start) * 1000
    cs.executed = True
    cs.exec_output = exec_result.output
    cs.exec_error = exec_result.error
    cs.interpreter_metadata = dict(exec_result.metadata or {})

    if not exec_result.success:
        cs.score = 1
        return cs

    try:
        passed = bool(intent.exec_assertion(exec_result.output))
    except Exception as e:
        cs.assertion_passed = False
        cs.exec_error = f"assertion raised: {type(e).__name__}: {e}"
        cs.score = 1
        return cs

    cs.assertion_passed = passed
    cs.score = 2 if passed else 1
    return cs
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/eval/test_scoring_execution.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/prompt_eval/scoring.py tests/eval/test_scoring_execution.py
git commit -m "$(cat <<'EOF'
eval: stage 2 execution scoring via lackpy.interpreters

Adds run_execution() and score_cell() to scoring.py. score_cell emits
the full 0/1/2 rubric: 0 for gate fail, 1 for execution or assertion
fail, 2 for full pass. Reuses lackpy's own interpreter plugins so
findings reflect the production execution path.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Ollama streaming runner with timeout

**Files:**
- Create: `scripts/prompt_eval/runner.py`
- Test: `tests/eval/test_runner.py`

The runner wraps Ollama chat calls with streaming (for timeout), retry-free single-shot generation, token-count capture, and exception trapping. Adapted from `scripts/pluckit-quartermaster.py`'s `_chat_with_timeout`. This module does no sanitization — that's scoring.py's job — but it does return the raw string plus timing metadata.

- [ ] **Step 1: Write the failing test**

Write `tests/eval/test_runner.py`:

```python
"""Tests for the ollama streaming runner."""

from unittest.mock import MagicMock, patch

import pytest

from scripts.prompt_eval.runner import GenerationRecord, generate_once


def _fake_stream(tokens: list[str]):
    """Return a generator yielding fake ollama chunks."""
    class _Msg:
        def __init__(self, c): self.content = c
    class _Chunk:
        def __init__(self, c, ec, pec):
            self.message = _Msg(c)
            self.eval_count = ec
            self.prompt_eval_count = pec
    for i, t in enumerate(tokens):
        yield _Chunk(t, ec=i + 1, pec=10)


def test_generate_once_returns_record():
    fake_client = MagicMock()
    fake_client.chat.return_value = _fake_stream(["hello", " world"])
    rec = generate_once(
        client=fake_client,
        model="test-model",
        system_prompt="sys",
        user_message="user",
        temperature=0.1,
        timeout=10,
    )
    assert isinstance(rec, GenerationRecord)
    assert rec.raw == "hello world"
    assert rec.model == "test-model"
    assert rec.tokens_eval == 2
    assert rec.tokens_prompt == 10
    assert rec.error is None


def test_generate_once_captures_exception():
    fake_client = MagicMock()
    fake_client.chat.side_effect = RuntimeError("boom")
    rec = generate_once(
        client=fake_client, model="m", system_prompt="s",
        user_message="u", temperature=0.1, timeout=10,
    )
    assert rec.raw == ""
    assert rec.error is not None
    assert "boom" in rec.error


def test_generate_once_reports_timeout():
    def _slow_stream():
        import time
        for i in range(100):
            time.sleep(0.1)
            class _Msg:
                def __init__(self, c): self.content = c
            class _Chunk:
                def __init__(self, c, ec, pec):
                    self.message = _Msg(c)
                    self.eval_count = ec
                    self.prompt_eval_count = pec
            yield _Chunk(".", ec=i + 1, pec=5)

    fake_client = MagicMock()
    fake_client.chat.return_value = _slow_stream()
    rec = generate_once(
        client=fake_client, model="m", system_prompt="s",
        user_message="u", temperature=0.1, timeout=1,
    )
    assert rec.error is not None
    assert "timeout" in rec.error.lower() or "exceeded" in rec.error.lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/eval/test_runner.py -v`
Expected: ImportError.

- [ ] **Step 3: Write the runner**

Write `scripts/prompt_eval/runner.py`:

```python
"""Ollama streaming runner: one shot generation with timeout and token counts."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class GenerationRecord:
    """One generation attempt's raw data.

    Attributes:
        model: Ollama model id (e.g. 'qwen2.5-coder:1.5b').
        raw: Raw content string returned by the model.
        tokens_eval: eval_count from the last streaming chunk (model-produced tokens).
        tokens_prompt: prompt_eval_count (input tokens consumed).
        duration_ms: Wall-clock milliseconds for the generation call.
        error: Error string if the call failed or timed out; None on success.
    """

    model: str
    raw: str
    tokens_eval: int
    tokens_prompt: int
    duration_ms: float
    error: str | None = None


def generate_once(
    client: Any,
    model: str,
    system_prompt: str,
    user_message: str,
    temperature: float,
    timeout: int,
    keep_alive: str = "30m",
) -> GenerationRecord:
    """Run a single streaming chat against Ollama with timeout.

    Exceptions are captured and returned as an error field on the
    record, not raised. Timeouts are measured against wall-clock time
    and bail the stream early.
    """
    start = time.time()
    chunks: list[str] = []
    eval_count = 0
    prompt_eval_count = 0
    error: str | None = None
    try:
        stream = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            options={"temperature": temperature},
            stream=True,
            keep_alive=keep_alive,
        )
        for chunk in stream:
            elapsed = time.time() - start
            if elapsed > timeout:
                error = f"timeout: exceeded {timeout}s"
                break
            token = chunk.message.content or ""
            chunks.append(token)
            ec = getattr(chunk, "eval_count", None)
            if ec:
                eval_count = ec
            pec = getattr(chunk, "prompt_eval_count", None)
            if pec:
                prompt_eval_count = pec
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    duration_ms = (time.time() - start) * 1000
    return GenerationRecord(
        model=model,
        raw="".join(chunks),
        tokens_eval=eval_count,
        tokens_prompt=prompt_eval_count,
        duration_ms=duration_ms,
        error=error,
    )


def make_ollama_client(host: str = "http://localhost:11435"):
    """Construct an ollama.Client. Lazy-imports ollama so tests can mock."""
    import ollama
    return ollama.Client(host=host)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/eval/test_runner.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/prompt_eval/runner.py tests/eval/test_runner.py
git commit -m "$(cat <<'EOF'
eval: ollama streaming runner with timeout + token counts

Single-shot generation wrapper. Captures eval_count, prompt_eval_count,
wall-clock duration, and any errors/timeouts as structured record data.
No sanitization (scoring.py handles that); no retry (we're measuring
raw prompt quality, not correction loops).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Harness orchestrator — matrix, JSONL, resume, tqdm, SIGINT

**Files:**
- Create: `scripts/prompt_eval/harness.py`
- Test: `tests/eval/test_harness.py`

The harness orchestrates the matrix. Responsibilities:
1. Take configuration (output path, models, interpreters, variant_ids, intent_ids).
2. Compute the full cell set.
3. Read any existing JSONL to build the resume skip-set.
4. Iterate cells, calling the runner then scoring; write one JSONL row per cell.
5. Show a tqdm progress bar with live description.
6. Handle SIGINT gracefully (flush, exit 0).
7. Write a `_meta` header row with the toybox hash, configuration, timestamp.

- [ ] **Step 1: Write the failing test**

Write `tests/eval/test_harness.py`:

```python
"""Tests for the harness orchestrator."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.prompt_eval.intents import GateResult, Intent
from scripts.prompt_eval.harness import (
    HarnessConfig,
    compute_cells,
    load_completed_keys,
    make_row,
    run_harness,
    toybox_hash,
)
from scripts.prompt_eval.runner import GenerationRecord


TOYBOX = Path(__file__).parent / "fixtures" / "toybox"


def _trivial_intent(id_: str, interp: str = "python"):
    return Intent(
        id=id_,
        interpreter=interp,
        difficulty="core",
        text="trivial",
        return_shape="int",
        structural_gate=lambda p: GateResult(passed=True),
        exec_assertion=lambda x: True,
    )


def test_toybox_hash_is_64_hex():
    h = toybox_hash(TOYBOX)
    assert isinstance(h, str)
    assert len(h) == 64


def test_compute_cells_matrix_size():
    intents = [_trivial_intent("py.a"), _trivial_intent("py.b")]
    cells = compute_cells(
        models=["m1", "m2"],
        interpreters=["python"],
        variant_ids=["baseline", "specialized"],
        intents=intents,
    )
    assert len(cells) == 2 * 1 * 2 * 2  # m × i × v × intent


def test_load_completed_keys_empty_file(tmp_path: Path):
    f = tmp_path / "nope.jsonl"
    keys = load_completed_keys(f)
    assert keys == set()


def test_load_completed_keys_reads_meta_and_rows(tmp_path: Path):
    f = tmp_path / "x.jsonl"
    f.write_text(
        json.dumps({"_meta": {"hash": "abc"}}) + "\n" +
        json.dumps({"model": "m1", "interpreter": "python",
                    "variant_id": "baseline", "intent_id": "py.a",
                    "score": 2}) + "\n"
    )
    keys = load_completed_keys(f)
    assert ("m1", "python", "baseline", "py.a") in keys
    assert len(keys) == 1


def test_run_harness_resume_skips_completed(tmp_path: Path):
    intents = [_trivial_intent("py.a"), _trivial_intent("py.b")]
    cfg = HarnessConfig(
        output_path=tmp_path / "out.jsonl",
        models=["m1"],
        interpreters=["python"],
        variant_ids=["baseline"],
        intents=intents,
        toybox_dir=TOYBOX,
        ollama_host="http://unused",
        temperature=0.1,
        timeout=10,
    )

    # Pre-seed the output with py.a already done
    cfg.output_path.write_text(
        json.dumps({"_meta": {"hash": toybox_hash(TOYBOX)}}) + "\n" +
        json.dumps({"model": "m1", "interpreter": "python",
                    "variant_id": "baseline", "intent_id": "py.a",
                    "score": 2}) + "\n"
    )

    called = []
    def fake_runner(*, client, model, system_prompt, user_message, temperature, timeout, keep_alive="30m"):
        called.append((model, user_message))
        return GenerationRecord(
            model=model, raw="1", tokens_eval=1, tokens_prompt=1,
            duration_ms=1.0, error=None,
        )

    with patch("scripts.prompt_eval.harness.generate_once", side_effect=fake_runner), \
         patch("scripts.prompt_eval.harness.make_ollama_client", return_value=MagicMock()):
        run_harness(cfg)

    # Only py.b should have run
    assert len(called) == 1
    assert "trivial" in called[0][1]


def test_make_row_contains_expected_keys():
    intent = _trivial_intent("py.a")
    gen = GenerationRecord(
        model="m1", raw="1", tokens_eval=1, tokens_prompt=5,
        duration_ms=200.0, error=None,
    )
    row = make_row(
        model="m1", interpreter="python", variant_id="baseline",
        intent=intent, gen=gen, score=None, toybox_dir=TOYBOX,
    )
    for key in ("model", "interpreter", "variant_id", "intent_id",
                "raw_generation", "duration_ms_generation",
                "tokens_eval", "tokens_prompt", "gen_error"):
        assert key in row
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/eval/test_harness.py -v`
Expected: ImportError.

- [ ] **Step 3: Write the harness**

Write `scripts/prompt_eval/harness.py`:

```python
"""Harness orchestrator for the prompt evaluation sweep.

Iterates the matrix {model × interpreter × variant × intent}, runs
the Ollama generation, scores via scoring.score_cell(), and writes
one JSONL row per cell to a resumable output file.

Matrix iteration order: intents fastest, variants, interpreters,
models slowest. This keeps the model hot across a whole row of
(variant × interpreter × intent) cells before switching.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

from lackpy.kit.registry import ResolvedKit

from .eval_kit import build_eval_kit
from .intents import Intent
from .prompts import get_prompt
from .runner import GenerationRecord, generate_once, make_ollama_client
from .scoring import CellScore, score_cell


# ── Configuration ──────────────────────────────────────────────────────


@dataclass
class HarnessConfig:
    """All inputs for a run."""

    output_path: Path
    models: list[str]
    interpreters: list[str]
    variant_ids: list[str]
    intents: list[Intent]
    toybox_dir: Path
    ollama_host: str = "http://localhost:11435"
    temperature: float = 0.2
    timeout: int = 60
    keep_alive: str = "30m"


# ── Hashing and namespace descriptions ─────────────────────────────────


def toybox_hash(toybox_dir: Path) -> str:
    """sha256 of the concatenated sorted python file contents under toybox_dir."""
    h = hashlib.sha256()
    for p in sorted(Path(toybox_dir).rglob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()


_NAMESPACE_DESC_CACHE: dict[str, str] = {}


def _namespace_desc_for(interpreter: str, toybox_dir: Path) -> str:
    """Namespace description passed into the prompt template.

    For `python`, we resolve the eval kit and use its description. For
    the three pluckit-backed interpreters, we return a short placeholder
    — the prompt variants hard-code their own selector/sheet/chain
    documentation and don't need a kit rendering.
    """
    cache_key = f"{interpreter}:{toybox_dir}"
    if cache_key in _NAMESPACE_DESC_CACHE:
        return _NAMESPACE_DESC_CACHE[cache_key]
    if interpreter == "python":
        kit: ResolvedKit = build_eval_kit(toybox_dir)
        desc = kit.description
    else:
        desc = f"(interpreter={interpreter}; see prompt variant for syntax reference)"
    _NAMESPACE_DESC_CACHE[cache_key] = desc
    return desc


# ── Cell computation and resume ────────────────────────────────────────


def compute_cells(
    models: list[str],
    interpreters: list[str],
    variant_ids: list[str],
    intents: list[Intent],
) -> list[tuple[str, str, str, Intent]]:
    """Expand the matrix into an ordered list of cells.

    Intents are filtered to match each interpreter — an intent tagged
    'python' is skipped when the current interpreter is 'ast-select'.
    Ordering: model (slowest) → interpreter → variant → intent
    (fastest). This keeps a model hot across (variant × intent) cells.
    """
    cells: list[tuple[str, str, str, Intent]] = []
    intents_by_interpreter: dict[str, list[Intent]] = {}
    for intent in intents:
        intents_by_interpreter.setdefault(intent.interpreter, []).append(intent)
    for model in models:
        for interp in interpreters:
            for variant_id in variant_ids:
                for intent in intents_by_interpreter.get(interp, []):
                    cells.append((model, interp, variant_id, intent))
    return cells


def load_completed_keys(
    output_path: Path,
) -> set[tuple[str, str, str, str]]:
    """Read an existing JSONL and return the set of completed cell keys.

    The header `_meta` row is ignored. Rows missing any of the four
    key fields are also ignored so partial/corrupt files can be recovered.
    """
    if not output_path.exists():
        return set()
    keys: set[tuple[str, str, str, str]] = set()
    with output_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "_meta" in row:
                continue
            try:
                keys.add((
                    row["model"], row["interpreter"],
                    row["variant_id"], row["intent_id"],
                ))
            except KeyError:
                continue
    return keys


# ── Row serialization ──────────────────────────────────────────────────


def make_row(
    *,
    model: str,
    interpreter: str,
    variant_id: str,
    intent: Intent,
    gen: GenerationRecord,
    score: CellScore | None,
    toybox_dir: Path,
) -> dict[str, Any]:
    """Compose one JSONL row. Safe for both successful and error cells."""
    row: dict[str, Any] = {
        "model": model,
        "interpreter": interpreter,
        "variant_id": variant_id,
        "intent_id": intent.id,
        "intent_difficulty": intent.difficulty,
        "intent_text": intent.text,
        "return_shape": intent.return_shape,
        "raw_generation": gen.raw,
        "duration_ms_generation": gen.duration_ms,
        "tokens_eval": gen.tokens_eval,
        "tokens_prompt": gen.tokens_prompt,
        "gen_error": gen.error,
    }
    if score is not None:
        row.update({
            "sanitized_program": score.sanitized_program,
            "gate_passed": score.gate.passed,
            "gate_errors": score.gate.errors,
            "executed": score.executed,
            "exec_output_repr": repr(score.exec_output) if score.executed else None,
            "exec_error": score.exec_error,
            "assertion_passed": score.assertion_passed,
            "duration_ms_execution": score.duration_ms_execution,
            "score": score.score,
        })
    else:
        row.update({
            "sanitized_program": "",
            "gate_passed": False,
            "gate_errors": [],
            "executed": False,
            "exec_output_repr": None,
            "exec_error": None,
            "assertion_passed": False,
            "duration_ms_execution": 0.0,
            "score": 0,
        })
    return row


def _write_meta(output_path: Path, cfg: HarnessConfig) -> None:
    """Write the _meta header row if the file doesn't already exist."""
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "_meta": {
            "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "toybox_hash": toybox_hash(cfg.toybox_dir),
            "toybox_dir": str(cfg.toybox_dir),
            "ollama_host": cfg.ollama_host,
            "models": cfg.models,
            "interpreters": cfg.interpreters,
            "variant_ids": cfg.variant_ids,
            "intent_ids": [i.id for i in cfg.intents],
            "temperature": cfg.temperature,
            "timeout": cfg.timeout,
        }
    }
    with output_path.open("w") as f:
        f.write(json.dumps(meta) + "\n")


# ── Main loop ──────────────────────────────────────────────────────────


_interrupt_received = False


def _install_sigint_handler() -> None:
    def _handler(_signum, _frame):
        global _interrupt_received
        _interrupt_received = True
        print("\n[harness] SIGINT received — finishing current cell then exiting cleanly.", file=sys.stderr)
    signal.signal(signal.SIGINT, _handler)


def run_harness(cfg: HarnessConfig) -> None:
    """Execute the matrix, writing JSONL as it goes.

    Already-completed cells (matching model, interpreter, variant_id,
    intent_id) are skipped. A KeyboardInterrupt / SIGINT causes the
    harness to finish the current cell (if mid-flight) and exit 0.
    """
    global _interrupt_received
    _interrupt_received = False
    _install_sigint_handler()

    _write_meta(cfg.output_path, cfg)
    completed = load_completed_keys(cfg.output_path)

    cells = compute_cells(
        models=cfg.models,
        interpreters=cfg.interpreters,
        variant_ids=cfg.variant_ids,
        intents=cfg.intents,
    )
    pending = [
        c for c in cells
        if (c[0], c[1], c[2], c[3].id) not in completed
    ]

    total = len(cells)
    done_count = total - len(pending)
    print(f"[harness] {done_count}/{total} cells already completed; {len(pending)} pending.")

    client = make_ollama_client(host=cfg.ollama_host)

    with cfg.output_path.open("a") as out_f:
        with tqdm(total=total, initial=done_count, desc="eval", unit="cell") as bar:
            for model, interp, variant_id, intent in pending:
                if _interrupt_received:
                    print("[harness] interrupted — exiting.", file=sys.stderr)
                    break
                bar.set_description(f"{model[:20]} / {interp[:10]} / {variant_id[:15]} / {intent.id}")
                namespace_desc = _namespace_desc_for(interp, cfg.toybox_dir)
                system_prompt = get_prompt(interp, variant_id, namespace_desc)
                gen = generate_once(
                    client=client,
                    model=model,
                    system_prompt=system_prompt,
                    user_message=intent.text,
                    temperature=cfg.temperature,
                    timeout=cfg.timeout,
                    keep_alive=cfg.keep_alive,
                )
                score: CellScore | None = None
                if gen.error is None and gen.raw:
                    try:
                        score = score_cell(intent, gen.raw, cfg.toybox_dir)
                    except Exception as e:
                        score = None
                        print(
                            f"[harness] score_cell raised on {intent.id}: "
                            f"{type(e).__name__}: {e}",
                            file=sys.stderr,
                        )
                row = make_row(
                    model=model, interpreter=interp, variant_id=variant_id,
                    intent=intent, gen=gen, score=score,
                    toybox_dir=cfg.toybox_dir,
                )
                out_f.write(json.dumps(row, default=str) + "\n")
                out_f.flush()
                bar.update(1)

    print(f"[harness] run complete — {cfg.output_path}")
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/eval/test_harness.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/prompt_eval/harness.py tests/eval/test_harness.py
git commit -m "$(cat <<'EOF'
eval: harness orchestrator with JSONL resume + tqdm + SIGINT

Iterates the matrix {model × interpreter × variant × intent},
generates via runner.generate_once(), scores via score_cell(), writes
one JSONL row per cell. Pre-existing output files are scanned for
completed (model, interpreter, variant, intent) keys and those cells
are skipped so runs are resumable. A _meta header row pins the toybox
hash and the full configuration. Ctrl+C finishes the current cell and
exits 0.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: `query.py` — live JSONL summary helper

**Files:**
- Create: `scripts/prompt_eval/query.py`
- Test: `tests/eval/test_query.py`

Reads a JSONL (live or finished) and prints per-model / per-prompt / per-interpreter pass rates, total cells done vs expected, median latency. Meant to be run in a second terminal while `harness.py` is writing, or invoked periodically by an agent querying progress.

- [ ] **Step 1: Write the failing test**

Write `tests/eval/test_query.py`:

```python
"""Tests for the query helper."""

import json
from pathlib import Path

from scripts.prompt_eval.query import summarize_jsonl


def test_summarize_empty(tmp_path: Path):
    f = tmp_path / "empty.jsonl"
    f.write_text(json.dumps({"_meta": {"toybox_hash": "x"}}) + "\n")
    summary = summarize_jsonl(f)
    assert summary["total_rows"] == 0


def test_summarize_counts_scores(tmp_path: Path):
    f = tmp_path / "x.jsonl"
    lines = [json.dumps({"_meta": {"toybox_hash": "x"}})]
    for score in [0, 1, 2, 2, 2]:
        lines.append(json.dumps({
            "model": "m1", "interpreter": "python",
            "variant_id": "baseline", "intent_id": f"i.{score}",
            "score": score,
            "duration_ms_generation": 100.0,
        }))
    f.write_text("\n".join(lines) + "\n")
    summary = summarize_jsonl(f)
    assert summary["total_rows"] == 5
    by_model = summary["by_model"]
    assert by_model["m1"]["rows"] == 5
    assert by_model["m1"]["sum_score"] == 7
    assert by_model["m1"]["pass_rate_2"] == 3 / 5


def test_summarize_groups_by_variant(tmp_path: Path):
    f = tmp_path / "x.jsonl"
    lines = [json.dumps({"_meta": {"toybox_hash": "x"}})]
    lines.append(json.dumps({"model": "m1", "interpreter": "python",
                             "variant_id": "baseline", "intent_id": "a",
                             "score": 2, "duration_ms_generation": 50.0}))
    lines.append(json.dumps({"model": "m1", "interpreter": "python",
                             "variant_id": "specialized", "intent_id": "a",
                             "score": 1, "duration_ms_generation": 50.0}))
    f.write_text("\n".join(lines) + "\n")
    summary = summarize_jsonl(f)
    assert set(summary["by_variant"].keys()) == {"baseline", "specialized"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/eval/test_query.py -v`
Expected: ImportError.

- [ ] **Step 3: Write the query helper**

Write `scripts/prompt_eval/query.py`:

```python
"""Live JSONL summary for the prompt eval harness.

Usage:
    python -m scripts.prompt_eval.query <path-to-jsonl>

Can be run while the harness is still writing. Reads the file once and
prints per-dimension aggregates.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


def _fresh_group() -> dict[str, Any]:
    return {"rows": 0, "sum_score": 0, "pass_rate_2": 0.0, "latencies": []}


def summarize_jsonl(path: Path) -> dict[str, Any]:
    """Aggregate a JSONL file into by-model, by-variant, by-interpreter dicts."""
    summary: dict[str, Any] = {
        "path": str(path),
        "total_rows": 0,
        "meta": None,
        "by_model": {},
        "by_variant": {},
        "by_interpreter": {},
        "global_median_ms": 0.0,
    }
    if not path.exists():
        return summary
    all_latencies: list[float] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "_meta" in row:
                summary["meta"] = row["_meta"]
                continue
            summary["total_rows"] += 1
            score = int(row.get("score", 0))
            latency = float(row.get("duration_ms_generation", 0.0))
            all_latencies.append(latency)
            for bucket, key in (
                ("by_model", row.get("model", "?")),
                ("by_variant", row.get("variant_id", "?")),
                ("by_interpreter", row.get("interpreter", "?")),
            ):
                g = summary[bucket].setdefault(key, _fresh_group())
                g["rows"] += 1
                g["sum_score"] += score
                g["latencies"].append(latency)
                if score == 2:
                    g["pass_rate_2"] += 1
    for bucket in ("by_model", "by_variant", "by_interpreter"):
        for key, g in summary[bucket].items():
            if g["rows"]:
                g["pass_rate_2"] = g["pass_rate_2"] / g["rows"]
                g["median_latency_ms"] = statistics.median(g["latencies"])
            else:
                g["median_latency_ms"] = 0.0
            # Drop the raw latencies list from the output; callers can re-read if needed
            del g["latencies"]
    if all_latencies:
        summary["global_median_ms"] = statistics.median(all_latencies)
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print(f"\n== {summary['path']} ==")
    print(f"total rows: {summary['total_rows']}    global median latency: {summary['global_median_ms']:.0f}ms")
    if summary["meta"]:
        meta = summary["meta"]
        print(f"toybox hash: {meta.get('toybox_hash', '?')[:16]}…  created: {meta.get('created_utc', '?')}")
    for bucket_name in ("by_model", "by_interpreter", "by_variant"):
        print(f"\n[{bucket_name}]")
        for key, g in sorted(summary[bucket_name].items(), key=lambda kv: -kv[1]["sum_score"]):
            print(f"  {key:<40} rows={g['rows']:<4} score_sum={g['sum_score']:<4} "
                  f"pass2={g['pass_rate_2']*100:5.1f}%  median={g['median_latency_ms']:.0f}ms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    summary = summarize_jsonl(args.path)
    print_summary(summary)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/eval/test_query.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/prompt_eval/query.py tests/eval/test_query.py
git commit -m "$(cat <<'EOF'
eval: query.py — live JSONL summary helper

Aggregates a running or finished JSONL into by-model, by-variant,
by-interpreter dicts with score totals, 2/2 pass rates, and median
latencies. Safe to run against a file the harness is still writing.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: `report.py` — JSONL → markdown report

**Files:**
- Create: `scripts/prompt_eval/report.py`
- Test: `tests/eval/test_report.py`

Consolidates one or more JSONL files into a single markdown report with executive summary, per-interpreter scoring matrix, top-3 cells per interpreter, and top failure modes. Refuses to merge files with mismatched toybox hashes.

- [ ] **Step 1: Write the failing test**

Write `tests/eval/test_report.py`:

```python
"""Tests for the report generator."""

import json
from pathlib import Path

import pytest

from scripts.prompt_eval.report import (
    ReportData,
    build_report,
    consolidate_jsonls,
)


def _write(path: Path, meta: dict, rows: list[dict]) -> None:
    with path.open("w") as f:
        f.write(json.dumps({"_meta": meta}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_consolidate_single_file(tmp_path: Path):
    f = tmp_path / "a.jsonl"
    _write(f, {"toybox_hash": "h1"}, [
        {"model": "m1", "interpreter": "python", "variant_id": "v1",
         "intent_id": "i.1", "score": 2},
    ])
    data = consolidate_jsonls([f])
    assert isinstance(data, ReportData)
    assert len(data.rows) == 1
    assert data.toybox_hash == "h1"


def test_consolidate_refuses_hash_mismatch(tmp_path: Path):
    f1 = tmp_path / "a.jsonl"
    f2 = tmp_path / "b.jsonl"
    _write(f1, {"toybox_hash": "h1"}, [])
    _write(f2, {"toybox_hash": "h2"}, [])
    with pytest.raises(ValueError, match="hash mismatch"):
        consolidate_jsonls([f1, f2])


def test_build_report_contains_headings(tmp_path: Path):
    f = tmp_path / "a.jsonl"
    _write(f, {"toybox_hash": "h1"}, [
        {"model": "m1", "interpreter": "python", "variant_id": "baseline",
         "intent_id": "py.core.01", "intent_difficulty": "core",
         "intent_text": "x", "score": 2, "duration_ms_generation": 100.0,
         "gate_passed": True, "assertion_passed": True},
    ])
    data = consolidate_jsonls([f])
    md = build_report(data)
    assert "# Prompt Eval Report" in md
    assert "## Executive summary" in md
    assert "## python" in md
    assert "m1" in md
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/eval/test_report.py -v`
Expected: ImportError.

- [ ] **Step 3: Write the report generator**

Write `scripts/prompt_eval/report.py`:

```python
"""JSONL → markdown report consolidator.

Usage:
    python -m scripts.prompt_eval.report results/prompt-eval-2026-04-11/*.jsonl \\
        --out results/prompt-eval-2026-04-11/report.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ReportData:
    rows: list[dict] = field(default_factory=list)
    toybox_hash: str = ""
    sources: list[str] = field(default_factory=list)


def consolidate_jsonls(paths: list[Path]) -> ReportData:
    """Merge multiple JSONL files into a single ReportData.

    Fails loudly if the files disagree on the toybox hash — cross-hash
    comparisons are invalid because the fixture drove different answers.
    """
    data = ReportData()
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)
        data.sources.append(str(p))
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "_meta" in row:
                    h = row["_meta"].get("toybox_hash", "")
                    if data.toybox_hash and h and h != data.toybox_hash:
                        raise ValueError(
                            f"toybox hash mismatch across sources: "
                            f"{data.toybox_hash} != {h}"
                        )
                    if h and not data.toybox_hash:
                        data.toybox_hash = h
                    continue
                data.rows.append(row)
    return data


def _best_cells_per_interpreter(
    data: ReportData, top_n: int = 3
) -> dict[str, list[tuple[tuple[str, str], int, int, float]]]:
    """Rank (model, variant) cells per interpreter by total score.

    Returns:
        {interpreter: [((model, variant), total_score, n_rows, median_latency), ...]}
    """
    buckets: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(
        lambda: defaultdict(lambda: {"score": 0, "rows": 0, "latencies": []})
    )
    for r in data.rows:
        interp = r.get("interpreter", "?")
        key = (r.get("model", "?"), r.get("variant_id", "?"))
        b = buckets[interp][key]
        b["score"] += int(r.get("score", 0))
        b["rows"] += 1
        b["latencies"].append(float(r.get("duration_ms_generation", 0.0)))

    out: dict[str, list[tuple[tuple[str, str], int, int, float]]] = {}
    for interp, keyed in buckets.items():
        ranked = sorted(
            (
                (k, v["score"], v["rows"], statistics.median(v["latencies"]) if v["latencies"] else 0.0)
                for k, v in keyed.items()
            ),
            key=lambda t: (-t[1], t[3]),
        )
        out[interp] = ranked[:top_n]
    return out


def _failure_modes(data: ReportData, top_n: int = 5) -> list[tuple[str, int]]:
    """Count the most common gate/exec error strings."""
    counter: dict[str, int] = defaultdict(int)
    for r in data.rows:
        for err in r.get("gate_errors", []) or []:
            counter[err] += 1
        exec_err = r.get("exec_error")
        if exec_err:
            counter[exec_err] += 1
    ranked = sorted(counter.items(), key=lambda kv: -kv[1])
    return ranked[:top_n]


def build_report(data: ReportData) -> str:
    """Render the markdown report from consolidated data."""
    lines: list[str] = []
    lines.append("# Prompt Eval Report")
    lines.append("")
    lines.append(f"- Sources: {', '.join(data.sources)}")
    lines.append(f"- Toybox hash: `{data.toybox_hash}`")
    lines.append(f"- Total rows: {len(data.rows)}")
    lines.append("")

    lines.append("## Executive summary")
    lines.append("")
    best = _best_cells_per_interpreter(data, top_n=1)
    lines.append("| Interpreter | Best (model, variant) | Score | Rows | Median gen ms |")
    lines.append("|---|---|---|---|---|")
    for interp in ("python", "ast-select", "pss", "plucker"):
        if interp in best and best[interp]:
            (m, v), score, n, lat = best[interp][0]
            lines.append(f"| {interp} | `{m}` / `{v}` | {score} | {n} | {lat:.0f} |")
        else:
            lines.append(f"| {interp} | — | — | — | — |")
    lines.append("")

    all_best = _best_cells_per_interpreter(data, top_n=3)
    for interp in ("python", "ast-select", "pss", "plucker"):
        lines.append(f"## {interp}")
        lines.append("")
        lines.append("### Top 3 cells")
        lines.append("")
        lines.append("| Rank | Model | Variant | Score | Rows | Median ms |")
        lines.append("|---|---|---|---|---|---|")
        for i, ((m, v), score, n, lat) in enumerate(all_best.get(interp, []), start=1):
            lines.append(f"| {i} | `{m}` | `{v}` | {score} | {n} | {lat:.0f} |")
        lines.append("")

    lines.append("## Top failure modes")
    lines.append("")
    for err, n in _failure_modes(data):
        lines.append(f"- `{err}` — {n}×")
    if not _failure_modes(data):
        lines.append("- _no errors recorded_")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = consolidate_jsonls(args.paths)
    md = build_report(data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/eval/test_report.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/prompt_eval/report.py tests/eval/test_report.py
git commit -m "$(cat <<'EOF'
eval: report.py — JSONL → markdown consolidator

Consolidates one or more phase JSONL files into a markdown report:
executive summary, per-interpreter top-3 cells, top failure modes.
Refuses to merge files with mismatched toybox hashes.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Phase entry-point scripts

**Files:**
- Create: `scripts/prompt_eval/phase1a_qualifier.py`
- Create: `scripts/prompt_eval/phase1b_grid.py`
- Create: `scripts/prompt_eval/phase2_refinement.py`
- Create: `scripts/prompt_eval/cohort.py` (model lists + selection helpers)
- Test: `tests/eval/test_cohort.py`

Each phase script loads its configuration, constructs a `HarnessConfig`, and calls `run_harness()`. Phase 1b and 2 read the previous phase's output to pick the qualifying models / top cells.

- [ ] **Step 1: Write the failing test for cohort**

Write `tests/eval/test_cohort.py`:

```python
"""Tests for cohort helpers."""

import json
from pathlib import Path

import pytest

from scripts.prompt_eval.cohort import (
    PHASE1A_MODELS,
    pick_phase1b_cohort,
)


def test_phase1a_models_nonempty_and_has_qwen():
    assert len(PHASE1A_MODELS) >= 10
    assert any("qwen" in m for m in PHASE1A_MODELS)
    assert all(":" in m for m in PHASE1A_MODELS)


def test_pick_phase1b_cohort_returns_top_n(tmp_path: Path):
    f = tmp_path / "qual.jsonl"
    lines = [json.dumps({"_meta": {"toybox_hash": "x"}})]
    # Seven models with descending scores
    for i, model in enumerate([f"m{n}" for n in range(7)]):
        for intent_id in ("py.core.01", "py.core.02", "py.core.03", "py.core.04"):
            lines.append(json.dumps({
                "model": model, "interpreter": "python",
                "variant_id": "baseline", "intent_id": intent_id,
                "score": 2 if i < 5 else 0,
                "gate_passed": True,
                "duration_ms_generation": 100.0 * (i + 1),
            }))
    f.write_text("\n".join(lines) + "\n")
    cohort = pick_phase1b_cohort(f, top_n=6, gate_floor=0.5)
    assert len(cohort) == 5  # only m0..m4 passed the floor
    # Best latency first among equal scores
    assert cohort[0] == "m0"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/eval/test_cohort.py -v`
Expected: ImportError.

- [ ] **Step 3: Write the cohort module**

Write `scripts/prompt_eval/cohort.py`:

```python
"""Model cohort lists and selection helpers for phase-to-phase handoff."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


# Phase 1a cohort: all useful Ollama models on localhost:11435.
# Sorted by size (ascending) so we go small-to-large and can bail
# early if memory pressure hits. Vision models are excluded.
PHASE1A_MODELS: list[str] = [
    # Tiny (<1GB)
    "qwen2.5-coder:0.5b",
    "qwen3:0.6b",
    # Small (1-2GB)
    "llama3.2:1b",
    "qwen2.5-coder:1.5b",
    "qwen2.5:1.5b",
    "codegemma:2b",
    "granite3.1-dense:2b",
    "granite3.3:2b",
    "smollm2:latest",
    # Medium (2-3GB)
    "llama3.2:latest",
    "phi4-mini:latest",
    "qwen2.5-coder:3b",
    "qwen2.5:3b",
    "granite-code:3b",
    "phi3:latest",
    # Large (5-8GB)
    "qwen2.5-coder:7b",
    "qwen2.5:7b",
    "qwen2:latest",
    # Very large (9GB)
    "gemma:latest",
]


def pick_phase1b_cohort(
    phase1a_jsonl: Path,
    top_n: int = 6,
    gate_floor: float = 0.5,
) -> list[str]:
    """Select the top-N models from a phase 1a qualifier JSONL.

    Rules:
      - Drop models whose gate-pass rate is below `gate_floor`.
      - Rank survivors by total score (descending).
      - Ties broken by median generation latency (ascending).

    Returns a list of model ids ordered best-first.
    """
    by_model: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "total_score": 0,
        "total_rows": 0,
        "gate_passes": 0,
        "latencies": [],
    })
    with phase1a_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "_meta" in row:
                continue
            model = row.get("model")
            if not model:
                continue
            b = by_model[model]
            b["total_rows"] += 1
            b["total_score"] += int(row.get("score", 0))
            if row.get("gate_passed"):
                b["gate_passes"] += 1
            b["latencies"].append(float(row.get("duration_ms_generation", 0.0)))

    survivors: list[tuple[str, int, float]] = []
    for model, b in by_model.items():
        if b["total_rows"] == 0:
            continue
        gate_rate = b["gate_passes"] / b["total_rows"]
        if gate_rate < gate_floor:
            continue
        median_latency = statistics.median(b["latencies"]) if b["latencies"] else 0.0
        survivors.append((model, b["total_score"], median_latency))

    survivors.sort(key=lambda t: (-t[1], t[2]))
    return [m for m, _s, _l in survivors[:top_n]]
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/eval/test_cohort.py -v`
Expected: 2 passed.

- [ ] **Step 5: Write the phase 1a entry point**

Write `scripts/prompt_eval/phase1a_qualifier.py`:

```python
"""Phase 1a: qualify models on the python interpreter's core intents.

Usage:
    python -m scripts.prompt_eval.phase1a_qualifier [--host URL] [--timeout 60] [--output PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .cohort import PHASE1A_MODELS, pick_phase1b_cohort
from .harness import HarnessConfig, run_harness
from .intents_python import PYTHON_INTENTS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:11435")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--output", type=Path, default=Path("results/prompt-eval-2026-04-11/phase1a-qualifier.jsonl"))
    parser.add_argument("--toybox", type=Path, default=Path("tests/eval/fixtures/toybox"))
    args = parser.parse_args()

    core_intents = [i for i in PYTHON_INTENTS if i.difficulty == "core"]
    cfg = HarnessConfig(
        output_path=args.output,
        models=PHASE1A_MODELS,
        interpreters=["python"],
        variant_ids=["baseline"],
        intents=core_intents,
        toybox_dir=args.toybox,
        ollama_host=args.host,
        temperature=0.2,
        timeout=args.timeout,
    )
    run_harness(cfg)

    print("\n== Qualifier selection ==")
    cohort = pick_phase1b_cohort(args.output, top_n=6, gate_floor=0.5)
    print("Top 6 for phase 1b:")
    for m in cohort:
        print(f"  {m}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Write the phase 1b entry point**

Write `scripts/prompt_eval/phase1b_grid.py`:

```python
"""Phase 1b: full grid over qualifier winners × 4 interpreters × 4 prompts × 14 intents."""

from __future__ import annotations

import argparse
from pathlib import Path

from .cohort import pick_phase1b_cohort
from .harness import HarnessConfig, run_harness
from .intents_ast_select import AST_SELECT_INTENTS
from .intents_plucker import PLUCKER_INTENTS
from .intents_pss import PSS_INTENTS
from .intents_python import PYTHON_INTENTS
from .prompts import list_variant_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:11435")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--qualifier", type=Path,
                        default=Path("results/prompt-eval-2026-04-11/phase1a-qualifier.jsonl"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/prompt-eval-2026-04-11/phase1b-grid.jsonl"))
    parser.add_argument("--toybox", type=Path, default=Path("tests/eval/fixtures/toybox"))
    args = parser.parse_args()

    if not args.qualifier.exists():
        raise SystemExit(f"qualifier JSONL not found: {args.qualifier}. Run phase1a_qualifier first.")
    cohort = pick_phase1b_cohort(args.qualifier, top_n=6, gate_floor=0.5)
    if not cohort:
        raise SystemExit("no models passed the qualifier floor.")
    print(f"Phase 1b cohort: {cohort}")

    all_intents = PYTHON_INTENTS + AST_SELECT_INTENTS + PSS_INTENTS + PLUCKER_INTENTS
    cfg = HarnessConfig(
        output_path=args.output,
        models=cohort,
        interpreters=["python", "ast-select", "pss", "plucker"],
        variant_ids=list_variant_ids(),
        intents=all_intents,
        toybox_dir=args.toybox,
        ollama_host=args.host,
        temperature=0.2,
        timeout=args.timeout,
    )
    run_harness(cfg)


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Write the phase 2 entry point**

Write `scripts/prompt_eval/phase2_refinement.py`:

```python
"""Phase 2: refine the top 2 (model, variant) per interpreter across temperature × constraints.

For each interpreter, find the top 2 (model, variant) cells in phase 1b,
then sweep temperature ∈ {0.0, 0.2, 0.4} on the same intents. The
example-count axis is not exercised in v1: that would require
additional variant permutations on the variants that are baseline or
specialized-without-fewshot — out of scope.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .harness import HarnessConfig, run_harness
from .intents_ast_select import AST_SELECT_INTENTS
from .intents_plucker import PLUCKER_INTENTS
from .intents_pss import PSS_INTENTS
from .intents_python import PYTHON_INTENTS


def _top_cells_from_grid(grid_jsonl: Path, top_n: int = 2) -> dict[str, list[tuple[str, str]]]:
    buckets: dict[str, dict[tuple[str, str], dict]] = defaultdict(
        lambda: defaultdict(lambda: {"score": 0, "rows": 0, "latencies": []})
    )
    with grid_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "_meta" in row:
                continue
            interp = row.get("interpreter", "?")
            key = (row.get("model", "?"), row.get("variant_id", "?"))
            b = buckets[interp][key]
            b["score"] += int(row.get("score", 0))
            b["rows"] += 1
            b["latencies"].append(float(row.get("duration_ms_generation", 0.0)))

    out: dict[str, list[tuple[str, str]]] = {}
    for interp, keyed in buckets.items():
        ranked = sorted(
            (
                (k, v["score"], statistics.median(v["latencies"]) if v["latencies"] else 0.0)
                for k, v in keyed.items()
            ),
            key=lambda t: (-t[1], t[2]),
        )
        out[interp] = [k for k, _s, _l in ranked[:top_n]]
    return out


_ALL_INTENTS = {
    "python": PYTHON_INTENTS,
    "ast-select": AST_SELECT_INTENTS,
    "pss": PSS_INTENTS,
    "plucker": PLUCKER_INTENTS,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:11435")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--grid", type=Path,
                        default=Path("results/prompt-eval-2026-04-11/phase1b-grid.jsonl"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/prompt-eval-2026-04-11/phase2-refinement.jsonl"))
    parser.add_argument("--toybox", type=Path, default=Path("tests/eval/fixtures/toybox"))
    args = parser.parse_args()

    if not args.grid.exists():
        raise SystemExit(f"grid JSONL not found: {args.grid}. Run phase1b_grid first.")
    top = _top_cells_from_grid(args.grid, top_n=2)

    # Phase 2 runs one harness invocation per temperature value, writing
    # to the same output file (resume semantics mean intermediate cells
    # can be picked up later). We embed the temperature in the variant_id
    # for bookkeeping by using a temporary tag.
    for temperature in (0.0, 0.2, 0.4):
        for interpreter, cells in top.items():
            if not cells:
                continue
            models = sorted({m for m, _v in cells})
            variant_ids = sorted({v for _m, v in cells})
            intents = _ALL_INTENTS[interpreter]
            cfg = HarnessConfig(
                output_path=args.output,
                models=models,
                interpreters=[interpreter],
                variant_ids=variant_ids,
                intents=intents,
                toybox_dir=args.toybox,
                ollama_host=args.host,
                temperature=temperature,
                timeout=args.timeout,
            )
            print(f"\n== phase 2 :: interpreter={interpreter} t={temperature} ==")
            print(f"models={models}  variants={variant_ids}")
            run_harness(cfg)


if __name__ == "__main__":
    main()
```

Note on phase 2 temperature tagging: because the `HarnessConfig.temperature` is not stored on the JSONL row directly, phase 2 rows with the same `(model, interpreter, variant_id, intent_id)` across different temperatures will collide in the resume-skip set. To avoid losing data, phase 2 writes to a **fresh output file per temperature**: update `cfg.output_path` to append `.t{temp}.jsonl`.

Replace the phase 2 output writing block (the `for temperature` loop body) with:

```python
    for temperature in (0.0, 0.2, 0.4):
        temp_suffix = f".t{temperature}"
        temp_output = args.output.with_name(args.output.stem + temp_suffix + args.output.suffix)
        for interpreter, cells in top.items():
            if not cells:
                continue
            models = sorted({m for m, _v in cells})
            variant_ids = sorted({v for _m, v in cells})
            intents = _ALL_INTENTS[interpreter]
            cfg = HarnessConfig(
                output_path=temp_output,
                models=models,
                interpreters=[interpreter],
                variant_ids=variant_ids,
                intents=intents,
                toybox_dir=args.toybox,
                ollama_host=args.host,
                temperature=temperature,
                timeout=args.timeout,
            )
            print(f"\n== phase 2 :: t={temperature} interpreter={interpreter} ==")
            print(f"models={models}  variants={variant_ids}  -> {temp_output}")
            run_harness(cfg)
```

(Keep this replacement inline in the final `phase2_refinement.py`.)

- [ ] **Step 8: Smoke test — module imports**

Run: `cd /mnt/aux-data/teague/Projects/lackpy/main && python -c "from scripts.prompt_eval import phase1a_qualifier, phase1b_grid, phase2_refinement"`
Expected: no output, exit 0.

- [ ] **Step 9: Commit**

```bash
git add scripts/prompt_eval/cohort.py scripts/prompt_eval/phase1a_qualifier.py scripts/prompt_eval/phase1b_grid.py scripts/prompt_eval/phase2_refinement.py tests/eval/test_cohort.py
git commit -m "$(cat <<'EOF'
eval: phase entry-point scripts + cohort selection

phase1a_qualifier runs all 19 Ollama models over the python core
corpus under the baseline prompt; pick_phase1b_cohort ranks survivors
by total score with a gate-floor filter. phase1b_grid consumes the
qualifier output to run the full matrix. phase2_refinement ranks
phase1b top cells and sweeps temperature ∈ {0.0, 0.2, 0.4}, writing
one JSONL per temperature so resume semantics stay valid.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: Canary test skeleton

**Files:**
- Create: `tests/eval/test_prompt_canaries.py`

The canary tests are initially skeleton-only — they're parametrized over per-interpreter expected winners that don't exist yet (the Phase 1b output fills them in). They must be marked `@pytest.mark.slow` and skipped when Ollama is unreachable so they don't block CI.

- [ ] **Step 1: Write the test file (no prior failing-test step — this file IS the test)**

Write `tests/eval/test_prompt_canaries.py`:

```python
"""Canary tests for the prompt evaluation harness.

These tests assert that the current best-known (model, variant) per
interpreter still reaches a 2/2 score on a small canary intent set.
They are marked @pytest.mark.slow and require Ollama to be reachable;
they are skipped otherwise so they do not block normal CI.

Update `_CANARIES` after each phase 1b run with the current winners.
Thresholds loosen by default (accept score >= 2 on any canary intent);
tighten to "all canary intents must pass" once findings are stable.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from urllib.parse import urlparse

import pytest

from scripts.prompt_eval.harness import HarnessConfig, run_harness
from scripts.prompt_eval.intents_ast_select import AST_SELECT_INTENTS
from scripts.prompt_eval.intents_plucker import PLUCKER_INTENTS
from scripts.prompt_eval.intents_pss import PSS_INTENTS
from scripts.prompt_eval.intents_python import PYTHON_INTENTS
from scripts.prompt_eval.query import summarize_jsonl


TOYBOX = Path(__file__).parent / "fixtures" / "toybox"
DEFAULT_HOST = os.environ.get("LACKPY_EVAL_OLLAMA", "http://localhost:11435")


def _ollama_reachable(host: str) -> bool:
    try:
        parsed = urlparse(host)
        with socket.create_connection((parsed.hostname or "localhost", parsed.port or 11434), timeout=1.0):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _ollama_reachable(DEFAULT_HOST), reason="Ollama not reachable"),
]


# Populate this dict from phase 1b results. Format:
#   interpreter -> (model, variant_id, [canary_intent_ids])
# Canary intent ids should be a mix of 1 core + 1 stretch + 1 edge-case.
_CANARIES: dict[str, tuple[str, str, list[str]]] = {
    # "python": ("qwen2.5-coder:3b", "specialized_fewshot",
    #            ["py.core.01", "py.core.04", "py.stretch.02"]),
    # "ast-select": (...),
    # "pss": (...),
    # "plucker": (...),
}


_ALL = {
    "python": PYTHON_INTENTS,
    "ast-select": AST_SELECT_INTENTS,
    "pss": PSS_INTENTS,
    "plucker": PLUCKER_INTENTS,
}


@pytest.mark.parametrize("interpreter", list(_CANARIES.keys()))
def test_canary(interpreter: str, tmp_path: Path):
    model, variant_id, intent_ids = _CANARIES[interpreter]
    intents = [i for i in _ALL[interpreter] if i.id in intent_ids]
    assert len(intents) == len(intent_ids), f"missing canary intents for {interpreter}"

    output = tmp_path / f"canary-{interpreter}.jsonl"
    cfg = HarnessConfig(
        output_path=output,
        models=[model],
        interpreters=[interpreter],
        variant_ids=[variant_id],
        intents=intents,
        toybox_dir=TOYBOX,
        ollama_host=DEFAULT_HOST,
        temperature=0.2,
        timeout=60,
    )
    run_harness(cfg)
    summary = summarize_jsonl(output)
    assert summary["total_rows"] == len(intents)
    # Loose threshold for v1: at least half of canaries must score 2
    passes = summary["by_model"][model]["pass_rate_2"]
    assert passes >= 0.5, (
        f"{interpreter} canary regressed: pass-rate={passes:.2%} "
        f"for model={model} variant={variant_id}"
    )
```

Note: `_CANARIES` is empty until after Phase 1b. Parametrized tests over an empty list run zero tests — a valid "no canaries registered yet" state.

- [ ] **Step 2: Run the test file to verify it collects cleanly**

Run: `cd /mnt/aux-data/teague/Projects/lackpy/main && python -m pytest tests/eval/test_prompt_canaries.py -v`
Expected: 0 tests run (empty parametrize) or all skipped (if ollama not reachable). Either way, exit 0.

- [ ] **Step 3: Commit**

```bash
git add tests/eval/test_prompt_canaries.py
git commit -m "$(cat <<'EOF'
eval: canary test skeleton for per-interpreter winners

tests/eval/test_prompt_canaries.py defines a slow-marked,
ollama-gated canary test that parametrizes over the current best-known
(model, variant) per interpreter. _CANARIES starts empty and is
populated after phase 1b lands.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 16: End-to-end dry run — shakeout

**Files:** (no new files; exercises the harness end-to-end)

Run the harness against a single small model and 3 intents to confirm the full pipeline works before committing to a full Phase 1a run.

- [ ] **Step 1: Run a minimal harness invocation manually**

Run:

```bash
cd /mnt/aux-data/teague/Projects/lackpy/main && python -c "
from pathlib import Path
from scripts.prompt_eval.harness import HarnessConfig, run_harness
from scripts.prompt_eval.intents_python import PYTHON_INTENTS

cfg = HarnessConfig(
    output_path=Path('results/prompt-eval-dry-run/dry.jsonl'),
    models=['qwen2.5-coder:1.5b'],
    interpreters=['python'],
    variant_ids=['baseline', 'specialized'],
    intents=[i for i in PYTHON_INTENTS if i.id in {'py.core.01', 'py.core.04', 'py.core.07'}],
    toybox_dir=Path('tests/eval/fixtures/toybox'),
    ollama_host='http://localhost:11435',
    temperature=0.2,
    timeout=60,
)
run_harness(cfg)
"
```

Expected: a progress bar reaches 6/6 cells. The JSONL at `results/prompt-eval-dry-run/dry.jsonl` contains a `_meta` row and 6 cell rows.

- [ ] **Step 2: Query the dry-run output**

Run:

```bash
python -m scripts.prompt_eval.query results/prompt-eval-dry-run/dry.jsonl
```

Expected: a summary table showing `qwen2.5-coder:1.5b` with 6 rows, some score total, and a median latency.

- [ ] **Step 3: Generate a dry-run report**

Run:

```bash
python -m scripts.prompt_eval.report results/prompt-eval-dry-run/dry.jsonl --out results/prompt-eval-dry-run/report.md
cat results/prompt-eval-dry-run/report.md | head -40
```

Expected: markdown with `# Prompt Eval Report`, a per-interpreter section for `python`, and top 3 cells listed (2 available, one for each variant).

- [ ] **Step 4: Re-run the harness to verify resume**

Run the same command from Step 1 again.

Expected: `[harness] 6/6 cells already completed; 0 pending.` and exit 0.

- [ ] **Step 5: Clean up the dry-run directory (optional — leaves a reference point)**

Do not commit the dry-run results. They are scratch output useful for future reruns:

```bash
echo "results/prompt-eval-dry-run/" >> .gitignore
git add .gitignore
git commit -m "$(cat <<'EOF'
eval: gitignore dry-run output

Dry-run artifacts are scratch evidence for sanity-checking the harness.
They should not be committed alongside the research results under
results/prompt-eval-YYYY-MM-DD/.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6: Final verification — full test suite**

Run: `cd /mnt/aux-data/teague/Projects/lackpy/main && python -m pytest tests/eval/ -v`
Expected: all tests pass (the canary tests collect to zero).

---

## Post-implementation handoff

After Task 16 completes, the harness is ready to run. The user can then:

1. Run `python -m scripts.prompt_eval.phase1a_qualifier` (~20 min wall clock).
2. Review the qualifier printout; re-run or adjust the floor if the cohort looks wrong.
3. Run `python -m scripts.prompt_eval.phase1b_grid` (~2–4 hours).
4. Run `python -m scripts.prompt_eval.phase2_refinement` (~1–2 hours).
5. Run `python -m scripts.prompt_eval.report` to consolidate all three phases.
6. Populate `_CANARIES` in `tests/eval/test_prompt_canaries.py` with the winners and run it to confirm the canaries pass.
7. File a follow-up plan for wiring the winning prompts into `src/lackpy/infer/prompt.py` (the explicitly-deferred Approach 3 from the spec).

During phases 1b and 2, I can query the live JSONL from a separate agent invocation to give incremental progress updates — the JSONL format is valid after every flushed row.
