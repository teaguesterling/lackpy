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
    assert text.count("@deprecated") == 2


def test_route_count():
    text = (TOYBOX / "app.py").read_text()
    # 4 @route handlers in app.py
    assert text.count("@route(") == 4


def test_sql_concat_count():
    # 3 SQL-building-via-string-concat sites
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
