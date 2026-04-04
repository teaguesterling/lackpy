"""Tests for pattern-specific error hints."""

from lackpy.infer.hints import enrich_errors


def test_open_hint_when_read_available():
    errors = ["Forbidden name: 'open' at line 3"]
    result = enrich_errors(errors, "  read_file(path) -> str: Read file")
    assert any("read_file(path)" in h for h in result)


def test_no_open_hint_when_read_not_available():
    errors = ["Forbidden name: 'open' at line 3"]
    result = enrich_errors(errors, "  find_files(pattern) -> list: Find files")
    assert not any("read_file(path)" in h for h in result)


def test_functiondef_hint():
    errors = ["Forbidden AST node: FunctionDef at line 1"]
    result = enrich_errors(errors, "")
    assert any("Use the tools to find" in h for h in result)


def test_lambda_hint():
    errors = ["Forbidden AST node: Lambda at line 1"]
    result = enrich_errors(errors, "")
    assert any("sort_by" in h for h in result)


def test_import_hint():
    errors = ["Forbidden AST node: Import at line 1"]
    result = enrich_errors(errors, "")
    assert any("already available" in h for h in result)


def test_no_hints_for_unknown_errors():
    errors = ["Unknown function: 'foo' at line 1"]
    result = enrich_errors(errors, "")
    assert result == errors  # no hints added


def test_multiple_hints():
    errors = [
        "Forbidden name: 'open' at line 1",
        "Forbidden AST node: FunctionDef at line 3",
    ]
    result = enrich_errors(errors, "  read_file(path) -> str: Read file")
    hints = [h for h in result if h != "--- Suggestions ---" and h not in errors]
    assert len(hints) >= 2
