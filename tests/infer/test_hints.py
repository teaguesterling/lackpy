"""Tests for pattern-specific error hints.

stepy100 Task 8: hints name the approved Path forms from
``dictionary.UNAPPROVED``. The kit-era tests pinned "Use read_file(path)",
"Use the tools to find" and "already available in the namespace"; the kit
that those hints named is gone (Plan 2, Task 7).
"""

from lackpy.infer.hints import enrich_errors
from lackpy.lang.dictionary import MODULES, UNAPPROVED
from lackpy.lang.validator import validate

KIT_WORDS = ["read_file", "find_files", "find_definitions", "edit(", "namespace", "tools"]


def _hints(result, errors):
    return [h for h in result if h != "--- Suggestions ---" and h not in errors]


def test_open_hint_is_the_dictionary_entry():
    errors = ["Forbidden name: 'open' at line 3"]
    result = enrich_errors(errors, "")
    assert any(UNAPPROVED["open"] in h for h in result)


def test_open_hint_does_not_depend_on_the_namespace():
    errors = ["Forbidden name: 'open' at line 3"]
    assert enrich_errors(errors, "") == enrich_errors(errors, "  read_file(path) -> str")


def test_unapproved_member_hint_from_real_validator_text():
    errors = validate("import json\nx = json.load(1)").errors
    result = enrich_errors(errors, "")
    assert any(UNAPPROVED["json.load"] in h for h in result)


def test_every_unapproved_word_has_its_hint():
    for word, hint in UNAPPROVED.items():
        result = enrich_errors([f"Forbidden name: '{word}' at line 1"], "")
        assert any(hint in h for h in result), word


def test_unknown_path_hint_names_the_import():
    errors = validate('print(Path("a").read_text())').errors
    result = enrich_errors(errors, "")
    assert any("from pathlib import Path" in h for h in _hints(result, errors))


def test_lambda_hint():
    errors = ["Forbidden AST node: Lambda at line 1"]
    result = enrich_errors(errors, "")
    assert not any("sort_by" in h for h in result)
    assert any("key=" in h and "sorted" in h and "min" in h and "max" in h for h in result)


def test_import_hint_lists_the_approved_modules():
    errors = ["Forbidden AST node: Import at line 1"]
    hints = _hints(enrich_errors(errors, ""), errors)
    assert hints
    for module in MODULES:
        assert any(module in h for h in hints), module
    assert not any("already available" in h for h in hints)


def test_no_hint_names_the_kit():
    errors = [
        "Forbidden name: 'open' at line 1",
        "Forbidden AST node: FunctionDef at line 3",
        "Forbidden AST node: ClassDef at line 4",
        "Forbidden AST node: Import at line 1",
        "Bare except at line 3 (name the error class, for example KeyError)",
    ]
    for h in _hints(enrich_errors(errors, ""), errors):
        for word in KIT_WORDS:
            assert word not in h, (word, h)


def test_no_hints_for_unknown_errors():
    errors = ["Unknown function: 'foo' at line 1"]
    result = enrich_errors(errors, "")
    assert result == errors  # no hints added


def test_multiple_hints():
    errors = [
        "Forbidden name: 'open' at line 1",
        "Forbidden AST node: Import at line 3",
    ]
    result = enrich_errors(errors, "")
    assert len(_hints(result, errors)) >= 2
