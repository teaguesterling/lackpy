"""Imports, the dictionary of approved modules, and module attributes (Task 4)."""

import pytest

from lackpy.lang.validator import validate


def _ids(result):
    return [v.rule_id for v in result.violations]


@pytest.mark.parametrize("program", [
    "import json",
    "import re as regex",
    "from pathlib import Path",
    "from collections import Counter as C",
    "import statistics",
])
def test_approved_imports_validate(program):
    result = validate(program)
    assert result.valid, result.errors


def test_unapproved_imports_rejected_with_hint():
    result = validate("import os")
    assert not result.valid
    imports = [v for v in result.violations if v.rule_id == "safety.import"]
    assert imports, result.violations
    assert imports[0].hint == "use pathlib.Path"

    for program in ("import subprocess", "import os.path", "from json import *", "from . import x"):
        result = validate(program)
        assert not result.valid, program
        assert "safety.import" in _ids(result), (program, result.violations)


def test_import_outside_top_level_rejected():
    # Bindings come from top-level imports only, so a nested import is rejected.
    result = validate("for i in [1]:\n    import json")
    assert "safety.import" in _ids(result), result.violations


def test_module_attribute_checked():
    assert "safety.module-attr" in _ids(validate("import statistics\nx = statistics.sys"))
    assert "safety.module-attr" in _ids(validate("import json\njson.codecs"))
    result = validate("import json\njson.loads('1')")
    assert result.valid, result.errors


def test_module_attribute_store_rejected():
    # An assignment to a module attribute would change the host's module.
    assert "safety.module-attr" in _ids(validate("import json\njson.loads = len"))
    # Counter is a Python class: a store through the chain changes it for the host.
    assert "safety.module-attr" in _ids(validate("import collections\ncollections.Counter.most_common = len"))
    assert "safety.module-attr" in _ids(validate("from pathlib import Path\nPath.read_text = len"))


def test_from_import_binding_is_callable():
    result = validate("from pathlib import Path\np = Path('a')")
    assert result.valid, result.errors
    assert "Path" not in result.calls


@pytest.mark.xfail(strict=True, reason="runtime proxy, Plan 2")
def test_alias_gap_is_runtime_work():
    result = validate("import statistics\ns = statistics\nx = s.sys")
    assert not result.valid


def test_open_hint():
    assert validate("open('f')").violations[0].hint == "use Path(p).read_text() or Path(p).write_text(s)"
