"""Underscore names, the dunder allowlist and the type(x).__name__ form (Task 5).

The underscore guard is the main defence against the ``__class__`` ->
``__subclasses__`` escape chain. These tests pin the few dunder forms that
are allowed, and that every other dunder name and every other use of
``type`` stays rejected.
"""

import pytest

from lackpy.lang.validator import validate


def _ids(result):
    return {v.rule_id for v in result.violations}


@pytest.mark.parametrize("program", [
    'if __name__ == "__main__":\n    print(1)',
    "p = __file__",
    "import json\nv = json.__version__",
    "n = type(3).__name__",
    "pairs = [(1, 2)]\nfor _, v in pairs:\n    print(v)",
    "_tmp = 1\nprint(_tmp)",
])
def test_allowed_forms_validate(program):
    result = validate(program)
    assert result.valid, result.errors


def test_single_underscore_loop_target_has_no_safety_violation():
    # The brief's program, verbatim. style.for-iter rejects a list literal
    # until Task 6 changes that rule, so only the safety tier is asserted here.
    result = validate("for _, v in [(1, 2)]:\n    print(v)")
    assert not [v for v in result.violations if v.rule_id.startswith("safety.")], result.violations


@pytest.mark.parametrize("program, rule_id", [
    ('__name__ = "x"', "safety.dunder-name"),
    ('__import__("os")', "safety.dunder-name"),
    ("x = __builtins__", "safety.dunder-name"),
    ("t = type(3)", "safety.type-form"),
    ("type(3)()", "safety.type-form"),
    ("x = (3).__class__.__name__", "safety.attr-private"),
    ('x = "a".__version__', "safety.attr-private"),
    ("t = type", "safety.type-form"),
])
def test_brief_rejections(program, rule_id):
    result = validate(program)
    assert not result.valid
    assert rule_id in _ids(result), result.violations


@pytest.mark.parametrize("program", [
    '__file__ = "x"',
    "for __x in [1]:\n    print(1)",
    "sorted([1], key=lambda __x: 1)",
    "import json as __j",
    "x = __spec__",
    "del __name__",
    "__name__ += 'x'",
])
def test_dunder_names_other_than_loads_of_the_allowlist_rejected(program):
    # __name__ and __file__ are allowed only as a load. Every other __x name is
    # rejected wherever it binds: a store, a parameter, an import alias.
    result = validate(program)
    assert "safety.dunder-name" in _ids(result), result.violations


@pytest.mark.parametrize("program", [
    'type(3).__name__ = "y"',      # the form must be a load
    "n = type(3, 4).__name__",     # exactly one positional argument
    "n = type().__name__",
    "a = [3]\nn = type(*a).__name__",
    "n = type(x=3).__name__",
    "d = {}\nn = type(**d).__name__",
    "n = type(3).__qualname__",
    "type = len",
    "n = sorted([1], key=lambda type: 1)",
    "for type in [1]:\n    print(1)",
])
def test_type_outside_the_name_form_rejected(program):
    result = validate(program)
    assert not result.valid
    assert "safety.type-form" in _ids(result), result.violations


def test_type_is_not_a_forbidden_name_any_more():
    # safety.type-form owns every use of the name `type`.
    result = validate("t = type(3)")
    assert "safety.forbidden-name" not in _ids(result), result.violations
    assert "safety.unknown-call" not in _ids(result), result.violations


def test_module_version_passes_the_module_attribute_check():
    # __version__ is not in MODULES["json"], but a module binding may read it.
    result = validate("import json\nv = json.__version__")
    assert "safety.module-attr" not in _ids(result), result.violations
    assert "safety.attr-private" not in _ids(result), result.violations


@pytest.mark.parametrize("program", [
    "import json\njson.__version__ = 'x'",
    "from json import loads\nv = loads.__version__",   # a member binding is not a module
    "import json\nv = json.loads.__version__",
    "x = __name__.__class__",
    "n = type(3).__name__.__class__",
])
def test_version_and_name_only_in_their_exact_form(program):
    result = validate(program)
    assert not result.valid
    assert _ids(result) & {"safety.attr-private", "safety.module-attr"}, result.violations
