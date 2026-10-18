"""Task 6: assert with a message, continue/break/pass, set operators, for-iter, lambda."""

import pytest

from lackpy.lang.validator import validate


def _ids(result):
    return {v.rule_id for v in result.violations}


@pytest.mark.parametrize("program", [
    'x = 1\nassert x == 1, "x is one"',
    'x = 1\nassert x, f"bad {x}"',
    "for k in [1, 2]:\n    print(k)",
    'd = {"items": [1]}\nfor k in d["items"]:\n    print(k)',
    "for i in range(3):\n    if i:\n        continue\n    break",
    "for i in range(3):\n    pass",
    "s = {1} & {2} | {3} ^ {4}",
    "xs = [(1, 2)]\nm = max(xs, key=lambda r: r[1])",
    "xs = [(1, 2)]\nm = sorted(xs, key=lambda r: r[1])",
])
def test_valid(program):
    result = validate(program)
    assert result.valid, result.violations


@pytest.mark.parametrize("program, rule_id", [
    ("x = 1\nassert x", "style.assert-message"),
    ('x = 1\nmsg = "m"\nassert x, msg', "style.assert-message"),
    ('x = 1\nassert x, "a" + "b"', "style.assert-message"),
    ("x = [3, 1]\nfor k in sorted(x)[:3]:\n    pass", "style.for-iter"),
    ("f = lambda x: x", "style.lambda-assign"),
    ("xs = [1]\nm = min(xs, default=lambda r: r)", "style.lambda-position"),
])
def test_invalid(program, rule_id):
    assert rule_id in _ids(validate(program))


def test_map_lambda_names_the_comprehension():
    result = validate("xs = [1]\ny = list(map(lambda x: x, xs))")
    hits = [v for v in result.violations if v.rule_id == "safety.forbidden-name"]
    assert hits and hits[0].hint == "use a comprehension"


@pytest.mark.parametrize("program", [
    'import re as regex\ns = regex.sub("a", lambda m: "b", "abc")',
    'from re import sub\ns = sub("a", lambda m: "b", "abc")',
    'import re\ns = re.subn("a", lambda m: "b", "abc")',
    'import re\np = re.compile("a")\ns = p.subn(lambda m: "b", "abc")',
    "import collections as c\nd = c.defaultdict(lambda: 0)",
    "from collections import defaultdict as dd\nd = dd(lambda: 0)",
])
def test_lambda_position_resolves_through_imports(program):
    result = validate(program)
    assert result.valid, result.violations


@pytest.mark.parametrize("program", [
    "import collections\nd = collections.defaultdict(int, lambda: 0)",
    'import re\ns = re.sub(lambda m: "b", "a", "abc")',
    'import json as re\ns = re.sub("a", lambda m: "b", "abc")',
])
def test_lambda_position_rejects_wrong_slot(program):
    assert "style.lambda-position" in _ids(validate(program))


def test_lambda_position_message_names_all_three_positions():
    v = [v for v in validate("list(lambda: 1)").violations if v.rule_id == "style.lambda-position"][0]
    assert "key=" in v.message and "defaultdict" in v.message and "re.sub" in v.message
