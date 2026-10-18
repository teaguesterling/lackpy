"""Task 7: top-level def with the STE function rules."""

import pytest

from lackpy.lang.validator import validate


def _ids(result):
    return {v.rule_id for v in result.violations}


def _body(n):
    return "def f():\n" + "".join(f"    a{i} = {i}\n" for i in range(n))


@pytest.mark.parametrize("program", [
    "def total(xs):\n    return sum(xs)\nprint(total([1]))",
    'def f(a, b=2, c="x", d=None, e=(1, 2)):\n    return a',
    "def f(a=-1, b=-2.5, c=((1, -2), None, True)):\n    return a",
    "def f(a, b):\n    if a:\n        return b\n    return a",
    "def f(a):\n    a.append(1)",  # no return at all
    "def f(a):\n    if a:\n        return\n    return",  # bare returns only
    "def f(xs):\n    return sorted(xs, key=lambda w: w)",
    "def f():\n    return 1\ndef g():\n    return f()\nprint(g())",
    "def f():\n    return 1\ndef g():\n    return 2\nprint(f(), g())",
    _body(12),
    "def f(a, /, b, *, c=1):\n    return a",
])
def test_valid(program):
    result = validate(program)
    assert result.valid, result.violations


@pytest.mark.parametrize("program, rule_id", [
    ("def f():\n    def g():\n        return 1\n    return 1", "style.def-top-level"),
    ("if True:\n    def f():\n        return 1", "style.def-top-level"),
    ("@dec\ndef f():\n    return 1", "style.def-top-level"),
    ("def f():\n    return f()", "style.def-before-use"),
    ("def f():\n    return g()\ndef g():\n    return 1", "style.def-before-use"),
    ("def f():\n    return 1\nx = f()\ndef g():\n    return 1\ny = g()\nz = h()\ndef h():\n    return 1",
     "style.def-before-use"),
    ("def f():\n    return 1\nprint(g())\ndef g():\n    return 1", "style.def-before-use"),
    ("def f(*a):\n    return 1", "style.def-signature"),
    ("def f(**k):\n    return 1", "style.def-signature"),
    ("def f(a=[]):\n    return a", "style.def-signature"),
    ("def f(a=g()):\n    return a", "style.def-signature"),
    ("def f(a=len):\n    return a", "style.def-signature"),
    ("def f(a=(1, [2])):\n    return a", "style.def-signature"),
    ("def f(*, a=[]):\n    return a", "style.def-signature"),
    ("def f(x):\n    if x:\n        return 1\n    return", "style.def-return-mode"),
    ("def f():\n    return lambda: 1", "style.def-returns-function"),
    ("def g():\n    return 1\ndef f():\n    return g", "style.def-returns-function"),
    (_body(13), "style.def-length"),
    ("def f(x):\n    if x:\n        a = 1\n        b = 2\n" + "    c = 3\n" * 10, "style.def-length"),
    ("def sorted(x):\n    return x", "style.def-shadow"),
    ("from json import loads\ndef loads(x):\n    return x", "style.def-shadow"),
    ("import json\ndef json(x):\n    return x", "style.def-shadow"),
    ("def f():\n    yield 1", "safety.node"),
    ("def f():\n    global x", "safety.node"),
    ("async def f():\n    return 1", "safety.node"),
    ("return 1", "safety.node"),
    ("for i in [1]:\n    return i", "safety.node"),
])
def test_invalid(program, rule_id):
    assert rule_id in _ids(validate(program))


def test_length_counts_nested_statements_and_excludes_def():
    assert validate(_body(12)).valid
    assert "style.def-length" in _ids(validate(_body(13)))


def test_return_inside_def_is_not_a_top_level_return():
    assert "safety.node" not in _ids(validate("def f():\n    return 1"))


def test_defined_functions_filled_and_calls_exclude_program_defs():
    result = validate("def total(xs):\n    return sum(xs)\nprint(total([1]))")
    assert result.valid
    assert "total" not in result.calls
    assert {"sum", "print"} <= set(result.calls)


def test_call_to_def_is_not_unknown():
    result = validate("def f():\n    return 1\nx = f()")
    assert "safety.unknown-call" not in _ids(result)


def test_def_name_rebound_hides_no_tool_call():
    # g is a program def; after `g = read_file`, g('x') reaches the tool.
    program = "def g():\n    return 1\ng = read_file\ng('secret')"
    result = validate(program, allowed_names={"read_file"})
    assert not result.valid
    assert "safety.unknown-call" in _ids(result)


def test_tool_call_before_same_named_def_is_rejected():
    program = "x = read_file('a')\ndef read_file(p):\n    return p"
    result = validate(program, allowed_names={"read_file"})
    assert "style.def-before-use" in _ids(result)


def test_import_rebinding_by_def_also_fires_safety_import():
    result = validate("from json import loads\ndef loads(x):\n    return x")
    assert {"safety.import", "style.def-shadow"} <= _ids(result)


def test_def_returns_lambda_fires_both_rules():
    assert {"style.def-returns-function", "style.lambda-position"} <= _ids(
        validate("def f():\n    return lambda: 1"))


# --- fix round 1 ---

def test_duplicate_def_blocks_mutual_recursion():
    program = "def g():\n    return 1\ndef h():\n    return g()\ndef g():\n    return h()\nx = h()"
    result = validate(program)
    assert "style.def-shadow" in _ids(result)
    msg = next(v.message for v in result.violations if v.rule_id == "style.def-shadow")
    assert "line 1" in msg and "line 5" in msg


@pytest.mark.parametrize("program", [
    "def f(x):\n    if x:\n        return 1\n    else:\n        return 0",
    "def f(x):\n    if x:\n        return 1\n    return 0",
    "def f(x):\n    if x:\n        return 1\n    elif x > 1:\n        return 2\n    else:\n        return 3",
    "def f(x):\n    x.append(1)",
])
def test_return_mode_valid_paths(program):
    assert validate(program).valid, validate(program).violations


@pytest.mark.parametrize("program", [
    "def f(x):\n    if x:\n        return 1",
    "def f(x):\n    if x:\n        return 1\n    elif x > 1:\n        return 2",
    "def f(x):\n    for i in x:\n        return i",
    "def f(x):\n    if x:\n        return 1\n    else:\n        pass",
])
def test_return_mode_fall_through_invalid(program):
    assert "style.def-return-mode" in _ids(validate(program))


def test_def_before_use_rationale_is_accurate():
    from lackpy.lang.registry import rule_by_id
    assert "direct call" in rule_by_id("style.def-before-use").rationale
