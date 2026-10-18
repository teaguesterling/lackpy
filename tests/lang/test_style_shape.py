"""Task 9: STE shape rules (design section 4.3)."""

import pytest

from lackpy.lang.validator import validate

P = "from pathlib import Path\n"
PJ = "import json\nfrom pathlib import Path\n"


def _ids(result):
    return {v.rule_id for v in result.violations}


@pytest.mark.parametrize("program", [
    P + "p = 'a'\ntext = Path(p).read_text().splitlines()",
    "x = [1]\nprint(len(x))",
    "xs = [1]\nn = sum(1 for x in xs if x)",
    "x = {}\na = x.b.c",
    "s = 'a'\ny = s.strip().lower().split()",
    # the generator is a call argument, positional or keyword
    "xs = [1]\nn = sorted(xs, key=len, reverse=any(x for x in xs))",
    # a call inside a comprehension counts from its own position
    "xs = [[1]]\nprint([len(sorted(x)) for x in xs])",
    "xs = [1]\nprint(sum(x for x in xs))",
    # a method-chain receiver does not add depth
    "s = 'a'\nprint(s.strip().lower())",
    # a comprehension passed as an argument adds no depth: its calls count from their own position
    "x = [1]\nprint(sum(len(sorted(x)) for _ in x))",
    # depth 3 is the maximum
    "x = [1]\nprint(len(sorted(x)))",
    # an I/O call may be an argument
    PJ + "p = 'a'\nd = json.loads(Path(p).read_text())",
    P + "p = 'a'\nn = len(Path(p).read_text())",
    PJ + "p = 'a'\nprint(Path(p).exists())",
    PJ + "p = 'a'\nprint(sorted(Path(p).glob('*')))",
    # an I/O call that is not an argument
    P + "p = 'a'\nif Path(p).exists():\n    print(1)",
    P + "p = 'a'\nok = Path(p).exists()\nprint(ok)",
    # two branches only
    "x = 1\nif x:\n    y = 1\nelif x > 1:\n    y = 2",
    "x = 1\nif x:\n    y = 1\nelse:\n    y = 2",
    "xs = [1]\ny = [a for a in xs if a]",
])
def test_valid(program):
    result = validate(program)
    assert result.valid, result.violations


@pytest.mark.parametrize("program, rule_id", [
    # the I/O call is not the innermost method call of its chain
    (P + "s = 'a'\nt = s.strip().read_text()", "style.io-position"),
    ("xs = [1]\ng = (x for x in xs)", "style.genexp-consumed"),
    ("xs = [1]\nfor y in (x for x in xs):\n    print(y)", "style.genexp-consumed"),
    ("xs = [1]\ndef f():\n    return (x for x in xs)\nprint(f())", "style.genexp-consumed"),
    ("xs = [1]\nt = [(x for x in xs)]", "style.genexp-consumed"),
    ("x = [1]\nprint(len(sorted(list(x))))", "style.call-nesting"),
    ("x = [1]\nprint(len(sorted(list(x))) + 1)", "style.call-nesting"),
    ("x = [1]\nprint(len(sorted(list(x), key=len)))", "style.call-nesting"),
    ("x = [1]\nprint(sum(len(sorted(list(set(x)))) for _ in x))", "style.call-nesting"),
    # an I/O call is never the receiver of an I/O call
    (P + "p = 'a'\nt = Path(p).read_text().read_text()", "style.io-position"),
    ("x = {}\na = x.b.c.d", "style.attr-chain"),
    ("s = 'a'\ny = s.strip().lower().split().pop()", "style.method-chain"),
    ("rows = [[1]]\ny = [[a for a in r] for r in rows]", "style.comprehension-nested"),
    ("rows = [[1]]\ny = [a for r in rows for a in r]", "style.comprehension-multi-for"),
    ("x = 1\nif x == 1:\n    y = 1\nelif x == 2:\n    y = 2\nelse:\n    y = 3", "style.elif-chain"),
    ("x = 1\nif x == 1:\n    y = 1\nelif x == 2:\n    y = 2\nelif x == 3:\n    y = 3", "style.elif-chain"),
])
def test_invalid(program, rule_id):
    result = validate(program)
    assert rule_id in _ids(result), (rule_id, result.violations)


def test_elif_chain_reports_once_on_the_top_if():
    src = ("x = 1\nif x == 1:\n    y = 1\nelif x == 2:\n    y = 2\n"
           "elif x == 3:\n    y = 3\nelif x == 4:\n    y = 4\nelse:\n    y = 5")
    hits = [v for v in validate(src).violations if v.rule_id == "style.elif-chain"]
    assert [v.line for v in hits] == [2]


def test_call_nesting_reports_once_per_outer_call():
    hits = [v for v in validate("x = [1]\nprint(len(sorted(list(x))))").violations
            if v.rule_id == "style.call-nesting"]
    assert len(hits) == 1
