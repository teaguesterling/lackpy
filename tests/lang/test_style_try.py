"""Task 8: try with named errors and approved handler shapes (design section 4.3)."""

import sys

import pytest

from lackpy.lang.grammar import DENIED_ATTRIBUTES
from lackpy.lang.validator import validate


def _ids(result):
    return {v.rule_id for v in result.violations}


# One program per handler shape in the design table, then the accepted name forms.
@pytest.mark.parametrize("program", [
    # skip: continue, inside a for body
    "d = {}\nfor k in ['a']:\n    try:\n        x = d[k]\n    except KeyError:\n        continue\n    print(x)",
    # skip, after a report
    "d = {}\nfor k in ['a']:\n    try:\n        x = d[k]\n    except KeyError:\n        print(f'no {k}')\n"
    "        continue\n    print(x)",
    # return: return <value>, inside a def
    "def f(s):\n    try:\n        n = int(s)\n    except ValueError:\n        return 0\n    return n",
    # return, with a try as the last statement of the def (def-return-mode sees both paths)
    "def f(s):\n    try:\n        return int(s)\n    except ValueError:\n        return 0",
    # ignore: pass, when the body assigns no name
    "d = {}\ntry:\n    print(d['k'])\nexcept KeyError:\n    pass",
    # ignore: an augmented assignment needs an earlier binding, so it is not a new name
    "n = 0\ntry:\n    n += int('3')\nexcept ValueError:\n    pass",
    # default: assign every name the body assigns
    "d = {}\ntry:\n    x = d['k']\nexcept KeyError:\n    x = 0\nprint(x)",
    "d = {}\ntry:\n    x = d['k']\n    y = d['j']\nexcept KeyError as e:\n    print(e, e.args, str(e))\n"
    "    x = 0\n    y = 1\nprint(x, y)",
    "d = {}\ntry:\n    x, y = d['k']\nexcept (KeyError, ValueError):\n    x, y = 0, 1\nprint(x, y)",
    # named through the dictionary modules
    "import json\ntry:\n    v = json.loads('[')\nexcept json.JSONDecodeError:\n    v = None",
    "from json import loads, JSONDecodeError\ntry:\n    v = loads('[')\nexcept JSONDecodeError:\n    v = None",
    "import re\ntry:\n    p = re.compile('(')\nexcept re.error:\n    p = None",
    "import re as regex\ntry:\n    p = regex.compile('(')\nexcept regex.error:\n    p = None",
    "from pathlib import Path\ntry:\n    t = Path('x').read_text()\nexcept (FileNotFoundError, UnicodeDecodeError):\n    t = ''",
    # report, then fall through (amendment ruling: design 4.3 definite assignment)
    'd = {}\nk = 1\ntry:\n    d.pop(k)\nexcept KeyError:\n    print("missing")',
    'd = {}\nk = 1\ntry:\n    x = d[k]\nexcept KeyError:\n    print("missing")\n    x = 0',
    "try:\n    print(1)\nexcept KeyError:\n    print(2)",
    "try:\n    x = 1\nexcept KeyError:\n    x = 0\n    print(x)",
    # assignments may also come before continue or return
    "for i in [1]:\n    try:\n        x = 1\n    except KeyError:\n        y = 2\n        continue",
    # NotImplementedError is RecursionError's sibling, not its base
    "try:\n    x = 1\nexcept NotImplementedError:\n    x = 0",
    # two handlers may reuse one as-name: each binds and deletes its own value
    "d = {}\ntry:\n    x = d['a']\nexcept KeyError as e:\n    print(e)\n    x = 0\n"
    "try:\n    y = d['b']\nexcept KeyError as e:\n    print(e)\n    y = 0",
])
def test_valid(program):
    result = validate(program)
    assert result.valid, result.violations


@pytest.mark.parametrize("program, rule_id", [
    # from the brief
    ("d = {}\nk = 1\ntry:\n    x = d[k]\nexcept Exception:\n    x = 0", "safety.except-base"),
    ("d = {}\nk = 1\ntry:\n    x = d[k]\nexcept BaseException:\n    x = 0", "safety.except-base"),
    ("d = {}\nk = 1\ntry:\n    x = d[k]\nexcept BaseException:  # ste: safety.except-base, test\n    x = 0",
     "safety.except-base"),
    ("d = {}\nk = 1\ntry:\n    x = d[k]\nexcept KeyError:\n    pass\nprint(x)", "style.try-handler"),
    ("try:\n    a = 1\n    b = 2\n    c = 3\n    e = 4\nexcept KeyError:\n    a = b = c = e = 0",
     "style.try-body"),
    ("try:\n    x = 1\nexcept KeyError:\n    x = 0\nelse:\n    x = 2", "style.try-else-finally"),
    ("try:\n    x = 1\nexcept KeyError:\n    x = 0\nfinally:\n    print(1)", "style.try-else-finally"),
    ("try:\n    x = 1\nexcept KeyError as e:\n    t = e.__traceback__", "safety.attr-private"),
    ("try:\n    x = 1\nexcept KeyError as e:\n    t = e.with_traceback(None).tb_frame", "safety.attr-denied"),
    # controller ruling: AttributeError.obj can hold any object
    ("d = {}\ntry:\n    x = d.k\nexcept AttributeError as e:\n    o = e.obj", "safety.attr-denied"),
    # except-base: bare, in a tuple, and the other non-Exception classes
    ("try:\n    x = 1\nexcept:\n    x = 0", "safety.except-base"),
    ("try:\n    x = 1\nexcept (KeyError, BaseException):\n    x = 0", "safety.except-base"),
    ("try:\n    x = 1\nexcept SystemExit:\n    x = 0", "safety.except-base"),
    ("try:\n    x = 1\nexcept KeyboardInterrupt:\n    x = 0", "safety.except-base"),
    ("try:\n    x = 1\nexcept GeneratorExit:\n    x = 0", "safety.except-base"),
    # fix round 2: every handler-class check is safety.except-base (style.try-named removed)
    ("try:\n    x = 1\nexcept (KeyError, Exception):\n    x = 0", "safety.except-base"),
    ("E = KeyError\ntry:\n    x = 1\nexcept E:\n    x = 0", "safety.except-base"),
    ("try:\n    x = 1\nexcept len([]):\n    x = 0", "safety.except-base"),  # ruling 4: a call
    ("import json\ntry:\n    x = 1\nexcept json.loads:\n    x = 0", "safety.except-base"),
    ("from collections import Counter\ntry:\n    x = 1\nexcept Counter:\n    x = 0", "safety.except-base"),
    ("try:\n    x = 1\nexcept print:\n    x = 0", "safety.except-base"),
    # try-handler: the loop and def positions (controller ruling)
    ("try:\n    x = 1\nexcept KeyError:\n    continue", "style.try-handler"),
    ("for i in [1]:\n    print(i)\ntry:\n    x = 1\nexcept KeyError:\n    continue", "style.try-handler"),
    ("for i in [1]:\n    try:\n        x = 1\n    except KeyError:\n        return 0", "style.try-handler"),
    ("def f():\n    try:\n        x = 1\n    except KeyError:\n        continue\n    return x",
     "style.try-handler"),
    # try-handler: the shapes themselves
    ("def f():\n    try:\n        x = 1\n    except KeyError:\n        return\n    print(x)", "style.try-handler"),
    ("for i in [1]:\n    try:\n        x = 1\n    except KeyError:\n        break", "style.try-handler"),
    ('d = {}\nk = 1\ntry:\n    x = d[k]\nexcept KeyError:\n    print("missing")\nprint(x)',
     "style.try-handler"),
    ("d = {}\nfor i in [1]:\n    try:\n        x = 1\n    except KeyError:\n        d.clear()\n        continue",
     "style.try-handler"),
    ("for i in [1]:\n    try:\n        x = 1\n    except KeyError:\n        pass", "style.try-handler"),
    ("try:\n    x = 1\n    y = 2\nexcept KeyError:\n    x = 0", "style.try-handler"),
    ("try:\n    for i in [1]:\n        x = i\nexcept KeyError:\n    pass", "style.try-handler"),
    # fix round 1, ruling 1: limit signals are never caught
    ("try:\n    x = 1\nexcept RecursionError:\n    x = 0", "safety.except-base"),
    ("try:\n    x = 1\nexcept RuntimeError:\n    x = 0", "safety.except-base"),
    ("try:\n    x = 1\nexcept MemoryError:\n    x = 0", "safety.except-base"),
    ("try:\n    x = 1\nexcept (KeyError, RecursionError):\n    x = 0", "safety.except-base"),
    ("def k(x):\n    try:\n        y = sorted([x], key=k)\n    except RecursionError:\n"
     "        y = sorted([x], key=k)\n    return 0\nprint(sorted([1], key=k))", "safety.except-base"),
    # fix round 1, ruling 2: an exception class name bound anywhere in the program
    ("for KeyError in [Exception]:\n    try:\n        x = 1\n    except KeyError:\n        x = 0",
     "safety.except-base"),
    ("def f(KeyError):\n    try:\n        x = 1\n    except KeyError:\n        x = 0\n    return x\n"
     "print(f(Exception))", "safety.except-base"),
    ("KeyError = 3\ntry:\n    x = 1\nexcept KeyError:\n    x = 0", "safety.except-base"),
    ("try:\n    x = 1\nexcept KeyError as ValueError:\n    x = 0\n"
     "try:\n    y = 1\nexcept ValueError:\n    y = 0", "safety.except-base"),
    # fix round 1, ruling 3: the as-name is deleted when the handler ends
    ("d = {}\ntry:\n    x = d['a']\nexcept KeyError as x:\n    x = 0\nprint(x)", "style.try-handler"),
    ("d = {}\ntry:\n    e = d['a']\nexcept KeyError as e:\n    e = 0\nprint(e)", "style.try-handler"),
    # fix round 1, ruling 4: the class expression is a name, a dotted name, or a tuple of them
    ("try:\n    x = 1\nexcept read_file('x'):\n    x = 0", "safety.except-base"),
    ("def f():\n    return KeyError\ntry:\n    x = 1\nexcept (KeyError, f()):\n    x = 0",
     "safety.except-base"),
    # fix round 2, ruling 1: routes to a limit signal inside the key= recursion
    ("def k(x):\n    try:\n        y = sorted([x], key=k)\n    except Exception:\n"
     "        y = sorted([x], key=k)\n    return 0\nprint(sorted([1], key=k))", "safety.except-base"),
    ("E = RecursionError\ndef k(x):\n    try:\n        y = sorted([x], key=k)\n    except E:\n"
     "        y = sorted([x], key=k)\n    return 0\nprint(sorted([1], key=k))", "safety.except-base"),
    ("errs = (KeyError, RecursionError)\ndef k(x):\n    try:\n        y = sorted([x], key=k)\n"
     "    except errs:\n        y = sorted([x], key=k)\n    return 0\nprint(sorted([1], key=k))",
     "safety.except-base"),
    # fix round 2, ruling 2: the as-name must not be bound elsewhere (the handler deletes it)
    ("d = {}\nx = 1\ntry:\n    y = d['a']\nexcept KeyError as x:\n    y = 0\nprint(x)", "style.try-handler"),
    # except-as names are bindings (Tasks 4 and 5 rules)
    ("try:\n    x = 1\nexcept KeyError as __x:\n    x = 0", "safety.dunder-name"),
])
def test_invalid(program, rule_id):
    result = validate(program)
    assert not result.valid
    assert rule_id in _ids(result), result.violations


def test_skip_shape_needs_the_for_body_not_only_a_later_position():
    inside = "for i in [1]:\n    try:\n        x = 1\n    except KeyError:\n        continue"
    assert validate(inside).valid
    after = "for i in [1]:\n    print(i)\ntry:\n    x = 1\nexcept KeyError:\n    continue"
    assert "style.try-handler" in _ids(validate(after))


def test_obj_is_denied_and_string_attributes_stay_allowed():
    assert "obj" in DENIED_ATTRIBUTES
    for attr in ("name", "path", "args"):
        assert attr not in DENIED_ATTRIBUTES


def test_except_as_name_cannot_rebind_an_import():
    # Task 4's rebind rule covers except-as; the program now parses past safety.node too.
    result = validate("from collections import Counter\ntry:\n    x = 1\n"
                      "except ValueError as Counter:\n    x = 0")
    assert any(v.rule_id == "safety.import" and "Rebinding" in v.message
               for v in result.violations), result.violations


def test_try_and_handler_are_allowed_nodes():
    result = validate("try:\n    x = 1\nexcept KeyError:\n    x = 0")
    assert result.valid, result.violations


@pytest.mark.skipif(sys.version_info < (3, 11), reason="except* parses from Python 3.11")
def test_try_star_stays_forbidden():
    result = validate("try:\n    x = 1\nexcept* KeyError:\n    x = 0")
    assert "safety.node" in _ids(result)


def test_try_named_is_not_a_rule():
    # Fix round 2: every handler-class check is safety-tier, so no waiver can reopen a limit signal.
    from lackpy.lang.registry import RULES
    ids = {r.id for r in RULES}
    assert "style.try-named" not in ids
    assert "safety.except-base" in ids


@pytest.mark.parametrize("program", [
    "try:\n    x = 1\nexcept Exception:\n    x = 0",
    "E = KeyError\ntry:\n    x = 1\nexcept E:\n    x = 0",
    "import json\ntry:\n    x = 1\nexcept json.loads:\n    x = 0",
    "try:\n    x = 1\nexcept RecursionError:\n    x = 0",
])
def test_except_base_messages_name_the_alternative(program):
    messages = [v.message for v in validate(program).violations if v.rule_id == "safety.except-base"]
    assert messages and all("KeyError" in m for m in messages), messages
