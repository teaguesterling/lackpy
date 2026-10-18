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


@pytest.mark.parametrize("program", [
    "import statistics\ns = statistics\nx = s.sys",
    "import json\nx = [json][0].codecs",
    "import json\nd = {'a': json}\nx = d['a'].codecs",
    "import json\nx = sorted([json], key=lambda m: m.codecs)",
])
def test_module_binding_only_as_attribute_base(program):
    # A module used as a value escapes the safety.module-attr check, so only
    # ``binding.member`` is allowed.
    result = validate(program)
    assert "safety.module-attr" in _ids(result), result.violations


def test_member_binding_is_a_value():
    # Members such as Path and Counter stay usable as values.
    result = validate("from pathlib import Path\nmake = [Path]\np = Path('a')")
    assert result.valid, result.errors


@pytest.mark.parametrize("program, allowed", [
    ("x = eval('1')\nfrom json import loads as eval", {"read_file"}),
    ("x = exec('1')\nfrom json import loads as exec", {"read_file"}),
    ("x = compile('1')\nfrom json import loads as compile", {"read_file"}),
    ("x = read_file('secret')\nfrom json import loads as read_file", {"read_file"}),
    ("import json as len", set()),
    ("import json as print", set()),
    ("import json as read_file", {"read_file"}),
    ("import json as __builtins__", set()),
    ("from collections import Counter as getattr", set()),
])
def test_binding_never_shadows_a_builtin_or_tool(program, allowed):
    # A binding named after a builtin or a kit tool would make safety.unknown-call
    # admit the name, and would hide the call from ``calls``.
    result = validate(program, allowed_names=allowed)
    assert not result.valid
    assert "safety.import" in _ids(result), result.violations


def test_open_hint():
    assert validate("open('f')").violations[0].hint == "use Path(p).read_text() or Path(p).write_text(s)"


def test_member_call_above_its_import_is_not_admitted():
    # At run time the call runs before the import binds the name, so it reaches
    # whatever the namespace holds. It must stay unknown, and stay in ``calls``.
    result = validate("x = read_file('secret')\nfrom json import loads as read_file",
                      allowed_names=set())
    assert not result.valid
    assert "safety.unknown-call" in _ids(result), result.violations
    assert "read_file" in result.calls


def test_member_call_after_import_on_same_line():
    # Order is by source position, not by line: a call after ``;`` follows the import.
    result = validate("from collections import Counter; c = Counter([1])")
    assert result.valid, result.errors


@pytest.mark.parametrize("program", [
    # measured: each validated with calls=[] and ran a hidden tool call
    "from collections import Counter\nCounter = read_file\nx = Counter('secret')",
    "from collections import Counter\nCounter, y = read_file, 1\nx = Counter('s')",
    "from collections import Counter\nx = [Counter('s') for Counter in [read_file]]",
    "from collections import Counter\nCounter = eval\nCounter('1')",
    # module binding plus member binding on one name
    "import collections as Counter\nfrom collections import Counter",
    # the same import twice: one name has one binding
    "from collections import Counter\nfrom collections import Counter",
    "import json\nimport json",
    # assignment forms
    "import json\njson += 1",
    "from collections import Counter\nCounter: int = 1",
    "from collections import Counter\na, *Counter = [1, 2]",
    # loop and comprehension targets
    "from collections import Counter\nfor Counter in [1]:\n    pass",
    "from collections import Counter\nx = {Counter for Counter in [1]}",
    "from collections import Counter\nx = sorted(Counter for Counter in [1])",
    # with ... as
    "from collections import Counter\nwith open_thing() as Counter:\n    pass",
    # parameters and def names
    "from collections import Counter\nx = sorted([1], key=lambda Counter: Counter)",
    "from collections import Counter\ndef f(Counter):\n    return 1",
    "from collections import Counter\ndef Counter():\n    return 1",
    # except ... as (Try is allowed since Task 8)
    "from collections import Counter\ntry:\n    x = 1\nexcept ValueError as Counter:\n    pass",
])
def test_imported_name_is_never_rebound(program):
    result = validate(program)
    rebinds = [v for v in result.violations
               if v.rule_id == "safety.import" and "imported at line" in v.message]
    assert rebinds, result.violations
    assert "'Counter'" in rebinds[0].message or "'json'" in rebinds[0].message


def test_rebind_message_names_the_import():
    result = validate("from collections import Counter\nCounter = read_file")
    messages = [v.message for v in result.violations if v.rule_id == "safety.import"]
    assert messages == ["Rebinding of imported name: 'Counter' at line 2 (imported at line 1)"]
