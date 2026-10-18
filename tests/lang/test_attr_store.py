"""safety.attr-store: a program never assigns to or deletes an attribute.

An attribute store on a class or object that the host passes in changes the host.
The rule is a safety rule so that no ``# ste:`` waiver lifts it.
"""

import pytest

from lackpy.lang.validator import validate


def _ids(result):
    return {v.rule_id for v in result.violations}


@pytest.mark.parametrize("program", [
    "from collections import Counter\nC = Counter\nC.most_common = len",
    "from pathlib import Path\n[Path][0].read_text = print",
    "x = []\nx.y = 1",
    "x = []\nx.y += 1",
    "x = []\ndel x.y",
])
def test_attribute_store_is_invalid(program):
    result = validate(program)
    assert "safety.attr-store" in _ids(result), result.violations


def test_message_and_hint():
    (v,) = [v for v in validate("x = []\nx.y = 1").violations if v.rule_id == "safety.attr-store"]
    assert v.message == "Assignment to an attribute is not allowed: 'x.y' at line 2"
    assert v.hint == "build a new value and assign it to a name"


def test_attribute_read_and_name_store_stay_valid():
    assert validate("x = [3]\nx = x + [1]\nn = x.count(1)\nx.append(2)").valid
