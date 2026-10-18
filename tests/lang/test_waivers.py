"""Task 10: ``# ste:`` waivers. Style rules are waivable. Safety rules never are."""

import pytest

from lackpy.lang.registry import RULES, Waiver
from lackpy.lang.validator import validate
from lackpy.lang.waivers import parse_waivers

GEN = "xs = [1]\ng = (x for x in xs)"
WAIVER = "# ste: style.genexp-consumed, read twice below"


def _ids(result):
    return {v.rule_id for v in result.violations}


def test_trailing_waiver_removes_style_violation():
    r = validate(f"xs = [1]\ng = (x for x in xs)  {WAIVER}")
    assert r.valid and r.violations == [] and r.errors == []
    assert len(r.waivers) == 1
    w = r.waivers[0]
    assert (w.rule_id, w.line, w.why) == ("style.genexp-consumed", 2, "read twice below")
    assert w.used is True


def test_standalone_waiver_applies_to_next_line():
    r = validate(f"xs = [1]\n{WAIVER}\ng = (x for x in xs)")
    assert r.valid and len(r.waivers) == 1 and r.waivers[0].used is True


def test_trailing_waiver_does_not_apply_to_next_line():
    r = validate(f"xs = [1]\nh = 1  {WAIVER}\ng = (x for x in xs)")
    assert "style.genexp-consumed" in _ids(r)
    assert r.waivers[0].used is False


def test_waiver_does_not_leak_to_unrelated_lines():
    r = validate(f"xs = [1]\ng = (x for x in xs)  {WAIVER}\nk = (x for x in xs)")
    assert [v.line for v in r.violations] == [3]
    assert not r.valid


def test_waiver_for_a_different_rule_does_not_hide_the_violation():
    r = validate("xs = [1]\ng = (x for x in xs)  # ste: style.attr-chain, wrong rule")
    assert "style.genexp-consumed" in _ids(r)
    assert r.waivers[0].used is False


def test_unused_waiver_is_recorded_without_error():
    r = validate(f"x = 1  {WAIVER}")
    assert r.valid and r.errors == []
    assert len(r.waivers) == 1 and r.waivers[0].used is False


def test_multiline_statement_first_line_waiver_covers():
    r = validate(f"xs = [1]\ng = (x  {WAIVER}\n     for x in xs)")
    assert r.valid and r.waivers[0].used is True


def test_multiline_statement_later_line_waiver_does_not_cover():
    r = validate(f"xs = [1]\ng = (x\n     for x in xs)  {WAIVER}")
    assert "style.genexp-consumed" in _ids(r)
    assert r.waivers[0].used is False


def test_safety_violation_stays_and_waiver_is_rejected():
    r = validate("open('f')  # ste: safety.forbidden-name, need it")
    assert not r.valid
    assert {"safety.forbidden-name", "meta.waiver-safety"} <= _ids(r)
    assert r.waivers == []


SAFETY_IDS = [r.id for r in RULES if r.tier == "safety"]


def test_safety_ids_exist():
    assert SAFETY_IDS


@pytest.mark.parametrize("rule_id", SAFETY_IDS)
def test_no_safety_rule_is_waivable(rule_id):
    r = validate(f"x = 1  # ste: {rule_id}, trying anyway")
    assert not r.valid
    assert "meta.waiver-safety" in _ids(r)
    assert r.waivers == []


@pytest.mark.parametrize("rule_id", SAFETY_IDS)
def test_safety_violation_survives_a_waiver(rule_id):
    rule = next(x for x in RULES if x.id == rule_id)
    if rule_id == "safety.parse":
        pytest.skip("a program that does not parse has no comments to read")
    bad = rule.bad.rstrip("\n").split("\n")
    waived = "\n".join(f"{line}  # ste: {rule_id}, trying anyway" for line in bad)
    r = validate(waived, allowed_names=set())
    assert rule_id in _ids(r)
    assert "meta.waiver-safety" in _ids(r)


@pytest.mark.parametrize("comment", [
    "# ste: style.nope, x",
    "# ste: style.try-named, removed rule id",
    "# ste: style.genexp-consumed",
    "# ste: style.genexp-consumed,",
    "# ste: style.genexp-consumed,   ",
    "# ste:",
    "# ste: , why",
])
def test_malformed_or_unknown_waiver(comment):
    r = validate(f"x = 1  {comment}")
    assert not r.valid
    assert _ids(r) == {"meta.waiver-unknown"}
    assert r.waivers == []


def test_string_literal_is_not_a_waiver():
    src = "xs = [1]\ns = '# ste: style.genexp-consumed, nope'\ng = (x for x in xs)"
    r = validate(src)
    assert "style.genexp-consumed" in _ids(r)
    assert r.waivers == []
    assert parse_waivers(src) == []


def test_ordinary_comments_are_ignored():
    assert parse_waivers("x = 1  # steady as she goes\n# not ste: here\n") == []


def test_parse_waivers_returns_three_field_waivers():
    ws = parse_waivers(f"x = 1  {WAIVER}")
    assert ws == [Waiver("style.genexp-consumed", 1, "read twice below")]


def test_why_may_contain_commas():
    (w,) = parse_waivers("x = 1  # ste: style.genexp-consumed, a, b")
    assert w.why == "a, b"
