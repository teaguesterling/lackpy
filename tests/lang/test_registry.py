"""Tests for the rule registry: every validator check is a rule with an id and a tier."""

import pytest

from lackpy.lang.registry import RULES, rule_by_id
from lackpy.lang.validator import validate


def test_every_rule_id_is_unique_and_tiered():
    ids = [r.id for r in RULES]
    assert len(ids) == len(set(ids))
    assert all(r.tier in ("safety", "style") for r in RULES)


@pytest.mark.parametrize("rule", RULES, ids=lambda r: r.id)
def test_rule_examples_behave_as_documented(rule):
    bad = validate(rule.bad)
    assert any(v.rule_id == rule.id for v in bad.violations), bad.violations
    good = validate(rule.good)
    assert good.valid, good.errors


def test_errors_mirror_violations():
    result = validate("import os\nopen('x')")
    assert result.errors == [v.message for v in result.violations]
    assert "Forbidden name: 'open' at line 2" in result.errors


def test_ported_rules_have_ids():
    for rule_id in (
        "safety.parse",
        "safety.node",
        "safety.forbidden-name",
        "safety.attr-private",
        "safety.attr-denied",
        "safety.call-target",
        "safety.unknown-call",
    ):
        assert rule_by_id(rule_id).id == rule_id


def test_handler_class_checks_are_safety_tier():
    # Task 8 fix round 2: style.try-named was folded into safety.except-base.
    assert rule_by_id("safety.except-base").tier == "safety"
    with pytest.raises(KeyError):
        rule_by_id("style.try-named")
