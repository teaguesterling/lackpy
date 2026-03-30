"""Tests for grade computation."""

from lackpy.lang.grader import Grade, compute_grade


def test_grade_from_read_only_tools():
    tools = {"read": {"grade_w": 1, "effects_ceiling": 1}, "glob": {"grade_w": 1, "effects_ceiling": 1}}
    grade = compute_grade(tools)
    assert grade == Grade(w=1, d=1)


def test_grade_from_mixed_tools():
    tools = {"read": {"grade_w": 1, "effects_ceiling": 1}, "edit": {"grade_w": 3, "effects_ceiling": 3}}
    grade = compute_grade(tools)
    assert grade == Grade(w=3, d=3)


def test_grade_from_empty_tools():
    grade = compute_grade({})
    assert grade == Grade(w=0, d=0)


def test_grade_defaults_for_missing_fields():
    tools = {"custom": {}}
    grade = compute_grade(tools)
    assert grade == Grade(w=3, d=3)


def test_grade_str():
    g = Grade(w=1, d=1)
    assert "1" in str(g)
