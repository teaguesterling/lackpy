"""The generated style guide must not drift from the registry."""

from pathlib import Path

from lackpy.lang.registry import RULES
from lackpy.lang.spec import format_style_guide, get_spec

GUIDE = Path(__file__).resolve().parents[2] / "docs" / "reference" / "style-guide.md"


def test_guide_file_matches_generator():
    assert GUIDE.read_text() == format_style_guide()


def test_guide_lists_every_rule():
    headings = format_style_guide().splitlines()
    for rule in RULES:
        assert f"### {rule.id}" in headings


def test_spec_exposes_rules():
    rules = get_spec()["rules"]
    assert [r["id"] for r in rules] == [r.id for r in RULES]
    assert all(set(r) == {"id", "tier", "summary"} for r in rules)
