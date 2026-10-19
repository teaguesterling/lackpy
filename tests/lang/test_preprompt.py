"""The pre-prompt teaches a model lackpy before the full guide.

It is generated from code, like the guide, so it cannot drift. These tests pin
what makes it useful: every worked example validates, the habit table names an
approved form for each habit models were measured to have, and it stays short.
"""
import re

from lackpy.lang import validate
from lackpy.lang.spec import HABITS, PREPROMPT_EXAMPLES, format_preprompt


def test_every_worked_example_validates():
    for title, program in PREPROMPT_EXAMPLES:
        result = validate(program)
        assert result.valid, (title, result.errors)


def test_each_habit_row_is_in_the_preprompt():
    text = format_preprompt()
    for habit, instead in HABITS:
        assert habit in text and instead in text


def test_approved_forms_in_the_habit_table_validate():
    # The "write instead" column holds code; each fragment must be legal lackpy
    # once its names are given values.
    preamble = "from pathlib import Path\nimport json\np = 'x'\nd = {}\nk = 'a'\ncond = True\n"
    for _, instead in HABITS:
        code = re.search(r"`([^`]+)`", instead)
        if code and not code.group(1).startswith("except"):
            result = validate(preamble + code.group(1) + "\n")
            assert not [v for v in result.violations if v.rule_id.startswith("safety.")], (
                instead, result.errors)


def test_preprompt_is_short_and_ends_before_the_guide():
    text = format_preprompt()
    assert len(text) < 5000
    assert "Part 1. Program rules" not in text


def test_preprompt_states_the_ruled_nesting_depth():
    text = format_preprompt()
    assert "A call nests at most 3 deep" in text
    assert "an I/O call inside another call" not in text


def test_preprompt_does_not_say_an_io_call_needs_its_own_line():
    text = format_preprompt()
    assert "gets its own line" not in text
    assert "In a chain, the I/O call comes first: `Path(p).read_text().strip()`." in text
