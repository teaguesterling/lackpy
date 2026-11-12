"""The ``# /// intent`` block: parsing and the [params] checks (spec E5, E6)."""

import pytest

from lackpy.intent import (
    Intent, IntentError, Param, check_params, document_intent, parse_intent, program_intent,
)

EXAMPLE = '''\
# /// intent
# goal   = "Count the TODO markers in each C++ file under a directory."
# reads  = ["{root}/*.cpp"]
# writes = []
# output = "printed"
#
# [params]
# root  = { type = "str", doc = "directory to scan, relative to the workspace" }
# limit = { type = "int", doc = "most files to report" }
# ///
from pathlib import Path

params = {"root": "src", "limit": 20}
print(params)
'''


def _block(*lines: str) -> str:
    return "# /// intent\n" + "".join(f"# {line}\n" for line in lines) + "# ///\n"


def test_the_spec_example_parses_to_the_expected_intent():
    assert program_intent(EXAMPLE) == Intent(
        goal="Count the TODO markers in each C++ file under a directory.",
        reads=("{root}/*.cpp",),
        writes=(),
        checks=(),
        output="printed",
        params=(
            Param("root", "str", "directory to scan, relative to the workspace", None),
            Param("limit", "int", "most files to report", None),
        ),
    )


def test_a_program_without_a_block_has_no_intent():
    assert parse_intent("x = 1\n") is None
    assert program_intent("x = 1\n") is None


def test_a_shebang_and_comments_may_come_before_the_block():
    program = "#!/usr/bin/env python3\n\n# a note\n" + _block('goal = "g"') + "x = 1\n"
    assert program_intent(program).goal == "g"


def test_code_before_the_block_hides_it_from_a_program():
    assert program_intent("x = 1\n" + _block('goal = "g"')) is None


def test_an_unknown_key_is_named_in_the_error():
    with pytest.raises(IntentError, match="gaol"):
        parse_intent(_block('gaol = "typo"'))


def test_a_second_block_is_an_error():
    with pytest.raises(IntentError, match="second"):
        parse_intent(_block('goal = "a"') + _block('goal = "b"'))


def test_a_bad_output_is_an_error():
    with pytest.raises(IntentError, match="output"):
        parse_intent(_block('output = "everything"'))


def test_a_bad_param_type_is_an_error():
    with pytest.raises(IntentError, match="dict"):
        parse_intent(_block("[params]", 'n = { type = "dict", doc = "d" }'))


def test_absent_writes_is_none_and_empty_writes_is_an_empty_tuple():
    assert parse_intent(_block('goal = "g"')).writes is None
    assert parse_intent(_block("writes = []")).writes == ()


def test_a_toml_error_is_an_intent_error():
    with pytest.raises(IntentError):
        parse_intent(_block("goal = "))


def test_a_param_key_missing_from_the_literal_is_an_error():
    intent = parse_intent(_block("[params]", 'a = { type = "int", doc = "d" }',
                                 'b = { type = "int", doc = "d" }'))
    errors = check_params(intent, "params = {'a': 1}\n")
    assert len(errors) == 1 and "'b'" in errors[0]


def test_a_literal_key_missing_from_params_is_an_error():
    intent = parse_intent(_block("[params]", 'a = { type = "int", doc = "d" }'))
    errors = check_params(intent, "params = {'a': 1, 'c': 2}\n")
    assert len(errors) == 1 and "'c'" in errors[0]


def test_a_default_of_the_wrong_type_is_an_error():
    intent = parse_intent(_block("[params]", 'limit = { type = "int", doc = "d" }'))
    errors = check_params(intent, "params = {'limit': '20'}\n")
    assert len(errors) == 1 and "limit" in errors[0] and "int" in errors[0]


def test_a_bool_is_not_an_int_and_list_items_must_be_str():
    intent = parse_intent(_block("[params]", 'n = { type = "int", doc = "d" }',
                                 'xs = { type = "list[str]", doc = "d" }'))
    assert len(check_params(intent, "params = {'n': True, 'xs': ['a']}\n")) == 1
    assert len(check_params(intent, "params = {'n': 1, 'xs': ['a', 2]}\n")) == 1
    assert check_params(intent, "params = {'n': 1, 'xs': ['a']}\n") == []


def test_check_params_fills_the_defaults():
    intent = program_intent(EXAMPLE)
    assert check_params(intent, EXAMPLE) == []
    assert [p.default for p in intent.params] == ["src", 20]


def test_params_without_a_literal_is_an_error():
    intent = parse_intent(_block("[params]", 'a = { type = "int", doc = "d" }'))
    assert len(check_params(intent, "x = 1\n")) == 1


def test_a_literal_without_params_is_an_error_only_when_a_block_exists():
    assert len(check_params(parse_intent(_block('goal = "g"')), "params = {'a': 1}\n")) == 1
    assert check_params(None, "params = {'a': 1}\n") == []


def test_the_block_in_the_first_code_cell_of_a_document_is_found():
    document = "Some prose.\n\n```lackpy\n" + _block('goal = "doc"') + "x = 1\n```\n"
    assert document_intent(document).goal == "doc"


def test_a_block_in_the_second_cell_of_a_document_is_not_found():
    document = "```lackpy\nx = 1\n```\n\n```lackpy\n" + _block('goal = "late"') + "```\n"
    assert document_intent(document) is None
