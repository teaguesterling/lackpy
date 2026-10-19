"""A validated try program runs: Task 3's exception classes and Task 8's rules together."""

from lackpy.lang.validator import validate
from lackpy.run.runner import RestrictedRunner


def test_validated_try_program_runs_and_uses_the_default():
    program = 'd = {}\ntry:\n    x = d["k"]\nexcept KeyError:\n    x = 0\nx'
    assert validate(program).valid
    result = RestrictedRunner().run(program)
    assert result.success, result.error
    assert result.output == 0
