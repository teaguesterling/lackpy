"""No set of style waivers admits a dangerous program (final review, recommendation 4).

Each program below is a route out of a guarantee: a discarded exception, a limit
signal caught, code run at def time, a forbidden module, or object internals.
Every check that closes one of them must be a safety rule, because a ``# ste:``
waiver lifts any style rule.

The harness waives every style rule id in ``RULES`` on every statement. The
waiver parser attaches a standalone waiver to the line directly below it only,
and one comment holds one rule id, so one statement can carry at most two
waivers (one above, one trailing). Stacked standalone lines would leave all but
the last one misplaced (``meta.waiver-unknown``). The harness therefore builds
one variant per style id and per placement, with that id waived on every
statement. A waiver removes only violations of its own id, so the variants
together cover the program with every style id waived at once.
"""

import ast

import pytest

from lackpy.lang.registry import RULES
from lackpy.lang.validator import validate

ALLOWED = {"read_file", "write_file", "guarded"}

_KEY_RECURSION = ("def k(x):\n    try:\n        y = sorted([x], key=k)\n    except {cls}:\n"
                  "        y = sorted([x], key=k)\n    return 0\nz = sorted([1], key=k)\nz")

DANGEROUS = {
    # C1: finally discards the exception in flight
    "finally_continue_key_recursion": (
        "def k(x):\n    for i in [1, 2]:\n        try:\n            y = sorted([x], key=k)\n"
        "        finally:\n            continue\n    return 0\nz = sorted([1], key=k)\nz"),
    "finally_return_recursion": (
        "def f(n):\n    try:\n        return f(n + 1)\n    finally:\n        return 0\ny = f(0)\ny"),
    "finally_return_swallows_refusal": (
        "def f(p):\n    try:\n        return guarded(p)\n    finally:\n        return 'swallowed'\n"
        "x = f('/etc/passwd')\nx"),
    # I1: a decorator runs a tool at def time
    "decorator_runs_tool": "@write_file\ndef f():\n    return 1\nf",
    # limit signals caught around key= recursion
    "except_exception_key_recursion": _KEY_RECURSION.format(cls="Exception"),
    "except_recursion_error_key_recursion": _KEY_RECURSION.format(cls="RecursionError"),
    "except_runtime_error_key_recursion": _KEY_RECURSION.format(cls="RuntimeError"),
    "bare_except": "d = {}\ntry:\n    x = d['k']\nexcept:\n    x = 0",
    "import_os": "import os\nx = os.listdir('.')",
    "class_traversal": "x = ().__class__",
    # I2: compiles only inside a loop
    "top_level_break": "x = 1\nbreak",
}

STYLE_IDS = [r.id for r in RULES if r.tier == "style"]


def _statement_lines(program: str) -> list[int]:
    return sorted({n.lineno for n in ast.walk(ast.parse(program)) if isinstance(n, ast.stmt)})


def waive(program: str, rule_id: str, placement: str) -> str:
    """Waive ``rule_id`` on every statement of ``program``, above it or trailing on its first line."""
    lines = program.split("\n")
    for lineno in reversed(_statement_lines(program)):
        i = lineno - 1
        if placement == "trailing":
            lines[i] += f"  # ste: {rule_id}, regression"
        else:
            indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
            lines.insert(i, f"{indent}# ste: {rule_id}, regression")
    return "\n".join(lines)


def test_harness_has_style_ids():
    assert len(STYLE_IDS) > 10


@pytest.mark.parametrize("placement", ["above", "trailing"])
@pytest.mark.parametrize("name", sorted(DANGEROUS))
def test_dangerous_program_stays_invalid_with_every_style_waiver(name, placement):
    program = DANGEROUS[name]
    for rule_id in STYLE_IDS:
        waived = waive(program, rule_id, placement)
        result = validate(waived, allowed_names=ALLOWED)
        misplaced = [v for v in result.violations if v.rule_id.startswith("meta.")]
        assert not misplaced, (rule_id, misplaced)  # every waiver attached to a statement
        assert not result.valid, (rule_id, waived)
        assert any(v.rule_id.startswith("safety.") for v in result.violations), \
            (rule_id, result.violations)


def test_harness_waivers_do_lift_style_violations():
    # Control: a program held back by one style rule becomes valid in that rule's variant,
    # so the harness's waivers are real.
    program = "d = {}\ntry:\n    x = d['k']\nexcept KeyError:\n    x = 0\nelse:\n    print(x)"
    styled = {v.rule_id for v in validate(program).violations}
    assert styled and all(i.startswith("style.") for i in styled)
    assert len(styled) == 1
    (rule_id,) = styled
    for placement in ("above", "trailing"):
        result = validate(waive(program, rule_id, placement))
        assert result.valid, (placement, result.violations)
