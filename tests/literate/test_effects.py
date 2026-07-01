"""Tests for static effect classification of literate cells.

classify_effects() operates on *compiled* cell source and never executes it, so
every case here is pure input -> CellEffects with no kernel, no filesystem.
"""

import pytest

from lackpy.interpreters.literate.effects import (
    CellEffects,
    ToolEffect,
    LITERATE_TOOL_EFFECTS,
    as_grade,
    classify_effects,
    combine,
    exceeds_ceiling,
)
from lackpy.interpreters.literate.compiler import compile_document
from lackpy.lang.grader import Grade


def test_pure_compute_is_grade_zero_and_transactional():
    eff = classify_effects("x = 1 + 2\ny = x * 10")
    assert eff.grade.w == 0
    assert eff.writes == frozenset() and eff.reads == frozenset()
    assert not eff.unanalyzable and not eff.dynamic_paths
    assert not eff.needs_sandbox
    assert eff.transactional  # nothing to roll back


def test_literal_read_is_pinhole_and_transactional():
    eff = classify_effects("print(read_file('README.md'))")
    assert eff.grade.w == 1
    assert eff.reads == frozenset({"README.md"})
    assert eff.writes == frozenset()
    assert eff.transactional  # reads need no rollback
    assert not eff.needs_sandbox


def test_literal_write_is_scoped_write_and_journalable():
    eff = classify_effects("write_file('src/utils.py', 'x = 1')")
    assert eff.grade.w == 3
    assert eff.writes == frozenset({"src/utils.py"})
    assert eff.transactional  # literal path -> journalable
    assert not eff.needs_sandbox


def test_apply_diff_is_a_write_target():
    eff = classify_effects("apply_diff('a.py', '--- a\\n+++ b\\n')")
    assert eff.grade.w == 3
    assert eff.writes == frozenset({"a.py"})
    assert eff.transactional


def test_write_kwarg_path_is_recognized():
    eff = classify_effects("write_file(path='k.py', content='x')")
    assert eff.writes == frozenset({"k.py"})


def test_dynamic_write_path_is_not_transactional():
    eff = classify_effects("p = 'out.py'\nwrite_file(p, 'x')")
    assert eff.grade.w == 3
    assert eff.writes == frozenset()  # path not statically known
    assert eff.dynamic_paths
    assert not eff.transactional  # can't journal an unknown target


def test_search_content_path_is_second_arg():
    eff = classify_effects("search_content('TODO', 'src')")
    assert eff.grade.w == 1
    assert eff.reads == frozenset({"src"})
    assert eff.transactional


def test_run_command_needs_sandbox_and_is_not_transactional():
    eff = classify_effects("run_command('ls -la')")
    assert eff.grade.w == 3
    assert eff.exec_calls == frozenset({"run_command"})
    assert eff.needs_sandbox
    assert not eff.transactional  # exec can't be rolled back


def test_run_tests_is_exec():
    eff = classify_effects("run_tests('tests')")
    assert "run_tests" in eff.exec_calls
    assert eff.needs_sandbox
    assert not eff.transactional


def test_import_makes_cell_unanalyzable_and_conservative():
    eff = classify_effects("import os\nos.remove('x')")
    assert eff.unanalyzable
    assert eff.grade.w == 3  # forced conservative
    assert eff.needs_sandbox
    assert not eff.transactional


def test_open_is_an_escape_hatch():
    eff = classify_effects("open('x', 'w').write('y')")
    assert eff.unanalyzable
    assert not eff.transactional


def test_syntax_error_is_unanalyzable_not_a_crash():
    eff = classify_effects("write_file('x',")  # truncated
    assert eff.unanalyzable
    assert not eff.transactional


def test_method_calls_are_not_blanket_unanalyzable():
    # str.join / list.append etc. must NOT trip the escape-hatch flag, or every
    # ordinary code cell would be marked unanalyzable.
    eff = classify_effects("parts = []\nparts.append('a')\nout = ', '.join(parts)")
    assert not eff.unanalyzable
    assert eff.grade.w == 0
    assert eff.transactional


def test_combine_takes_max_grade_and_unions_paths():
    a = classify_effects("print(read_file('a.txt'))")
    b = classify_effects("write_file('b.py', 'x')")
    agg = combine([a, b])
    assert agg.grade.w == 3  # max(read=1, write=3)
    assert agg.reads == frozenset({"a.txt"})
    assert agg.writes == frozenset({"b.py"})
    assert agg.transactional  # both halves are transactional


def test_combine_is_poisoned_by_one_exec_cell():
    safe = classify_effects("write_file('b.py', 'x')")
    risky = classify_effects("run_command('rm -rf /tmp/x')")
    agg = combine([safe, risky])
    assert agg.needs_sandbox
    assert not agg.transactional  # one non-transactional cell taints the segment


def test_empty_combine_is_pure():
    agg = combine([])
    assert agg.grade.w == 0
    assert agg.transactional
    assert not agg.needs_sandbox


def test_classifies_through_the_real_compiler():
    # The annotated forms compile to literal-path tool calls, so they classify
    # identically to hand-written tool calls -- the uniform-path claim.
    doc = (
        "```lackpy @write(src/out.py)\n"
        "value = 1\n"
        "```\n"
    )
    compiled = compile_document(doc)
    eff = classify_effects(compiled)
    assert "src/out.py" in eff.writes
    assert eff.grade.w == 3
    assert eff.transactional


def test_celleffects_is_frozen():
    eff = classify_effects("x = 1")
    assert isinstance(eff, CellEffects)
    import dataclasses
    assert dataclasses.is_dataclass(eff)


# --- config-driven grade table (the "(b)" mechanism: grades from data, injectable) ---

def test_grades_load_from_toml_covering_the_literate_tools():
    assert set(LITERATE_TOOL_EFFECTS) == {
        "read_file", "search_content", "write_file",
        "apply_diff", "run_tests", "run_command",
    }
    assert LITERATE_TOOL_EFFECTS["write_file"].grade.w == 3
    assert LITERATE_TOOL_EFFECTS["read_file"].kind == "read"
    # run_command: write-ceiling blast radius but exec mechanism (sandbox).
    rc = LITERATE_TOOL_EFFECTS["run_command"]
    assert rc.grade.w == 3 and rc.kind == "exec" and rc.path_arg is None


def test_injected_tool_effects_override_the_default_table():
    # A caller (e.g. a session sourcing grades from the resolved toolbox) can
    # supply its own map; the built-in names are then NOT classified.
    custom = {"my_writer": ToolEffect(Grade(3, 3), "write", "path", 0)}
    eff = classify_effects("my_writer('out.x', 'data')", tool_effects=custom)
    assert eff.writes == frozenset({"out.x"})
    assert eff.grade.w == 3
    # read_file is absent from the injected map -> treated as a plain call.
    eff2 = classify_effects("read_file('x')", tool_effects=custom)
    assert eff2.reads == frozenset()
    assert eff2.grade.w == 0


def test_unknown_tool_name_is_ignored():
    eff = classify_effects("frobnicate('x')")
    assert eff.grade.w == 0
    assert not eff.needs_sandbox
    assert eff.transactional


# --- the ceiling gate (slice 1: refuse before running if effects exceed grade) ---

def test_exceeds_ceiling_on_world_coupling():
    write = classify_effects("write_file('a.py', 'x')")  # w=3
    assert exceeds_ceiling(write, Grade(1, 3)) is not None  # w=3 > ceiling w=1
    assert "w=3" in exceeds_ceiling(write, Grade(1, 3))


def test_within_ceiling_passes():
    read = classify_effects("print(read_file('a.py'))")  # w=1
    assert exceeds_ceiling(read, Grade(1, 1)) is None  # exactly at ceiling -> ok
    assert exceeds_ceiling(read, Grade(3, 3)) is None


def test_pure_doc_passes_a_zero_ceiling():
    pure = classify_effects("x = 1 + 2")
    assert exceeds_ceiling(pure, Grade(0, 0)) is None


def test_exceeds_ceiling_on_effects_depth():
    # d over, w within: still a violation, reported on the d axis.
    eff = CellEffects(Grade(1, 3), frozenset(), frozenset(), frozenset(), False, False)
    msg = exceeds_ceiling(eff, Grade(2, 2))
    assert msg is not None and "d=3" in msg


def test_unanalyzable_doc_is_caught_by_a_low_ceiling():
    # import -> unanalyzable -> conservative w=3, so a read-only ceiling refuses it.
    eff = classify_effects("import os\nos.remove('x')")
    assert exceeds_ceiling(eff, Grade(1, 1)) is not None


# --- review fixes: introspection not a hatch, as_grade coercion ---

def test_introspection_is_not_an_escape_hatch():
    # locals/globals/vars are pure namespace introspection (no world effect) and
    # the first-party @scratch directive compiles to locals() -- flagging them
    # would falsely refuse benign read-only docs (review finding #2/#8).
    for src in ("locals()", "globals()", "vars()"):
        eff = classify_effects(src)
        assert not eff.unanalyzable, src
        assert eff.grade.w == 0, src


def test_scratch_directive_classifies_as_pure():
    from lackpy.interpreters.literate.compiler import compile_document
    compiled = compile_document("```lackpy @scratch\na = 10\nb = 'x'\n```")
    eff = classify_effects(compiled)
    assert not eff.unanalyzable
    assert eff.grade.w == 0  # benign introspection, allowed under any ceiling


def test_open_still_an_escape_hatch_after_introspection_removed():
    # Removing locals/globals/vars must not weaken the genuine hatches.
    assert classify_effects("open('x','w')").unanalyzable
    assert classify_effects("eval('1+1')").unanalyzable


def test_as_grade_coerces_grade_int_and_pair():
    assert as_grade(Grade(2, 3)) == Grade(2, 3)
    assert as_grade(2) == Grade(2, 2)
    assert as_grade((1, 3)) == Grade(1, 3)
    assert as_grade([0, 0]) == Grade(0, 0)


def test_as_grade_rejects_garbage_and_bool():
    with pytest.raises(TypeError):
        as_grade("high")
    with pytest.raises(TypeError):
        as_grade(True)  # bool is an int subclass; reject the ambiguity


# --- tool_effect_from_spec: one derivation for injected tools (spec-declared
#     effect metadata makes an injected write tool precise + journalable) ---

def test_spec_with_effect_metadata_is_precise_and_journalable():
    from lackpy.interpreters.literate.effects import tool_effect_from_spec
    from lackpy.tools.toolbox import ToolSpec
    spec = ToolSpec(name="my_write", provider="python", description="w", args=[],
                    returns="None", grade_w=3, effects_ceiling=3,
                    effect_kind="write", path_arg="path", path_index=0)
    te = tool_effect_from_spec(spec)
    assert te.kind == "write" and te.path_arg == "path" and te.path_index == 0
    # a literal-path call to it is journalable (its target lands in writes)
    eff = classify_effects("my_write('out.txt', 'data')", tool_effects={"my_write": te})
    assert eff.writes == frozenset({"out.txt"})
    assert eff.transactional


def test_spec_without_effect_kind_falls_back_to_the_grade_heuristic():
    from lackpy.interpreters.literate.effects import tool_effect_from_spec
    from lackpy.tools.toolbox import ToolSpec
    spec = ToolSpec(name="w2", provider="python", description="w", args=[],
                    returns="None", grade_w=3, effects_ceiling=3)  # no effect metadata
    te = tool_effect_from_spec(spec)
    assert te.kind == "write"          # heuristic from grade_w
    assert te.path_arg is None
    # no declared path -> the heuristic can't journal it (dynamic)
    eff = classify_effects("w2('out.txt', 'data')", tool_effects={"w2": te})
    assert eff.writes == frozenset()
    assert eff.dynamic_paths and not eff.transactional


def test_w3_exec_spec_is_not_mistaken_for_a_write():
    # The heuristic maps w>=3 -> write; an explicit effect_kind="exec" fixes it.
    from lackpy.interpreters.literate.effects import tool_effect_from_spec
    from lackpy.tools.toolbox import ToolSpec
    spec = ToolSpec(name="sh", provider="python", description="x", args=[],
                    returns="None", grade_w=3, effects_ceiling=3, effect_kind="exec")
    te = tool_effect_from_spec(spec)
    assert te.kind == "exec"
    eff = classify_effects("sh('rm -rf /tmp/x')", tool_effects={"sh": te})
    assert eff.exec_calls == frozenset({"sh"}) and eff.needs_sandbox
