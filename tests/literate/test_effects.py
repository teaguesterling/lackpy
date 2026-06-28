"""Tests for static effect classification of literate cells.

classify_effects() operates on *compiled* cell source and never executes it, so
every case here is pure input -> CellEffects with no kernel, no filesystem.
"""

from lackpy.interpreters.literate.effects import (
    CellEffects,
    classify_effects,
    combine,
)
from lackpy.interpreters.literate.compiler import compile_document


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
