"""Tests for the literate effect helpers (literate stepy Task 3, design D6/D7).

``classify_effects`` grades a *compiled* cell from the ``Path`` members it
names, without running it: a read member is a pinhole read (w=1), a write
member is a scoped write (w=3). The kit grade map (``tool_effects.toml``) is
gone. The classifier is the ceiling gate's cooperative planner; the policy
(``read_only_policy``) is the boundary.

``path_writes`` is strict reactive mode's evidence: the ``Path`` write
entries of a cell's slice of the session trace.
"""

import pytest

from lackpy.interpreters.literate.compiler import compile_document
from lackpy.interpreters.literate.effects import (
    PATH_WRITE_TOOLS,
    CellEffects,
    as_grade,
    classify_effects,
    exceeds_ceiling,
    path_writes,
)
from lackpy.lang.grader import Grade
from lackpy.run.trace import TraceEntry


def test_pure_compute_is_grade_zero():
    assert classify_effects("x = 1 + 2\ny = x * 10").grade == Grade(0, 0)


@pytest.mark.parametrize("member", ["read_text", "read_bytes", "exists", "is_file",
                                    "is_dir", "glob", "rglob", "iterdir"])
def test_a_path_read_member_is_a_pinhole_read(member):
    eff = classify_effects(f"from pathlib import Path\nx = Path('a').{member}")
    assert eff.grade == Grade(1, 1)


@pytest.mark.parametrize("member", ["write_text", "write_bytes", "mkdir"])
def test_a_path_write_member_is_a_scoped_write(member):
    eff = classify_effects(f"from pathlib import Path\nPath('a').{member}('x')")
    assert eff.grade == Grade(3, 3)


def test_an_aliased_write_member_is_still_a_write():
    # The attribute is graded where it is named, called or not.
    eff = classify_effects("from pathlib import Path\nw = Path('a').write_text\nw('x')")
    assert eff.grade.w == 3


def test_a_write_wins_over_a_read():
    eff = classify_effects("from pathlib import Path\np = Path('a')\np.write_text(p.read_text())")
    assert eff.grade == Grade(3, 3)


def test_importing_pathlib_is_not_an_effect():
    # The validator closes the import set; an import is no longer an escape hatch.
    assert classify_effects("from pathlib import Path\nimport json").grade == Grade(0, 0)


def test_a_string_method_is_not_an_effect():
    assert classify_effects("s = 'a b'.split()\nt = ', '.join(s).upper()").grade.w == 0


def test_syntax_error_is_conservative_not_a_crash():
    assert classify_effects("x = (1 +").grade == Grade(3, 3)


@pytest.mark.parametrize("doc,w", [
    ("```lackpy @read(a.txt)\n```", 1),
    ("```lackpy @write(a.txt)\nbody\n```", 3),
    ("```lackpy @scratch\na = 10\nb = 'x'\n```", 0),
])
def test_classifies_through_the_real_compiler(doc, w):
    assert classify_effects(compile_document(doc)).grade.w == w


# --- the ceiling gate --------------------------------------------------------

def test_exceeds_ceiling_on_world_coupling():
    write = classify_effects("from pathlib import Path\nPath('a').write_text('x')")
    assert "w=3" in exceeds_ceiling(write, Grade(1, 3))


def test_within_ceiling_passes():
    read = classify_effects("from pathlib import Path\nprint(Path('a').read_text())")
    assert exceeds_ceiling(read, Grade(1, 1)) is None  # exactly at the ceiling
    assert exceeds_ceiling(read, Grade(3, 3)) is None


def test_pure_doc_passes_a_zero_ceiling():
    assert exceeds_ceiling(classify_effects("x = 1 + 2"), Grade(0, 0)) is None


def test_exceeds_ceiling_on_effects_depth():
    msg = exceeds_ceiling(CellEffects(Grade(1, 3)), Grade(2, 2))
    assert msg is not None and "d=3" in msg


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


# --- strict mode's evidence: Path writes in the trace ----------------------------

def _entry(tool, success=True):
    return TraceEntry(step=0, tool=tool, args=["p"], result=None, duration_ms=0.0,
                      success=success, error=None if success else "boom")


def test_the_write_tools_are_the_path_write_members():
    assert PATH_WRITE_TOOLS == {"Path.write_text", "Path.write_bytes", "Path.mkdir"}


def test_path_writes_names_each_write_once_sorted():
    entries = [_entry("Path.read_text"), _entry("Path.write_text"), _entry("Path.mkdir"),
               _entry("Path.write_text"), _entry("Path.exists")]
    assert path_writes(entries) == ("Path.mkdir", "Path.write_text")


def test_a_failed_write_still_counts():
    # A write that raised part way may have changed the file: strict mode must
    # not replay it.
    assert path_writes([_entry("Path.write_bytes", success=False)]) == ("Path.write_bytes",)


def test_reads_only_is_no_write():
    assert path_writes([_entry("Path.read_text"), _entry("Path.glob")]) == ()
