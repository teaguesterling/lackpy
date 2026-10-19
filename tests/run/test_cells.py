"""CellSession: one namespace, cumulative validation, refusals end the cell (literate stepy Task 1)."""

from __future__ import annotations

import pytest

from lackpy.lang import validate
from lackpy.run.cells import CellSession
from lackpy.run.paths import load_policy


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


@pytest.fixture
def session(workspace, tmp_home):
    return CellSession(path_policy=load_policy(workspace, home=tmp_home, managed=None))


def _ok(outcome):
    assert outcome.ok, (outcome.phase, outcome.error)
    assert outcome.phase is None and outcome.error is None
    return outcome


# -- names, imports and defs flow across cells ---------------------------------

def test_names_imports_and_defs_flow_across_cells(session):
    _ok(session.run_cell("import json\nrows = [1, 2, 3]"))
    _ok(session.run_cell("def total(xs):\n    return sum(xs)"))
    out = _ok(session.run_cell("print(json.dumps(total(rows)))"))
    assert out.stdout == "6\n"


def test_bound_names_lists_user_names_only(session):
    _ok(session.run_cell("import json\nrows = [1]\ndef f():\n    return 1"))
    assert session.bound_names() == {"json", "rows", "f"}


# -- the import gate and json.codecs across cells (Review Focus 2) -------------

def test_import_in_a_later_cell_is_refused(session):
    _ok(session.run_cell("import json"))
    out = session.run_cell("import os")
    assert not out.ok and out.phase == "validate"
    # The rule names the node, and its hint the approved alternative.
    assert "Import at line 1" in out.error and "pathlib.Path" in out.error


def test_json_codecs_across_cells_is_refused_statically(session):
    # Alone, ``json.codecs`` validates: the validator does not know ``json`` is a module.
    assert validate("json.codecs", allowed_names=set()).valid
    _ok(session.run_cell("import json"))
    out = session.run_cell("x = json.codecs")
    assert not out.ok and out.phase == "validate"
    assert "json" in out.error
    assert "x" not in session.bound_names()


def test_json_codecs_through_a_def_from_an_earlier_cell_is_refused(session):
    # The def aliases the module; the validator stops it in its own cell.
    _ok(session.run_cell("import json"))
    out = session.run_cell("def mod():\n    return json")
    assert not out.ok and out.phase == "validate"


# -- violations are reported on this cell's lines only -------------------------

def test_violation_is_reported_on_this_cells_lines(session):
    _ok(session.run_cell("rows = [1, 2]\ntotal = 0"))
    out = session.run_cell("n = len(rows)\nk = rows.__class__")
    assert not out.ok and out.phase == "validate"
    # The cell's line 2, not the cumulative line 4.
    assert "line 2" in out.error
    assert "line 4" not in out.error
    assert "n" not in session.bound_names()  # nothing ran


def test_a_cell_that_failed_validation_is_not_in_the_cumulative_program(session):
    assert session.run_cell("import os").phase == "validate"
    # Had ``import os`` joined the prefix, this cell would carry no violation of its own
    # and still be invalid; it validates because the rejected cell is left out.
    _ok(session.run_cell("x = 1"))


def test_a_parse_error_is_reported_on_this_cells_line(session):
    _ok(session.run_cell("a = 1\nb = 2\nc = 3"))
    out = session.run_cell("x = 1\ny = (")
    assert not out.ok and out.phase == "validate"
    assert "line 2" in out.error


def test_a_cell_that_continues_an_earlier_block_is_rejected(session):
    # Concatenated, ``else:`` would join the earlier ``if``; alone it does not parse.
    _ok(session.run_cell("c = True\nif c:\n    x = 1"))
    out = session.run_cell("else:\n    x = 2")
    assert not out.ok and out.phase == "validate"


def test_a_runtime_failure_gives_phase_runtime_and_keeps_what_ran(session):
    out = session.run_cell("a = 1\nb = {}['k']")
    assert not out.ok and out.phase == "runtime"
    assert "k" in out.error
    assert session.bound_names() == {"a"}
    out = _ok(session.run_cell("print(a)"))
    assert out.stdout == "1\n"


# -- refusals -------------------------------------------------------------------

READ_OUTSIDE = ('try:\n    text = Path("/etc/hostname").read_text()\n'
                'except OSError:\n    text = ""')


def test_a_refusal_gives_phase_refused(session):
    _ok(session.run_cell("from pathlib import Path"))
    out = session.run_cell('text = Path("/etc/hostname").read_text()')
    assert not out.ok and out.phase == "refused"
    assert "outside the workspace" in out.error


def test_a_refusal_inside_try_gives_phase_refused(session):
    # Gate 4: the handler cannot catch a refusal (it is a BaseException).
    _ok(session.run_cell("from pathlib import Path"))
    assert validate("from pathlib import Path\n" + READ_OUTSIDE, allowed_names=set()).valid
    out = session.run_cell(READ_OUTSIDE)
    assert not out.ok and out.phase == "refused"
    assert "text" not in session.bound_names()


def test_except_exception_is_rejected_before_it_can_run(session):
    # The brief's ``except Exception`` shape never reaches the runtime: safety.except-base.
    _ok(session.run_cell("from pathlib import Path"))
    out = session.run_cell(READ_OUTSIDE.replace("OSError", "Exception"))
    assert not out.ok and out.phase == "validate"


def test_json_in_a_container_from_an_earlier_cell_is_refused_statically(session):
    _ok(session.run_cell("import json"))
    out = session.run_cell("mods = {'j': json}")
    assert not out.ok and out.phase == "validate"


# -- eval_expr ------------------------------------------------------------------

def _value(outcome):
    assert outcome.ok, (outcome.phase, outcome.error)
    assert outcome.phase is None and outcome.error is None
    return outcome.value


def test_eval_expr_evaluates_in_the_namespace(session):
    _ok(session.run_cell("rows = [1, 2, 3]"))
    assert _value(session.eval_expr("len(rows)")) == 3


def test_eval_expr_rejects_a_private_attribute(session):
    _ok(session.run_cell("rows = [1, 2, 3]"))
    out = session.eval_expr("rows.__class__")
    assert not out.ok and out.phase == "validate" and out.value is None
    assert "line 1" in out.error


def test_eval_expr_uses_the_cumulative_context(session):
    _ok(session.run_cell("import json"))
    out = session.eval_expr("json.codecs")
    assert not out.ok and out.phase == "validate"


def test_eval_expr_rejects_a_statement_and_a_second_line(session):
    assert session.eval_expr("x = 1").phase == "validate"
    assert session.eval_expr("1\n2").phase == "validate"
    assert "x" not in session.bound_names()


def test_eval_expr_gives_phase_runtime_for_an_exception(session):
    _ok(session.run_cell("d = {}"))
    out = session.eval_expr("d['k']")
    assert not out.ok and out.phase == "runtime" and "k" in out.error


def test_eval_expr_never_raises_a_refusal(session):
    # Ruling (M2): the kernel's ``except BaseException`` must never see a refusal.
    _ok(session.run_cell("from pathlib import Path"))
    out = session.eval_expr('Path("/etc/hostname").read_text()')
    assert not out.ok and out.phase == "refused"
    assert "outside the workspace" in out.error


def test_eval_expr_returns_the_value_unfiltered(session):
    _ok(session.run_cell("def f():\n    return 1"))
    assert callable(_value(session.eval_expr("f")))


# -- I1: no violation the prefix lacked, wherever it lands -----------------------

def test_a_forward_reference_def_cannot_alias_json(session):
    # As one program: "Forbidden use of module: 'json' at line 2" (review I1).
    _ok(session.run_cell("def f():\n    return json"))
    out = session.run_cell("import json")
    assert not out.ok and out.phase == "validate"
    assert "line 2 of cell 1" in out.error and "made invalid by this cell" in out.error
    assert "json" not in session.bound_names()


def test_a_forward_reference_def_cannot_alias_pathlib(session):
    _ok(session.run_cell("def m():\n    return pathlib"))
    out = session.run_cell("import pathlib")
    assert not out.ok and out.phase == "validate"
    assert "line 2 of cell 1" in out.error


def test_a_forward_reference_def_cannot_reach_json_codecs(session):
    _ok(session.run_cell("def g():\n    return json.codecs"))
    out = session.run_cell("import json")
    assert not out.ok and out.phase == "validate"
    assert "json.codecs" in out.error


def test_json_through_a_cell_that_failed_at_runtime_is_refused_statically(session):
    # The failed cell stays in the program, so the validator knows json is a module.
    assert session.run_cell("import json\nx = {}['k']").phase == "runtime"
    out = session.run_cell("y = json.codecs")
    assert not out.ok and out.phase == "validate"


# -- the latch: a refusal ends the session (D4) ---------------------------------

def test_after_a_refusal_every_cell_is_refused_without_running(session):
    _ok(session.run_cell("from pathlib import Path"))
    first = session.run_cell('import json\ntext = Path("/etc/hostname").read_text()')
    assert first.phase == "refused"
    # The refused cell bound json; a later cell must not reach it (security finding, M1).
    out = session.run_cell("x = json")
    assert not out.ok and out.phase == "refused"
    assert "/etc/hostname" in out.error
    assert "x" not in session.bound_names()
    expr = session.eval_expr("1")
    assert not expr.ok and expr.phase == "refused" and "/etc/hostname" in expr.error


def test_a_refusal_from_eval_expr_latches_the_session(session):
    _ok(session.run_cell("from pathlib import Path"))
    assert session.eval_expr('Path("/etc/hostname").read_text()').phase == "refused"
    out = session.run_cell("print(1)")
    assert out.phase == "refused" and out.stdout == "" and "/etc/hostname" in out.error


# -- re-runs ---------------------------------------------------------------------

def test_each_cell_gets_its_own_cell_id(session):
    a = session.run_cell("x = 1")
    b = session.run_cell("import os")
    assert a.cell_id is not None and b.cell_id is not None and a.cell_id != b.cell_id


def test_rerun_cell_re_executes_the_stored_code(session):
    _ok(session.run_cell("n = 0"))
    bump = _ok(session.run_cell("n = n + 1\nprint(n)"))
    again = _ok(session.rerun_cell(bump.cell_id))
    assert again.cell_id == bump.cell_id
    assert again.stdout == "2\n" and again.delta == {"n": 2}


def test_rerun_of_a_def_cell_does_not_append_it_again(session):
    cell = _ok(session.run_cell("def f():\n    return 1"))
    _ok(session.rerun_cell(cell.cell_id))
    # Appended twice, style.def-shadow would now reject the next cell.
    _ok(session.run_cell("x = f()"))


def test_rerun_of_an_unknown_or_failed_cell_fails(session):
    assert not session.rerun_cell(999).ok
    rejected = session.run_cell("import os")
    out = session.rerun_cell(rejected.cell_id)
    assert not out.ok and out.error


def test_rerun_in_a_latched_session_is_refused(session):
    cell = _ok(session.run_cell("from pathlib import Path"))
    assert session.run_cell('Path("/etc/hostname").read_text()').phase == "refused"
    out = session.rerun_cell(cell.cell_id)
    assert not out.ok and out.phase == "refused"


# -- stdout and delta -------------------------------------------------------------

def test_stdout_is_captured_per_cell(session):
    assert _ok(session.run_cell("print('a')")).stdout == "a\n"
    assert _ok(session.run_cell("print('b', 'c', sep=None)")).stdout == "b c\n"
    assert _ok(session.run_cell("x = 1")).stdout == ""


def test_stdout_is_kept_when_a_cell_fails(session):
    out = session.run_cell("print('before')\nx = {}['k']")
    assert out.phase == "runtime" and out.stdout == "before\n"


def test_delta_holds_data_this_cell_bound_or_rebound(session):
    first = _ok(session.run_cell("import json\na = 1\nb = [1]\ndef f():\n    return 1"))
    assert first.delta == {"a": 1, "b": [1]}
    second = _ok(session.run_cell("a = 2\nc = 'x'"))
    assert second.delta == {"a": 2, "c": "x"}
    third = _ok(session.run_cell("print(a)"))
    assert third.delta == {}


def test_delta_leaves_out_a_rebinding_to_a_non_data_value(session):
    _ok(session.run_cell("a = 1"))
    out = _ok(session.run_cell("def a():\n    return 1"))
    assert out.delta == {}


# -- Path under a tmp policy ----------------------------------------------------

def test_path_works_under_a_tmp_policy(session, workspace):
    target = workspace / "out.txt"
    _ok(session.run_cell("from pathlib import Path"))
    _ok(session.run_cell(f"p = Path({str(target)!r})\np.write_text('hi')"))
    out = _ok(session.run_cell("print(p.read_text())"))
    assert out.stdout == "hi\n"
    assert target.read_text() == "hi"
    assert any(e.tool.startswith("Path.") for e in session.trace.entries)


def test_path_refuses_without_a_policy():
    out = CellSession(path_policy=None).run_cell("from pathlib import Path")
    assert not out.ok and out.phase == "refused"


def test_a_line_of_an_earlier_cell_in_a_message_names_the_cell(session):
    # style.def-shadow names both defs; the earlier one is named by cell and line.
    _ok(session.run_cell("x = 1\ndef f():\n    return 1"))
    out = session.run_cell("def f():\n    return 2")
    assert not out.ok and out.phase == "validate"
    assert "line 2 of cell 1" in out.error and "line 1" in out.error


def test_quoted_program_text_is_not_renumbered(session):
    # M4: only the validator's own line references change, not text it quotes.
    _ok(session.run_cell("p = [1]"))
    out = session.run_cell('p["line 9"].y = 2')
    assert not out.ok and out.phase == "validate"
    assert "'p['line 9'].y' at line 1" in out.error


def test_rerun_of_a_cell_that_failed_at_runtime_runs_it_again(session):
    # Forgiveness: a cell that failed on missing data is re-run once a later cell fixes it.
    _ok(session.run_cell("d = {}"))
    failed = session.run_cell("v = d['k']")
    assert failed.phase == "runtime"
    _ok(session.run_cell("d = {'k': 1}"))
    again = _ok(session.rerun_cell(failed.cell_id))
    assert again.delta == {"v": 1}
