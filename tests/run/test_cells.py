"""CellSession: one namespace, cumulative validation, refusals end the cell (literate stepy Task 1)."""

from __future__ import annotations

import pytest

from lackpy.lang import validate
from lackpy.run.cells import CellRejected, CellSession
from lackpy.run.paths import load_policy
from lackpy.run.refusal import LackpyRefusal


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

def test_eval_expr_evaluates_in_the_namespace(session):
    _ok(session.run_cell("rows = [1, 2, 3]"))
    assert session.eval_expr("len(rows)") == 3


def test_eval_expr_rejects_a_private_attribute(session):
    _ok(session.run_cell("rows = [1, 2, 3]"))
    with pytest.raises(CellRejected) as info:
        session.eval_expr("rows.__class__")
    assert "line 1" in str(info.value)


def test_eval_expr_uses_the_cumulative_context(session):
    _ok(session.run_cell("import json"))
    with pytest.raises(CellRejected):
        session.eval_expr("json.codecs")


def test_eval_expr_rejects_a_statement_and_a_second_line(session):
    with pytest.raises(CellRejected):
        session.eval_expr("x = 1")
    with pytest.raises(CellRejected):
        session.eval_expr("1\n2")
    assert "x" not in session.bound_names()


def test_eval_expr_raises_a_refusal_to_the_caller(session):
    _ok(session.run_cell("from pathlib import Path"))
    with pytest.raises(LackpyRefusal):
        session.eval_expr('Path("/etc/hostname").read_text()')


def test_eval_expr_returns_the_value_unfiltered(session):
    _ok(session.run_cell("def f():\n    return 1"))
    assert callable(session.eval_expr("f"))


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


def test_a_line_of_an_earlier_cell_in_a_message_says_so(session):
    # style.def-shadow names both defs; the earlier one keeps its cumulative number.
    # It also means a cell re-run through run_cell with a def fails validation.
    _ok(session.run_cell("x = 1\ndef f():\n    return 1"))
    out = session.run_cell("def f():\n    return 2")
    assert not out.ok and out.phase == "validate"
    assert "line 2 of the earlier cells" in out.error and "line 1" in out.error
