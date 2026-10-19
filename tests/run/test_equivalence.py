"""Design §5.3, measured: a reference program does the same under python3 and under lackpy.

Each of the 30 eval reference programs runs twice, in two fresh copies of
``scripts/eval/ste_usability/fixtures``: once as ``python3 prog.py`` (cwd = the
copy), once through the lackpy helper in ``conftest.run_in_lackpy`` (which
switches to ``run_program`` in Task 7). Both must succeed and give equal stdout
and equal file trees (every file's bytes, and the set of paths).
"""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "eval" / "ste_usability"))
from tasks import TASKS  # noqa: E402

from .conftest import run_in_lackpy


def snapshot(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.mark.parametrize("task", TASKS, ids=[t["id"] for t in TASKS])
def test_reference_is_equivalent_under_python3_and_lackpy(task, fixtures_copy, tmp_home):
    py_dir, lp_dir = fixtures_copy("py"), fixtures_copy("lackpy")
    before = snapshot(py_dir)
    assert before == snapshot(lp_dir)

    (py_dir / "prog.py").write_text(task["reference"])
    proc = subprocess.run([sys.executable, "prog.py"], cwd=py_dir, capture_output=True, text=True, timeout=60)
    (py_dir / "prog.py").unlink()
    assert proc.returncode == 0, proc.stderr

    result = run_in_lackpy(task["reference"], lp_dir, tmp_home)
    assert result.success, result.error

    assert result.stdout == proc.stdout
    after_py, after_lp = snapshot(py_dir), snapshot(lp_dir)
    assert after_lp == after_py
    assert after_py != before or not task["writes"], "a task with writes changed nothing"
