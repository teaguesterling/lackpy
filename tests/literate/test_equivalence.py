"""Literate stepy gate 3: the plain-Python invariant on the equivalence fixtures.

Each fixture in ``fixtures/`` is a literate document that mirrors an eval task
(a guarded patch, a JSON read with ``try``, a scan with ``def``, a regex task,
and a document that writes files). Its code cells, joined in order, are a
program. That program runs under ``python3 -I`` in a fresh copy of
``fixtures/workspace``. The document runs in two more fresh copies:

* cell by cell, through the kernel the service builds
  (``literate._build_kernel``: one ``CellSession`` under the copy's
  ``PathPolicy``), which gives the stdout of each cell;
* through ``LackpyService.run_program`` with ``execution="literate"``.

python3's stdout must equal the code cells' stdout joined, exactly. The
service's rendered stdout must equal every cell's output joined, exactly. All
three copies must hold the same files and directories afterwards. Prose output is excluded: python3 has no
prose. A document with ``@diff`` has no python3 twin (Task 3 ruling I2), so no
fixture uses ``@diff``, and the test checks that.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from lackpy.interpreters.base import ExecutionContext
from lackpy.interpreters.literate import _build_kernel
from lackpy.interpreters.literate.compiler import compile_cells
from lackpy.interpreters.literate.parser import ParseResult, parse
from lackpy.profiles import Profile
from lackpy.run.paths import load_policy
from lackpy.service import LackpyService

FIXTURES = Path(__file__).parent / "fixtures"
SEED = FIXTURES / "workspace"
DOCS = sorted(p.name for p in FIXTURES.glob("*.md"))


def _copy(dest: Path) -> Path:
    shutil.copytree(SEED, dest)
    return dest


def _files(root: Path) -> dict[str, bytes | None]:
    """Every file (its bytes) and every directory (``None``) under ``root``."""
    return {str(p.relative_to(root)): p.read_bytes() if p.is_file() else None
            for p in sorted(root.rglob("*"))}


def _joined(parsed: ParseResult) -> str:
    """The document's code cells joined in order: prose is not code."""
    code = [c for c in parsed.cells if c.cell_type != "prose"]
    return compile_cells(ParseResult(frontmatter=parsed.frontmatter, cells=code)) + "\n"


def test_there_are_six_fixtures_and_none_uses_diff():
    # Six since Teague's C1 decision: rebind.md rebinds a name across cells.
    assert len(DOCS) == 6
    for name in DOCS:
        parsed = parse((FIXTURES / name).read_text())
        assert not parsed.errors, (name, parsed.errors)
        assert "diff" not in {c.cell_type for c in parsed.cells}, name
        assert {c.cell_type for c in parsed.cells} & {"prose"}, f"{name} has no prose"


@pytest.mark.parametrize("name", DOCS)
def test_code_cells_match_python3(name, tmp_path):
    doc = (FIXTURES / name).read_text()
    parsed = parse(doc)
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)

    # python3 -I on the joined code cells.
    py_ws = _copy(tmp_path / "py")
    script = tmp_path / "joined.py"
    script.write_text(_joined(parsed))
    proc = subprocess.run([sys.executable, "-I", str(script)], cwd=py_ws,
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr

    # The document, cell by cell, through the service's kernel builder.
    cell_ws = _copy(tmp_path / "cells")
    kernel = _build_kernel(ExecutionContext(
        base_dir=str(cell_ws),
        config={"path_policy": load_policy(cell_ws, home=home, managed=None)}))
    code_out, all_out = [], []
    for index, cell in enumerate(parsed.cells):
        result = kernel.execute_cell(cell, index)
        assert result.success, (name, index, result.error)
        all_out.append(result.output or "")
        if cell.cell_type != "prose":
            code_out.append(result.output or "")

    # The document through the service entry point.
    svc_ws = _copy(tmp_path / "svc")
    svc = LackpyService(workspace=svc_ws,
                        path_policy_loader=lambda w: load_policy(w, home=home, managed=None))
    run = asyncio.run(svc.run_program(doc, profile=Profile(execution="literate")))
    assert run.success, run.error

    assert proc.stdout == "".join(code_out)
    assert proc.stdout, f"{name}: the code cells print nothing to compare"
    assert _files(py_ws) == _files(cell_ws) == _files(svc_ws)
    assert _files(py_ws) != _files(SEED) or name in {"json_try.md", "rebind.md", "regex.md", "scan_def.md"}
    # The service's rendered document is every cell's output, exactly.
    assert run.stdout == "".join(all_out)
