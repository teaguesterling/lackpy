"""Run and score lackpy programs for the STE usability eval.

A program is scored on two independent axes:

  valid    - lackpy.lang.validate() accepts it (static, the language).
  correct  - run under plain python3 in a fresh copy of fixtures/, it gives the
             same stdout and the same written files as the task's reference.

Every program runs inside bubblewrap: the host is read-only, only the scratch
copy of fixtures/ is writable, and there is no network. Invalid programs run
too, so "correct but invalid" measures over-restriction.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
sys.path.insert(0, str(HERE))
from tasks import TASKS  # noqa: E402

from lackpy.lang import validate  # noqa: E402


def _snapshot(root: Path) -> dict[str, str]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "prog.py":
            out[str(p.relative_to(root))] = p.read_text(errors="replace")
    return out


def _norm(text: str) -> list[str]:
    """Compare output by content, not format: drop ':' and ',' and whitespace runs."""
    return text.replace(":", " ").replace(",", " ").split()


def run(program: str, timeout: int = 15) -> dict:
    """Run a program in a sandboxed copy of fixtures/. Return stdout, rc, files."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "ws"
        shutil.copytree(FIXTURES, work)
        (work / "prog.py").write_text(program)
        cmd = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
               "--tmpfs", "/tmp", "--bind", str(work), str(work),
               "--unshare-net", "--die-with-parent", "--chdir", str(work),
               "python3", "prog.py"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            stdout, rc, err = r.stdout, r.returncode, r.stderr[-400:]
        except subprocess.TimeoutExpired:
            stdout, rc, err = "", -9, "timeout"
        return {"stdout": stdout, "rc": rc, "stderr": err, "files": _snapshot(work)}


def expected() -> dict[str, dict]:
    """Run every reference; each must validate and exit 0."""
    out = {}
    for t in TASKS:
        v = validate(t["reference"])
        if not v.valid:
            raise SystemExit(f"reference {t['id']} does not validate: {v.errors}")
        r = run(t["reference"])
        if r["rc"] != 0:
            raise SystemExit(f"reference {t['id']} failed: rc={r['rc']} {r['stderr']}")
        out[t["id"]] = r
    return out


def score(task_id: str, program: str, exp: dict) -> dict:
    v = validate(program)
    r = run(program)
    e = exp[task_id]
    correct = r["rc"] == 0 and r["stdout"] == e["stdout"] and r["files"] == e["files"]
    loose = r["rc"] == 0 and _norm(r["stdout"]) == _norm(e["stdout"]) and \
        {k: _norm(c) for k, c in r["files"].items()} == {k: _norm(c) for k, c in e["files"].items()}
    return {
        "task": task_id,
        "valid": v.valid,
        "rules": sorted({x.rule_id for x in v.violations}),
        "errors": [x.message for x in v.violations][:6],
        "correct": correct,
        "correct_loose": loose,
        "rc": r["rc"],
        "stderr": r["stderr"] if r["rc"] != 0 else "",
    }


if __name__ == "__main__":
    exp = expected()
    print(f"{len(exp)} references validate and run")
