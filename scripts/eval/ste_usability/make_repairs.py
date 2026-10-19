"""Prepare round 2 for a model that answers through files (haiku subagents).

usage: make_repairs.py RUN_DIR MODEL_LABEL
Copies valid r1 programs to r2 unchanged. For each invalid one, writes a repair
prompt (validator messages and hints included) to RUN_DIR/<label>/r2-prompts/.
Prints the ids that need a repair answer.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prompts import repair_prompt  # noqa: E402
from tasks import TASKS  # noqa: E402

from lackpy.lang import validate  # noqa: E402


def main() -> None:
    run_dir, label = Path(sys.argv[1]), sys.argv[2]
    r1, r2, rp = (run_dir / label / d for d in ("r1", "r2", "r2-prompts"))
    r2.mkdir(parents=True, exist_ok=True)
    rp.mkdir(parents=True, exist_ok=True)
    todo = []
    for t in TASKS:
        f = r1 / f"{t['id']}.py"
        prev = f.read_text() if f.exists() else ""
        v = validate(prev)
        if v.valid:
            shutil.copy(f, r2 / f.name)
            continue
        msgs = [x.message + (f" (hint: {x.hint})" if x.hint else "") for x in v.violations]
        (rp / f"{t['id']}.md").write_text(repair_prompt(t, prev, msgs))
        todo.append(t["id"])
    print(" ".join(todo))


if __name__ == "__main__":
    main()
