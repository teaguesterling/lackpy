"""Re-derive every <id>.py from its saved <id>.reply.txt with the current extract().

usage: reextract.py RUN_DIR
Runs before scoring, so every model's programs come from one extraction rule,
whatever runner version generated them. Writers without reply files (haiku
subagents wrote programs directly) are left alone. Prints what changed.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prompts import extract  # noqa: E402


def main() -> None:
    run_dir = Path(sys.argv[1])
    changed = 0
    for reply in sorted(run_dir.glob("*/r*/*.reply.txt")):
        if reply.parts[-3].startswith(("_", "prompts")):
            continue
        prog = reply.with_name(reply.name.replace(".reply.txt", ".py"))
        new = extract(reply.read_text())
        if not prog.exists() or prog.read_text() != new:
            prog.write_text(new)
            changed += 1
            print("re-extracted", prog.relative_to(run_dir))
    print(f"{changed} program(s) re-extracted")


if __name__ == "__main__":
    main()
