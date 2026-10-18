"""Score every model/round under a run directory and write results.

usage: score.py RUN_DIR
Writes RUN_DIR/results.json (per program) and prints a summary table.
A missing program file counts as a failure; it is never skipped.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import expected, score  # noqa: E402
from tasks import TASKS  # noqa: E402


def main() -> None:
    run_dir = Path(sys.argv[1])
    exp = expected()
    results, summary = [], []
    for model_dir in sorted(p for p in run_dir.iterdir() if p.is_dir() and not p.name.startswith(("prompts", "_"))):
        for round_dir in sorted(p for p in model_dir.iterdir()
                                if p.is_dir() and re.fullmatch(r"r\d+", p.name)):
            rows = []
            for t in TASKS:
                f = round_dir / f"{t['id']}.py"
                if f.exists():
                    row = score(t["id"], f.read_text(), exp)
                else:
                    row = {"task": t["id"], "valid": False, "rules": ["missing"],
                           "errors": ["no program file"], "correct": False,
                           "correct_loose": False, "rc": None, "stderr": ""}
                row.update(model=model_dir.name, round=round_dir.name, family=t["family"])
                rows.append(row)
            results.extend(rows)
            n = len(rows)
            vc = sum(r["valid"] and r["correct"] for r in rows)
            cnv = sum(r["correct"] and not r["valid"] for r in rows)
            vnc = sum(r["valid"] and not r["correct"] for r in rows)
            rules = Counter(rule for r in rows for rule in r["rules"])
            summary.append(dict(model=model_dir.name, round=round_dir.name, n=n,
                                valid=sum(r["valid"] for r in rows),
                                correct=sum(r["correct"] for r in rows),
                                correct_loose=sum(r["correct_loose"] for r in rows),
                                valid_and_loose=sum(r["valid"] and r["correct_loose"] for r in rows),
                                valid_and_correct=vc, correct_not_valid=cnv,
                                valid_not_correct=vnc, rules=dict(rules.most_common())))
    (run_dir / "results.json").write_text(json.dumps({"summary": summary, "programs": results}, indent=1))
    print(f"{'model':12} {'rnd':4} {'valid':>6} {'correct':>8} {'loose':>6} {'both':>5} {'both~':>6} {'corr!val':>9} {'val!corr':>9}  top rules")
    for s in summary:
        top = ", ".join(f"{k} {v}" for k, v in list(s["rules"].items())[:4])
        print(f"{s['model']:12} {s['round']:4} {s['valid']:>3}/{s['n']} {s['correct']:>5}/{s['n']} "
              f"{s['correct_loose']:>6} {s['valid_and_correct']:>5} {s['valid_and_loose']:>6} {s['correct_not_valid']:>9} {s['valid_not_correct']:>9}  {top}")


if __name__ == "__main__":
    main()
