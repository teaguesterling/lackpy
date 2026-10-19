"""Select fixture candidates from the frozen corpus, for human review.

Writes NN-<family>.py files OUTSIDE git. Teague reviews them for private content
before anything is copied into tests/fixtures/corpus/.
Shape regexes follow section 2.2 of the design. Selection is deterministic:
the shortest bodies of 8 to 60 lines that match, one per distinct session first.

Usage: python scripts/corpus/select_fixtures.py [--snapshot PATH] [--out DIR]
"""
import argparse
import ast
import re
import sys
import warnings
from pathlib import Path

import duckdb

DEFAULT = Path("~/.local/share/lackpy/corpus/heredocs-2026-10-18.parquet").expanduser()
OUT = Path("~/.local/share/lackpy/corpus/fixture-candidates").expanduser()

# (family, count, predicate over body text and tree)
FAMILIES = [
    ("patch", 15, lambda s, t: re.search(r"\.replace\(", s) and re.search(r"read_text|\.read\(", s)
        and re.search(r"write_text|\.write\(", s)),
    ("json", 10, lambda s, t: re.search(r"json\.loads?|jsonl|json\.load", s)),
    ("scan", 10, lambda s, t: re.search(r"\.glob\(|\.rglob\(|os\.walk|glob\.glob", s)),
    ("regex", 5, lambda s, t: re.search(r"\bre\.(sub|search|findall|match|compile|finditer)", s)),
    ("try", 5, lambda s, t: any(isinstance(n, ast.Try) for n in ast.walk(t))),
    ("def", 5, lambda s, t: any(isinstance(n, ast.FunctionDef) for n in ast.walk(t))),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, default=DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    snap = args.snapshot.expanduser()
    if not snap.exists():
        print(f"corpus not found: {snap}", file=sys.stderr)
        return 2
    rows = duckdb.connect().execute(
        f"SELECT session_file, body FROM read_parquet('{snap}') WHERE body <> ''").fetchall()
    parsed = []
    for sess, body in rows:
        n = body.count("\n") + 1
        if not 8 <= n <= 60:
            continue
        try:
            parsed.append((n, body, sess, ast.parse(body)))
        except SyntaxError:
            pass
    parsed.sort(key=lambda x: (x[0], x[1]))
    args.out.mkdir(parents=True, exist_ok=True)
    used, idx = set(), 0
    for fam, want, pred in FAMILIES:
        hits = [p for p in parsed if p[1] not in used and pred(p[1], p[3])]
        picked, seen = [], set()
        for p in hits:  # one per session first, then fill
            if p[2] not in seen:
                picked.append(p); seen.add(p[2])
        picked += [p for p in hits if p not in picked]
        for n, body, sess, _ in picked[:want]:
            idx += 1
            used.add(body)
            (args.out / f"{idx:02d}-{fam}.py").write_text(body + "\n")
        got = min(want, len(picked))
        print(f"{fam}: {got}/{want}")
    print(f"wrote {idx} candidates to {args.out}")
    return 0 if idx == 50 else 1


if __name__ == "__main__":
    sys.exit(main())
