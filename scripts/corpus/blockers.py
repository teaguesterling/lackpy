"""Rank the non-vocabulary rules by how many corpus programs each blocks ALONE.

Usage: python scripts/corpus/blockers.py [--snapshot PATH] [--top N] [--json]

"blocks alone" (structural view): the program's only non-vocabulary violations
come from that one rule. Loosening the rule would move the program into the
structural count. "blocks alone (strict)": the program has no other violation
of any kind, vocabulary included, so loosening would admit it.
Uses the same snapshot check and vocabulary definition as coverage.py.
"""
import argparse
import ast
import collections
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import coverage as cov  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, default=cov.DEFAULT)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    import duckdb
    from lackpy.lang import validate
    snap = args.snapshot.expanduser()
    if not snap.exists():
        print(f"corpus not found: {snap}", file=sys.stderr)
        return 2
    bodies = [r[0] for r in duckdb.connect().execute(
        f"SELECT body FROM read_parquet('{snap}') WHERE body <> ''").fetchall()]
    if not cov.check_identity(bodies, snap)[2]:
        return 3
    total = collections.Counter()
    alone = collections.Counter()
    strict = collections.Counter()
    nonvocab_sets = []
    parsed = 0
    for b in bodies:
        try:
            ast.parse(b)
        except SyntaxError:
            continue
        parsed += 1
        res = validate(b, allowed_names=cov.KIT)
        nv = {v.rule_id for v in res.violations if not cov.is_vocabulary(v)}
        every = {v.rule_id for v in res.violations}
        nonvocab_sets.append(nv)
        for r in nv:
            total[r] += 1
        if len(nv) == 1:
            (r,) = nv
            alone[r] += 1
            if every == nv:
                strict[r] += 1
    top = [r for r, _ in sorted(total.items(), key=lambda kv: (-kv[1], kv[0]))[:args.top]]
    structural_now = sum(1 for s in nonvocab_sets if not s)
    structural_if = sum(1 for s in nonvocab_sets if not (s - set(top)))
    out = {
        "parsed": parsed,
        "structural_now": structural_now,
        "structural_if_top_dropped": structural_if,
        "rules": [{"rule": r, "programs": total[r], "alone": alone[r], "alone_strict": strict[r]} for r in top],
    }
    if args.json:
        print(json.dumps(out, indent=1))
        return 0
    print(f"parsed {parsed}; structural now {structural_now}; structural if all top {args.top} dropped {structural_if}")
    print(f"{'rule':34s} {'programs':>8s} {'alone':>7s} {'alone(strict)':>14s}")
    for x in out["rules"]:
        print(f"{x['rule']:34s} {x['programs']:8d} {x['alone']:7d} {x['alone_strict']:14d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
