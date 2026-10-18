"""Measure how much of the frozen heredoc corpus today's lackpy validator admits.

Usage: python scripts/corpus/coverage.py [--snapshot PATH] [--json]
Exit 3 on "snapshot mismatch" (default snapshot only).
Exit 2 with "corpus not found: <path>" on stderr when the snapshot is missing.
Exit 4 with "unknown rule id" on stderr when a violation names an id that is not
in the rule registry (and is not "extra"). The figures are not printed then.

"admitted" counts programs where ValidationResult.valid is true.
"by_rule" counts, per Violation.rule_id, the programs with at least one such violation.
"structural" counts programs whose every violation is vocabulary: VOCAB_IDS,
plus ``.format`` (ruled vocabulary: its replacement is an f-string).
"""
import argparse
import ast
import collections
import hashlib
import json
import sys
import warnings
from pathlib import Path

DEFAULT = Path("~/.local/share/lackpy/corpus/heredocs-2026-10-18.parquet").expanduser()
# Identity of the canonical snapshot: distinct non-empty bodies, sorted, each followed by a NUL byte, SHA-256.
CANON_ROWS = 7003
CANON_SHA256 = "fc7ba975c23d207ade7cd072ec71719a065680ba5f5e3ed62f33232844c9672e"
KIT = {"read_file", "find_files", "write_file", "edit_file"}
VOCAB_IDS = {"safety.forbidden-name", "safety.unknown-call", "safety.import", "safety.module-attr"}
# The baseline counted ``.format`` as vocabulary (the replacement is an f-string).
FORMAT_MESSAGE = "Forbidden attribute access: '.format'"


def is_vocabulary(v):
    if v.rule_id in VOCAB_IDS:
        return True
    if v.rule_id == "safety.attr-denied":
        return v.message.split(" at line")[0] == FORMAT_MESSAGE
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, default=DEFAULT)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=SyntaxWarning)  # corpus bodies hold stray \$ and \`
    snap = args.snapshot.expanduser()
    if not snap.exists():
        print(f"corpus not found: {snap}", file=sys.stderr)
        return 2
    import duckdb
    from lackpy.lang import RULES, validate
    known_ids = {r.id for r in RULES} | {"extra"}

    bodies = [r[0] for r in duckdb.connect().execute(
        f"SELECT body FROM read_parquet('{snap}') WHERE body <> ''").fetchall()]
    h = hashlib.sha256()
    for b in sorted(bodies):
        h.update(b.encode() + b"\0")
    rows, sha = len(bodies), h.hexdigest()
    if snap == DEFAULT and (rows, sha) != (CANON_ROWS, CANON_SHA256):
        print(f"snapshot mismatch: expected {CANON_ROWS}/{CANON_SHA256[:12]}, got {rows}/{sha[:12]}",
              file=sys.stderr)
        return 3
    results = []
    for b in bodies:
        try:
            ast.parse(b)
        except SyntaxError:
            continue
        results.append(validate(b, allowed_names=KIT))
    n = len(results)
    if n == 0:
        print(f"corpus empty: no parsable bodies in {snap}", file=sys.stderr)
        return 2
    unknown = sorted({v.rule_id for r in results for v in r.violations} - known_ids)
    if unknown:
        print(f"unknown rule id: {', '.join(unknown)}", file=sys.stderr)
        return 4
    admitted = sum(1 for r in results if r.valid)
    structural = sum(1 for r in results if all(is_vocabulary(v) for v in r.violations))
    by_rule = collections.Counter(i for r in results for i in {v.rule_id for v in r.violations})
    out = {
        "snapshot_rows": rows,
        "snapshot_sha256": sha,
        "parsed": n,
        "admitted": admitted,
        "admitted_pct": round(100 * admitted / n, 1),
        "structural": structural,
        "structural_pct": round(100 * structural / n, 1),
        "by_rule": dict(sorted(by_rule.items(), key=lambda kv: (-kv[1], kv[0]))),
    }
    if args.json:
        print(json.dumps(out, indent=1))
    else:
        print(f"snapshot {rows}/{sha[:12]}")
        print(f"parsed {n}; admitted {admitted} ({out['admitted_pct']}%); "
              f"structural {structural} ({out['structural_pct']}%)")
        for k, v in list(out["by_rule"].items())[:20]:
            print(f"  {v:6d}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
