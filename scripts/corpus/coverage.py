"""Measure how much of the frozen heredoc corpus today's lackpy validator admits.

Usage: python scripts/corpus/coverage.py [--snapshot PATH] [--json]
Exit 3 on "snapshot mismatch" (default snapshot only).
Exit 2 with "corpus not found: <path>" on stderr when the snapshot is missing.

Until the rule registry exists, validate() error strings map to keys per
Appendix A of docs/superpowers/specs/2026-10-18-lackpy-ste-design.md.
"structural" ignores the vocabulary keys: name:, import:, call:, attr:format.
"""
import argparse
import ast
import collections
import hashlib
import json
import re
import sys
import warnings
from pathlib import Path

DEFAULT = Path("~/.local/share/lackpy/corpus/heredocs-2026-10-18.parquet").expanduser()
# Identity of the canonical snapshot: distinct non-empty bodies, sorted, each followed by a NUL byte, SHA-256.
CANON_ROWS = 7003
CANON_SHA256 = "fc7ba975c23d207ade7cd072ec71719a065680ba5f5e3ed62f33232844c9672e"
KIT = {"read_file", "find_files", "write_file", "edit_file"}
VOCAB_PREFIXES = ("name:", "import:", "call:")
VOCAB_KEYS = {"attr:format"}


def keys_for(body, tree, validate):
    res = validate(body, allowed_names=KIT)
    if res.valid:
        return set()
    imports_by_line = collections.defaultdict(list)
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imports_by_line[n.lineno] += [a.name.split(".")[0] for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            imports_by_line[n.lineno].append((n.module or ".").split(".")[0])
    keys = set()
    for e in res.errors:
        m = re.match(r"Forbidden AST node: (\w+)(?: at line (\d+))?", e)
        if m:
            if m.group(1) in ("Import", "ImportFrom") and m.group(2):
                keys |= {f"import:{x}" for x in imports_by_line[int(m.group(2))]}
            elif m.group(1) not in ("alias", "ExceptHandler"):
                keys.add(f"node:{m.group(1)}")
            continue
        m = re.match(r"Forbidden name: '(\w+)'", e)
        if m:
            keys.add(f"name:{m.group(1)}")
            continue
        m = re.match(r"Unknown function: '(\w+)'", e)
        if m:
            keys.add(f"call:{m.group(1)}")
            continue
        m = re.match(r"Forbidden attribute access: '\.(\w+)'", e)
        if m:
            a = m.group(1)
            keys.add("attr:_underscore" if a.startswith("_") else f"attr:{a}")
            continue
        if e.startswith("Lambda"):
            keys.add("rule:lambda-not-key")
        elif e.startswith("For-loop"):
            keys.add("rule:for-iter-shape")
        elif e.startswith("Unsupported call target"):
            keys.add("rule:call-target")
        else:
            keys.add("other:" + e[:40])
    return keys


def is_structural(keys):
    return not {k for k in keys if not k.startswith(VOCAB_PREFIXES) and k not in VOCAB_KEYS}


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
    from lackpy.lang import validate

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
    progs = []
    for b in bodies:
        try:
            tree = ast.parse(b)
        except SyntaxError:
            continue
        progs.append(keys_for(b, tree, validate))
    n = len(progs)
    if n == 0:
        print(f"corpus empty: no parsable bodies in {snap}", file=sys.stderr)
        return 2
    admitted = sum(1 for k in progs if not k)
    structural = sum(1 for k in progs if is_structural(k))
    by_rule = collections.Counter(k for ks in progs for k in ks)
    out = {
        "snapshot_rows": rows,
        "snapshot_sha256": sha,
        "parsed": n,
        "admitted": admitted,
        "admitted_pct": round(100 * admitted / n, 1),
        "structural": structural,
        "structural_pct": round(100 * structural / n, 1),
        "by_rule": dict(by_rule.most_common()),
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
