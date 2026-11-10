"""Choose the eval projects for balance, not by hash alone.

The first attempt ranked projects by hash and took them until the size target passed. One
project holds 17.4% of the corpus, so the eval overshot to 27.9%, and the family mix came out
wrong: eval held MORE `scan` programs than train, and 45% of all `json` work. A split that
over-represents a family makes every later eval number unrepresentative of the corpus.

This enumerates every subset of 3 to 8 projects, keeps those whose size share falls in the
target band, and picks the one whose family shares sit closest to the corpus. Ties break on
the subset hash, so the choice is reproducible.
"""
import collections, hashlib, itertools, json, re
from pathlib import Path
import duckdb

D = Path("~/.local/share/lackpy/corpus").expanduser()
CALLS, SNAP = D / "heredoc_calls-2026-10-18.parquet", D / "heredocs-2026-10-18.parquet"
OUT = Path("~/.local/share/lackpy/rightsize").expanduser()
LO, HI = 0.17, 0.25

FAMILY = [("patch", r"\.replace\("), ("assert", r"\bassert\b"),
          ("json", r"\bjson\.(loads?|dumps?)\b"), ("scan", r"\b(glob|rglob|walk|iterdir)\b"),
          ("regex", r"\bre\.(search|match|sub|findall|compile)\b"), ("shell", r"\bsubprocess\b")]
FAMILY = [(n, re.compile(p)) for n, p in FAMILY]


def main():
    calls = duckdb.sql("SELECT filename, cmd FROM read_parquet('%s')" % CALLS).fetchall()
    body_set = {b[0] for b in duckdb.sql(
        "SELECT body FROM read_parquet('%s') WHERE body IS NOT NULL" % SNAP).fetchall()}
    rx = re.compile(r"(?s)python3? (?:- )?<<-?['\"]?(PY|EOF|PYEOF|PYTHON|PYCODE|END)['\"]?\n(.*?)\n(PY|EOF|PYEOF|PYTHON|PYCODE|END)\b")
    proj_of = {}
    for fn, cmd in calls:
        m = rx.search(cmd or "")
        if m and m.group(2) in body_set:
            proj_of.setdefault(m.group(2), Path(fn).parent.name)

    N = len(proj_of)
    per = collections.Counter(proj_of.values())
    famcount = {n: collections.Counter() for n, _ in FAMILY}
    for b, p in proj_of.items():
        for n, r in FAMILY:
            if r.search(b):
                famcount[n][p] += 1
    corpus_fam = {n: sum(famcount[n].values()) for n, _ in FAMILY}
    projects = sorted(per)

    best = None
    for k in range(3, 9):
        for sub in itertools.combinations(projects, k):
            size = sum(per[p] for p in sub)
            if not (LO * N <= size <= HI * N):
                continue
            # deviation of each family's eval share from its corpus share
            # No single project may dominate the eval set: a majority-one-project eval
            # measures that project, not the corpus.
            if max(per[p] for p in sub) > 0.40 * size:
                continue
            dev = max(abs(sum(famcount[n][p] for p in sub) / corpus_fam[n] - size / N)
                      for n, _ in FAMILY if corpus_fam[n])
            keyh = hashlib.sha256("|".join(sub).encode()).hexdigest()
            cand = (round(dev, 5), keyh, sub, size)
            if best is None or cand < best:
                best = cand
    dev, _, sub, size = best
    evalset = set(sub)
    print("EVAL projects (%d), %d programs, %.1f%% of %d" % (len(sub), size, 100.0*size/N, N))
    for p in sorted(sub, key=lambda q: -per[q]):
        print("   %-48s %5d" % (p[:48], per[p]))
    print("worst family deviation from the corpus share: %.1f points" % (100*dev))

    print("\n%-8s %8s %8s %10s %10s" % ("family", "train", "eval", "eval share", "corpus share"))
    for n, _ in FAMILY:
        e = sum(famcount[n][p] for p in sub); t = corpus_fam[n] - e
        print("%-8s %8d %8d %9.1f%% %11.1f%%" % (n, t, e, 100.0*e/corpus_fam[n], 100.0*size/N))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "split.json").write_text(json.dumps(
        {"eval_projects": sorted(evalset), "n": N, "n_eval": size,
         "eval_share": size / N, "worst_family_deviation": dev,
         "family_eval": {n: sum(famcount[n][p] for p in sub) for n, _ in FAMILY},
         "family_corpus": corpus_fam}, indent=1))
    print("\nwrote %s" % (OUT / "split.json"))


main()
