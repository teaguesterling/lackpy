"""Recover corpus provenance so the train/eval split can be honest.

The frozen snapshot flattened `session_file` to the constant "corpus", so a session-level
split cannot be made from it. The saved `calls` table still carries `filename`. This joins
the two on the body text and reports what a split has to work with.

Prints aggregates only. Program bodies and descriptions are never printed: they come from
session transcripts and can carry private content.
"""
import collections, hashlib, re, sys
from pathlib import Path
import duckdb

D = Path("~/.local/share/lackpy/corpus").expanduser()
CALLS, SNAP = D / "heredoc_calls-2026-10-18.parquet", D / "heredocs-2026-10-18.parquet"

print("calls columns:", [c[0] for c in duckdb.sql("DESCRIBE SELECT * FROM read_parquet('%s')" % CALLS).fetchall()])
print("snap  columns:", [c[0] for c in duckdb.sql("DESCRIBE SELECT * FROM read_parquet('%s')" % SNAP).fetchall()])

calls = duckdb.sql("SELECT * FROM read_parquet('%s')" % CALLS).fetchall()
cols = [c[0] for c in duckdb.sql("DESCRIBE SELECT * FROM read_parquet('%s')" % CALLS).fetchall()]
ix = {c: i for i, c in enumerate(cols)}
print("\ncalls rows: %d" % len(calls))

fn = ix.get("filename")
sessions = collections.Counter(r[fn] for r in calls) if fn is not None else {}
print("distinct session files: %d" % len(sessions))
projects = collections.Counter(Path(f).parent.name for f in sessions)
print("distinct projects: %d" % len(projects))
cnt = sorted(sessions.values(), reverse=True)
if cnt:
    print("calls per session: max %d  median %d  min %d" % (cnt[0], cnt[len(cnt)//2], cnt[-1]))
    print("top 5 sessions hold %.0f%% of calls" % (100.0*sum(cnt[:5])/sum(cnt)))
print("\nprograms per project (the unit a split should respect):")
for p, c in projects.most_common(12):
    print("   %-46s %5d sessions" % (p[:46], c))

d = ix.get("descr")
if d is not None:
    desc = [r[d] or "" for r in calls]
    lens = sorted(len(x.split()) for x in desc)
    print("\nintent (Bash description) quality:")
    print("   present %d of %d" % (sum(1 for x in desc if x.strip()), len(desc)))
    print("   words: median %d  p90 %d  max %d" % (lens[len(lens)//2], lens[int(len(lens)*0.9)], lens[-1]))
    print("   distinct: %d (%.0f%% unique)" % (len(set(desc)), 100.0*len(set(desc))/len(desc)))
    rep = collections.Counter(desc)
    print("   most repeated intent appears %d times" % rep.most_common(1)[0][1])
    vague = sum(1 for x in desc if len(x.split()) <= 4)
    print("   4 words or fewer: %d (%.0f%%)  <- these cannot specify a program" % (vague, 100.0*vague/len(desc)))
