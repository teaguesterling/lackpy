"""Right-size the lackpy-STE training set, measured per program against the CURRENT validator.

Each layer is a transform someone would actually build, ordered cheapest-first. A layer is credited
only when it is the LAST thing blocking a program, so the curve shows what each one buys on top of
the ones before it -- not what it would buy alone.

Classification comes from the AST plus the violation rule_ids, not from message text: the import
rule reports "Forbidden AST node: Import" and never names the module, so message parsing silently
lumped 1,841 programs into one bucket on the first pass.
"""
import ast, collections, inspect, json, sys, warnings
from pathlib import Path
import duckdb

# `lackpy` is a namespace package in two halves: the core runtime in src/ and the lang leaf in
# packages/lackpy-lang/src/. Put both on the path from THIS tree, anchored to __file__ rather than
# ~, so the numbers cannot come from a different checkout on a machine where ~/Projects differs.
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "packages/lackpy-lang/src")]
from lackpy.lang import validate

# Prepending is not proof on its own: which `lackpy` wins depends on the editable installs the
# running interpreter has registered, and those differ per machine. This reads back where the
# validator really came from, so a mismatch stops the run instead of measuring another tree.
_vsrc = inspect.getsourcefile(validate)
if _vsrc is None or not Path(_vsrc).resolve().is_relative_to(ROOT):
    raise SystemExit(
        f"refusing to measure: lackpy.lang came from outside this tree\n"
        f"  resolved: {_vsrc}\n  expected under: {ROOT}"
    )

SNAP = Path("~/.local/share/lackpy/corpus/heredocs-2026-10-18.parquet").expanduser()
KIT = {"read_file", "find_files", "write_file", "edit_file"}
APPROVED = {"pathlib", "json", "re", "collections", "math", "statistics", "hashlib", "datetime"}

# rule_ids a deterministic rewriter can clear, with the transform that clears each
MECHANICAL = {
    "style.io-position":  "hoist the I/O call into its own statement",
    "style.call-nesting": "hoist the inner call into its own statement",
    "style.lambda-assign": "name = lambda -> def",
    "safety.attr-denied": ".format -> f-string",
}
TEMPLATABLE = {"style.assert-message": "add the message the assert already implies"}


def modules_of(tree):
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            mods.add(n.module.split(".")[0])
    return mods


def main():
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    rows = duckdb.sql("SELECT body FROM read_parquet('%s')" % SNAP).fetchall()
    recs = []
    for (b,) in rows:
        if not b:
            continue
        try:
            tree = ast.parse(b)
        except Exception:
            continue
        try:
            res = validate(b, allowed_names=KIT)
        except Exception:
            continue
        rids = collections.Counter(v.rule_id for v in res.violations)
        recs.append({"valid": res.valid, "rids": rids, "mods": modules_of(tree),
                     "uses_open": any(isinstance(n, ast.Name) and n.id == "open" for n in ast.walk(tree))})
    N = len(recs)
    print("programs analysed: %d\n" % N)

    def remaining(r, cleared_rids, drop_imports, fix_open):
        """rule_ids still blocking this program after the given transforms."""
        out = collections.Counter()
        for rid, k in r["rids"].items():
            if rid in cleared_rids:
                continue
            if rid == "safety.import" and drop_imports and r["mods"] <= APPROVED:
                continue
            if rid == "safety.forbidden-name" and fix_open and not (r["mods"] - APPROVED):
                # `open` is the dominant forbidden name; credit the layer only when it is
                # plausibly the only one, i.e. the program imports nothing unapproved.
                continue
            out[rid] += k
        return out

    layers = [
        ("L0  today, unmodified",                                            set(), False, False),
        ("L1  + delete imports of approved modules (lackpy injects them)",   set(), True,  False),
        ("L2  + `open(...)` -> read_file / Path.read_text",                  set(), True,  True),
        ("L3  + mechanical statement splits (io-position, call-nesting, lambda-assign, .format)",
                                                                             set(MECHANICAL), True, True),
        ("L4  + templated assert messages",                                  set(MECHANICAL) | set(TEMPLATABLE), True, True),
    ]
    prev = None
    print("%-86s %7s %7s %8s" % ("layer", "admits", "share", "delta"))
    for name, cleared, di, fo in layers:
        n = sum(1 for r in recs if not remaining(r, cleared, di, fo))
        print("%-86s %7d %6.1f%% %8s" % (name, n, 100.0 * n / N, "" if prev is None else "%+d" % (n - prev)))
        prev = n

    cleared, di, fo = set(MECHANICAL) | set(TEMPLATABLE), True, True
    resid = collections.Counter(); blocked = 0
    for r in recs:
        rem = remaining(r, cleared, di, fo)
        if rem:
            blocked += 1
            for rid in rem:
                resid[rid] += 1
    print("\n%d of %d (%.1f%%) still blocked. These are the DECISIONS, not transforms:" % (blocked, N, 100.0*blocked/N))
    for rid, n in resid.most_common(14):
        print("   %-28s %5d  %5.1f%%" % (rid, n, 100.0 * n / N))
    unapproved = collections.Counter()
    for r in recs:
        for m in (r["mods"] - APPROVED):
            unapproved[m] += 1
    print("\nunapproved modules, by programs that import them:")
    for m, n in unapproved.most_common(12):
        print("   %-14s %5d  %5.1f%%" % (m, n, 100.0 * n / N))
    json.dump({"n": N, "residual": dict(resid), "unapproved": dict(unapproved)},
              open(Path("~/.local/share/lackpy/rightsize/rightsize2.json").expanduser(), "w"), indent=1)


main()
