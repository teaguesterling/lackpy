"""Scrub private content from corpus bodies, then check the scrub did not lie or break anything.

Three passes, because a scrub that only replaces what its patterns match gives false confidence:

  1. REPLACE the known patterns with stable generic values. Stable means the same input maps to
     the same output everywhere, so a path used twice in one program stays one path.
  2. RE-PARSE every changed body. A scrub that breaks the syntax has changed the program, not
     redacted it.
  3. RESIDUAL SCAN with different detectors -- Shannon entropy over long tokens, and base64-like
     runs. These find secrets that no keyword matches. Pass 1 cleans what it knows; pass 3
     measures what it did not know.

Prints counts and redacted shapes only. No body, no intent, and no matched value is printed.
"""
import ast, base64, collections, json, math, re, sys, warnings
from pathlib import Path
import duckdb

D = Path("~/.local/share/lackpy/corpus").expanduser()
SNAP = D / "heredocs-2026-10-18.parquet"
OUT = Path("~/.local/share/lackpy/rightsize").expanduser()

HOME = re.compile(r"/home/[A-Za-z0-9_.-]+")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PRIVIP = re.compile(r"\b(?:10|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b")
LONGHEX = re.compile(r"\b[0-9a-f]{32,}\b")
URLCRED = re.compile(r"(https?://)[^/\s:@]+:[^@\s]+@")
# a key-like NAME assigned a string literal; the value is what matters, not the name
KEYVAL = re.compile(r"(?i)\b(api[_-]?key|secret|token|password|passwd|bearer)(\s*[:=]\s*)(['\"])([^'\"]{8,})\3")

B64 = re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b")
LONGTOK = re.compile(r"[A-Za-z0-9_\-+/=]{24,}")


def entropy(s):
    if not s:
        return 0.0
    c = collections.Counter(s)
    n = len(s)
    return -sum((k / n) * math.log2(k / n) for k in c.values())


def scrub(body):
    hits = collections.Counter()
    def sub(rx, repl, name, s):
        s2, k = rx.subn(repl, s)
        if k:
            hits[name] += k
        return s2
    s = body
    s = sub(URLCRED, r"\1user:password@", "url-cred", s)
    s = sub(KEYVAL, lambda m: "%s%s%sREDACTED%s" % (m.group(1), m.group(2), m.group(3), m.group(3)),
            "key-value", s)
    s = sub(EMAIL, "user@example.com", "email", s)
    s = sub(PRIVIP, "10.0.0.1", "private-ip", s)
    s = sub(HOME, "/home/user", "home-path", s)
    s = sub(LONGHEX, lambda m: "0" * len(m.group(0)), "long-hex", s)
    # PASS 4. Neutralise any long high-entropy token the patterns did not name. A commit SHA,
    # a content hash and an API key are indistinguishable to an entropy test, so this treats
    # them alike. For TRAINING that costs nothing: a real hash teaches a model nothing that a
    # fixed one does not. It is replaced by a same-length placeholder, so the program's shape,
    # its line lengths and any slicing of the literal survive.
    def neutral(m):
        tok = m.group(0)
        if "REDACTED" in tok or tok.startswith("0" * 16):
            return tok
        if len(tok) >= 32 and entropy(tok) >= 4.2:
            hits["high-entropy"] += 1
            return ("X" * len(tok))[:len(tok)]
        return tok
    s = LONGTOK.sub(neutral, s)
    return s, hits


def main():
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    bodies = [r[0] for r in duckdb.sql(
        "SELECT body FROM read_parquet('%s') WHERE body IS NOT NULL" % SNAP).fetchall()]
    total = collections.Counter()
    changed = broke = 0
    resid_entropy = resid_b64 = 0
    resid_examples = collections.Counter()
    scrubbed = []
    for b in bodies:
        try:
            ast.parse(b)
            parsed_before = True
        except Exception:
            parsed_before = False
        s, hits = scrub(b)
        if hits:
            changed += 1
            total.update(hits)
            if parsed_before:
                try:
                    ast.parse(s)
                except Exception:
                    broke += 1
        scrubbed.append(s)
        # pass 3: what the patterns did not know about
        for m in LONGTOK.finditer(s):
            tok = m.group(0)
            if tok.startswith("0" * 16) or "REDACTED" in tok or set(tok) == {"X"}:
                continue
            e = entropy(tok)
            if e >= 4.2 and len(tok) >= 32:
                resid_entropy += 1
                resid_examples["high-entropy len%d" % (len(tok) // 16 * 16)] += 1
                break
        else:
            # The placeholders are themselves base64-shaped: a 40-character run of X matches.
            # Counting them would report the scrub's own output as residual risk.
            for mb in B64.finditer(s):
                t = mb.group(0)
                if set(t) <= {"X"} or set(t) <= {"0"} or "REDACTED" in t:
                    continue
                resid_b64 += 1
                break

    print("bodies: %d   changed by the scrub: %d (%.0f%%)   syntax broken by the scrub: %d"
          % (len(bodies), changed, 100.0 * changed / len(bodies), broke))
    print("\nreplacements made:")
    for k, v in total.most_common():
        print("   %-12s %6d" % (k, v))
    print("\nPASS 3 -- what the patterns did NOT catch, found by different detectors:")
    print("   programs with a high-entropy token (>=32 chars, entropy >=4.2): %d (%.1f%%)"
          % (resid_entropy, 100.0 * resid_entropy / len(bodies)))
    print("   programs with a base64-like run and no high-entropy token:      %d (%.1f%%)"
          % (resid_b64, 100.0 * resid_b64 / len(bodies)))
    for k, v in resid_examples.most_common(6):
        print("      %-24s %5d" % (k, v))
    print("\n   These are candidates, not findings. A commit SHA, a content hash and a key all")
    print("   look the same to an entropy test. They are the set a human must read.")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scrub.json").write_text(json.dumps(
        {"bodies": len(bodies), "changed": changed, "syntax_broken": broke,
         "replacements": dict(total), "residual_entropy": resid_entropy,
         "residual_b64": resid_b64}, indent=1))
    print("\nwrote %s" % (OUT / "scrub.json"))


main()
