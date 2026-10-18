# Right-sizing the lackpy-STE training set

**Status:** measured. The numbers come from `scripts/corpus/rightsize.py` against the
frozen snapshot `heredocs-2026-10-18.parquet` (7,003 rows, sha `fc7ba975c23d`),
6,984 of which parse. The validator is this branch at `6855fa1`.

**Why:** the usability eval shows Qwen3-30B-A3B writes correct programs that are not
valid lackpy: 26 of 30 correct, 8 of 30 valid. The gap is conformance, not capability.
A conformance gap is a training target, because a deterministic checker scores it.

---

## 1. How much of the corpus is reachable

Each layer below is a transform that someone builds once. A layer gets credit only when
it removes the **last** block on a program. The curve therefore shows what each layer
adds on top of the layers above it.

| Layer | Admits | Share | Delta |
|---|---|---|---|
| L0 today, unmodified | 608 | 8.7% | |
| L1 delete imports of approved modules | 609 | 8.7% | +1 |
| L2 `open(...)` becomes `read_file` or `Path.read_text` | 2,293 | 32.8% | **+1,684** |
| L3 mechanical statement splits | 2,395 | 34.3% | +102 |
| L4 templated assert messages | 4,582 | **65.6%** | **+2,187** |

**4,582 programs are reachable by deterministic transform.** Two layers carry the
result. The `open` rewrite adds 1,684 programs. The assert message adds 2,187.

L1 adds one program. That is not an error in the layer. A program that imports an
approved module almost always also uses `open`, or imports an unapproved module.

## 2. What blocks the other 2,402

These need a **decision**, not a transform.

| Blocker | Programs | Share |
|---|---|---|
| `safety.import` (unapproved module) | 1,834 | 26.3% |
| `safety.forbidden-name` | 1,211 | 17.3% |
| `safety.module-attr` (for example `json.load`) | 658 | 9.4% |
| `safety.except-base` (`except Exception`) | 366 | 5.2% |
| `safety.unknown-call` | 295 | 4.2% |
| `safety.node` (`while`, `MatMult`, `Pow`) | 290 | 4.2% |

The unapproved modules, counted by the programs that import them:

| Module | Programs | Share | Note |
|---|---|---|---|
| `glob` | 563 | 8.1% | `find_files` already covers this |
| `os` | 388 | 5.6% | mostly `os.path`, which `pathlib` covers |
| `io` | 346 | 5.0% | |
| `sys` | 244 | 3.5% | |
| `duckdb` | 210 | 3.0% | the design already removed DuckDB from v1 |
| `subprocess` | 185 | 2.6% | out of scope: the sandbox forbids it |
| `time` | 160 | 2.3% | |
| `urllib` | 148 | 2.1% | out of scope: no network |

Three allowlist decisions are worth pricing, because each maps onto something lackpy
already has: `glob` to `find_files`, `os.path` to `pathlib`, and `json.load` to
`json.loads` over `read_file`. Their combined value is not yet measured. The counts
above overlap, so the decisions do not add up.

## 3. What one training pair is

A pair is one record:

```
intent    the declared goal, in the frontmatter format of design section 4.1
program   a program that the validator admits
checks    the assertions the program makes about its own effect
```

The corpus supplies the program and a weak intent. Every heredoc call carries the Bash
`description` field. Those intents are loose: "Fix registrations and rebuild" does not
say which registrations, or what makes them fixed. A pair therefore needs an intent that
is rewritten to the frontmatter format. That rewrite is the one step in this pipeline
that a model must do, and it is the step that needs review.

## 4. The gates

A candidate pair enters the training set only if it passes all four.

1. **Conformance.** `validate()` admits the program. Deterministic, no model.
2. **Behaviour.** The transformed program produces the same result as the original on
   the original inputs. This gate protects the L2, L3 and L4 transforms. Without it, a
   rewrite that changes behaviour enters the corpus as a correct answer.
3. **Intent fidelity.** The declared intent and the intent derived from the AST agree.
   Design section 1.2 defines this comparison.
4. **No leak.** No training pair shares a request text or a program with the eval set.

Gate 2 is the one to build first. Gates 1 and 4 are cheap. Gate 3 needs the derivation.

## 5. Split discipline

Split by **session**, and then by **identifier**. A split by file is not enough. In a
related corpus, the same function name appeared on both sides of a split, so the
request text and the answer text were identical across it. An automated overlap check
found 18 such pairs, after a split by file looked correct.

## 6. Size, and what size buys

4,582 reachable programs is a large training set for this task. For comparison, 491
verified pairs moved a 0.8B model from 10.6% to 84.1% on a held-out construct suite in
a related project. That effect replicated across six seeds at t=74.

Do not read that as a promise here. It states that a small, verified, construct-focused
set can move a small model a long way, when the gap is vocabulary rather than capability.
The eval says this gap is vocabulary: 26 of 30 correct, 8 of 30 valid.

## 7. Limits

  - **The eval is 30 tasks.** The binomial standard error at n=30 is about 9 points.
    The current 8/30 against 30/30 is far outside that. A later comparison of two
    trained models will not be. Grow the eval before training, not after.
  - **Seed variance is not known for this task.** In the related project, six seeds of
    one recipe spanned 11.6 points on the primary metric, and every arm had used the
    best of those seeds. Measure the seed variance before any comparison.
  - **A conforming program can still be wrong.** This set teaches form. Gate 2 is what
    keeps form from replacing behaviour.
  - **The corpus is one author's history.** It comes from about 12 repositories, mostly
    DuckDB extensions and evaluation harnesses. lackpy's users may write different
    programs. This is an assumption, and it is not measured.

---

## 8. The split, and what the snapshot lost

**The frozen snapshot cannot produce a split.** Its `session_file` column holds the constant
`"corpus"` for all 7,003 rows, so provenance did not survive the write. The saved calls table
`heredoc_calls-2026-10-18.parquet` still holds `filename`. `scripts/corpus/provenance.py`
rejoins the two on the body text and recovers it: **32 sessions across 20 projects**.

Recommend that `extract.py` keep `filename` in the snapshot. A later re-extract cannot recover
it, because Claude Code prunes old sessions; the pruned history now yields 3,104 bodies against
the frozen 7,003.

**The split unit is the project.** Programs inside one session patch the same files repeatedly.
Sessions inside one project share those files. Any split below the project leaks.

`scripts/corpus/split.py` enumerates every subset of 3 to 8 projects, keeps those inside the
size band, and picks the one whose family shares sit closest to the corpus. Two constraints
decide the result, and the second was added after the first attempt failed:

  - Size between 17% and 25%.
  - **No project may hold more than 40% of the eval set.** Without this, the best-balanced
    answer put 1,021 of 1,748 eval programs (58%) in one project. That eval measures the
    project, not the corpus.

The chosen eval set: 7 projects, 1,202 programs, 17.2%.

| Family | Train | Eval | Eval share |
|---|---|---|---|
| patch | 3,646 | 885 | 19.5% |
| assert | 2,430 | 497 | 17.0% |
| json | 1,283 | 176 | 12.1% |
| scan | 743 | 106 | 12.5% |
| regex | 493 | 139 | 22.0% |
| shell | 231 | 34 | 12.8% |

The worst family deviates 5.1 points from its corpus share. The balance-first answer reached
3.8 points, and it is the one this rejects on the dominance rule.

**Report per family, never as one number.** The imbalance above stops mattering when each
family carries its own result. It is also the only way to see this:

| Family | Eval n | SE at p=0.5 | 95% CI | Verdict |
|---|---|---|---|---|
| patch | 885 | 1.7 | ±3.3 | usable |
| assert | 497 | 2.2 | ±4.4 | usable |
| json | 176 | 3.8 | ±7.4 | usable |
| scan | 106 | 4.9 | ±9.5 | marginal |
| regex | 139 | 4.2 | ±8.3 | marginal |
| shell | 34 | 8.6 | ±16.8 | **too small** |

`shell` cannot support a claim at any effect size worth acting on. Either drop it from the
reported families, or hold out more of it. Do not report it beside the others as if it were
equal evidence.

## 9. Private content

`split.py` screens every body by pattern. **1,188 of 7,003 programs (17%) carry a flag.**

| Flag | Programs |
|---|---|
| home path | 829 |
| key-like word | 317 |
| private IP | 98 |
| long hex | 64 |
| email | 24 |

Most of the home-path flags are benign. The key-like flags need a read. Decision D4 requires a
human review of every committed fixture. This screen narrows that read; it does not replace it.
No body and no intent is printed by any script here, for the same reason.

## 10. What is ready, and what waits

Ready now, and independent of the language work:

  - the reachability measurement (section 1)
  - the split, with its manifest at `~/.local/share/lackpy/rightsize/split.json`
  - the private-content screen
  - the per-family power table, which sets what the eval can claim

Waits for build-order steps 2 to 7:

  - the transforms themselves, which need the final rule set
  - gate 2, behavioural equivalence, which needs the transformed program
  - intent rewriting to the frontmatter format, which needs the parser

---

## 11. Scrubbing, and what it does not buy

`scripts/corpus/scrub.py` replaces private content with stable generic values. Stable means
one input maps to one output everywhere, so a path used twice in a program stays one path.

**Result: 1,777 of 7,003 programs changed (25%). No program's syntax broke.** The syntax check
is part of the scrub: a replacement that stops a body parsing has changed the program, not
redacted it.

| Replacement | Count |
|---|---|
| high-entropy token | 1,243 |
| home path | 985 |
| private IP | 147 |
| long hex | 137 |
| email | 37 |

**The `key-like` flags were noise.** The screen in section 9 flagged 317 programs for words such
as `token` and `secret`. The scrub matches a key-like name *assigned a string literal*, and it
replaced **zero**. Those 317 were variable names, dictionary keys and comments.

**The pattern scrub alone was not enough, and a second detector proved it.** After replacing
every known pattern, a Shannon-entropy scan found **1,066 programs (15.2%) holding a long
high-entropy token that no pattern had named** — more programs than the scrub had changed. A
commit SHA, a content hash and an API key are the same thing to an entropy test.

So the scrub also neutralises any token of 32 characters or more whose entropy is 4.2 or
higher, replacing it with a placeholder of the same length. For training this costs nothing: a
real hash teaches a model nothing that a fixed one does not, and the same length keeps the
program's shape and any slicing of the literal.

After that pass the high-entropy residual is **0**. 182 programs (2.6%) still hold a
base64-shaped run, all of low entropy, which is to say repetitive and structured rather than
random. They are the remaining read.

**A measurement warning, from making the mistake.** The first residual scan after this pass
reported 1,213 base64-like programs, up from 180. The placeholder is a run of `X`, which is
itself base64-shaped, so the detector was counting the scrub's own output. A detector that runs
after a transform must exclude what the transform writes.

### Sequencing: scrub last

Scrubbing is a semantic transform. A scrubbed program does not read the file the original read.
Gate 2 of section 4 compares a transformed program against the original on the original inputs,
so the two steps conflict:

  - Run gate 2 on **unscrubbed** pairs, then scrub. Correct, and it needs the unscrubbed corpus
    present while the gate runs.
  - Or scrub first and run gate 2 on scrubbed inputs on both sides. Also correct, and it tests a
    program nobody wrote.

Prefer the first. It keeps the behavioural check on the program as the author wrote it.

### What scrubbing still does not buy

It does not make the corpus publishable. It removes what two detectors can name, and the second
one found a population the first missed. A third detector would probably find a smaller one
again. Decision D4 requires a human read of every committed fixture; about 50 fixtures is a
tractable read, and the scrub narrows what that read is looking for.
