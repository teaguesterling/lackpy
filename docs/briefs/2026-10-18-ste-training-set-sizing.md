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
