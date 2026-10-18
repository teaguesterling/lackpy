# lackpy as a controlled language: design

**Status:** approved by Teague on 2026-10-18. Nothing is built yet. Reviewed once by a builder-check subagent; its questions are answered in sections 4.2, 4.3, 5.1 and 7.
**Date:** 2026-10-18
**Origin:** `docs/briefs/2026-10-18-ste-for-python.md` and a brainstorming session
with Teague on 2026-10-18. Where this document and the brief disagree, this
document is the newer decision, and the disagreement is named.

---

## 1. Intent

lackpy becomes a controlled language for Python, in the way ASD-STE100 is a
controlled language for English. The aim is the one STE has. **A reader must not
misread the text.** Here the text is an intent, and the reader is a model that
writes a program from it.

The program side has a second job. Its form is restricted so that a
deterministic walk over its AST can say what the program does. lackpy then
compares that derived intent with the declared intent. No model is in that
comparison.

### 1.1 The unit of restriction

Both the intent and the program are restricted. One vocabulary joins them.

  - The **intent** is the primary restriction. It is what a model reads.
  - The **program** is restricted so that its intent can be derived from the AST.
    The current safety rules stay. They are a separate concern (section 5.1).

### 1.2 Who writes and who reads

The intent is a first-class artifact. It travels with the program, in the
program's frontmatter, and every reader sees it.

| Case | Intent written by | Program written by | What the intent comparison catches |
|---|---|---|---|
| 1. Agent writes both | Claude | Claude | drift between what Claude said and what Claude wrote |
| 2. Delegation | Claude | a sub-agent, often a small model | the sub-agent misreading the intent (the main worry) |
| 3. Human request | a human, free-form, then normalized | a model | the model misreading the human |

The readers are a human, or an auditor agent. They get one bundle: the
declared intent, the program, the derived intent, the differences between the
two intents, and the output.

Case 1 is the weakest check, because one author writes both halves. A
misunderstanding that both halves share will pass. Case 2 is the strongest
check, and it is the main reason to build this.

### 1.3 What success looks like

Claude writes lackpy programs where it writes `python3 - <<EOF` heredocs today.
It calls them mainly through the lackpy MCP tool. The CLI is an acceptable
second path.

---

## 2. Evidence: the heredoc corpus

All numbers in this section are **measured**. The method is in Appendix A.

### 2.1 The corpus

  - Source: assistant `Bash` tool calls in `~/.claude/projects/*/*.jsonl` (top
    level only, not subagent transcripts) whose command matches
    `python3? (- )?<<`.
  - 7,419 calls, from 32 sessions. Python bodies extracted: 7,023. Distinct bodies
    that parse: **6,984**. Body length: median 20 lines, p75 37, p90 60.
  - Every call already has an intent: the Bash `description` field. These
    intents are loose. "Fix registrations and rebuild" does not say which
    registrations, or what makes them fixed.
  - The sessions came from about 12 repositories, mostly DuckDB extensions,
    knowledge-base work and evaluation harnesses. lackpy's own future users may
    differ from that mix. This is an assumption, and it is not measured.

### 2.2 What the programs do

The counts overlap, and they come from regex matches, so they are approximate.

| Shape | Share of bodies |
|---|---|
| guarded patching: read, `.replace`, write | 50% |
| `assert` (mostly `assert s.count(old) == 1`) | 42% |
| read JSON or JSONL | 20% |
| scan files (glob, rglob, walk) | 13% |
| regex | 10% |
| shell-variable interpolation (`$SP`) | 8% |
| shell out (subprocess) | 4% |
| `import duckdb` | 3% |

**Decision (Teague):** v1 covers the activity mix the history shows: patching,
structured reads, scans and reports. It is not DuckDB plus data munging, as the
brief said. DuckDB may not be the best fit for lackpy.

### 2.3 Today's validator against the corpus

`lackpy.lang.validate(body, allowed_names={read_file, find_files, write_file,
edit_file})` **admits 7 of 6,984 programs (0.1%).**

Most of that is vocabulary. Programs use `open`, `pathlib` and `json`, where
lackpy forbids those names today. When I (Claude) ignore vocabulary and count only
structural rejections, **2,550 of 6,984 (36.5%)** pass. The structural blockers:

| Construct rejected today | Programs | Share |
|---|---|---|
| `assert` | 2,821 | 40.4% |
| `for` over anything but a call or a name | 970 | 13.9% |
| `def` | 654 | 9.4% |
| `continue` | 567 | 8.1% |
| `return` (inside a `def`) | 476 | 6.8% |
| `try` | 381 | 5.5% |
| `lambda` outside `key=` | 150 | 2.1% |
| `break` | 136 | 1.9% |
| `&`, `\|` (mostly set algebra) | 170 | 2.4% |
| `global` | 107 | 1.5% |
| underscore attribute | 102 | 1.5% |
| `while` | 78 | 1.1% |

The rejected `for` iterables: tuple literal 483, subscript 311, list literal 250,
`BoolOp` (`x or []`) 64, attribute 50, `BinOp` 42. None of these is a safety
problem. The rule is stricter than its reason.

Top vocabulary blockers: `open` 64.7%, `pathlib` 22.9%, `json` 19.2%, `re` 12.3%,
`glob` 8.1%, `os` 5.6%, `io` 5.0%, `collections` 4.7%.

### 2.4 The brief's STE rules against the corpus

These rules measure the cost of tightening. Section 2.3 measured the cost of the
current rules.

| Proposed rule | Programs in violation |
|---|---|
| no method call on a call result (`open(p).read()`) | 65.5% |
| no call nested in call arguments (`print(len(x))`) | 42.0% |
| more than 30 statements | 4.2% |
| bare or swallowing `except` | 1.6% |
| nested comprehension | 1.4% |
| multi-`for` comprehension | 0.9% |
| `if`/`elif` chain of 3 or more branches | 0.4% |
| attribute chain longer than 3 | 0.0% |
| walrus | 0.0% |
| **conforms to all of the above** | **15.4%** |

Most of the first two rows are I/O idioms, such as `open(p).read()` and
`json.loads(open(p).read())`. Splitting the I/O call into its own statement removes them. The rest are mostly
`print(len(x))`. A reader does not misread that. Section 4.3 uses these numbers
to set limits.

### 2.5 A defect in a builtin tool

`src/lackpy/sources/builtins.py:27` `edit_file` returns `False` when `old_str` is
absent. When `old_str` occurs more than once, it replaces the first occurrence
without a signal. The corpus guards against exactly this with
`assert s.count(old) == 1`, in 26% of programs. The tool swallows the failure
that the authors check for. This breaks the STE rule "preserve modality".

---

## 3. Goals, non-goals, anti-goals

### Goals

  1. An official style guide for lackpy, written in the form of ASD-STE100: a
     numbered rule set (Part 1) and a dictionary (Part 2).
  2. A static validator that enforces the guide. It has two rule tiers, safety
     and style (section 5.1), and it reports `# ste:` waivers.
  3. A `# /// intent` frontmatter block, a template that agents fill, and a
     deterministic derivation of intent from the AST.
  4. An intent comparison, returned by the MCP tool together with the program and
     the output.
  5. Loosen the language until the corpus fits, measured by the corpus harness.
     Today's subset is too constrained (Teague).
  6. A corpus-coverage harness that is committed and can be run again. It is the
     regression metric for every later change to the allowlist.

### Non-goals (v1)

  - Constrained decoding with a grammar (GBNF, `outlines`). See section 7,
    decision D1.
  - General Python. The domain is the corpus mix in section 2.2.
  - Subprocess, network and dynamic import. Together they are under 7% of the
    corpus, and they are the riskiest surface.
  - Proof that a program is correct. See the limits in section 3.1.

### Anti-goals: these make it a failure even if it works

  - **Loosening that opens the sandbox.** No safety rule may be weakened to gain
    coverage. A safety rule is never waivable by `# ste:`.
  - **An intent comparison that passes by default.** If the derivation cannot
    determine a field, the result says "undetermined". It never says "matches".
  - **A guide nobody reads.** The rules must fit on a page that an agent loads
    into its context. The dictionary is data that the validator loads, not prose.
  - **Rewrites for their own sake.** If a corpus idiom is not a misreading risk,
    the guide accepts it, even where the brief's table would forbid it.

### 3.1 Limits, carried from the brief without change

  - **A conforming program can still be wrong.** The intent comparison narrows
    this. It does not remove it. A wrong intent can be matched exactly by a wrong
    program. The program still needs its test.
  - **Over-restriction has a real cost.** Section 2.4 measures part of it. The
    reviewer-cost half needs a human reading the output (section 6, step 8).
  - **The allowlist is the product, and it is maintenance.** v1 derives it from
    the corpus. The harness is how it is maintained.
  - **The escape hatch exists.** `# ste: <construct>, <why>` (section 4.4).

---

## 4. The design

### 4.1 Intent frontmatter

A program carries its intent in a comment block at the top of the program. The
block follows the form of PEP 723 inline script metadata, so the program stays
valid Python and is easy to parse. The contents are TOML.

```python
# /// intent
# goal = "Record the table support in the docx, odt and rtf registry notes."
# reads  = ["src/supported_extensions.cpp"]
# writes = ["src/supported_extensions.cpp"]
# checks = ["each old note occurs exactly once"]
# output = "none"
# ///
from pathlib import Path

path = Path("src/supported_extensions.cpp")
text = path.read_text()
old = '"images and footnotes are not read yet"}'
assert text.count(old) == 1, "the docx note occurs exactly once"
text = text.replace(old, '"... TABLES ARE read ..."}')
path.write_text(text)
```

The fields have two kinds:

  - **Declared, and compared** (`reads`, `writes`, `output`): the AST
    derivation also produces these. The comparison checks them.
  - **Declared, and matched to the code** (`[params]`): one description per
    parameter. See "Parameters" below.
  - **Declared, and displayed** (`goal`, `checks`): the comparison does not check
    them. They are for the human or auditor reader. `goal` must itself pass the
    prose rules in the guide (Part 3, a short STE subset for intents).

The exact field set is the builder's call, within these constraints: each
compared field must be derivable from the AST alone, and each field must have a
measured use in the corpus.

**Parameters** (decision D15, Teague). A program receives parameters through
one dict literal at the top level, named `params`. It holds the default
values. The intent block documents each key:

```python
# /// intent
# goal   = "Count the TODO markers in each C++ file under a directory."
# reads  = ["{root}/*.cpp"]
# writes = []
# output = "printed"
#
# [params]
# root  = { type = "str", doc = "directory to scan, relative to the workspace" }
# limit = { type = "int", doc = "most files to report" }
# ///
from pathlib import Path

params = {"root": "src", "limit": 20}

counts = []
for p in Path(params["root"]).glob("*.cpp"):
    text = p.read_text()
    counts.append((text.count("TODO"), p.name))
top = sorted(counts, reverse=True)[: params["limit"]]
for n, name in top:
    print(name, n)
```

  - The file runs as-is under `python3`, with its defaults.
  - lackpy replaces values in the `params` literal before execution. It never
    adds a key. A passed key that the literal does not hold is an error. A passed
    value whose type differs from the default's type is an error.
  - The keys of `[params]` and the keys of the literal must be the same set.
    The validator checks both directions, so the documentation cannot drift
    from the code.
  - **Each parameter declares a type** (Teague). v1 types: `str`, `int`,
    `float`, `bool`, `list[str]` (my call; extend from demand). Each default
    in the literal must match its declared type, and the validator checks this.
  - **Internally, `[params]` is a `TypedDict`** (Teague). lackpy builds it with
    the functional form, `TypedDict("Params", {"root": str, "limit": int})`,
    and checks every passed value against it before execution. The program
    itself stays a plain dict literal, so it needs no `typing` import.
  - The literal must be at the top level, before any use. Its values must be
    constants. The program must not assign to `params` or to a key of it again.
  - A path that depends on a parameter derives as a pattern over the parameter,
    for example `{root}/*.cpp`. The declared `reads` and `writes` may use the
    same `{name}` form.

**The skeleton** (Teague, 2026-10-18). Agents start from a program skeleton
that lackpy serves. It replaces the kit. The skeleton holds the empty intent
block, an imports section, and a place for the implementation:

```python
# /// intent
# goal   = ""
# reads  = []
# writes = []
# checks = []
# output = "none"          # none | printed | value
#
# [params]                 # per key of the params dict: { type = "...", doc = "..." }
# ///

# TODO: imports. Approved: pathlib (Path), json, re, collections, math,
#       statistics, hashlib, datetime

# TODO: implementation
```

  - **No stubs** (Teague: stubs need the kit machinery that this design
    removes). Every name a program uses comes from Python's builtins, the
    standard library, or the program itself. See section 5.3.
  - **No kit in the agent's view** (Teague: "leave the kit out"). The agent picks
    no profile, no tool set and no `extra_tools`. The existing kit and profile
    machinery in `src/lackpy` can stay internal for now. Removing it is not part
    of this design.
  - **Term.** lackpy already uses "template" for a saved, reusable program (the
    `create` MCP tool, and the templates tier of inference). This design says
    "skeleton" so that the two terms do not collide (my call).
  - **Delivery.** The MCP surface serves the skeleton (a tool or a resource, the
    builder's call), and so does a CLI verb. The `lackpy-delegation` skill shows
    it too. That skill lives outside this repo, in the lackpy plugin, so
    updating it is a separate change.

An agent fills the intent block before it writes the program (case 1), or
before it delegates (case 2). For case 3, a model normalizes the human's
free-form request into the block. In v1, the human reads that block in the
bundle after the run (section 4.2).

### 4.2 Derived intent and comparison

A deterministic pass over the AST produces:

  1. **Effects:** every I/O call on a `Path` (section 4.3), with its path
     resolved to a constant, a pattern over `params` keys, a glob, or
     `undetermined`. The
     method name decides the kind: `read_text` is a read, and `write_text` is a
     write. From these: `reads` and `writes`.
  2. **Checks:** every `assert`, rendered as text.
  3. **Procedure:** one STE-style sentence per top-level statement, from a
     template for each dictionary entry. The derivation also recognizes
     the guarded-patch idiom as one unit: `text = p.read_text()`, then
     `assert text.count(a) == 1, ...`, then `text = text.replace(a, b)`, then
     `p.write_text(text)`. It renders that as "In P, replace the one occurrence
     of A with B. Fail if A does not occur exactly once." This is display only. It is what a reviewer reads instead of
     the code.
  4. **Output:** what the program prints or returns, by shape.

The comparison reports, for each compared field: `match`, `missing`
(declared, not derived), `extra` (derived, not declared), or `undetermined`.
An `extra` write is the most severe result.

**Write policy (decision D2, Teague):**

  - A declared `writes` entry may be a glob. A path matches when the glob,
    resolved against the workspace, contains the normalized path.
  - The derivation carries a glob pattern through the loop variable that
    iterates over its result. `for p in Path("src").glob("*.cpp"):` followed by
    `p.write_text(...)` derives the write `src/*.cpp`.
  - A write whose target is still `undetermined` **blocks the run**. So does an
    `extra` write.
  - Every other difference is reported, and the run continues.

**`output`** (my call) has a closed vocabulary: `none`, `printed`, `value`. The
runner captures both printed text and the last expression. A program that does
both is `printed+value`. The template offers only these words.

**Style violations** (my call): a style-tier violation that has no `# ste:`
waiver **rejects** the program. For cases 2 and 3, the rejection goes into the
existing correction chain, in the same way as a safety rejection. The error text
names the rule id and the approved alternative, so the fixer has something to
act on.

**Case 3** (my call): the step where a human confirms the normalized intent
block before generation is **out of v1**. In v1, case 3 shows the normalized
block in the bundle after the run, and the human reads it there.

### 4.3 The language: what changes

Each change below is my (Claude's) proposal. Section 2 holds its measured cost or
benefit. The builder confirms each change with the harness before it ships.

**Loosen (style tier):**

| Change | Corpus programs unblocked | Note |
|---|---|---|
| `assert <cond>, "<message>"` (message required) | 40.4% | the precondition idiom; loud failure |
| `for` over any expression without a call inside it, plus the current call/name forms | 13.9% | literals, subscripts, attributes |
| top-level `def`, with rules below | 9.4% | with `return` |
| `continue`, `break` | 8.1%, 1.9% | |
| `&`, `\|`, `-` on sets | 2.4% | |
| `lambda` as an argument to an allowed higher-order call (`sorted`, `max`, `min`) | 2.1% | still single expression |
| `try`/`except`, with the rules below | 5.5% | Teague, 2026-10-18 |

Measured together, with vocabulary ignored: the loosening without `try` raises
structural admission from 2,550 to **6,293 of 6,984 (90.1%)**. With `try` it
reaches **6,557 (93.9%)**. That figure counts the construct only. The handler
rules below reject most corpus handlers as written, because 87% of them catch
`Exception` or use a bare `except`. Allowing `assert` alone reaches 73.5%.
Allowing every style-tier construct, including `while`, reaches 97.6%.

Rules for `def`. The corpus has 849 function definitions. Body size in
statements: median 4, p75 7, p90 12, p95 16, maximum 35.

  - **Top level only.** No nested `def`, no closures, no decorators. Corpus
    cost: 29 definitions are not at top level, 12 of them nested.
  - **Define before use.** A function may call only functions that are defined
    above it. This removes recursion, and it is STE's "define a term before use".
    Corpus cost: 34 definitions recurse directly, mostly tree walkers over JSON or
    AST data. They stay out of v1 because no execution timeout is enforced
    (section 5.1).
  - **One way to define a function.** `name = lambda ...` is an error, and the
    author writes `def` instead (107 programs do this today).
  - **No function returns a function** ("one part of speech per word").
  - **Generator expressions, yes. Generator functions, no.** Measured: 731
    programs (10.5%) use a generator expression, and 1,440 of 1,444 of them are
    passed directly to a consumer (`sum` 588, `Counter` 203, `join` 183, `any`
    127, `next` 116). Only 3 of 849 definitions use `yield`. The rule for clean
    use: **a generator expression is consumed where it is written**, as the
    argument of a call. It is never assigned to a name. A stored generator can
    be consumed twice, and the second pass silently sees nothing. That is a
    misreading that STE exists to prevent. The rule costs 0 corpus programs.
    Generator functions stay out (my call, after Teague asked about
    memory-efficient filtering). Generator expressions already give lazy
    filtering. Also, an effect inside a generator function runs when the caller
    iterates, not when the caller calls, so the effect summary at the call site
    would be wrong. Where memory matters, a program streams in plain Python:
    `for line in Path(p).open()` is out, because `open` is out, so it reads the
    text and iterates over `splitlines()`. Files that do not fit in memory are
    outside v1's domain.
  - **Plain signatures.** Positional and keyword parameters only. No `*args` and
    no `**kwargs` (7 definitions). A default value must be a constant (`None`, a
    number, a string, a tuple of constants), never a list, a dict or a call.
  - **One return mode.** A function either returns a value on every path, or it
    never returns a value. A mix of bare `return` and `return x` is an error.
  - **Body at most 12 statements** (p90 of the corpus), waivable with `# ste:`.
  - **No `global`.** 108 corpus definitions use it, to accumulate results. They
    return the value instead.
  - **Effects inside functions are allowed** (132 definitions, 16%, do I/O). The
    derivation summarizes each function's effects in terms of its parameters,
    for example "writes `{path}`", and substitutes the arguments at each call
    site. The define-before-use rule makes the call graph acyclic, so one pass
    in definition order computes every summary. Without that rule, the
    derivation would need a fixed-point iteration.
  - **No method declarations**, because there are no classes (see "Data" below).
  - A one-line docstring is optional. When present, the derived procedure shows
    it as the function's goal.

**Data.** The corpus says that v1 programs move records. They do not model a
domain. Measured share of programs:

| Container | Programs |
|---|---|
| tuple unpacking | 19.2% |
| record access by string key, `d["path"]` | 18.1% |
| dict literal | 12.6% |
| dict comprehension | 4.5% |
| `collections.Counter` | 3.6% |
| `collections.defaultdict` | 0.9% |
| `class` | 0.1% (5 programs) |
| `dataclass`, `namedtuple`, `SimpleNamespace` | 0 |

The v1 guidance, as my (Claude's) call:

  - A record is a dict with string-literal keys. Literal keys let the derivation
    name the fields that a program reads and writes.
  - A fixed group of two or three values is a tuple, unpacked at the point of
    use. A longer group is a record.
  - `Counter` and `defaultdict` are in the dictionary, as approved functions of
    the `collections` proxy.
  - No `class`, no `dataclass`, no `namedtuple`.

Data modelling is a different problem space, and it would need a different rule
set. ASD-STE100 makes the same split. It has separate rules for procedural
writing (instructions) and descriptive writing (how a system is built). lackpy
v1 is procedural writing. Typed records would be its descriptive part. That part
belongs in a later version, built only when demand shows, for example in `# ste:`
waiver counts or in a new corpus.

Rules for `try` (Teague asked for `try` with restrictions; the shape below is my
(Claude's) proposal from the measurements):

  - **Name the error.** Each `except` names one or more exception types from the
    dictionary. `except Exception`, `except BaseException` and a bare `except`
    are errors. Each dictionary entry lists the exceptions it raises (for
    example `json.loads` raises `JSONDecodeError`, and `Path.read_text` raises
    `FileNotFoundError`).
  - **No "reachable error" check** (my call, in the plain-Python review). An
    earlier draft required each named type to be one that a call in the body is
    documented to raise. That needs a hand-maintained table of which stdlib
    call raises what, which is kit-like metadata, and it would reject correct
    code such as `except KeyError` around `d[k]`. The small-body rule below
    already ties the handler to a few statements.
  - **Keep the body small.** At most 3 statements. Measured on the corpus's 459
    `try` blocks: 1 statement 301 (66%), 2 or fewer 394 (86%), 3 or fewer 421
    (92%).
  - **No `else`, no `finally`.** Use `with` for cleanup. `finally` occurs once
    in the corpus.
  - **Approved handler shapes** (Teague: allow `pass` in specific cases, and
    define the shape "if it fails, skip it"). A handler may begin with
    reporting statements, such as `print`. It must end in one of these shapes:

    | Shape | Form | Derived sentence | Corpus handlers |
    |---|---|---|---|
    | skip | `except E: continue`, in a loop | "If F raises E, skip this item." | 170 (+18 after a print) |
    | return | `except E: return <value>`, in a `def` | "If F raises E, return V." | 89 (+9) |
    | ignore | `except E: pass`, when the body assigns no name | "If F raises E, ignore it." | 23 |
    | default | `except E: x = <value>`, assigning every name the body assigns | "If F raises E, use V for X." | 11 |

    One rule covers all four shapes: **after a `try`, every name the body
    assigns is either assigned in each handler, or the handler leaves the
    flow** (`continue` or `return`). This is definite assignment. It rejects
    the real defect in 15 corpus handlers: `pass` after a body that assigns, so
    the name may be unbound later. Re-raising (16 handlers) is out, because
    `raise` is out. Most of the other 145 handlers report an error and then fall
    through. They conform when they satisfy the same rule.
  - **No `log()` requirement** (Teague: it is undefined and would be noisy).
    Two mechanisms keep a caught failure visible without author code:
    (a) the derived procedure states every handler, for example "If
    `json.loads(line)` raises `JSONDecodeError`, skip the record"; (b) the runtime
    trace counts how often each handler caught, one counter per handler, not one
    line per catch.
  - **No dedicated `def`.** Teague's first answer chose "a dedicated def and a
    one-statement body". In the same answer, Teague asked whether that
    over-thinks it and suggested minimal-body rules instead. I took the
    minimal-body form. Reason: 63% of corpus `try` blocks sit in a top-level
    loop and skip a bad record. A dedicated `def` turns each of those into a
    sentinel return plus a check at the call site, which is more for the reader
    to read. Teague confirmed this ("I like your try rules").

**Keep out:** `raise` (0.8%), `while` (1.1%), `global` (1.5%), `class`,
`yield`, `await`, walrus, `del`, and `import` of anything that is not in the
dictionary. Teague confirmed `raise`, `while` and `global`.

**Tighten (style tier), at measured cost:**

| Rule | Cost on corpus |
|---|---|
| attribute chain at most 3 | 0.0% |
| no walrus | 0.0% |
| `if`/`elif` chain of 3 or more becomes a dispatch dict | 0.4% |
| no multi-`for` comprehension | 0.9% |
| no nested comprehension | 1.4% |
| method chains: at most 3 links | 10 chains in the corpus have 4 or more links |
| no nested call in arguments | **not adopted** as stated. Allow builtins nested to depth 2 (`print(len(x))`). An I/O call may never be an argument to another call. Measure again after the corpus is rewritten to `Path`. |

**Method chains** (Teague: allow them where they are idiomatic). A chain of
method calls is allowed up to 3 links, which is STE's limit of 3 nouns in a
cluster. Measured: 2,342 chains of 2 links, 90 of 3, and 10 of 4 or more. A
call that does I/O may only be the root of a chain, as in
`Path(p).read_text().splitlines()`. The derivation then still sees one effect per
statement. A longer chain needs a `# ste:` waiver. pandas is the example of a
fluent API, but no corpus program imports it, so it is not in the v1
dictionary. When it is added, its entries can carry a `chainable` mark with its
own limit.

The four rules adopted unconditionally (walrus, `if`/`elif` chain, multi-`for`,
nested comprehension) together leave **6,806 of 6,984 (97.5%)** conforming.

The last rule is the "point where a reviewer reads more rather than less" from
the brief. `n = len(x)` followed by `print(n)` is longer, and it is no clearer.
An I/O call is an effect. It gets its own statement, so the procedure
derivation sees one effect per sentence. So `json.loads(Path(p).read_text())`
becomes `text = Path(p).read_text()` followed by `data = json.loads(text)`.

**Vocabulary (the dictionary):**

  - **`pathlib.Path` is the one way to do file I/O** (decision D13, Teague).
    `open` is rejected, and the error names the `Path` equivalent:
    `open(p).read()` becomes `Path(p).read_text()`. Reasons: STE's "one word,
    one meaning", and the method name tells the derivation whether a call reads
    or writes, with no mode string to parse. Corpus: `open` is in 64.7% of
    programs and `pathlib` in 22.9%.
      - `Path` reaches the program through the same import-to-proxy rewrite as
        the approved modules. The proxy confines every path to the workspace,
        and it rejects a path that resolves outside it, symlinks included.
      - Approved members, as a starting list for the builder to set from the
        corpus: `read_text`, `write_text`, `read_bytes`, `write_bytes`,
        `exists`, `is_file`, `is_dir`, `glob`, `rglob`, `iterdir`, `mkdir`,
        `name`, `stem`, `suffix`, `parent`, `parts`, `with_suffix`,
        `relative_to`, and the `/` operator. Every `Path` that a member returns
        is a proxy too.
      - Denied: `unlink`, `rmdir`, `rename`, `replace`, `chmod`, `symlink_to`,
        `hardlink_to`, `touch`, `open`, `home`, `cwd`, `expanduser`, `resolve`
        to an outside path, and every underscore attribute.
      - The `edit_file` defect (section 2.5) no longer reaches agents. The
        corpus's guarded-patch idiom (read, `assert` the count, replace, write)
        conforms as written, and the derivation names it (section 4.2).
  - **Approved modules: written as imports, rewritten to proxy namespaces**
    (decision D3, Teague). Agents write ordinary Python: `import json`,
    `import statistics as st`, `from re import findall`. Before execution,
    lackpy removes each import statement and binds the name to a **proxy
    namespace**. The proxy holds only that module's approved functions from the
    dictionary. Candidate modules: `json` (parse and dump only), `re`,
    `collections`, `math`, `statistics`, `hashlib`, `datetime`. The builder
    sets the final list from the corpus.
      - Why not real modules: the attribute guard is a name filter. I (Claude)
        measured that `validate("x = statistics.sys.modules")` returns no
        errors, and `json`, `re` and `collections` expose other modules the same
        way. A static check cannot follow an alias such as `s = statistics`.
        The proxy is the boundary.
      - Why not wrapper functions (`json_loads`): Teague's view is that they are
        clumsy and confuse agents.
      - Why imports and not pre-bound names: pre-bound proxies would need
        boilerplate in every agent prompt to say they exist. Imports are what
        agents already write.
      - The validator also rejects, statically, an attribute of an import-bound
        name that is not in the dictionary (`json.codecs`), so the error comes
        early and has a good message. The runtime proxy still catches the alias
        case.
      - An import of a module that is not in the dictionary is an error, with
        the approved alternative when the dictionary has one (`os.path.exists`
        → `Path(p).exists()`, `glob.glob` → `Path(d).glob`).
      - The rewrite is a safety-tier mechanism. Its tests must include each
        known escape route: `statistics.sys`, `json.codecs`, `re.functools`,
        `collections.abc`, an alias of each, and `from x import y as z`.
  - **Unapproved words with approved alternatives**, in the way the STE
    dictionary lists them. Examples from the corpus:

    | Unapproved | Approved | Why |
    |---|---|---|
    | `open(p).read()` | `Path(p).read_text()` | one way per operation; no mode string |
    | `open(p, "w").write(s)` | `Path(p).write_text(s)` | the method name says it writes |
    | `json.loads(open(p).read())` | `text = Path(p).read_text()`, then `json.loads(text)` | one effect per statement |
    | `glob.glob(pat)`, `os.walk(d)` | `Path(d).glob(pat)`, `Path(d).rglob(pat)` | one name per operation |
    | `os.path.join(a, b)` | `Path(a) / b` | one way per operation |
    | `map(f, xs)` | a comprehension | "one word, one meaning" |
    | `"{}".format(x)` | an f-string | safety (`format` is a sink) |

### 4.4 The escape hatch: `# ste:`

  - Form: `# ste: <rule-id>, <why>`, on the line it waives, or on the line above
    it. The form follows the `shortcut:` comment form in the `ponytail` skill.
  - It waives **one style rule on one line**. It never waives a safety rule. The
    validator rejects a `# ste:` that names a safety rule.
  - Every waiver appears in the bundle that the reviewer reads. A waiver has no
    effect on the intent comparison.
  - The harness counts waivers by rule. A rule that is waived often is a
    candidate to loosen. That is how the dictionary gets maintained.

### 4.5 The style guide document

Location: `docs/reference/style-guide.md` (my call), with the machine-readable
dictionary in `packages/lackpy-lang` as data.

  - **Part 1, Program rules.** Numbered, in the form of STE: rule id, rule,
    rationale, a non-conforming example, a conforming example. Each rule has a
    tier (safety or style) and the rule id that `# ste:` uses.
  - **Part 2, Dictionary.** Each entry: the approved word (builtin, method, module
    function or `Path` member), its one approved meaning, its derived-procedure
    template, and the unapproved alternatives it replaces.
  - **Part 3, Intent rules.** A short STE subset for the `goal` field. Take it
    from the adopted prose rules in `~/.CLAUDE.md`: 25 words at most, active
    voice, one idea, approved names (such as `read_text`) used as Technical Names.

The validator and the guide must not drift. The rule ids and the dictionary come
from one source, and the guide's tables are generated from it.

---

## 5. Constraints

### 5.1 Two tiers of rules

  - **Safety tier.** These are the current rules that keep the sandbox closed:
    the node allowlist where it concerns escape, `FORBIDDEN_NAMES`, the
    underscore attribute rule, `DENIED_ATTRIBUTES` (`format` and the frame
    internals), and the call-target rule. They are
    **never waivable**. They are documented as safety, not as style.
  - **Style tier.** These are the STE rules. They are waivable with `# ste:`.

Every loosening in section 4.3 must be checked against the safety tier. Known
interactions:

  - `def` enables recursion. The define-before-use rule prevents it. Confirm that
    the runner has an execution timeout. I (Claude) have not verified that.
  - `lambda` in more positions exposes more callables. It remains a single
    expression, and the attribute rules apply inside it.
  - Each approved module adds attribute surface. The proxy namespace (section
    4.3) is what closes it. A real module object never reaches the program.
  - Return values from approved functions are a second surface. Examples:
    `re.compile` returns a `Pattern`, and `re.search` returns a `Match`. The
    builder must list the non-underscore attributes of each returned type, and
    confirm that none of them leads to a module or a namespace. That is a test,
    not an assumption.
  - `try` makes exception objects reachable (`except JSONDecodeError as e`).
    The comment above `DENIED_ATTRIBUTES` in `grammar.py` relies on "Try /
    Raise / sys remain excluded" to keep tracebacks out of reach. That premise
    no longer holds. `e.__traceback__` is blocked by the underscore rule, and
    `tb_frame` and `tb_next` are on the deny list. The builder must list every
    non-underscore attribute of each approved exception type, confirm that no
    route reaches a frame, and update that comment.
  - **No execution timeout is enforced today** (measured by the builder check).
    `config.sandbox_timeout` is read but used only in a report
    (`src/lackpy/service.py:1109`). `RestrictedRunner.run` calls `exec` in
    process. `range(10**12)` can already hang a program. The define-before-use
    rule means `def` adds no new route to a hang. An enforced timeout needs a
    subprocess runner, and that is **out of v1** (my call). It gets its own issue.

### 5.2 Other constraints

  - `lackpy-lang` stays free of runtime dependencies. The validator, the
    derivation and the dictionary belong there. The runtime proxies for modules and `Path` belong in `lackpy`.
  - lackpy has no installed users (teaguesterling/retritis#16). Breaking changes
    to the language are acceptable.
  - The existing correction chain (`src/lackpy/infer/correction.py`: cleanup,
    few-shot retry, fixer) stays as the generation path for cases 2 and 3.

---

### 5.3 The plain-Python invariant

Teague's requirement, 2026-10-18: no kit, and no lackpy-only call where normal
Python has one. The design states that as a testable invariant:

> **A conforming lackpy program is a plain Python 3 program. It runs unchanged
> under CPython with only the standard library, from the workspace directory,
> and gives the same output there as inside lackpy.** Inside lackpy it can only
> do less: a sandbox refusal is an error, never a different result.

I (Claude) re-reviewed the design and today's runner
(`src/lackpy/run/runner.py`) against that invariant. Each finding below is
measured from the code or the corpus.

| # | Finding | Today | Change |
|---|---|---|---|
| 1 | `sort_by` is a lackpy-only builtin | `grammar.py:99`, `runner.py:14`, advertised in `infer/hints.py:39` | remove it; `sorted(xs, key=lambda r: r["k"])` is the one way |
| 2 | parameters are injected as bare global names | `runner.py:108`. Measured: a program that assigns `n = 3` returns 3 when `params={'n': 5}`, so the program's own assignment silently wins | a single `params` dict literal holds the defaults, and lackpy replaces its values only. The intent block's `[params]` table documents each key (section 4.1, D15). |
| 3 | exception classes are not in the namespace | `__builtins__` is `{}`, and only `ALLOWED_BUILTINS` is injected (`runner.py:99`, `:131`), so `except KeyError:` fails. Measured with `RestrictedRunner().run(...)`: `name 'KeyError' is not defined` | resolve names against the real `builtins` module, with every name classified (D16). Every exception class is allowed except `SystemExit`, `KeyboardInterrupt` and `GeneratorExit`. Module exceptions such as `json.JSONDecodeError` come through the module proxy. |
| 4 | the builtin allowlist is narrower than ordinary code | no `repr`, `chr`, `ord`, `divmod`, `hash`, `iter` ... | the same classification (D16). A test fails when `dir(builtins)` holds a name that is neither allowed nor denied, so a new Python release cannot add a builtin silently. The corpus informs which names to allow. |
| 5 | effects are traced only for kit calls | `make_traced` wraps the kit namespace (`runner.py:97`) | the `Path` proxy records each read and write in the trace |
| 6 | the final bare expression becomes the output value | `runner.py:116` | keep it: under plain Python it is a no-op, so it does not break the invariant. `output = "value"` is the one lackpy convention, and the guide says so. |
| 7 | the inference prompts teach kit tools and `sort_by` | `src/lackpy/infer/` (prompts, hints, examples) | cases 2 and 3 must generate from the skeleton; this is a work item, not a footnote |
| 8 | the "reachable error" rule needs a raises table | section 4.3 | dropped (see the `try` rules) |
| 9 | proxies could change behaviour | a proxy is a new object | the equivalence test below catches any difference |
| 10 | a refusal at run time must be clear and must not be caught | aliases, computed paths and returned objects pass static checks | stand-ins raise a lackpy refusal with the approved alternative as a hint (D17) |

**Scope of the restricted builtins** (measured 2026-10-18). The classified
builtins apply only to code compiled from the program. Standard-library modules
keep their own globals and the real builtins. With an empty program
`__builtins__`, `json.loads("[1, 2, 3]")` and `statistics.mean` still returned
`[1, 2, 3]` and `2`, while the program's own `len([1])` failed with
`NameError`. Functions and lambdas that the program defines see the program's
restricted builtins, which is the intent.

**Underscore names** (my calls, after Teague asked whether dunders and hidden
names should be reachable at all):

| Form | Rule | Tier | Corpus |
|---|---|---|---|
| attribute beginning with `_` (`x.__class__`, `x._module`) | denied, as today | safety | 102 programs (1.5%) |
| name beginning with `__` (`__name__`, `__import__`, `__result__`) | denied (new) | safety | 5 programs, all `__import__` |
| name with one leading `_` (`_`, `_tmp`) | allowed | n/a | 175 programs, mostly `_` as a throwaway (305 uses) |

The dunder-name ban also reserves the runner's internal `__result__` name.

**The dunder allowlist** (Teague asked for `__name__`, `__version__` and
`__file__`; the shapes are my call). Each entry is read-only, its value is a
plain string, and an assignment to it is an error. Measured corpus use is in the
last column.

| Dunder | Value inside lackpy | Plain-Python behaviour it matches | Corpus |
|---|---|---|---|
| `__name__` (a name) | always `"__main__"` | `python3 program.py` | 0 |
| `__file__` (a name) | the program's absolute path, when lackpy runs a file inside the workspace; `"<stdin>"` when the program arrives as a string | `python3 program.py`, and `python3 -`. Measured on Python 3.14.6: an absolute path, and `<stdin>` | 0 |
| `.__version__` (an attribute) | only on approved module proxies; a version string, or `AttributeError` where the module has none | the same attribute on the real module | 6, all `duckdb.__version__` |
| `type(x).__name__` (one exact form) | the type's name | the same expression | 56 of 56 `.__name__` uses (54 `type(x)`, 2 `x.__class__`) |

  - `type` stays out of the namespace as a callable. A stand-in occupies the
    name `type`. It answers `.__name__` on its result, and it raises a refusal
    (D17) for any other use: `type(x) is int`, `type(x)()`, `type(x).mro()`.
    The validator accepts only the exact form `type(<expr>).__name__`. This is
    the one entry that adds machinery. Teague approved it with the design.
  - `x.__class__.__name__` stays denied, because `__class__` is the start of
    the classic escape chain. The approved spelling is `type(x).__name__`.
  - `__file__` is an absolute path (measured), so it reveals
    where the workspace sits on the host. Any I/O through it still goes
    through the workspace-confined `Path`.
  - Every other dunder stays denied: `__dict__`, `__class__`, `__init__`,
    `__globals__`, `__builtins__` and the rest.

**Refusals at run time** (decision D17, Teague). The validator catches nearly
every unapproved feature before execution, with a message that names the
approved alternative. The runtime stand-ins (module proxies, the `Path` proxy,
the classified builtins) are the backstop for what a static check cannot see.

  - A stand-in that reaches a masked feature raises a lackpy refusal. Examples:
    `open(...)`, a `Path` that resolves outside the workspace, `Path.unlink`, or
    `s.sys` where `s` is an alias of `statistics`.
  - The message names the approved alternative, for example "`open` is not
    approved: use `Path(p).read_text()`". The hint comes from the same table of
    unapproved words that the validator uses (section 4.3), so the static
    message and the runtime message cannot disagree.
  - **The refusal derives from `BaseException`, not from `Exception`.** If it
    subclassed `AttributeError`, an approved `except AttributeError:` would
    swallow a sandbox refusal. Its class is also not nameable in a program.
    Approved handlers name specific `Exception` subclasses, and `except
    BaseException` is an error, so no handler can catch a refusal.
  - The stand-ins add no names. Each one occupies a standard-library name
    (`open`, `Path`, `json`, ...). The plain-Python invariant still holds.

**The equivalence test.** For each committed fixture program, run it twice in a
scratch copy of its workspace: once with `python3 program.py`, once through
lackpy. Compare stdout, the output files and the exit status. Any difference
fails the test. This is the check that proves there is no hidden kit. It also
makes lackpy programs debuggable with ordinary tools.

Known limits of the invariant:

  - Proxies may make `isinstance(p, Path)` or `type(p)` behave differently.
    `type` is already a forbidden name. The builder must check `isinstance`
    against the proxy.
  - A program that lackpy refuses (a path outside the workspace) may still run
    under plain Python. The invariant is one-directional on purpose.

## 6. Build order (for the plan)

  1. Commit the corpus harness (Appendix A) as a script. Freeze the snapshot
     outside git (D4) and choose the fixtures. Record today's baseline: 0.1%
     admitted, 36.5% structurally admitted.
  2. Rule registry with ids and tiers. Port the current rules into it with no
     change in behaviour. Test: the current test suite passes without change.
     Then remove `sort_by`, move parameters to top-level assignments, and add the
     approved exception classes to the namespace (section 5.3, findings 1-4).
  3. Loosen per section 4.3, one rule per change, with the harness delta recorded
     for each change.
  4. The dictionary data, plus the import-to-proxy rewrite for modules and
     `Path`, with its escape-route and workspace-confinement tests (section 5.1).
  5. Intent frontmatter parser, the derivation, and the comparison.
  6. MCP and CLI: return the bundle. Rewrite the inference prompts and examples
     for the skeleton (section 5.3, finding 7).
  7. The equivalence test (section 5.3), run on every fixture in CI.
  8. The style guide, generated in part from the registry. Then a human read of
     about 20 rewritten corpus programs, to check the reviewer-cost limit.
  9. A mechanical translation of a corpus sample into lackpy, to measure the real
     admit rate rather than the structural rate. Target: my (Claude's) call is
     80% of the sampled patch, read and scan programs, after translation, with
     every waiver counted.

---

## 7. Decisions

All four were decided by Teague on 2026-10-18.

  - **D1. Constrained decoding: deferred to v2.** The brief said build it
    ("prefer 2"). This design supersedes the brief on that point. Reasons: (a) inference goes through woollama, and I found
    no grammar or schema hook in `src/lackpy/infer/providers/`; (b) I believe
    Ollama accepts a JSON schema but not a GBNF grammar, and I have not verified
    that; (c) the intent comparison catches the semantic misreading that a
    grammar cannot catch. The cost of deferral: a small model in case 2 still
    produces invalid programs, and the correction chain absorbs them.
  - **D2. Write policy: globs allowed, undetermined writes block.** See section
    4.2.
  - **D3. Modules: imports rewritten to proxy namespaces.** See section 4.3.
  - **D4. Corpus storage: outside git, plus fixtures.** The frozen snapshot of
    6,984 programs lives outside the repo, at
    `~/.local/share/lackpy/corpus/heredocs-2026-10-18.parquet` (the path is my
    call). The harness script and the baseline numbers are committed. About 50
    hand-picked programs, reviewed for private content, are committed as test
    fixtures. They cover each family in section 2.2. The harness reports
    "corpus not found" and exits non-zero when the snapshot is absent. It never
    reports a zero.

Added on 2026-10-18, after the first review of the artifact:

  - **D5. `try`/`except` is allowed, with restrictions** (Teague). The rules
    are in section 4.3. The minimal-body form instead of a dedicated `def` is
    my call, and Teague confirmed it ("I like your try rules").
  - **D6. No `log()` requirement in handlers** (Teague). The derived procedure
    and the per-handler counters in the trace keep a caught failure visible.
  - **D7. Method chains of up to 3 links are allowed** (Teague asked for
    idiomatic chaining; the limit of 3 is my call).
  - **D8. Handlers have four approved shapes:** skip, return, ignore and
    default. `pass` is allowed in the ignore shape (Teague). Definite
    assignment, as the single rule behind the shapes, is my call.
  - **D9. Function declaration rules** are in section 4.3 (my calls, measured).
  - **D10. No classes or typed records in v1.** Records are dicts. Data
    modelling is a separate rule set for a later version, in the way that STE
    separates descriptive writing from procedural writing (my call; Teague
    raised the question).
  - **D11. Generator expressions are consumed where they are written; generator
    functions are out** (Teague agreed).
  - **D12. No kit and no stubs. Agents start from a skeleton** (Teague). The
    agent never sees a kit, a profile or a tool set. Stubs were dropped too,
    because they need the kit machinery.
  - **D13. `pathlib.Path` is the one way to do file I/O, and `open` is
    rejected** (Teague). `Path` goes through a workspace-confined proxy.
  - **D14. The plain-Python invariant** (Teague's requirement; the wording and
    the equivalence test are my call). See section 5.3.
  - **D15. Parameters arrive in one `params` dict literal, documented in the
    intent block's `[params]` table** (Teague). See section 4.1.
  - **D16. Builtins resolve against the real `builtins` module, and every name
    is classified** (Teague). A test fails on an unclassified name.
  - **D17. A refusal at run time is uncatchable, and it carries the approved
    alternative** (Teague). See section 5.3.
  - **D18. Parameters are typed.** `[params]` declares a type for each key,
    and lackpy holds it as a `TypedDict` (Teague). The v1 type list is my call.
  - **D19. No name may begin with `__`, except the allowlist.** Attributes
    beginning with `_` stay denied. Single-underscore names stay allowed (my
    calls; Teague raised the question).
  - **D20. Dunder allowlist: `__name__`, `__file__`, module `__version__`, and
    the exact form `type(x).__name__`** (Teague named the first three; the
    `type` stand-in was my proposal, approved with the design).

---

## Appendix A. Method

The corpus extraction runs in DuckDB 1.5.5 CLI. The analysis runs in lackpy's
`.venv` Python with `lackpy.lang.validate` from this checkout (`8569e41`).

```sql
CREATE TABLE calls AS
WITH m AS (
  SELECT filename, unnest(json_extract(message, '$.content[*]')) AS c
  FROM read_json('~/.claude/projects/*/*.jsonl', format='newline_delimited',
       columns={type:'VARCHAR', message:'JSON'}, filename=true, ignore_errors=true)
  WHERE type='assistant')
SELECT filename, json_extract_string(c,'$.input.command') AS cmd,
       json_extract_string(c,'$.input.description') AS descr
FROM m
WHERE json_extract_string(c,'$.type')='tool_use'
  AND json_extract_string(c,'$.name')='Bash'
  AND regexp_matches(json_extract_string(c,'$.input.command'), 'python3? (- )?<<');

CREATE TABLE bodies AS
SELECT *, regexp_extract(cmd,
  '(?s)python3? (?:- )?<<-?[''"]?(PY|EOF|PYEOF|PYTHON|PYCODE|END)[''"]?\n(.*?)\n(PY|EOF|PYEOF|PYTHON|PYCODE|END)\b', 2) AS body
FROM calls;
```

The validator pass maps each `validate()` error to a key (`import:<module>`,
`name:<n>`, `node:<Type>`, `call:<n>`, `attr:<a>`, `rule:<shape>`). The
structural pass drops the `name:`, `import:` and `call:` keys. The STE-rule pass
is a separate AST walk. The scripts were `coverage.py`, `structural.py` and
`ste_rules.py` in the session scratchpad. Step 1 of section 6 commits them.

Known gaps in the method: subagent transcripts are not included. Heredocs with
uncommon delimiters (5% of calls) are not extracted. Bodies with unexpanded
`$VAR` text parse as Python only when the variable sits inside a string.

---

## 8. Changes during Plan 1 implementation

Plan 1 built the static language. The ledger rulings below changed or corrected
sections 4.3, 4.4, 4.5 and 5.1. Earlier sections stay as written.

- **Lambda positions (4.3).** A lambda is allowed as any `key=` argument, as the
  factory of `defaultdict`, and as the `repl` argument of `re.sub` and `re.subn`.
  The design allowed it for any argument of `sorted`, `max` and `min`. A
  positional lambda to those calls always raises `TypeError`. The corpus has no such use.
- **`style.try-named` removed (4.3, 5.1).** `safety.except-base` now rejects every
  bad handler class, and no waiver can lift it. The design made `except Exception`
  waivable. `Exception` also catches the limit signals, so it cannot be waivable.
- **Uncatchable limit signals (4.3).** `RecursionError`, `RuntimeError` and
  `MemoryError` cannot be caught. `RuntimeError` is the base of `RecursionError`.
  A recursion bound needs a signal that no handler can hide.
- **Recursion claim narrowed (4.3).** Define-before-use removes recursion through
  direct calls only. A function passed as `key=` can still recurse. The
  uncatchable `RecursionError` bounds that case.
- **Return mode on every path (4.3).** A function with a valued `return` must end
  every path in a valued `return`. The design said this. The first implementation
  checked only the `return` statements, so a path that ended without a `return`
  passed. Fix commit 197f2b0 closed that gap.
- **Report-and-fall-through handlers (4.3).** A handler may end without a
  `raise` or `return` when it assigns every name that the `try` body assigns.
  The first implementation rejected all of them, against the design.
  Of 114 such corpus handlers, 28 then pass.
- **Waiver scope (4.4).** A waiver covers one statement. For a compound
  statement, it covers the header lines only. A misplaced `# ste:` comment draws
  `meta.waiver-unknown`. The design did not define the scope.
- **Guide names no Python version (4.5).** The guide must read the same on every
  supported Python. Part 2 states the allowed exception classes as a rule. Python
  3.13 added `PythonFinalizationError`, so a list would differ by version.
- **Use before import, and rebinding (5.1).** A call through an import binding passes
  only after the import statement. A name bound by an import cannot be
  bound again. Both close ways to hide a tool call behind an imported name.
- **Module binding only as attribute (5.1).** A module binding may appear only as
  the object of an attribute load, as in `json.loads`. `safety.module-attr` rejects assigning,
  passing or returning it. This closes aliasing of a module until
  the Plan 2 proxies exist.

### Measured gap to the section 4.3 prediction

The design predicted 93.9% structural admission, with `try` included. The final
corpus run measures 3,030 of 6,984 (43.4%). The gap is a finding. It has two
main causes:

- **`style.assert-message`.** The design's 73.5% for `assert` alone assumed no
  message rule. The rule fires on 2,511 programs. In 2,346 of them it is the only
  non-vocabulary violation.
- **`style.call-nesting` at depth 2.** The rule fires on 781 programs, measured
  before translation to `Path`. In 307 it is the only non-vocabulary violation.

`scripts/corpus/blockers.py` reproduces these figures.

### Changes from the final whole-branch review

- `finally` is now safety tier (`safety.try-finally`). A `return` or `continue`
  in `finally` discards an in-flight exception. A style waiver therefore let a
  program swallow RecursionError, and the review measured a hang. `else` stays
  style (`style.try-else`).
- Decorators are now safety tier (`safety.node`). A waived decorator ran a tool
  at def time, and the call was hidden from `calls`.
- `safety.node` rejects `break` and `continue` outside a `for` body in the same
  function scope. `validate()` compiles a rule-clean program, and a compile
  error becomes `safety.parse`.
- `RestrictedRunner.run` returns a failed result on a compile error. It no
  longer raises.
- `ValidationResult.calls` is advisory. It lists named calls on a best-effort
  basis, and it is not a soundness boundary. Plan 3 derives effects soundly.
- `tests/lang/test_waive_everything.py` checks that dangerous programs stay
  invalid when every style rule is waived.

### Decisions after Plan 1 (Teague, 2026-10-18)

- **`style.assert-message` stays required.** A bare `assert` fails with an empty
  `AssertionError`. The message tells the reader which expectation broke. The
  measured cost is 2,346 corpus programs that need a message added.
- **Nesting relaxes to depth 3, I/O calls included.** `style.call-nesting`
  allows a call nested 3 deep. `style.io-position` allows an I/O call as an
  argument, so `json.loads(Path(p).read_text())` is valid. The derivation
  (Plan 3) must therefore find effects nested inside arguments. Apply this as
  the first change of the next plan. Measure the corpus before and after.

- **The `Path` stand-in enforces read-only and read-write permissions**
  (Teague, 2026-10-18). The rules come mechanically from a configuration, in
  the style of Claude Code. lackpy reads the same `permissions` rules that
  Claude Code reads, so one source governs both. Design calls (Claude):
  - Sources, merged as Claude Code merges them (the lists union, and a deny
    at any level wins). The sources are `~/.claude/settings.json`,
    `<workspace>/.claude/settings.json`, `<workspace>/.claude/settings.local.json`
    and a `[paths]` table in `<workspace>/.lackpy/config.toml`.
  - Only `Read(pattern)` and `Edit(pattern)` rules count. `Edit` covers every
    write. A `Read` deny also blocks writes. `ask` counts as deny, because a
    sandboxed program cannot ask anyone.
  - Patterns use gitignore syntax with Claude Code's anchors. `//p` is
    absolute, `~/p` is home, `/p` is relative to the settings file's source,
    and `p` or `./p` is relative to the workspace.
  - Order: deny, then allow. With no match, a path inside the workspace or an
    `additionalDirectories` entry may be read and written. Every other path is
    refused.
  - Symlinks: an allow must match both the requested path and its resolved
    target. A deny matches either.
- **No attribute stores** (Claude, measured). `safety.attr-store` rejects an
  assignment to, or a deletion of, any attribute. 65 of 6,984 corpus programs
  (0.9%) do this, mostly openpyxl and `sys`, both outside v1. The rule closes
  host-class mutation through any alias, statically.

### Carried into Plan 2 as gate requirements

Plan 2 must turn each item into a test that passes before `lackpy-ste` merges.

1. The module proxies close `statistics.sys`, and the `Path` proxy closes
   `Path('x').parser.os`. On Python 3.13 that attribute is the real `os` module.
2. The proxies refuse `setattr` on member bindings. Today these validate:
   `C = Counter; C.most_common = len`, and `[Path][0].read_text = print`.
3. The trace records a `BaseException` refusal. `src/lackpy/run/trace.py`
   catches `Exception` only.
4. `RestrictedRunner.run` never raises. A validated 3,000-deep unary expression
   raises RecursionError from `ast.fix_missing_locations`. `validate()` raises
   MemoryError from `ast.parse` at a depth of about 50,000.
5. Decide if `TimeoutError` from `src/lackpy/run/bridge.py` becomes a refusal.
   Today a program can catch it and retry a hung tool.
6. Define `__file__` deliberately. `Path(__file__).read_text()` validates.
7. Decide if `ExecutionResult.variables` may carry function objects.
8. `datetime.UTC` exists only on Python 3.11 and later. `lackpy-lang` still
   claims Python 3.10.
9. Plan 2 replaces `tests/run/test_import_fail_closed.py`. That test pins the
   fact that imports fail closed until the proxies exist.
