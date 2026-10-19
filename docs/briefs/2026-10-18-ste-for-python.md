# Brief: make lackpy "ASD-STE100 for Python"

**Status:** idea, approved for design. Not yet specced. Start with brainstorming.
**Origin:** Teague, 2026-10-18. Reasoning in
claude.ai/code/session_01XoHU19qnngy1CJnXyX1csz

---

## The prompt

You are designing a change to lackpy, the intent-to-sandboxed-program
micro-inferencer that runs a local model.

Constrain the Python lackpy generates the way ASD-STE100 constrains English.
ASD-STE100 is the aerospace maintenance-manual standard: 53 writing rules plus
a dictionary of about 850 approved words, so that a technician under time
pressure, often not a native speaker, cannot misread an instruction. lackpy has
the same problem shape. A small local model under-generates correct code, and a
reviewer has to verify it fast. Restrict the language so the dangerous
constructions are unrepresentable.

Run `superpowers:brainstorming` first. This is architectural: it changes what
lackpy emits and adds a verification stage. Do not start writing code.

## The mapping, as a starting point

| ASD-STE100 rule | Python analogue |
|---|---|
| ~850 approved words, plus Technical Names | allowlist of builtins and stdlib calls, plus the project's domain API as the technical names |
| one word, one meaning | one canonical way per operation; not both `map()` and a comprehension for the same thing |
| one part of speech per word | a name is a value or a function, never both; no callables returning callables |
| sentence ≤ 20 words | one operation per statement; no chained expressions |
| paragraph ≤ 6 sentences | function ≤ N statements |
| noun cluster ≤ 3 | attribute chains ≤ 3: `a.b.c` yes, `a.b.c.d` no |
| active voice, imperative | explicit control flow; no decorators, metaclasses, `__getattr__` |
| one instruction per sentence | one assignment or call per line; no walrus, no nested comprehensions |
| no ambiguous phrasal verbs | no operator-overloading surprises, no dunder tricks |
| preserve modality | explicit error handling; never swallow an exception |
| 3+ steps become a list | 3+ branches become an explicit dispatch table |
| define a term once | one definition site per concept; no shadowing |

## Why this is worth building

A controlled subset is **mechanically checkable**. ASD-STE100 has conformance
checkers; a Python subset is an AST walk. That turns lackpy's output from
trusted into verified, with no model in the verification path. It also satisfies
the rule that a model should do judgment, not deterministic transforms.

## Build the second version, not the first

1. **Lint after generation.** An AST validator rejects non-conforming output and
   lackpy retries. Easy, but it wastes generations, and a small model may never
   converge.
2. **Constrain the decoding.** A grammar (GBNF, `outlines`, or similar) makes a
   non-conforming program unrepresentable rather than merely detectable. For a
   local model that is the difference between "usually works" and "cannot emit
   an invalid program."

Prefer 2. Keep the AST checker anyway, as an independent gate: a grammar
constrains syntax, while the allowlist is semantic.

## Limits, agreed up front — do not design these away

  - **A conforming program can still be wrong.** STE guarantees the instruction
    cannot be misread; it does not guarantee the instruction is correct. This
    buys verifiable form, not correct behaviour, so it does not remove the need
    for the test lackpy's output should carry.
  - **Over-restriction has a real cost.** STE accepts verbosity as the price of
    unambiguity. Restricted Python will be longer and sometimes clumsy, and
    there is a point where a reviewer reads more rather than less. Find it.
  - **The allowlist is the product, and it is maintenance.** ASD-STE100 took an
    industry consortium and nine issues to settle ~850 words. Scope v1 to one
    domain — DuckDB queries plus data munging — not general Python.
  - **It needs STE's own escape hatch.** Technical Names exist because the
    standard would be useless without them. Use a `# ste: <construct>, <why>`
    annotation, mirroring the `shortcut:` comment form in the `ponytail` skill.

## Context you should pick up

  - `~/.CLAUDE.md` has the ASD-STE100 subset we adopted for prose, including the
    four rules we dropped and why. Read it; the reasoning transfers.
  - lackpy currently has **zero recorded invocations** across 1,913 sessions,
    because it is not installed. See teaguesterling/retritis#16. Its zero is
    absence, not rejection — but it also means there is no usage to regress, so
    this redesign is unusually free.
  - `ponytail` skill, for the house form of a decision ladder.

## First question to settle

Is the unit of restriction the **program** lackpy emits, or the **intent
language** a caller writes? Those lead to different products. Decide that before
anything else.
