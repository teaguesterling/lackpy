# Corpus harness

Measures how much of a frozen set of real agent Python programs the lackpy validator
admits. Method: Appendix A of `docs/superpowers/specs/2026-10-18-lackpy-ste-design.md`.

The snapshot lives outside git: `~/.local/share/lackpy/corpus/heredocs-2026-10-18.parquet`
(columns `session_file`, `description`, `body`; distinct bodies only).

| Script | Use |
|---|---|
| `extract.py [--calls PARQUET] [--force]` | Build the snapshot. Refuses to overwrite without `--force`. |
| `coverage.py [--snapshot PATH] [--json]` | Print coverage. Exit 2 with `corpus not found: <path>` if the snapshot is missing. Exit 4 with `unknown rule id: ...` if a violation names an id outside the registry. |
| `select_fixtures.py` | Write 50 fixture candidates to `~/.local/share/lackpy/corpus/fixture-candidates/` (outside git, for human review). |

`baseline.json` holds the `coverage.py --json` output recorded on 2026-10-18 (before the
rule registry): 6,984 parsed, 7 admitted (0.1%), 2,550 structural (36.5%). Its `by_rule`
keys are message-derived (`name:open`, `node:Assert`, ...).
"structural" ignores `name:`, `import:`, `call:` and `attr:format` keys.

Since the rule registry (Task 2), `coverage.py` keys `by_rule` by `Violation.rule_id`,
sorted by count, then id. "admitted" is `ValidationResult.valid`. "structural" ignores
`safety.forbidden-name`, `safety.unknown-call`, `safety.import` and `safety.module-attr`.
Until `safety.import` exists, it also ignores the `safety.node` violations on import
nodes and the `safety.attr-denied` violation on `.format`, as the baseline did. The
figures stay 7 admitted and 2,550 structural.

## Snapshot provenance

The snapshot holds **7,003** distinct non-empty bodies (6,984 parse). It was built with
`extract.py --calls heredoc_calls-2026-10-18.parquet`, from a saved copy of the 7,419
`calls` rows (32 sessions). Claude Code has since pruned old session files: extracting
from the live history on 2026-10-18 later gave only 3,351 calls (11 sessions) and 3,104
bodies. The live extract is not comparable with the baseline, so do not use it for
`baseline.json`.

## Snapshot identity

Canonical: **7,003** bodies, SHA-256 `fc7ba975c23d207ade7cd072ec71719a065680ba5f5e3ed62f33232844c9672e`.
Definition: distinct non-empty bodies, sorted, each UTF-8 encoded and followed by a NUL byte, hashed.
`coverage.py` checks both on the default snapshot (mismatch: stderr message, exit 3). With
`--snapshot PATH` it skips the check. Its JSON always carries `snapshot_rows` and `snapshot_sha256`.
`extract.py` without `--calls` warns on stderr when the count differs from 7,003.

## Plan 1 close-out figures (2026-10-18)

Final run (blq `corpus`, run `corpus:93`): snapshot 7,003 rows, sha `fc7ba975c23d`; 6,984 parsed;
**608 admitted (8.7%)**; **3,030 structural (43.4%)**. Design section 4.3 predicted 93.9% structural.

Per task, from the commit subjects (last figures of the task):

| Task | Change | Admitted | Structural |
|---|---|---:|---:|
| 1 | Harness, baseline (derived: subject gave percentages only; figures from `baseline.json`) | 7 | 2,550 |
| 2 | Rule registry (derived: subject said "unchanged") | 7 | 2,550 |
| 3 | Classified builtins; `sort_by` removed; `mro` denied | 7 | 2,550 |
| 4 | Dictionary, imports, module attributes | 516 | 2,550 |
| 5 | Underscore names, dunder allowlist | 516 | 2,545 |
| 6 | Simple constructs loosened | 640 | 3,444 |
| 7 | Function rules | 658 | 3,648 |
| 8 | `try` rules | 658 | 3,654 |
| 9 | STE shape rules | 608 | 3,030 |
| 10 | `# ste:` waivers | 608 | 3,030 |
| 11 | Style guide (docs only, no figure in subject) | 608 | 3,030 |

Task 6 figures come from `f642431`; its later fix `94a399c` carries none. Task 11 figures are
from the Task 10 commit and the final run.

### Blockers (`blockers.py`)

`blockers.py [--top N] [--json]` ranks non-vocabulary rules. "Alone" means the program's only
non-vocabulary violation is that rule: loosening the rule moves the program into "structural".
"Alone (strict)" means no other violation of any kind: loosening admits the program.

| Rule | Programs | Alone | Alone (strict) |
|---|---:|---:|---:|
| `style.assert-message` | 2,511 | 2,346 | 799 |
| `style.call-nesting` | 781 | 307 | 4 |
| `style.io-position` | 422 | 104 | 32 |
| `safety.except-base` | 366 | 70 | 0 |
| `safety.node` | 290 | 51 | 12 |

Dropping all five would give 6,258 structural (89.6%). `safety.except-base` and `safety.node`
are safety tier, so only the first three are candidates for loosening.

## Plan 2 close-out figures (2026-10-18, after stepy100 Plan 2)

Measured at commit `f936537` (branch `mcp-tool-metadata`, clean tree). blq `corpus` run `corpus:217`:
snapshot 7,003 rows, sha `fc7ba975c23d`; 6,984 parsed; **615 admitted (8.8%)**;
**3,372 structural (48.3%)**. Plan 1 close-out was 608 admitted and 3,030 structural; the change comes
from `dd6b283` (nesting to depth 3, I/O calls as arguments, no attribute stores), before Plan 2 began.
No Plan 2 task changed the corpus figures.

blq `test-all` (run `test-all:216`, same commit): **2,497 passed, 32 skipped, 102 xfailed**.
