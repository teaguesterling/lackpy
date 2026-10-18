# Corpus harness

Measures how much of a frozen set of real agent Python programs the lackpy validator
admits. Method: Appendix A of `docs/superpowers/specs/2026-10-18-lackpy-ste-design.md`.

The snapshot lives outside git: `~/.local/share/lackpy/corpus/heredocs-2026-10-18.parquet`
(columns `session_file`, `description`, `body`; distinct bodies only).

| Script | Use |
|---|---|
| `extract.py [--calls PARQUET] [--force]` | Build the snapshot. Refuses to overwrite without `--force`. |
| `coverage.py [--snapshot PATH] [--json]` | Print coverage. Exit 2 with `corpus not found: <path>` if the snapshot is missing. |
| `select_fixtures.py` | Write 50 fixture candidates to `~/.local/share/lackpy/corpus/fixture-candidates/` (outside git, for human review). |

`baseline.json` holds the `coverage.py --json` output recorded on 2026-10-18 (before the
rule registry): 6,984 parsed, 7 admitted (0.1%), 2,550 structural (36.5%). Its `by_rule`
keys are message-derived (`name:open`, `node:Assert`, ...).
"structural" ignores `name:`, `import:`, `call:` and `attr:format` keys.

## Snapshot provenance

The snapshot holds **7,003** distinct non-empty bodies (6,984 parse). It was built with
`extract.py --calls heredoc_calls-2026-10-18.parquet`, from a saved copy of the 7,419
`calls` rows (32 sessions). Claude Code has since pruned old session files: extracting
from the live history on 2026-10-18 later gave only 3,351 calls (11 sessions) and 3,104
bodies. The live extract is not comparable with the baseline, so do not use it for
`baseline.json`.
