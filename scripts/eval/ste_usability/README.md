# STE usability eval

Can a model write lackpy from the style guide alone, and does a one-page
pre-prompt help? Each task is scored on two independent axes:

- **valid**: `lackpy.lang.validate()` accepts the program.
- **correct**: under plain `python3` in a fresh, bubblewrapped copy of `fixtures/`,
  the program gives the same stdout and written files as the task's reference.
  `correct_loose` ignores `:`, `,` and whitespace differences (finding 3).

30 synthetic tasks (`tasks.py`), 5 in each of the corpus families. Each task has
a reference program that must itself validate and run.

## Run it

```
.venv/bin/python scripts/eval/ste_usability/harness.py        # references validate and run
.venv/bin/python scripts/eval/ste_usability/prompts.py RUN/prompts
PROMPT_VARIANT=pre .venv/bin/python scripts/eval/ste_usability/prompts.py RUN/prompts-pre
THINKING=off .venv/bin/python scripts/eval/ste_usability/run_local.py RUN LABEL r1 tiiny-fast/<model id>
THINKING=off .venv/bin/python scripts/eval/ste_usability/run_local.py RUN LABEL r2 tiiny-fast/<model id>
.venv/bin/python scripts/eval/ste_usability/reextract.py RUN
.venv/bin/python scripts/eval/ste_usability/score.py RUN
```

Haiku runs through Claude Code subagents: six subagents answer from the prompt
files and write `RUN/haiku*/r1/<id>.py`, and `make_repairs.py` prepares round 2.

Rules learned the hard way (FINDINGS.md 5 and 8):

- Use explicit model routes. `default` does not say which model answered.
- Never send other requests to the shared device during a run. Doing so
  evicted the model mid-run.
- Stream, because the gateway has a 120 s idle timeout. Retry transport failures.
- `THINKING=off` for Qwen3.6. With thinking on, it returns empty content.

Results: `runs/2026-10-18/results.json` and `FINDINGS.md` finding 9.
