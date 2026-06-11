<!-- blq:agent-instructions -->
## blq - Build Log Query

Run builds and tests via blq MCP tools, not via Bash directly:
- `mcp__blq_mcp__commands` - list available commands
- `mcp__blq_mcp__run` - run a registered command (e.g., `run(command="test")`)
- `mcp__blq_mcp__register_command` - register new commands
- `mcp__blq_mcp__status` - check current build/test status
- `mcp__blq_mcp__events` - filter events; pass `severity="error"` for errors-only (replaces the non-existent `errors` tool)
- `mcp__blq_mcp__info` - detailed run info (supports relative refs like `+1`, `latest`)
- `mcp__blq_mcp__output` - search/filter captured logs (grep, tail, head, lines)

Do NOT use shell pipes or redirects in commands (e.g., `pytest | tail -20`).
Instead: run the command, then use `output(run_id=N, tail=20)` to filter.
<!-- /blq:agent-instructions -->

## Worktrees & sub-agents

lackpy is a **flat monorepo** (`packages/lackpy-lang` is a sibling distribution, not a
git submodule). For isolated or parallel work — and **especially when fanning out
sub-agents that *edit* the repo** — give each its own git worktree rather than sharing
one working tree. Concurrent edits in a shared tree entangle (uncommitted changes get
swept into the wrong commit) and churn the index. (Read-only/Explore sub-agents don't
need this — only parallel *mutating* ones do.)

- Ephemeral fan-out: spawn the sub-agent with the Agent tool's `isolation: "worktree"`.
- Persistent/human worktrees: use `git-wt` (see the global worktree guidance).

When using `git-wt` on lackpy, share `.lackpy/` (config, kits, templates — read-mostly)
from `main/` across worktrees via `.git-worktree-shared`. Do **not** blanket-share
`.bird/` (blq's DuckDB) when parallel sub-agents run builds — DuckDB is single-writer,
so a shared DB invites lock contention; one history is only worth it for serial work.
Never share source or `packages/` — those must diverge per branch.
