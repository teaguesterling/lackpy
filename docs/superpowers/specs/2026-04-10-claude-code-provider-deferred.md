# Claude Code Provider (Deferred)

*Status: deferred. Waiting on umwelt's nsjail executor to land before building.*

## Why it's deferred

The value of a `ClaudeCodeProvider` is **safe delegation** — running a
Claude Code session inside bounds we can actually enforce. The whole
point of `allowed_tools`, `disallowed_tools`, `mcp_servers`, and
`strict_mcp` is that those limits are real. Without a sandboxed
subprocess runner underneath, the provider is a cosmetic wrapper over a
`claude -p` invocation that honors none of its apparent bounds.

That sandboxed runner naturally lives in umwelt — it's the output of
the nsjail compiler operating on a compiled view. Rather than
pre-inventing a subprocess layer in lackpy that umwelt will later
replace, we wait for the real thing.

## What's preserved here

The flag mapping, config schema, output parsing, and executor-injection
shape are the valuable parts of the work that happened in the
conversation where this was designed. Keeping them in a spec means when
umwelt lands we can build the provider in a couple of hours instead of
rediscovering the CLI flag surface.

## Claude Code CLI flags worth wrapping

From `claude --help`:

| CLI flag | Provider kwarg | Purpose |
|---|---|---|
| `--model <alias-or-id>` | `model` | claude model (default `"haiku"`) |
| `--print` | always on | one-shot mode, no interactive session |
| `--output-format <fmt>` | `output_format` | `"text"`, `"json"`, `"stream-json"` — default `"json"` |
| `--permission-mode <mode>` | `permission_mode` | `"acceptEdits"` default for delegates |
| `--bare` | `bare` | skip hooks, LSP, auto-memory, CLAUDE.md — default `True` for delegates |
| `--add-dir <dir>` | `add_dirs` + `working_dir` | directories the session can access |
| `--allowed-tools <...>` | `allowed_tools` | tool allow list (space or comma separated) |
| `--disallowed-tools <...>` | `disallowed_tools` | tool deny list (default `["Bash"]`) |
| `--mcp-config <file-or-json...>` | `mcp_servers` | MCP server JSON configs (file paths or dicts) |
| `--strict-mcp-config` | `strict_mcp` | ignore ambient MCP config (default `True` for delegates) |
| `--append-system-prompt <text>` | `append_system_prompt` + `namespace_desc` | add tool descriptions to the system prompt |

## Provider API (target shape)

```python
from lackpy.infer.providers.claude_code import ClaudeCodeProvider

provider = ClaudeCodeProvider(
    model="haiku",
    permission_mode="acceptEdits",
    working_dir=None,                          # None = CWD at call time
    add_dirs=None,                             # additional --add-dir entries
    append_system_prompt=None,                 # extra system prompt text
    allowed_tools=["Read", "Grep", "Glob"],    # --allowed-tools
    disallowed_tools=["Bash"],                 # --disallowed-tools (default)
    mcp_servers=None,                          # list[dict] or list[path]
    strict_mcp=True,                           # isolate from ambient MCP
    bare=True,                                 # minimal session for delegates
    output_format="json",                      # structured single-result output
    timeout_seconds=300,
    executor=None,                             # INJECTED: sandboxed runner
)

# Per-call config override, merged into the constructor defaults
# on a per-key basis. Config wins for any key it specifies;
# other keys fall through to constructor.
await provider.generate(
    intent="Fix the admin login bug",
    namespace_desc="",
    config={
        "allowed_tools": ["Read", "Edit"],
        "mcp_servers": [{"fledgling": {"command": "fledgling"}}],
        "working_dir": "/tmp/workspace",
    },
)
```

## Executor injection

The provider never imports `subprocess` directly. Instead it takes an
`executor` callable at construction time and delegates the actual
subprocess invocation. The executor has a narrow async interface:

```python
# Executor protocol (sketch)
async def executor(
    argv: list[str],
    cwd: Path | None,
    timeout: int,
) -> str | None:
    """Run a command in a sandbox and return stdout, or None on failure."""
    ...
```

When umwelt ships, it provides the concrete executor — probably
something like `umwelt.runners.nsjail.run_in_sandbox(view, argv, ...)`
that compiles the view to an nsjail config and runs the command inside
the jail. The provider passes its assembled argv to that executor and
doesn't know or care about the sandbox mechanics.

For testing in v1, the executor is a mock that captures the argv and
returns a canned response.

## Flag composition logic

The provider's `generate` method does four things:

1. **Resolve config**: merge constructor kwargs with per-call `config`
   dict. Per-call values override constructor values on a per-key
   basis (not a wholesale replacement).
2. **Compose prompts**: user prompt is the intent; system prompt
   append combines `append_system_prompt` + `namespace_desc` (so the
   delegate knows what tools it has in its workspace).
3. **Build MCP args**: for each entry in `mcp_servers`, if it's a
   dict, serialize to a tempfile; if it's a path string, use it
   directly. Pass all paths as a single `--mcp-config` flag with
   space-separated values.
4. **Assemble argv**: produce the full `claude` command line, in the
   right order (model, output-format, permission-mode, bare flag,
   add-dirs, allow/deny lists, MCP config, strict-mcp flag, system
   prompt append, and finally the user prompt as the last positional
   argument).

Then delegate to the executor with the argv, cwd (resolved from
`working_dir`), and timeout. Parse the output based on
`output_format`:

- `"text"` → return stdout stripped
- `"json"` → parse as JSON, look for `result` / `content` / `text` /
  `message` keys in that order
- `"stream-json"` → iterate lines, return the last assistant message

## Output parsing shape (to verify empirically when ready)

Claude Code's `--output-format json` produces a single-result envelope.
The exact schema isn't documented in `claude --help`, so when we
implement this we'll need to run a real invocation and inspect the
actual JSON shape. Likely fields based on similar tools:

- `result` or `content` — the final assistant message text
- `messages` — full conversation log
- `cost_usd` / `total_cost_usd` — cost info
- `duration_ms` / `wall_time_ms` — timing
- `is_error` — error flag
- `stop_reason` — why the session ended

The provider's `_parse_output` helper should try these field names in
order and gracefully fall back to the raw JSON string if none match.

## Defaults rationale

- **`model="haiku"`**: delegates should be cheap and fast by default.
  Outer agent already did the expensive reasoning before deciding to
  delegate.
- **`permission_mode="acceptEdits"`**: delegate can edit files without
  asking. The outer agent already decided to delegate, so we don't
  want the inner session pausing for confirmation on every edit.
- **`bare=True`**: clean session with no ambient state (no CLAUDE.md
  discovery, no auto-memory, no LSP, no hooks). Designed for exactly
  the delegate use case.
- **`strict_mcp=True`**: delegate only sees MCP servers we explicitly
  provide. No surprise tools from whatever config the user happens to
  have.
- **`disallowed_tools=["Bash"]`**: conservative default — no shell
  access unless explicitly allowed. Callers can pass an empty list if
  they need shell.
- **`output_format="json"`**: structured result, no fragile text
  scraping.

## Why this provider is different from AnthropicProvider

The existing `AnthropicProvider` calls the Anthropic API directly. It
produces a single text completion — no tool use, no file access, no
multi-step reasoning inside the call.

The `ClaudeCodeProvider` produces a delegate *session* — a full Claude
Code invocation with tool use, file access, and multi-step reasoning.
The output is the *result* of whatever the delegate accomplished, not
a single completion. This is why claude-code is the right backend
rather than the API.

Put differently: `AnthropicProvider` asks Claude to write a program.
`ClaudeCodeProvider` asks Claude to *do a task*, where Claude has its
own tools for reading, editing, and testing, all operating inside the
bounded workspace umwelt defined.

## Dependencies on work that's not ready yet

- **umwelt nsjail compiler** — provides `view → nsjail config` translation
- **umwelt workspace executor** — provides the context manager that
  builds a workspace and yields it to an executor
- **umwelt runners.nsjail** — provides the concrete `argv → stdout`
  runner the provider injects as its executor

Until these exist, building the provider produces an unusable
artifact. Better to wait.

## Revisit trigger

Build the provider when any of these are true:

1. umwelt ships a minimum-viable `runners.nsjail` that can take an argv
   and a view and return stdout under real sandbox enforcement
2. A specific lackpy consumer needs claude-code delegation *today* and
   is willing to accept an unsandboxed version as an interim
3. The executor-injection shape proves useful for another provider
   (e.g., a local binary provider) that could share the same interface

## Related documents

- `docs/superpowers/specs/2026-04-10-views-and-nsjail-integration.md`
  — the umwelt-nsjail integration plan
- `docs/superpowers/specs/2026-04-10-nsjail-view-config-format.md` —
  the standalone config format spec
- `~/Projects/umwelt/docs/vision/` — umwelt's vision and architecture docs
