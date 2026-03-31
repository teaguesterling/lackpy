# lackpy v0.3 Design Spec: .lpy Format, CLI Redesign, Error Correction

## Overview

Three connected changes that evolve lackpy from a library-with-CLI into a proper tool:

1. **`.lpy` file format** — self-describing executable programs with metadata, provider mappings, typed params, and return declarations
2. **CLI redesign** — `lackpy` as a program runner (like `python`), `lackpyctl` as environment manager (like `pip`/`venv`)
3. **Error correction chain** — multi-strategy retry pipeline for fixing invalid generated programs

## 1. The `.lpy` File Format

### Structure

An `.lpy` file has three sections: frontmatter, setup, and implementation.

```yaml
---
name: count-lines
description: Count lines in files matching a pattern
pattern: "count lines in {pattern}"
tools:
  - read
  - glob
returns:
  type: int
  description: Total number of files found
---
# Setup
pattern: str = pattern if pattern else "**/*.py"

# Implementation
files = glob(pattern)
for f in files:
    content = read(f)
    print(f"{f}: {len(content.splitlines())}")
len(files)
```

### Frontmatter Fields

All fields are optional except `tools` (which can be inferred from the implementation if omitted).

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Program name (defaults to filename stem) |
| `description` | `str` | Human-readable description |
| `pattern` | `str` | Intent match pattern with `{placeholder}` captures (makes it a template) |
| `tools` | `list` | Tool names the program uses, with optional provider pinning |
| `providers` | `list` | Provider definitions for resolving tools |
| `returns` | `dict` or `str` | Return type and description |

### Tools Declaration

Each entry in `tools:` is either a bare name (resolved from toolbox/providers) or a `name: provider` pair for explicit pinning:

```yaml
tools:
  - read                        # resolve from providers list or toolbox
  - glob                        # resolve from providers list or toolbox
  - find_definitions: fledgling # explicitly from the "fledgling" provider
  - run_tests: blq              # explicitly from the "blq" provider
  - errors: blq                 # explicitly from the "blq" provider
```

### Providers Declaration

Providers define how to reach external tools. Each provider has a `name`, a `type`, and type-specific configuration. An optional `mappings` dict handles name differences between lackpy tool names and the provider's actual function/tool names.

```yaml
providers:
  - name: fledgling
    type: mcp
    server: fledgling
  - name: blq
    type: python
    package: blq
    module: blq.api
    mappings:
      errors: get_errors        # lackpy name "errors" -> function "get_errors"
```

Provider types:

| Type | Required Fields | Description |
|------|----------------|-------------|
| `builtin` | (none) | Built-in lackpy tools |
| `python` | `module` | Python import. Optional: `package` (for dependency checking), `function` (if only one tool), `mappings` |
| `mcp` | `server` | MCP server. Optional: `mappings` |
| `inline` | `path` | Python file in `.lackpy/tools/`. Optional: `mappings` |

Provider references with `@` resolve from `.lackpy/providers/`:

```yaml
providers:
  - @fledgling              # loads .lackpy/providers/fledgling.yml
  - @blq                    # loads .lackpy/providers/blq.yml
```

A `.lackpy/providers/fledgling.yml` file:

```yaml
name: fledgling
type: mcp
server: fledgling
```

### Returns Declaration

Simple form:

```yaml
returns: int
```

Full form:

```yaml
returns:
  type: list[dict]
  description: File info with name and line count
  schema:
    file: str
    lines: int
```

The `returns` field is used for:
- MCP tool schema generation (when `.lpy` is exposed as an MCP tool)
- Documentation (`lackpy --help script.lpy`)
- Optional runtime type checking of the last expression

### Setup Section

The setup section contains annotated assignments that declare parameters with defaults. It appears between the `# Setup` and `# Implementation` comments.

```python
# Setup
pattern: str = pattern if pattern else "**/*.py"
max_depth: int = max_depth if max_depth else 10
verbose: bool = verbose if verbose else False
```

Each line is an `ast.AnnAssign` node — an annotated assignment. This is valid Python syntax and valid lackpy (after adding `ast.AnnAssign` to `ALLOWED_NODES`).

The pattern `name: type = name if name else default` handles both cases:
- Caller provided the param: `name` is already set, keeps the value
- Caller didn't provide it: `name` is not set... which would be a `NameError`

**Problem:** `name if name else default` fails if `name` isn't defined at all. We need to inject `None` for missing params, or use a different pattern.

**Solution:** The runner injects `None` for any param declared in Setup that wasn't provided by the caller. So:

```python
pattern: str = pattern if pattern else "**/*.py"
```

Works because the runner pre-injects `pattern = None` if the caller didn't provide it. The annotated assignment then sets the default.

Alternatively, the simpler pattern:

```python
pattern: str = pattern or "**/*.py"
```

### Implementation Section

Everything after `# Implementation` is the lackpy program. This is what the inferencer generates. It uses only tools from the `tools:` list and allowed builtins.

### File Unification

`.lpy` files unify what were previously separate concepts:

| Previous | Now |
|----------|-----|
| Templates (`.tmpl` files) | `.lpy` with `pattern:` field |
| Scripts (manual programs) | `.lpy` without `pattern:` |
| Generated programs | Output of `lackpy -c`, saved as `.lpy` |

Templates are just `.lpy` files that have a `pattern:` field for matching intents.

### AST Changes

Add `ast.AnnAssign` to `ALLOWED_NODES` to support the Setup section's annotated assignments.

## 2. CLI Redesign

### Principle

`lackpy` is like `python` — it runs programs. `lackpyctl` is like `pip`/`venv` — it manages the environment.

### `lackpy` — Program Runner

```bash
# REPL — interactive mode
lackpy

# Run a file
lackpy script.lpy
lackpy script.lpy --param pattern="src/*.py"
lackpy script.lpy --param pattern="src/*.py" --param verbose=true

# One-shot from intent
lackpy -c "read file main.py and count its lines"
lackpy -c "find all python files" --kit read,glob

# Validate
lackpy --validate script.lpy
lackpy --validate < program.txt

# Stdin
echo "files = glob('**/*.py')\nlen(files)" | lackpy --kit read,glob
cat program.lpy | lackpy

# Create an .lpy from intent
lackpy --create -c "count lines in files" --kit read,glob --name count-lines

# Generate without running
lackpy --generate -c "find all python files" --kit read,glob

# Sandbox control
lackpy --sandbox=readonly script.lpy
lackpy --sandbox=build -c "edit the file" --kit read,edit

# Help for a specific script
lackpy --help script.lpy
```

Argument parsing:

| Argument | Description |
|----------|-------------|
| `<file>` | Run an `.lpy` file |
| `-c <intent>` | One-shot from intent (like `python -c`) |
| `--kit <tools>` | Comma-separated tool list (for `-c` and stdin) |
| `--param <name>=<value>` | Pass a parameter (repeatable) |
| `--validate` | Validate without running |
| `--generate` | Generate program without running (print to stdout) |
| `--create` | Generate and save as `.lpy` file |
| `--name <name>` | Name for `--create` |
| `--sandbox <preset>` | Sandbox preset |
| `--help <file>` | Show script info (params, returns, description) |

No positional argument and no `-c`: launch REPL.

### `lackpyctl` — Environment Manager

```bash
# Setup
lackpyctl init
lackpyctl init --ollama-url http://localhost:11435 --ollama-model qwen2.5-coder:1.5b

# Kit management
lackpyctl kit list
lackpyctl kit info debug
lackpyctl kit create my-kit --tools read,glob,find_callers

# Toolbox management
lackpyctl toolbox list
lackpyctl toolbox show read
lackpyctl toolbox add --name my_tool --type python --module foo.bar --function baz

# Provider management
lackpyctl provider list
lackpyctl provider add fledgling --type mcp --server fledgling
lackpyctl provider show fledgling

# Template management
lackpyctl template list
lackpyctl template test count-lines

# Info
lackpyctl status
lackpyctl spec
```

### REPL

The REPL is an interactive session where you type intents or lackpy code:

```
$ lackpy
lackpy v0.2.0 | kit: debug | ollama: qwen2.5-coder:1.5b
>>> find all python files
files = glob('**/*.py')
files
['src/main.py', 'tests/test_main.py']
>>> read file src/main.py
content = read('src/main.py')
content
'def hello():\n    print("hello")\n'
>>>
```

The REPL:
- Shows the generated program before output (so you can see what it did)
- Maintains a session namespace (variables persist between entries)
- Supports both intents (sent to inference) and raw lackpy code (executed directly)
- Kit is set at start (`lackpy --kit read,glob`) or defaults to config default

Distinguishing intent from code: if the input parses as valid lackpy, run it directly. If it doesn't parse, send it to inference.

### Entry Points

```toml
[project.scripts]
lackpy = "lackpy.cli:main"
lackpyctl = "lackpy.ctl:main"
```

The existing `cli.py` is refactored into:
- `cli.py` — the `lackpy` runner (file, -c, stdin, REPL, --validate, --generate, --create)
- `ctl.py` — the `lackpyctl` manager (init, kit, toolbox, provider, template, status, spec)

Both use `service.py` underneath.

## 3. Error Correction Chain

### Overview

When a generated program fails validation, lackpy runs a chain of correction strategies before giving up. The chain is hardcoded for v0.3, with the interface designed for future extensibility (including agent-riggs integration).

### The Chain

```
Generate program
  |
  v
Validate
  |
  +-- Valid? --> Execute
  |
  +-- Invalid:
      |
      v
  Strategy 1: Deterministic Cleanup
      Strip imports, fences, preamble.
      Re-validate.
      |
      +-- Valid? --> Execute
      |
      v
  Strategy 2: Few-Shot Correction (same provider)
      Show the model its bad output + positive error hints.
      Higher temperature (0.6).
      Re-validate.
      |
      +-- Valid? --> Execute
      |
      v
  Strategy 3: Fresh Fixer Prompt (same provider)
      New conversation with a "fix this code" system prompt.
      Provide: broken code, errors, namespace, intent.
      Re-validate.
      |
      +-- Valid? --> Execute
      |
      v
  Strategy 4: Next Provider
      Fall through to next inference provider in priority order.
      Full chain restarts for each provider.
      |
      +-- Valid? --> Execute
      |
      v
  All providers exhausted --> Error
```

### Strategy 1: Deterministic Cleanup

Already implemented (`sanitize.py`). Extended with:
- Strip `import` lines (the model adds them reflexively)
- Strip `from X import Y` lines
- Replace `open(path)` calls with `read(path)` where possible (AST transform)

These are safe transforms because:
- Import lines have no effect in the sandbox (no modules available)
- `open(path).read()` is semantically equivalent to `read(path)` for our use case

```python
def deterministic_cleanup(program: str) -> str:
    """Apply safe, deterministic fixes to common model mistakes."""
    # Strip import lines
    lines = program.split("\n")
    lines = [l for l in lines if not l.strip().startswith(("import ", "from "))]
    program = "\n".join(lines)

    # AST-level: replace open(path).read() with read(path)
    # AST-level: replace os.path.basename(x) with x.split('/')[-1]
    program = _fix_open_calls(program)

    return program
```

### Strategy 2: Few-Shot Correction

Already implemented. The model sees its bad output in the conversation history and gets positive hints at higher temperature.

### Strategy 3: Fresh Fixer Prompt

A new conversation with a different system prompt optimized for code correction, not generation:

```
You are a code fixer. The following code was generated for a restricted
Python environment but contains errors.

Fix the code to use ONLY these functions:
{namespace_description}

The code should accomplish: {intent}

Broken code:
{broken_program}

Errors:
{validation_errors}

Output ONLY the fixed code — no explanation, no markdown fences.
```

This uses the same model but a fresh context — no conversation history to anchor it to the wrong patterns. The fixer prompt is shorter and more focused than the generation prompt.

### Strategy 4: Provider Fallthrough

Same as current behavior — try the next provider in the configured order. Each provider gets the full correction chain (strategies 1-3) before falling through.

### Interface

```python
@dataclass
class CorrectionStrategy:
    name: str
    apply: Callable[[str, list[str], str, str], Awaitable[str | None]]
    # (program, errors, namespace_desc, intent) -> fixed_program or None

class CorrectionChain:
    def __init__(self, strategies: list[CorrectionStrategy]):
        self._strategies = strategies

    async def correct(
        self, program: str, errors: list[str],
        namespace_desc: str, intent: str,
    ) -> str | None:
        for strategy in self._strategies:
            result = await strategy.apply(program, errors, namespace_desc, intent)
            if result is not None:
                validation = validate(result, ...)
                if validation.valid:
                    return result
        return None
```

The chain is constructed in the dispatcher and passed the list of strategies. For v0.3 the strategies are hardcoded. Future versions could make them configurable or integrate with agent-riggs for trust-based strategy selection.

### Dispatch Integration

The dispatch loop changes from:

```
for provider in providers:
    program = provider.generate(intent)
    if valid: return
    program = provider.generate(intent, error_feedback)  # one retry
    if valid: return
```

To:

```
for provider in providers:
    program = provider.generate(intent)
    program = deterministic_cleanup(program)
    if valid: return
    program = correction_chain.correct(program, errors, namespace_desc, intent)
    if valid: return
```

The correction chain encapsulates all retry logic. The dispatcher just generates, cleans, validates, and optionally corrects.

## 4. Grammar Changes

### New ALLOWED_NODES

- `ast.AnnAssign` — for Setup section annotated assignments (`pattern: str = ...`)

### Validation Rule for AnnAssign

Annotated assignments are only allowed in the Setup section (before `# Implementation`). The validator can enforce this by checking that all `AnnAssign` nodes appear before any non-`AnnAssign` statements, or by splitting the program at the `# Implementation` comment.

Simpler approach: allow `AnnAssign` anywhere. It's just a typed assignment — no security implications beyond what `Assign` already allows. The `# Setup` / `# Implementation` comments are conventions for readability, not enforced boundaries.

## 5. Service Layer Changes

### New Methods

```python
async def run_lpy(
    path: Path,
    params: dict[str, Any] | None = None,
    sandbox: str | SandboxConfig | None = None,
) -> DelegateResult:
    """Load and run an .lpy file."""

def parse_lpy(path: Path) -> LpyFile:
    """Parse an .lpy file into structured components."""

def create_lpy(
    program: str,
    name: str,
    tools: list[str],
    params: dict[str, dict] | None = None,
    returns: str | dict | None = None,
    pattern: str | None = None,
    providers: list[dict] | None = None,
) -> Path:
    """Create and save an .lpy file."""
```

### LpyFile Dataclass

```python
@dataclass
class LpyFile:
    name: str
    description: str
    pattern: str | None
    tools: list[str | dict]        # bare names or {name: provider} pairs
    providers: list[dict]
    params: dict[str, dict]        # extracted from Setup section AnnAssign nodes
    returns: str | dict | None
    setup: str                     # Setup section source
    implementation: str            # Implementation section source
    full_program: str              # setup + implementation combined
    path: Path | None
```

## 6. MCP Server Changes

`.lpy` files with a `returns:` field can be exposed as MCP tools:

```python
@mcp.tool()
async def count_lines(pattern: str = "**/*.py") -> int:
    """Count lines in files matching a pattern."""
    return await service.run_lpy(
        Path(".lackpy/templates/count-lines.lpy"),
        params={"pattern": pattern},
    )
```

The MCP server could auto-discover `.lpy` files in `.lackpy/templates/` and expose them as tools, using the frontmatter for parameter schemas and return types.

## 7. Migration

### Template Migration

Existing `.tmpl` files are converted to `.lpy`:

```
# Old: .lackpy/templates/find-callers.tmpl
---
name: find-callers
pattern: "find (all )?(callers|usages|references) of {name}"
success_count: 10
fail_count: 0
---
results = find_callers("{name}")
results
```

Becomes:

```
# New: .lackpy/templates/find-callers.lpy
---
name: find-callers
description: Find all callers of a function
pattern: "find (all )?(callers|usages|references) of {name}"
tools:
  - find_callers
returns:
  type: list[dict]
  description: Caller locations
---
# Setup
name: str = name or ""

# Implementation
results = find_callers(name)
results
```

The templates provider is updated to read `.lpy` files instead of `.tmpl` files. A migration command `lackpyctl migrate-templates` converts existing `.tmpl` files.

### CLI Migration

The current `lackpy` CLI commands move to `lackpyctl`. The `lackpy` entry point becomes the runner. Both entry points coexist during transition — `lackpy delegate` still works but prints a deprecation warning pointing to `lackpy -c`.
