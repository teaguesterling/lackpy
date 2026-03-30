# lackpy Design Spec

*Python that lacks most of Python. A micro-inferencer that translates intent into sandboxed tool-composition programs.*

## Overview

lackpy is an MCP tool and CLI that accepts natural language intent from an outer agent or human, generates a restricted Python program using a pluggable inference pipeline, validates it against a strict AST whitelist and namespace, runs it inside a sandboxed environment, and returns a structured execution trace.

It is a complete star topology at micro scale: the outer agent is the Principal, the inference pipeline is the Inferencer, the AST filter + namespace is the Harness, and the executor is the Executor. The whole thing is exposed as both an MCP server and a CLI, both backed by a unified service layer.

lackpy is part of the Rigged developer tool suite (agent-riggs, kibitzer, fledgling, blq, jetsam) but is independently installable and usable. agent-riggs audits lackpy usage and can register templates via the lackpy API, but is not a dependency.

## Core Architecture

```
delegate(intent, kit, params, sandbox, rules) -> trace

  Kit Resolution
  |-- Predefined kit name ("debug", "implement")
  |-- Explicit tool list (["read", "glob", "find_callers"])
  |-- Full mapping ({"tool_name": {provider details}})
  +-- Quartermaster (v2: model-inferred kit from intent)

  Inference Pipeline (plugin-based, priority-ordered)
  |-- Templates (pattern match against .lackpy/templates/)
  |-- Rules (keyword -> program, deterministic)
  |-- Ollama (local model, configurable)
  |-- Anthropic API (Haiku fallback)
  +-- ... (extensible: Bedrock, OpenAI, remote Ollama, cache, etc.)

  -> AST validation (whitelist walk + custom rules)
  -> Namespace validation (all calls in resolved kit or allowed builtins)
  -> Grade computation (join of tool grades from kit)
  -> Execution (v1: AST compile + restricted run; v2: nsjail sandbox)
  -> Structured trace capture
  -> Return trace to caller
```

## Package Structure

```
lackpy/
|-- __init__.py              # Version, public API
|-- lang/
|   |-- __init__.py
|   |-- grammar.py           # ALLOWED_NODES, FORBIDDEN_NODES, FORBIDDEN_NAMES
|   |-- validator.py         # ast.parse -> walk -> validate against whitelist + namespace
|   |-- grader.py            # Compute grade from tool grades in resolved kit
|   |-- rules.py             # Custom validation rules (AST visitors)
|   +-- spec.py              # The lackpy language spec as structured data
|-- kit/
|   |-- __init__.py
|   |-- registry.py          # Kit resolution: name/list/dict/None -> ResolvedKit
|   |-- toolbox.py           # Plugin-based tool provider store
|   +-- providers/
|       |-- __init__.py
|       |-- base.py          # ToolProvider protocol
|       |-- builtin.py       # Built-in tools (read, glob, etc.)
|       +-- python.py        # Python import-based tools (fledgling, blq, etc.)
|-- infer/
|   |-- __init__.py
|   |-- dispatch.py          # Priority-ordered provider dispatch
|   |-- prompt.py            # System prompt template (Jupyter cell framing)
|   |-- sanitize.py          # Output sanitization (strip fences, preamble)
|   +-- providers/
|       |-- __init__.py
|       |-- base.py          # InferenceProvider protocol
|       |-- templates.py     # Tier 0: pattern library
|       |-- rules.py         # Tier 1: keyword-based generation
|       |-- ollama.py        # Tier 2: Ollama client
|       +-- anthropic.py     # Tier 3: Anthropic API fallback
|-- run/
|   |-- __init__.py
|   |-- base.py              # Executor protocol
|   |-- runner.py            # v1: AST compile + restricted run with traced namespace
|   +-- trace.py             # Trace data structures and capture
|-- service.py               # Unified service layer: delegate, generate, run_program, create
|-- mcp/
|   |-- __init__.py
|   +-- server.py            # FastMCP server: thin adapter over service
|-- cli.py                   # CLI: thin adapter over service
|-- config.py                # Configuration loading (config.toml)
+-- py.typed                 # PEP 561 marker
```

## Module Design

### `lang/` — Validation and Grading

The security-critical core. **Zero dependencies beyond Python stdlib.** Uses only the `ast` module.

#### Allowed AST Nodes

```python
ALLOWED_NODES = {
    # Structural
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign,
    ast.For, ast.If, ast.With,

    # Expressions
    ast.Call, ast.Name, ast.Attribute, ast.Subscript,
    ast.List, ast.Dict, ast.Tuple, ast.Set,
    ast.ListComp, ast.DictComp, ast.SetComp,
    ast.Compare, ast.BoolOp, ast.UnaryOp, ast.BinOp,
    ast.JoinedStr, ast.FormattedValue,  # f-strings
    ast.Constant, ast.Starred, ast.Index, ast.Slice,

    # Comprehension internals
    ast.comprehension, ast.IfExp,

    # Context nodes (required by Python AST)
    ast.Load, ast.Store, ast.Del,

    # Operators
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.FloorDiv,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Is, ast.IsNot, ast.In, ast.NotIn,
    ast.And, ast.Or, ast.Not, ast.USub, ast.UAdd,
}
```

#### Forbidden Constructs

```python
FORBIDDEN_NODES = {
    ast.Import, ast.ImportFrom,
    ast.FunctionDef, ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
    ast.While,                    # No unbounded iteration
    ast.Try, ast.ExceptHandler,   # No exception swallowing
    ast.Raise,                    # No exception raising
    ast.Global, ast.Nonlocal,     # No scope escape
    ast.Yield, ast.YieldFrom,     # No generators
    ast.Await, ast.AsyncFor, ast.AsyncWith,  # No async
    ast.Assert,                   # Outer agent evaluates, not the program
    ast.Delete,                   # No del statements
    ast.Match,                    # No structural pattern matching
}

FORBIDDEN_NAMES = {
    '__import__', 'open',
    'globals', 'locals', 'vars', 'dir',
    'getattr', 'setattr', 'delattr', 'hasattr',
    '__builtins__', '__build_class__',
    'breakpoint', 'exit', 'quit',
    'type', 'super', 'classmethod', 'staticmethod', 'property',
    'memoryview', 'bytearray', 'bytes',
    'map', 'filter', 'reduce',  # Use comprehensions instead
    'input',                     # No stdin
}
```

Note: `eval`, `exec`, and `compile` are not in FORBIDDEN_NAMES because they
are already blocked by the namespace check — they are not in ALLOWED_BUILTINS
and would not be in any kit. FORBIDDEN_NAMES is specifically for names that
might appear as variable references or could be confused with allowed constructs.

#### Allowed Builtins

```python
ALLOWED_BUILTINS = {
    'len', 'sorted', 'reversed', 'enumerate', 'zip', 'range',
    'min', 'max', 'sum', 'any', 'all', 'abs', 'round',
    'str', 'int', 'float', 'bool', 'list', 'dict', 'set', 'tuple',
    'isinstance', 'print',
}
```

#### Validation Pipeline

1. **Parse**: `ast.parse(program_text)` — reject on syntax error
2. **Node walk**: Walk entire AST, reject if any node type not in `ALLOWED_NODES`
3. **Name check**: All `ast.Name` nodes checked against `FORBIDDEN_NAMES`
4. **Namespace check**: All `ast.Call` nodes where function is a `Name` — reject if not in resolved kit or `ALLOWED_BUILTINS`
5. **For-loop check**: All `ast.For` nodes must iterate over a call result or variable
6. **String literal check**: No string contains `__` (blocks dunder access via string manipulation)
7. **Custom rules**: Any additional AST visitor rules provided by kit, caller, or config

Returns:

```python
@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]
    grade: Grade
    calls: list[str]      # tool calls found in the program
    variables: list[str]  # variables assigned in the program
```

#### Custom Validation Rules

Additional rules are AST visitors that can only **tighten** the core checks, never loosen them. Each rule is a callable that takes an `ast.Module` and returns a list of error strings (empty means pass).

Examples:
- `no_loops` — forbid all `ast.For` nodes
- `max_depth(n)` — limit AST nesting depth
- `max_calls(n)` — limit number of tool calls
- `no_nested_calls` — forbid calls as arguments to other calls (force intermediate variables)

Note: custom rules can only restrict within the bounds of `ALLOWED_NODES`. A rule cannot introduce new node types (e.g., requiring `Try`/`ExceptHandler` would conflict with `FORBIDDEN_NODES`). If a future use case needs expanded node support, that's a spec change, not a custom rule.

Rules can be specified by:
- The kit definition (kit-level constraints)
- The caller explicitly (`delegate(..., rules=["no_loops"])`)
- Configuration presets

#### Grade Computation

Grade is the join (max) of all tool grades in the resolved kit. Each tool in the toolbox optionally declares `grade_w` (world coupling) and `effects_ceiling`. Tools without declared grades default to a conservative maximum.

```python
@dataclass
class Grade:
    w: int  # world coupling (0=pure, 1=pinhole read, 2=scoped exec, 3=scoped write)
    d: int  # effects ceiling
```

### `kit/` — Tool Selection and Registry

#### The Toolbox

The toolbox is a **plugin-based provider store** containing all known tools and how to reach them. It lives at `.lackpy/toolbox/` as configuration, with provider plugins handling resolution to callables.

Each toolbox entry specifies:
- **name**: Tool identifier used in programs
- **provider**: Which provider plugin resolves this tool (`builtin`, `python`, `mcp-local`, `mcp-system`, `inline`, etc.)
- **provider_config**: Provider-specific configuration (module + function, server + tool name, code path, etc.)
- **description**: Human-readable description (used in inference prompts)
- **args**: Argument spec (names, types, descriptions)
- **returns**: Return type description
- **grade_w**: Optional world coupling grade
- **effects_ceiling**: Optional effects ceiling grade

```python
@dataclass
class ToolSpec:
    name: str
    provider: str
    provider_config: dict
    description: str
    args: list[ArgSpec]
    returns: str
    grade_w: int = 3       # conservative default
    effects_ceiling: int = 3
```

#### Tool Providers

Provider plugins implement a common protocol:

```python
class ToolProvider(Protocol):
    name: str
    def available(self) -> bool: ...
    def resolve(self, tool_spec: ToolSpec) -> Callable: ...
```

v1 providers:
- **`builtin`**: Tools implemented inside lackpy itself (read, glob, write, edit)
- **`python`**: Import a function from an installed Python package (e.g., `fledgling.api:find_definitions`)

Future providers (v2+):
- **`mcp-local`**: Connect to a project-local MCP server (`.mcp.json`)
- **`mcp-system`**: Connect to a system-level MCP server (`~/.claude/mcp.json`)
- **`inline`**: Load a Python file from `.lackpy/tools/`

#### Kit Resolution

The `kit` parameter accepts four forms:

| Form | Type | Behavior |
|------|------|----------|
| Predefined name | `str` | Look up kit file in `.lackpy/kits/`, resolve each tool from toolbox |
| Explicit list | `list[str]` | Resolve each named tool from toolbox |
| Full mapping | `dict` | Tool names mapped to toolbox names or inline provider configs |
| Quartermaster | `None` | v1: raises `NotImplementedError`. v2: model infers minimal kit from intent |

Predefined kits live as flat files in `.lackpy/kits/`:

```yaml
---
name: debug
description: Read-only code exploration and analysis
---
read
glob
find_definitions
find_callers
run_tests
errors
```

Kit resolution produces a `ResolvedKit`:

```python
@dataclass
class ResolvedKit:
    tools: dict[str, ToolSpec]       # name -> spec with resolved callable
    callables: dict[str, Callable]   # name -> actual callable for execution
    grade: Grade                     # aggregate grade (join of tool grades)
    description: str                 # formatted namespace description for prompts
```

### `infer/` — Inference Pipeline

#### Provider-Based Dispatch

The inference pipeline is a priority-ordered list of provider plugins. Each provider implements:

```python
class InferenceProvider(Protocol):
    name: str
    def available(self) -> bool: ...
    async def generate(
        self, intent: str, namespace_desc: str, config: dict,
        error_feedback: list[str] | None = None,
    ) -> str | None: ...
```

Dispatch iterates providers in configured order. For each provider:
1. Check `available()` — skip if not
2. Call `generate()` — skip if returns `None`
3. Sanitize output (strip markdown fences, preamble)
4. Validate against kit + rules
5. If valid, return. If invalid, retry once with error feedback (provider-dependent), then try next provider.

#### v1 Providers

**Templates** (`templates.py`):
- Pattern-match intent against `.lackpy/templates/` files
- Each template has a regex/glob pattern with `{placeholder}` captures
- Returns instantiated program if match confidence exceeds threshold
- Tracks success/fail counts for ratchet

Template file format (`.lackpy/templates/`):
```yaml
---
name: find-callers
pattern: "find (all )?(callers|usages|references) of {name}"
success_count: 12
fail_count: 1
---
results = find_callers("{name}")
[f"{r['file']}:{r['line']}" for r in results]
```

**Rules** (`rules.py`):
- Deterministic keyword-to-program mapping
- Finite set of high-confidence patterns (e.g., "read file X" -> `content = read("X")\ncontent`)
- No model call, zero cost
- Returns `None` if no rule matches

**Ollama** (`ollama.py`):
- Connects to local or remote Ollama instance
- Default model: `qwen2.5-coder:1.5b`
- System prompt uses Jupyter cell framing with namespace description from resolved kit
- One retry with validation error feedback before failing
- Requires `ollama` package (optional dependency)

**Anthropic** (`anthropic.py`):
- Haiku fallback when Ollama unavailable or fails
- Same prompt construction as Ollama
- Requires `anthropic` package (optional dependency)

#### Configuration

Providers are configured in `.lackpy/config.toml`:

```toml
[inference]
order = ["templates", "rules", "ollama-local", "anthropic-haiku"]

[inference.providers.templates]
plugin = "templates"
threshold = 0.8
min_uses = 5

[inference.providers.rules]
plugin = "rules"

[inference.providers.ollama-local]
plugin = "ollama"
host = "http://localhost:11434"
model = "qwen2.5-coder:1.5b"
keep_alive = "30m"
temperature = 0.2

[inference.providers.anthropic-haiku]
plugin = "anthropic"
model = "claude-haiku-4-5-20251001"
```

Adding a new provider (e.g., remote Ollama, Bedrock, OpenAI, cache) is a new plugin entry in configuration, not a code change to the dispatch pipeline.

#### System Prompt

```
You are a Jupyter notebook cell generator. Write a single cell
using ONLY the pre-loaded kernel namespace below.

Output ONLY the cell contents — no markdown, no explanation, no code fences.

Assign tool results to variables and reuse them. Never call the same function twice
when you can reuse a variable.

Kernel namespace:
{namespace_description}

Builtins: {builtins_list}
{params_section}
Not available: import, def, class, while, try/except, lambda, open

The cell's last expression is displayed as output.
```

The `{params_section}` is conditionally included when params are provided:
```
Pre-set variables (already defined, use directly):
  schema: str — the config schema definition
  target_dir: str — directory to search
```

The `{namespace_description}` is generated from the resolved kit:
```
  read(path) -> str: Read file contents
  edit(path, old_str, new_str) -> bool: Replace text in file
  glob(pattern) -> list[str]: Find files matching pattern
  find_definitions(name) -> list[dict{file,line,name,type}]: Find definitions
```

#### Output Sanitization

Applied to all model output before validation:
1. Strip leading/trailing whitespace
2. Remove ```python ... ``` wrapping
3. Remove "Here's the code:" style preamble lines

### `run/` — Execution

#### Executor Protocol

```python
class Executor(Protocol):
    def run(
        self, tree: ast.Module, namespace: dict[str, Callable],
        sandbox: SandboxConfig | None,
    ) -> ExecutionResult: ...
```

The executor takes a **validated AST** (not a string), compiles it, and runs it. The AST is the source of truth through the entire pipeline: parse -> validate -> transform -> compile -> run.

#### v1 Implementation (AST compile + restricted run)

1. Wrap each callable in the namespace to capture trace entries (tool name, args, result, duration, success/error)
2. Add allowed builtins to traced namespace
3. Rewrite last expression in AST to `__result__ = <expr>` to capture output
4. `compile(tree, '<lackpy>', 'exec')` to produce a code object
5. Run the code object with `{"__builtins__": {}}` + the traced namespace
6. Collect `__result__`, trace, and final variable state

Security posture: the AST validation is the primary safety layer. Running with empty `__builtins__` and a controlled namespace is the secondary layer. The AST validator blocks the constructs needed for known escape vectors (no attribute access to dunders, no `getattr`, no `type()`, no strings containing `__`).

#### v2: nsjail Executor (designed later)

Will use the `nsjail-python` package which provides:
- Fluent `Jail()` builder API
- `sandbox()` preset function
- `Runner` class with sync/async execution
- `NsJailResult` with returncode, stdout, stderr, timeout/OOM detection

The nsjail executor will implement the same `Executor` protocol. Design deferred until the generation pipeline is working and we understand the real shape of generated programs.

#### Sandbox Configuration

Sandbox policy is specified as either a named preset or a literal config object:

```python
SandboxConfig = str | NsJailConfig | None
# str -> resolve named preset to NsJailConfig
# NsJailConfig -> use directly
# None -> v1 default (no sandbox, restricted run fallback)
```

Named presets are resolved at the service layer. The executor only deals with concrete configs.

#### Trace Structure

```python
@dataclass
class TraceEntry:
    step: int
    tool: str
    args: dict
    result: Any           # summarized (truncated large outputs)
    duration_ms: float
    success: bool
    error: str | None

@dataclass
class Trace:
    entries: list[TraceEntry]
    program: str
    kit: str | list[str] | dict
    grade: Grade
    generation_tier: str          # provider name
    generation_model: str | None
    generation_time_ms: float
    execution_time_ms: float
    total_time_ms: float
    files_read: list[str]
    files_modified: list[str]
    variables: dict[str, Any]     # final variable state (summarized)
```

### `service.py` — Unified Service Layer

The service layer orchestrates the pipeline. Both MCP and CLI are thin adapters over these methods.

#### API

```python
# Params type: simple values or rich metadata
Params = dict[str, Any | ParamSpec]
# Where ParamSpec is:
#   {"value": Any, "type": str, "description": str}
# Simple form: {"schema": "<content>", "target_dir": "src/"}
# Rich form:   {"schema": {"value": "<content>", "type": "str",
#                           "description": "the config schema"}}

async def delegate(
    intent: str,
    kit: str | list[str] | dict | None = "debug",
    params: Params | None = None,
    sandbox: str | SandboxConfig | None = None,
    rules: list[str] | None = None,
) -> DelegateResult:
    """Full pipeline: resolve kit -> infer -> validate -> run -> trace."""

async def generate(
    intent: str,
    kit: str | list[str] | dict | None = "debug",
    params: Params | None = None,
    rules: list[str] | None = None,
) -> GenerateResult:
    """Infer + validate, no execution. Returns the program for inspection."""

async def run_program(
    program: str,
    kit: str | list[str] | dict | None = "debug",
    params: Params | None = None,
    sandbox: str | SandboxConfig | None = None,
    rules: list[str] | None = None,
) -> RunResult:
    """Validate + run a caller-provided program."""

async def create(
    program: str,
    kit: str | list[str] | dict | None = "debug",
    name: str = "",
    pattern: str | None = None,
) -> CreateResult:
    """Validate a caller-provided program and save as a template."""

def kit_info(
    kit: str | list[str] | dict,
) -> KitInfoResult:
    """Resolve a kit and return tools, grades, descriptions."""

def kit_list() -> list[KitSummary]:
    """List available predefined kits."""

def kit_create(
    name: str,
    tools: list[str],
    description: str | None = None,
) -> KitSummary:
    """Create and save a new predefined kit."""

def toolbox_list() -> list[ToolInfo]:
    """List all registered tools with metadata."""

def validate(
    program: str,
    kit: str | list[str] | dict | None = "debug",
    rules: list[str] | None = None,
) -> ValidationResult:
    """Validate a program without running it."""
```

#### Params

Params are caller-provided variables injected into both the inference prompt and the execution namespace. They allow the caller to pass context (file contents, search results, configuration) without baking it into the intent string.

**Prompt injection**: Param names, types, and descriptions appear in the system prompt as pre-set variables. The full values are NOT included in the inference prompt — only metadata. This keeps inference cheap even when params contain large content.

```
Pre-set variables:
  schema: str — the config schema definition
  target_dir: str — directory to search
```

**Execution injection**: Param values are injected as top-level variables in the execution namespace. The generated program references them directly (`schema`, not `params["schema"]`).

**Name collision**: Param names are validated at resolution time — they must not shadow tool names or builtin names. Collisions are reported as errors.

**Namespace check**: Param names are added to the set of allowed names during validation, so the validator accepts references to them.

#### `delegate` Pipeline

1. Resolve kit -> `ResolvedKit`
2. Resolve params -> validate names, extract metadata for prompt, extract values for namespace
3. Build namespace description from resolved kit + param metadata
4. Dispatch to inference providers in priority order -> program string
5. Parse program -> `ast.Module`
6. Validate AST against core checks + custom rules + kit namespace + param names
7. Compute grade from resolved kit tool grades
8. Inject param values into execution namespace
9. Run via executor (v1: AST compile + restricted run with traced namespace)
10. Capture trace
11. Return `DelegateResult` with program, trace, grade, output, metadata

### `mcp/` — MCP Server

Thin adapter over the service layer. All tools, no resources.

**Tools:**
- `delegate(intent, kit, params, sandbox, rules)` — full pipeline
- `generate(intent, kit, params, rules)` — infer + validate, no execution
- `run_program(program, kit, params, sandbox, rules)` — validate + run a provided program
- `create(program, kit, name, pattern)` — validate + save as template
- `validate(program, kit, rules)` — check without running
- `kit_info(kit)` — resolve kit, return tools/grades/descriptions
- `kit_list()` — available predefined kits
- `kit_create(name, tools, description)` — create a new predefined kit
- `toolbox_list()` — all registered tools with metadata

### `cli.py` — CLI

Thin adapter over the service layer.

```bash
# Primary operations
lackpy delegate "find callers of validate_token" --kit debug
lackpy generate "fix parse_header" --kit implement
lackpy run program.py --kit debug
lackpy create program.py --name find-callers --kit debug --pattern "find callers of {name}"

# Validation / inspection
lackpy validate program.py --kit debug
lackpy spec

# Kit management
lackpy kit list
lackpy kit info debug
lackpy kit info --tools read,glob,find_callers
lackpy kit create my-kit --tools read,glob,find_callers

# Toolbox management
lackpy toolbox list
lackpy toolbox show read

# Templates
lackpy template list
lackpy template test <name>

# Setup
lackpy init
lackpy init --ollama-model qwen2.5-coder:1.5b
lackpy status
```

## Configuration

`.lackpy/config.toml`:

```toml
[inference]
order = ["templates", "rules", "ollama-local", "anthropic-haiku"]

[inference.providers.templates]
plugin = "templates"
threshold = 0.8
min_uses = 5

[inference.providers.rules]
plugin = "rules"

[inference.providers.ollama-local]
plugin = "ollama"
host = "http://localhost:11434"
model = "qwen2.5-coder:1.5b"
keep_alive = "30m"
temperature = 0.2

[inference.providers.anthropic-haiku]
plugin = "anthropic"
model = "claude-haiku-4-5-20251001"

[sandbox]
enabled = false          # v1: restricted run fallback; v2: nsjail
timeout_seconds = 120
memory_mb = 512

[kit]
default = "debug"

[tool_providers]
# Builtin tools
read = { provider = "builtin" }
edit = { provider = "builtin" }
write = { provider = "builtin" }
glob = { provider = "builtin" }

# Python import tools
find_definitions = { provider = "python", module = "fledgling.api", function = "find_definitions" }
find_callers = { provider = "python", module = "fledgling.api", function = "find_callers" }
run_tests = { provider = "python", module = "blq.api", function = "run_tests" }
errors = { provider = "python", module = "blq.api", function = "get_errors" }
```

## Dependencies

```toml
[project]
name = "lackpy"
description = "Python that lacks most of Python. Restricted program generation and execution for tool composition."
requires-python = ">=3.10"
dependencies = []  # Zero required dependencies — core validation needs only stdlib

[project.optional-dependencies]
ollama = ["ollama>=0.4"]
sandbox = ["nsjail-python"]
anthropic = ["anthropic>=0.40"]
mcp = ["mcp[cli]"]
fledgling = ["fledgling"]
blq = ["blq-cli"]
full = ["lackpy[ollama,sandbox,anthropic,mcp,fledgling,blq]"]

[project.scripts]
lackpy = "lackpy.cli:main"
```

Core package (`lang/` validation + `run/` runner) has **zero dependencies** beyond Python stdlib.

## Design Principles

1. **Zero-dependency core.** The validator and restricted executor use only Python stdlib. Everything else is optional and pluggable.

2. **The AST is the security boundary.** Not the sandbox (belt), not the namespace (suspenders) — the AST whitelist is the primary safety layer. If a program passes validation, it can only do things the namespace allows. The sandbox catches bugs in the validator.

3. **The model is untrusted.** Every model output passes through the full validation pipeline. Safety comes from the pipeline, not the model.

4. **Plugin-based extensibility.** Both the toolbox (tool providers) and the inference pipeline (inference providers) are plugin-based. Adding a new tool backend or a new model provider is configuration, not code changes to the core.

5. **Graceful degradation everywhere.** No Ollama? Skip that provider. No nsjail? Use restricted run. No fledgling? Tool excluded from kits. No agent-riggs? Templates still work locally. The tool works with just Python stdlib + a model.

6. **The program is the trace.** A lackpy program is both the execution specification and the audit trail. The AST is fully parseable, the tool calls are enumerable, the grade is decidable at parse time.

7. **The ratchet is built in.** Every successful execution is a potential template. Templates replace model calls. The steady state is that the model is rarely called because its best outputs have been promoted. Two feed paths: explicit (`create`) and promotion (from successful traces).

8. **Separation of concerns.** lackpy owns tool selection (kits) and program generation/validation/execution. Kibitzer owns mode enforcement (permissions, write boundaries, sandboxing policy). agent-riggs audits and contributes templates but is not a dependency.

## v2 Roadmap

Items deferred from v1, with interfaces defined but not implemented:

- **nsjail executor**: Design the sandbox execution model once the generation pipeline is working and the shape of generated programs is understood. Will use the `nsjail-python` package.
- **Quartermaster**: `kit=None` triggers model-inferred kit selection. Uses same Ollama instance with a prompt listing available tools and the intent. Returns a kit.
- **MCP tool providers**: `mcp-local` and `mcp-system` provider plugins for the toolbox. lackpy becomes both MCP server and client, reading configs from `.mcp.json`, `~/.claude/mcp.json`, etc.
- **Additional inference providers**: Bedrock, OpenAI, Claude Code subprocess, cache layer, remote Ollama instances.

## Testing Strategy

### Unit Tests: Validator (highest priority)

For each forbidden construct, a program that uses it and an assertion that the validator rejects it:

```python
def test_rejects_import():
    assert not validate("import os").valid

def test_rejects_function_def():
    assert not validate("def f(): pass").valid

def test_rejects_while():
    assert not validate("while True: pass").valid

def test_rejects_forbidden_name():
    assert not validate("__import__('os')").valid

def test_rejects_unknown_function():
    assert not validate("unknown_tool()").valid

def test_accepts_valid_program():
    result = validate("x = read('test.py')\nlen(x)", allowed_names={"read"})
    assert result.valid
    assert result.calls == ["read", "len"]
```

### Unit Tests: Custom Rules

```python
def test_no_loops_rule():
    rule = no_loops()
    tree = ast.parse("for x in range(10): print(x)")
    assert rule(tree) != []

def test_max_calls_rule():
    rule = max_calls(2)
    tree = ast.parse("a = read('x')\nb = read('y')\nc = read('z')")
    assert rule(tree) != []
```

### Integration Tests: Execution

Mock tool implementations returning known values:

```python
def test_run_simple():
    namespace = {"read": lambda path: "hello world"}
    result = run_program("content = read('test.py')\nlen(content)", kit=namespace)
    assert result.output == 11
    assert result.trace.entries[0].tool == "read"
```

### Integration Tests: Kit Resolution

```python
def test_resolve_predefined_kit():
    kit = resolve_kit("debug")
    assert "read" in kit.tools
    assert "edit" not in kit.tools

def test_resolve_explicit_list():
    kit = resolve_kit(["read", "glob"])
    assert len(kit.tools) == 2
```

### Model Tests (require Ollama)

Marked `@pytest.mark.ollama`, skipped if unavailable:

```python
@pytest.mark.ollama
async def test_delegate_end_to_end():
    result = await delegate(
        "read the file src/main.py and print its length",
        kit="debug",
    )
    assert result.success
    assert any(e.tool == "read" for e in result.trace.entries)
```

### Inference Provider Tests

Each provider tested independently with known inputs:

```python
def test_template_matching():
    provider = TemplatesProvider(templates_dir=test_templates)
    result = await provider.generate("find callers of validate_token", namespace_desc="...")
    assert "find_callers" in result

def test_rules_provider():
    provider = RulesProvider()
    result = await provider.generate("read file src/main.py", namespace_desc="...")
    assert "read(" in result
```
