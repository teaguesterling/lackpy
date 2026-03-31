# lackpy v0.3 Design Spec: Lackey Class System

*Supersedes `2026-03-30-lpy-format-cli-errorcorrection.md`*

## Overview

Lackey files are valid Python modules built on a `Lackey` base class (like Spack's `Package` DSL). They serve as the unit of work in lackpy — self-describing, executable, composable programs that carry their tool bindings, parameters, return types, and creation provenance.

The same file works in two modes:
- **Native Python**: import it, instantiate it, call `.run()` — real providers, real execution
- **lackpy mode**: lackpy extracts the `run()` body, validates it against the AST whitelist, executes it in a sandboxed namespace

Three connected changes in this spec:
1. **Lackey class system** — the file format and DSL
2. **CLI redesign** — `lackpy` as runner, `lackpyctl` as manager
3. **Error correction chain** — multi-strategy retry for fixing generated programs

## 1. The Lackey Class System

### A Complete Example

```python
from lackpy.lackey import Lackey, Tool, Log, System, User, Assistant
from lackpy.lackey.mcp import fledgling

class AnalyzeFunction(Lackey):
    """Find a function's definition and report its structure."""

    # Tools
    read = Tool()
    glob = Tool()
    find_definitions = Tool(fledgling)

    # Parameters (type + default from annotation)
    target: str = "main"
    include_callers: bool = False

    # Return type
    returns: dict

    # Creation provenance
    creation_log = Log([
        System("You are a Jupyter notebook cell generator..."),
        User("find where a function is defined and show its structure"),
        Assistant(
            "defs = find_definitions(target)\n"
            "for d in defs:\n"
            "    content = read(d['file'])\n"
            "    print(f\"{d['file']}:{d['line']}\")",
            accepted=True,
        ),
    ])

    def run(self) -> dict:
        defs = self.find_definitions(self.target)
        results = []
        for d in defs:
            content = self.read(d['file'])
            lines = content.splitlines()
            results.append({
                'file': d['file'],
                'line': d['line'],
                'preview': lines[d['line'] - 1] if d['line'] <= len(lines) else '',
            })
        return {'target': self.target, 'definitions': results}
```

### The Lackey Base Class

```python
class Lackey:
    """Base class for lackpy programs.

    Subclass this to define a tool-composition program. The run() method
    body is the lackpy program — it uses tools declared as class attributes
    and parameters declared as type annotations.
    """

    # Subclasses override
    returns: type | None = None
    creation_log: Log | None = None

    def run(self):
        raise NotImplementedError
```

`Lackey` provides:
- Metaclass magic to collect `Tool()` descriptors and parameter annotations
- `self.tool_name(...)` access to tools (resolved from descriptors)
- Parameter injection from caller-provided values or annotation defaults
- Introspection for the validator, MCP schema generation, and CLI help

### The Tool Descriptor

`Tool()` is a descriptor that binds a tool name to a provider. It accepts multiple forms:

```python
# Builtin — name inferred from class attribute
read = Tool()

# From a provider module — tool name matches attribute name
find_definitions = Tool(fledgling)

# From a specific provider tool — with local alias
find_defs = Tool(fledgling.find_definitions)

# From an MCP server with partial binding
build = Tool(blq_mcp.run(command="build"))

# Any callable
read = Tool(some_custom_function)
```

Resolution rules:
- `Tool()` with no args → builtin provider, name from class attribute
- `Tool(module)` → look up a tool matching the attribute name from that module
- `Tool(module.tool_name)` → use that specific tool, attribute name is the local alias
- `Tool(callable)` → wrap the callable directly

```python
class Tool:
    """Descriptor that binds a tool to a provider.

    When accessed on a Lackey instance (self.read), returns a callable
    that delegates to the resolved provider implementation.

    When accessed on the class (CountLines.read), returns the Tool
    descriptor itself for introspection.
    """

    def __init__(self, provider=None):
        self._provider = provider
        self._name = None  # set by __set_name__

    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self  # class access returns descriptor
        return obj._resolved_tools[self._name]
```

### Provider Imports

Providers are Python modules that expose tool descriptors. They can be:

**Real Python packages** (installed via pip):
```python
import fledgling
find_definitions = Tool(fledgling)
```

**MCP server wrappers** (lazy-loaded, no-fail import):
```python
from lackpy.lackey.mcp import fledgling
find_definitions = Tool(fledgling)

from lackpy.lackey.mcp import blq_mcp
build = Tool(blq_mcp.run(command="build"))
```

`lackpy.lackey.mcp` is a dynamic module that generates provider objects from MCP server configuration (`.lackpy/providers/`, `.mcp.json`, `~/.claude/mcp.json`). Import does not fail if the server is unavailable — the Tool descriptor raises at resolution time instead.

**lackpy-generated providers** (from config):
```python
from lackpy.lackey.providers import fledgling
find_definitions = Tool(fledgling)
```

Same as MCP but resolved from `.lackpy/providers/fledgling.yml`:
```yaml
name: fledgling
type: mcp
server: fledgling
```

**Provider module protocol:**

A provider module exposes tools as attributes. Each attribute is either a callable or a tool descriptor that can be resolved to a callable:

```python
# fledgling.py (or auto-generated)
find_definitions = ToolRef(server="fledgling", tool="find_definitions")
find_callers = ToolRef(server="fledgling", tool="find_callers")
```

### MCP Tool Binding

MCP tools need special handling because they may require partial application (binding some params at definition time):

```python
from lackpy.lackey.mcp import blq_mcp

class Build(Lackey):
    # Full tool, all params passed at call time
    run_tests = Tool(blq_mcp.run_tests)

    # Partial binding — command="build" is fixed, other params passed at call time
    build = Tool(blq_mcp.run(command="build"))

    def run(self):
        self.build(target="src/")
        results = self.run_tests()
        return results
```

`blq_mcp.run(command="build")` returns a partially-bound tool reference. This is not a function call — it's a descriptor factory that captures the partial args. Similar to `sh` library's `.bake()` or `functools.partial`.

When the MCP tool is not callable at definition time (server not running), the binding is deferred — it captures the intent and resolves at execution time.

### Parameters

Parameters are declared as class-level type annotations with optional defaults:

```python
class CountLines(Lackey):
    pattern: str = "**/*.py"
    max_depth: int = 10
    verbose: bool = False
```

The Lackey metaclass collects these and uses them for:
- **CLI argument generation**: `lackpy count_lines.py --param pattern="src/*.py"`
- **MCP tool schema**: when exposed as an MCP tool
- **Validation**: optional type checking of caller-provided values
- **Documentation**: `lackpy --help count_lines.py`

Parameters with no default are required. Parameters with defaults are optional.

At instantiation, caller-provided values override defaults:
```python
task = CountLines(pattern="src/*.py")  # override pattern
task.run()
```

Or via the runner:
```python
result = await service.run_lackey(path, params={"pattern": "src/*.py"})
```

### Returns

The return type is declared as a class attribute or as the `run()` method's return annotation:

```python
class CountLines(Lackey):
    returns: int  # class-level declaration

    def run(self) -> int:  # or method annotation (takes precedence)
        ...
```

For complex return types:
```python
class AnalyzeFunction(Lackey):
    returns: list[dict]  # used for MCP schema generation

    def run(self) -> list[dict]:
        ...
```

### Creation Provenance (Log)

The `creation_log` records how the program was generated — the conversation between the user/agent and the inferencer:

```python
from lackpy.lackey import Log, System, User, Assistant

class CountLines(Lackey):
    creation_log = Log([
        System("You are a Jupyter notebook cell generator..."),
        User("count lines in files matching a pattern"),
        Assistant(
            "import os\nfor f in glob('**/*.py'):\n    print(len(open(f).readlines()))",
            accepted=False,
            errors=["Forbidden AST node: Import", "Forbidden name: 'open'"],
        ),
        User("Use read(path) to get file contents. All needed functions are already available."),
        Assistant(
            "files = glob(pattern)\nfor f in files:\n    content = read(f)\n    print(f\"{f}: {len(content.splitlines())}\")\nlen(files)",
            accepted=True,
        ),
    ])
```

Each message in the log:
- `System(content)` — the system prompt used
- `User(content)` — the intent or correction feedback
- `Assistant(content, accepted, errors=None)` — the model's output, whether it passed validation, and any errors

The log is:
- **Readable**: you can see exactly how the program was generated
- **Replayable**: the correction chain can use previous attempts to avoid repeating mistakes
- **Auditable**: agent-riggs can read logs to compute trust scores
- **The ratchet's memory**: when promoting a trace to a Lackey file, the log captures what worked

### Composition

Lackeys can embed other Lackeys as sub-tools:

```python
from lackpy.lackey import Lackey, Tool

class CountLines(Lackey):
    read = Tool()
    glob = Tool()

    pattern: str = "**/*.py"

    def run(self) -> int:
        files = self.glob(self.pattern)
        for f in files:
            content = self.read(f)
            print(f"{f}: {len(content.splitlines())}")
        return len(files)


class AnalyzeProject(Lackey):
    count = CountLines()  # embed as sub-tool
    read = Tool()

    def run(self) -> dict:
        n = self.count(pattern="src/*.py")
        readme = self.read("README.md")
        return {"file_count": n, "has_readme": len(readme) > 0}
```

When `self.count(pattern="src/*.py")` is called:
- A new `CountLines` instance is created with the provided params
- Its `run()` method is executed
- The result is returned

Composed Lackeys inherit the parent's sandbox and tracing context — tool calls from sub-tasks appear in the parent's trace.

### Extraction for Validation

lackpy doesn't validate the full Python file — it extracts the `run()` method body and validates only that:

```python
def extract_run_body(lackey_class: type) -> ast.Module:
    """Extract the run() method body as a standalone AST module.

    The body is the lackpy program. Tool references (self.read, self.glob)
    are rewritten to plain function calls (read, glob) for validation
    against the standard lackpy AST whitelist.
    """
    source = inspect.getsource(lackey_class.run)
    tree = ast.parse(source)
    # Extract function body (skip the def line)
    body = tree.body[0].body
    # Rewrite self.tool(args) -> tool(args)
    body = rewrite_self_calls(body)
    # Rewrite self.param -> param
    body = rewrite_self_params(body)
    return ast.Module(body=body, type_ignores=[])
```

After extraction, the standard validator checks the body against `ALLOWED_NODES`, `FORBIDDEN_NAMES`, namespace, etc. The `self.` prefix is syntactic sugar — the underlying program is the same restricted subset.

### Inferencer Integration

The inferencer generates only the `run()` body — plain lackpy code without `self.`:

```python
# Inferencer output (what qwen2.5-coder:1.5b generates):
files = glob(pattern)
for f in files:
    content = read(f)
    print(f"{f}: {len(content.splitlines())}")
len(files)
```

The `--create` command wraps this in a Lackey class:

```python
# What lackpy --create produces:
from lackpy.lackey import Lackey, Tool, Log, System, User, Assistant

class CountLines(Lackey):
    """Count lines in files matching a pattern."""

    read = Tool()
    glob = Tool()

    pattern: str = "**/*.py"

    returns: int

    creation_log = Log([
        System("You are a Jupyter notebook cell generator..."),
        User("count lines in files matching a pattern"),
        Assistant(
            "files = glob(pattern)\n"
            "for f in files:\n"
            "    content = read(f)\n"
            "    print(f\"{f}: {len(content.splitlines())}\")\n"
            "len(files)",
            accepted=True,
        ),
    ])

    def run(self) -> int:
        files = self.glob(self.pattern)
        for f in files:
            content = self.read(f)
            print(f"{f}: {len(content.splitlines())}")
        return len(files)
```

Note the `run()` body adds `self.` prefixes to tool and param references. This is a mechanical transform — the `--create` command does it automatically.

## 2. CLI Redesign

### Principle

`lackpy` is like `python` — it runs programs. `lackpyctl` is like `pip` — it manages the environment.

### `lackpy` — Program Runner

```bash
# REPL
lackpy

# Run a Lackey file
lackpy count_lines.py
lackpy count_lines.py --param pattern="src/*.py"

# One-shot from intent
lackpy -c "read file main.py and count its lines"
lackpy -c "find all python files" --kit read,glob

# Validate
lackpy --validate count_lines.py
lackpy --validate -c "files = glob('*.py')\nlen(files)" --kit read,glob

# Generate without running
lackpy --generate -c "find all python files" --kit read,glob

# Create a Lackey file from intent
lackpy --create -c "count lines in files" --kit read,glob --name CountLines

# Stdin
echo "files = glob('**/*.py')\nlen(files)" | lackpy --kit read,glob

# Sandbox control
lackpy --sandbox=readonly count_lines.py

# Help for a script
lackpy --help count_lines.py
```

| Argument | Description |
|----------|-------------|
| `<file>` | Run a Lackey `.py` file |
| `-c <intent>` | One-shot from intent (like `python -c`) |
| `--kit <tools>` | Comma-separated tool list (for `-c` and stdin) |
| `--param <name>=<value>` | Pass a parameter (repeatable) |
| `--validate` | Validate without running |
| `--generate` | Generate program, print to stdout |
| `--create` | Generate and save as Lackey file |
| `--name <name>` | Class name for `--create` |
| `--sandbox <preset>` | Sandbox preset |
| `--help <file>` | Show Lackey info |

No args and no `-c`: REPL.

### `lackpyctl` — Environment Manager

```bash
# Setup
lackpyctl init
lackpyctl init --ollama-url http://localhost:11435

# Kit management
lackpyctl kit list
lackpyctl kit info debug
lackpyctl kit create my-kit --tools read,glob

# Toolbox
lackpyctl toolbox list
lackpyctl toolbox show read

# Providers
lackpyctl provider list
lackpyctl provider add fledgling --type mcp --server fledgling
lackpyctl provider show fledgling

# Templates (Lackey files with pattern: field equivalent)
lackpyctl template list
lackpyctl template test CountLines

# Info
lackpyctl status
lackpyctl spec
```

### REPL

```
$ lackpy
lackpy v0.3.0 | kit: debug | ollama: qwen2.5-coder:1.5b
>>> find all python files
files = glob('**/*.py')
files
['src/main.py', 'tests/test_main.py']
>>> read file src/main.py
content = read('src/main.py')
content
'def hello():\n    print("hello")\n'
>>> files = glob('src/*.py')
>>> for f in files:
...     print(f)
src/main.py
>>>
```

Input that parses as valid lackpy → execute directly. Input that doesn't parse → send to inference.

### Entry Points

```toml
[project.scripts]
lackpy = "lackpy.cli:main"
lackpyctl = "lackpy.ctl:main"
```

## 3. Error Correction Chain

### Overview

When a generated program fails validation, lackpy runs a chain of correction strategies. The chain is hardcoded for v0.3, designed for future extensibility.

### The Chain

```
Generate program
    |
    v
Strategy 0: Deterministic Cleanup
    Strip imports, fences, preamble.
    Replace open(path).read() with read(path) via AST transform.
    Validate.
    |
    +-- Valid? --> Execute
    |
    v
Strategy 1: Few-Shot Correction
    Same provider, same conversation.
    Show model its bad output + positive hints.
    Temperature escalated to 0.6.
    Validate.
    |
    +-- Valid? --> Execute
    |
    v
Strategy 2: Fresh Fixer Prompt
    Same provider, new conversation.
    System prompt optimized for code correction (not generation).
    Provides: broken code, errors, namespace, intent.
    Validate.
    |
    +-- Valid? --> Execute
    |
    v
Strategy 3: Next Provider
    Fall through to next inference provider.
    Restart chain from Strategy 0.
    |
    +-- Valid? --> Execute
    |
    v
All providers exhausted --> Error
```

### Strategy 0: Deterministic Cleanup

Extends the existing sanitizer with AST-level transforms:

```python
def deterministic_cleanup(program: str) -> str:
    """Safe, deterministic fixes for common model mistakes."""
    # Text-level: strip fences and preamble (existing sanitizer)
    program = sanitize_output(program)

    # Text-level: strip import lines
    lines = program.split("\n")
    lines = [l for l in lines if not l.strip().startswith(("import ", "from "))]
    program = "\n".join(lines).strip()

    # AST-level: replace open(path).read() with read(path)
    program = rewrite_open_to_read(program)

    # AST-level: replace os.path.basename(x) with x.split('/')[-1]
    program = rewrite_path_calls(program)

    return program
```

These are safe because:
- Import lines have no effect in the sandbox
- `open(path).read()` is semantically equivalent to `read(path)`
- `os.path.basename(x)` is equivalent to `x.split('/')[-1]` for our use case

### Strategy 1: Few-Shot Correction

Already implemented. Model sees its bad output in conversation history, gets positive hints, higher temperature.

### Strategy 2: Fresh Fixer Prompt

New conversation with a correction-focused system prompt:

```
You are fixing code for a restricted Python environment.

The code below was generated but contains errors. Rewrite it using
ONLY these functions:
{namespace_description}

Intent: {intent}

Broken code:
{broken_program}

Errors found:
{enriched_errors}

Output ONLY the fixed code.
```

This uses the same model but fresh context — no conversation history anchoring it to wrong patterns. The fixer prompt is shorter and more focused than the generation prompt.

### Strategy 3: Provider Fallthrough

Try the next provider in configured order. Each provider gets the full chain (strategies 0-2).

### Interface

```python
@dataclass
class CorrectionResult:
    program: str
    strategy: str         # which strategy succeeded
    attempts: int         # total attempts across all strategies

class CorrectionChain:
    strategies: list[CorrectionStrategy]

    async def correct(
        self,
        program: str,
        errors: list[str],
        namespace_desc: str,
        intent: str,
        provider: InferenceProvider,
    ) -> CorrectionResult | None:
        ...
```

### Provenance in creation_log

Each correction attempt is recorded in the Lackey's `creation_log`:

```python
creation_log = Log([
    System("You are a Jupyter notebook cell generator..."),
    User("count lines in files"),
    Assistant(
        "import os\nfor f in glob('*.py'): print(len(open(f).readlines()))",
        accepted=False,
        errors=["Forbidden AST node: Import", "Forbidden name: 'open'"],
        strategy="deterministic_cleanup",
    ),
    Assistant(
        "files = glob('*.py')\nfor f in files: print(len(open(f).readlines()))",
        accepted=False,
        errors=["Forbidden name: 'open'"],
        strategy="deterministic_cleanup",  # imports stripped but open remained
    ),
    User("Use read(path) to get file contents."),  # positive hint
    Assistant(
        "files = glob(pattern)\nfor f in files:\n    content = read(f)\n    ...",
        accepted=True,
        strategy="few_shot_correction",
    ),
])
```

## 4. Grammar Changes

### New ALLOWED_NODES

- `ast.AnnAssign` — for typed parameter declarations in the run() body (e.g., `pattern: str = pattern or "**/*.py"`)

### Note on Lackey File Validation

The full `.py` file is NOT validated against the lackpy AST whitelist. Only the extracted `run()` body is validated. The file-level imports, class definition, Tool descriptors, and Log are Python scaffolding that lackpy reads but does not execute in the sandbox.

## 5. Service Layer Changes

### New Methods

```python
def parse_lackey(path: Path) -> LackeyInfo:
    """Parse a Lackey file and extract metadata."""

async def run_lackey(
    path: Path,
    params: dict[str, Any] | None = None,
    sandbox: str | SandboxConfig | None = None,
) -> DelegateResult:
    """Load and run a Lackey file."""

def create_lackey(
    program: str,
    name: str,
    tools: list[str],
    params: dict[str, dict] | None = None,
    returns: str | None = None,
    creation_log: list[dict] | None = None,
    providers: list | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Wrap a generated program in a Lackey class and save as .py."""
```

### LackeyInfo Dataclass

```python
@dataclass
class LackeyInfo:
    name: str
    description: str
    class_name: str
    tools: dict[str, ToolInfo]    # name -> provider info
    params: dict[str, ParamInfo]  # name -> type, default, description
    returns: str | None
    has_pattern: bool             # True if it's a template
    pattern: str | None
    run_body: str                 # extracted run() body as lackpy source
    creation_log: list[dict] | None
    path: Path
```

## 6. MCP Server Integration

Lackey files can be auto-discovered and exposed as MCP tools:

```python
# Auto-discover Lackey files in .lackpy/templates/
for path in templates_dir.glob("*.py"):
    info = parse_lackey(path)
    # Generate MCP tool from Lackey metadata
    register_mcp_tool(
        name=info.name,
        description=info.description,
        params=info.params,
        returns=info.returns,
        handler=lambda **kwargs: service.run_lackey(path, params=kwargs),
    )
```

The MCP tool schema is generated from the Lackey's parameter annotations and return type.

## 7. Template Unification

Existing `.tmpl` template files are replaced by Lackey files. A Lackey with a `pattern` class attribute (or detected via the creation_log's intent) is a template:

```python
class FindCallers(Lackey):
    """Find all callers of a function."""

    find_callers = Tool()

    # This makes it a template — pattern matching for intent dispatch
    pattern = r"find (all )?(callers|usages|references) of {name}"

    name: str = ""

    returns: list

    def run(self) -> list:
        results = self.find_callers(self.name)
        return results
```

The templates provider loads Lackey files, extracts `pattern`, and uses it for intent matching — same as current `.tmpl` behavior but with richer metadata.

### Migration

`lackpyctl migrate-templates` converts `.tmpl` files to Lackey `.py` files. The old format is deprecated.

## 8. File Organization

```
.lackpy/
├── config.toml
├── providers/
│   ├── fledgling.yml
│   └── blq.yml
├── templates/            # Lackey files with pattern (templates)
│   ├── find_callers.py
│   ├── read_file.py
│   └── count_lines.py
├── kits/
│   ├── debug.kit
│   └── implement.kit
└── tools/                # Inline tool implementations
    └── custom_tool.py
```

User Lackey files live wherever the user wants — they're just `.py` files. Template Lackeys (with `pattern`) live in `.lackpy/templates/` for auto-discovery by the inference pipeline.
