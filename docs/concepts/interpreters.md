# Interpreter Plugins

Lackpy was originally built around one execution model: restricted Python over
a resolved kit of callable tools. Starting in **v0.5.0**, that model is one
plugin among several. An **interpreter** is anything that takes a program
string, validates it, and executes it against an
[`ExecutionContext`](../reference/api.md) to produce a structured result.

!!! note "Library-only in v0.5.x"

    The interpreter plugin system is currently a **library API**, not yet
    plumbed through `LackpyService.delegate()` or the `lackpy delegate`
    CLI. To use any non-default interpreter, import it and call
    `run_interpreter()` directly (see the examples below). CLI and service
    integration is planned for a future release — when it lands, the
    `python` interpreter will remain the default so existing callers are
    unaffected.

The protocol is intentionally small:

```python
class Interpreter(Protocol):
    name: str
    description: str

    def validate(self, program: str, context: ExecutionContext) -> InterpreterValidationResult: ...

    async def execute(self, program: str, context: ExecutionContext) -> InterpreterExecutionResult: ...
```

Enforcement during execution is the interpreter's own responsibility — the
Python interpreter uses AST restrictions, ast-select relies on pluckit's
read-only AST queries, and so on.

## Using an interpreter directly

All bundled interpreters are exported from `lackpy.interpreters`. The
`run_interpreter()` helper validates then executes, returning a single
`InterpreterExecutionResult`:

```python
import asyncio
from lackpy.interpreters import (
    AstSelectInterpreter,
    ExecutionContext,
    run_interpreter,
)

async def main():
    interp = AstSelectInterpreter()
    ctx = ExecutionContext(config={"code": "src/greetings.py"})
    result = await run_interpreter(interp, ".fn#greet", ctx)
    print(result.output)        # markdown string
    print(result.output_format) # "markdown"
    print(result.metadata)      # {"selector": ".fn#greet", "match_count": 1, ...}

asyncio.run(main())
```

## Bundled interpreters

### `python` — restricted Python (the default execution path)

The original lackpy execution model, now exposed as a plugin. Accepts a
restricted subset of Python (no `import`, `def`, `class`, no attribute access
outside the kit) and runs it against the resolved kit's callables. Output is
whatever the program's last expression evaluated to.

```python
from lackpy.interpreters import PythonInterpreter, ExecutionContext, run_interpreter
from lackpy.kit.registry import resolve_kit

interp = PythonInterpreter()
kit = resolve_kit(["read_file"])
ctx = ExecutionContext(kit=kit)
result = await run_interpreter(
    interp,
    'content = read_file("README.md")\ncontent',
    ctx,
)
```

This is the same interpreter `LackpyService.delegate()` uses under the hood —
you rarely need to drive it yourself, but it's useful for testing validation
or running a program you already generated elsewhere.

### `ast-select` — bare CSS selectors

Evaluates a single [pluckit](https://github.com/teague/pluckit) CSS-style
selector against source code and renders the matches as markdown. The selector
itself is the program — no chaining, no function calls, just one selector per
invocation. The source file path is supplied via `context.config["code"]`.

```python
from lackpy.interpreters import AstSelectInterpreter, ExecutionContext, run_interpreter

interp = AstSelectInterpreter()
ctx = ExecutionContext(config={"code": "src/greetings.py"})
result = await run_interpreter(interp, ".fn#greet", ctx)
```

Output is markdown with the selector as an `H1`, one `H2` per match with the
qualified name and `file:line-range`, and a language-tagged code block for each
match body. Set `config={"mode": "brief"}` for single-line signature-only output.

### `pss` — pluckit selector sheets

Accepts a multi-rule selector sheet (selector + declaration blocks, like CSS)
and renders all matches as markdown via pluckit's `AstViewer` plugin. Useful
for building multi-section views of a codebase in one shot.

```python
from lackpy.interpreters import PssInterpreter, ExecutionContext, run_interpreter

sheet = """
.fn#main { show: signature; }
.class#User { show: body; }
"""

interp = PssInterpreter()
ctx = ExecutionContext(config={"code": "src/app.py"})
result = await run_interpreter(interp, sheet, ctx)
```

### `plucker` — fluent chain expressions

A thin wrapper over the `python` interpreter with a pluckit-specific kit. The
program is a fluent chain over pluckit's `Plucker` and `Selection` classes,
entered via `source(code)`:

```python
from lackpy.interpreters import PluckerInterpreter, ExecutionContext, run_interpreter

interp = PluckerInterpreter()
ctx = ExecutionContext(config={"code": "src/app.py"})
result = await run_interpreter(
    interp,
    'source().find(".fn#main").names()',
    ctx,
)
```

The chain's terminal operation determines the output type: `.names()` returns
a list of strings, `.count()` returns an int, `.view(...)` returns a markdown
string. Because plucker delegates to the python interpreter, the `output_format`
on the execution result stays `"python"` — inspect the actual value type for
the shape.

`source()` with no arguments uses the `code` key from
`ExecutionContext.config`, letting callers set a default source and chain
against it repeatedly. `source(path)` with an explicit argument overrides the
default.

## Registering a custom interpreter

Interpreters register themselves through the module-level registry. Bundled
interpreters register at import time; custom interpreters register wherever
their module is imported:

```python
from lackpy.interpreters import register_interpreter, get_interpreter

class MyInterpreter:
    name = "my-interp"
    description = "My custom interpreter"

    def validate(self, program, context): ...
    async def execute(self, program, context): ...

register_interpreter(MyInterpreter)

# Later:
cls = get_interpreter("my-interp")  # → MyInterpreter
instance = cls()
```

## Output formats

The `output_format` field on `InterpreterExecutionResult` identifies the shape
of the result so consumers can dispatch accordingly. Known values:

| Format     | Produced by         | Shape                                 |
|------------|---------------------|---------------------------------------|
| `python`   | `python`, `plucker` | Arbitrary Python value                |
| `markdown` | `ast-select`, `pss` | Markdown string                       |
| `text`     | any                 | Plain text string                     |
| `json`     | any                 | JSON-serializable structure           |
| `none`     | any                 | Failed execution or explicit no-op    |

Interpreters may define their own formats; consumers should treat unknown
values as opaque text.
