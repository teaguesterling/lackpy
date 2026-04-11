# Interpreter Plugins

Lackpy was originally built around one execution model: restricted Python over
a resolved kit of callable tools. Starting in **v0.5.0**, that model is one
plugin among several. An **interpreter** is anything that takes a program
string, validates it, and executes it against an
[`ExecutionContext`](../reference/api.md) to produce a structured result.

The protocol is intentionally small:

```python
class Interpreter(Protocol):
    name: str
    description: str

    def validate(self, program: str, context: ExecutionContext) -> InterpreterValidationResult: ...

    async def execute(self, program: str, context: ExecutionContext) -> InterpreterExecutionResult: ...
```

Interpreters are selected by name via `--interpreter` on the CLI or the
`interpreter=` parameter on service calls. Enforcement during execution is the
interpreter's own responsibility — the Python interpreter uses AST
restrictions, ast-select relies on pluckit's read-only AST queries, and so on.

## Bundled interpreters

### `python` — restricted Python (default)

The original lackpy execution path, now exposed as a plugin. Accepts a
restricted subset of Python (no `import`, `def`, `class`, no attribute access
outside the kit) and runs it against the resolved kit's callables. Output is
whatever the program's last expression evaluated to.

```bash
lackpy delegate "read the file README.md" --kit read_file
# uses --interpreter python (default)
```

### `ast-select` — bare CSS selectors

Evaluates a single [pluckit](https://github.com/teague/pluckit) CSS-style
selector against source code and renders the matches as markdown. The selector
itself is the program — no chaining, no function calls, just one selector per
invocation.

```bash
lackpy delegate ".fn#greet" --interpreter ast-select --code src/greetings.py
```

Output is markdown with the selector as an `H1`, one `H2` per match with the
qualified name and `file:line-range`, and a language-tagged code block for each
match body. Supports `mode: brief` for signature-only output.

### `pss` — pluckit selector sheets

Accepts a multi-rule selector sheet (selector + declaration blocks, like CSS)
and renders all matches as markdown via pluckit's `AstViewer` plugin. Useful
for building multi-section views of a codebase in one shot.

```
.fn#main { show: signature; }
.class#User { show: body; }
```

### `plucker` — fluent chain expressions

A thin wrapper over the `python` interpreter with a pluckit-specific kit. The
program is a fluent chain over pluckit's `Plucker` and `Selection` classes,
entered via `source(code)`:

```python
source("src/**/*.py").find(".fn#main").names()
```

The chain's terminal operation determines the output type: `.names()` returns
a list of strings, `.count()` returns an int, `.view(...)` returns a markdown
string. Because plucker delegates to the python interpreter, the output format
on the execution result stays `"python"` — inspect the actual value type for
the shape.

`source()` with no arguments uses the `code` key from
`ExecutionContext.config`, letting callers set a default source and chain
against it repeatedly.

## Registering a custom interpreter

Interpreters register themselves through the module-level registry:

```python
from lackpy.interpreters import register_interpreter, Interpreter

class MyInterpreter:
    name = "my-interp"
    description = "My custom interpreter"

    def validate(self, program, context): ...
    async def execute(self, program, context): ...

register_interpreter(MyInterpreter)
```

Once registered, the interpreter is selectable by name:

```bash
lackpy delegate "..." --interpreter my-interp
```

## Output formats

The `output_format` field on `InterpreterExecutionResult` identifies the shape
of the result so consumers can dispatch accordingly. Known values:

| Format     | Produced by       | Shape                                 |
|------------|-------------------|---------------------------------------|
| `python`   | `python`, `plucker` | Arbitrary Python value                |
| `markdown` | `ast-select`, `pss` | Markdown string                     |
| `text`     | any               | Plain text string                     |
| `json`     | any               | JSON-serializable structure           |
| `none`     | any               | Failed execution or explicit no-op    |

Interpreters may define their own formats; consumers should treat unknown
values as opaque text.
