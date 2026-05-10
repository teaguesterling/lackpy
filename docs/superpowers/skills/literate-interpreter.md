---
name: literate-interpreter
description: Use when writing, debugging, or reviewing literate interpreter documents (markdown with ```lackpy code blocks). Covers cell types, execution order, prose interpolation rules, common model pitfalls, and the gather-continue-report pattern.
---

# Literate Interpreter

Guide for writing and working with lackpy literate documents — markdown with embedded `lackpy` code blocks that execute as a program.

## When to Use

- Writing literate documents for the interpreter
- Debugging execution failures in literate documents
- Reviewing model-generated literate output
- Building agent harnesses that use the literate interpreter
- Working on the compiler, parser, or kernel code

## Core Mental Model

A literate document is a **program disguised as a markdown document**. Every line executes:

- **Prose** becomes `print(...)` — the reader sees it as rendered text
- **Code blocks** execute as Python — stdout joins the output stream
- **Variables persist** across cells — the namespace accumulates
- **Execution is top-to-bottom** — no forward references

The rendered output IS captured stdout. There is no side channel.

## Cell Types Quick Reference

| Fence line | Type | Behavior |
|---|---|---|
| `` ```lackpy `` | code | Execute, stdout visible |
| `` ```lackpy @hidden `` | hidden | Execute silently |
| `` ```lackpy @gather `` | gather | Execute silently (batch data collection) |
| `` ```lackpy @continue `` | continue | Pause execution, return variables to caller |
| `` ```lackpy @read(path) `` | read | Print file contents |
| `` ```lackpy @write(path) `` | write | Write block body to file |
| `` ```lackpy @diff(path) `` | diff | Apply unified diff |
| `` ```lackpy @scratch `` | scratch | Execute, print variable summary only |

## Prose Interpolation

Prose text can include `{expression}` for variable interpolation. The compiler splits prose into literal text and expression parts using brace-matching, then compiles each expression as a standalone f-string.

**Safe patterns:**
- `{variable_name}` — simple variable
- `{len(items)}` — function call
- `{value:.2f}` — format spec
- `{data['key']}` — dict subscript
- `{chr(10).join(items)}` — complex expression with nested braces

**The compiler handles nested braces correctly.** An expression like `{chr(10).join([f'- {x}' for x in items])}` is captured as a single expression because the splitter tracks brace depth.

**Escaping literal braces:**
- `{{variable}}` in prose produces literal `{variable}` — doubled braces are not interpolated

**What is NOT interpolation:**
- `{"key": "value"}` — starts with `"`, not an identifier
- `{123}` — starts with a digit
- `{` without matching `}` — treated as literal text

## Execution Rules

### Rule 1: Top-to-bottom, no forward references

Cells execute sequentially. A prose cell that references `{table}` MUST appear AFTER the code block that defines `table`.

```
WRONG:                          RIGHT:
Results: {table}                ```lackpy @hidden
                                table = build_table(data)
```lackpy @hidden                ```
table = build_table(data)
```                             Results: {table}
```

### Rule 2: Variables accumulate across cells

Variables defined in ANY cell (including `@gather` and `@hidden`) are available to all later cells and to prose interpolation.

### Rule 3: @gather and @hidden produce no output

Their code runs but stdout is not captured into the rendered document. Use them for data collection and computation.

### Rule 4: @continue pauses execution

When `@continue` is reached, execution stops and the interpreter returns partial results with `continue_requested: True` and all variables in scope. The agent loop feeds these back to the model for the next iteration. Think of it as "I need to see my results before writing the report."

### Rule 5: Errors are caught and recoverable

Static analysis (compile check + AST name resolution) catches syntax errors and undefined references *before* execution. Runtime errors are caught after. In both cases, the recovery handler may give the model a second chance — it sees the error, current variables, and can write replacement cells. Errors are **patch-forward**: you can't rewrite history, only emit corrections.

### Rule 6: Annotations go on the fence line

`@hidden`, `@gather`, `@read(path)` etc. go on the fence line, never inside the code body. The parser now detects this mistake and produces a helpful error message.

## The Gather-Continue-Report Pattern

The standard agent pattern for multi-step analysis:

````markdown
```lackpy @gather
data = search_content("TODO", "src/")
files = run_command("find src -name '*.py' | wc -l")
```

```lackpy @hidden
count = int(files.strip())
todos = [l.strip() for l in data.splitlines() if l.strip()]
```

# Report

Found {len(todos)} TODOs across {count} files.

```lackpy
for todo in todos[:5]:
    print(f"- {todo}")
```
````

**Phase 1 (gather):** `@gather` blocks collect raw data silently.
**Phase 2 (compute):** `@hidden` blocks process the data.
**Phase 3 (render):** Prose and visible code blocks present findings.

For multi-turn tasks, add `@continue` after the gather phase. `@continue` **pauses execution and returns control to the caller** with all variables defined so far. The caller feeds those variables back to the model, which writes the report in the next iteration. Without `@continue`, the entire document executes in one shot.

## Available Tools in Namespace

| Function | Purpose |
|---|---|
| `read_file(path)` | Read file contents |
| `write_file(path, content)` | Write file (creates parent dirs) |
| `apply_diff(path, diff)` | Apply unified diff |
| `search_content(pattern, path=".")` | Grep-like search |
| `run_command(cmd)` | Shell command |
| `run_tests(path=".")` | Run pytest |

All Python builtins and stdlib imports are available.

## Common Pitfalls

### Forward reference in prose
**Symptom:** `undefined names: 'variable'`
**Cause:** Prose references `{variable}` before the code block that defines it.
**Fix:** Move the defining code block above the prose.

### Token budget exhaustion
**Symptom:** `unterminated string literal` or truncated code
**Cause:** Model hit `num_predict` limit mid-expression.
**Fix:** Increase token limit or simplify the task. Keep code blocks short.

### @read syntax on wrong line
**Symptom:** `SyntaxError` in code block containing `@read(path)`
**Cause:** Model put `@read(path)` inside the code body instead of on the fence line.
**Fix:** The annotation goes on the fence: `` ```lackpy @read(path) ``, not as code.

### Complex prose interpolation
**Symptom:** `SyntaxError` in prose cell (pre-v0.10.1)
**Cause:** Old compiler used `repr()` which broke with nested quotes/backslashes.
**Fix:** Updated in v0.10.1 — compiler now uses brace-matching + per-expression f-strings.

## Key Files

| Component | Path |
|---|---|
| Interpreter | `src/lackpy/interpreters/literate/__init__.py` |
| Parser | `src/lackpy/interpreters/literate/parser.py` |
| Compiler | `src/lackpy/interpreters/literate/compiler.py` |
| Kernel | `src/lackpy/interpreters/literate/kernel/lightweight.py` |
| Static analysis | `src/lackpy/interpreters/literate/kernel/static_analysis.py` |
| Tools | `src/lackpy/interpreters/literate/tools.py` |
| System prompt | `src/lackpy/interpreters/literate/prompt.py` |
| Agent harness | `scripts/literate_agent.py` |
| Documentation | `docs/concepts/literate.md` |
