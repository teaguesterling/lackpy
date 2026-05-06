# Incremental Cell Execution Design

## Overview

Replace the literate interpreter's "compile all → exec once" model with an incremental, cell-by-cell execution engine inspired by Jupyter kernels. Cells execute as they arrive from the model stream, with pluggable recovery when a cell fails and a plugin hook API for coaching systems like Kibitzer.

**Goal:** A streaming execution pipeline where cells parse, analyze, execute, and recover independently — producing a re-executable notebook artifact.

**Key decisions:**
- Cell-by-cell is the primary execution model, not an alternative path
- Pluggable kernel backend (lightweight exec-into-dict or real Jupyter kernel)
- Recovery is pluggable; works without Kibitzer (just error + scope), enhanced with it
- Plugin API defines hooks; Kibitzer subscribes rather than being wired in
- Notebook (.ipynb) is the artifact format; literate markdown is the authoring format
- Parse-as-you-stream: cells execute as fence boundaries are detected in model output
- Patch-forward recovery: fixes operate on current state, no history rewriting

## Architecture

```
Model stream ──→ StreamingCellParser ──→ Cell objects
                                              │
                                              ▼
                          ┌──────── StreamingDriver ────────┐
                          │                                  │
                          │  for each cell:                  │
                          │    notify plugins (on_cell_start)│
                          │    kernel.execute_cell(cell)     │
                          │      ├─ success → log + output  │
                          │      └─ failure → recovery loop │
                          │           ├─ plugin advice      │
                          │           ├─ handler action     │
                          │           ├─ inspect via kernel │
                          │           └─ fix or abort       │
                          │                                  │
                          └──────────────────────────────────┘
                                              │
                                              ▼
                                      ExecutionLog
                                     ╱            ╲
                            to_notebook()    render_markdown()
                                 ↓                   ↓
                              .ipynb            clean markdown
```

## Component Specifications

### 1. Kernel Interface

The kernel executes cells, maintains persistent state, and supports inspection.

```python
@dataclass
class CellResult:
    success: bool
    output: str | None          # captured stdout
    error: str | None           # exception message if failed
    error_phase: str | None     # "static" | "runtime"
    namespace_delta: dict       # variables added/changed by this cell
    cell_index: int

class KernelInterface(Protocol):
    def execute_cell(self, cell: Cell) -> CellResult: ...
    def inspect(self, expr: str) -> str: ...
    def get_scope(self) -> dict[str, str]: ...
    def restart(self) -> None: ...
    def get_namespace(self) -> dict[str, Any]: ...
```

**Static analysis** runs inside `execute_cell` before the actual exec:
1. Compile the cell's Python — catches syntax errors and malformed f-strings
2. Walk the AST, check referenced names exist in current scope or builtins
3. If either fails → `CellResult(success=False, error_phase="static")`
4. If both pass → exec, catch runtime errors → `CellResult(error_phase="runtime")` on failure

**Lightweight backend** (`LightweightKernel`): exec-into-dict. The namespace dict is the kernel state. `execute_cell` compiles a single cell using the existing per-cell-type compiler functions (prose → `print(f"...")`, hidden → passthrough, @write → `write_file(...)`, etc.) then execs the result. `inspect()` evaluates an expression in the namespace and returns `repr()`. `restart()` re-initializes the namespace with tools and builtins.

**Jupyter backend** (optional, future): wraps `jupyter_client.KernelClient`, translates `execute_cell` to kernel protocol messages.

### 2. Streaming Cell Parser

Detects cell boundaries in partial model output, yielding Cell objects as they complete.

```python
class StreamingCellParser:
    def feed(self, chunk: str) -> list[Cell]: ...
    def flush(self) -> list[Cell]: ...
    def reset(self) -> None: ...
```

**Behavior:**
- Accumulates text in a buffer
- Watches for fence open (`` ```lackpy ``) and fence close (`` ``` ``)
- When a fence opens: yields any accumulated prose as a prose Cell
- When a fence closes: parses info string, yields the code Cell (with type + annotation)
- On flush: trailing prose becomes a final Cell; unclosed fences become error/partial cells

**Properties:**
- Each `feed()` can yield 0, 1, or multiple cells
- Non-lackpy fences (```python, ```json) are treated as prose
- Frontmatter (---...---) is detected at stream start, consumed, exposed via property
- Simpler than full-document parser — fence detection on a stream, no markdown-it-py needed

### 3. Recovery Handler Protocol

Pluggable callback invoked by the driver when a cell fails.

```python
@dataclass
class RecoveryAction:
    kind: str               # "fix" | "inspect" | "skip" | "abort"
    cells: list[Cell] | None = None    # replacement cells for "fix"
    expr: str | None = None            # expression for "inspect"
    target_index: int | None = None    # optional: cell index to target (escape hatch)

@dataclass
class RecoveryContext:
    failed_cell: Cell
    error: str
    error_phase: str            # "static" | "runtime"
    scope: dict[str, str]       # name → type + brief repr of value
    cell_index: int
    prior_output: str           # rendered output so far
    attempt: int                # 0-indexed retry count
    plugin_advice: PluginAdvice | None  # coaching from plugins, if any

class RecoveryHandler(Protocol):
    def on_cell_error(self, ctx: RecoveryContext) -> RecoveryAction: ...
    def on_inspect_result(self, ctx: RecoveryContext, result: str) -> RecoveryAction: ...
    max_attempts: int
```

**Built-in handlers:**
- `NoRecoveryHandler`: always returns `abort`. For tests, batch execution, artifact re-runs.
- `InferenceRecoveryHandler`: formats context + scope + error into a prompt, calls a model, parses the response. @scratch in response → `inspect` action. Otherwise → `fix` action. Accepts optional plugin advice to fold into prompt.

**Recovery flow:**
1. Cell fails → driver builds RecoveryContext (includes plugin advice if available)
2. Calls `handler.on_cell_error(ctx)`
3. Handler returns action:
   - `fix`: driver executes replacement cells through kernel. If they fail, increment attempt, loop.
   - `inspect`: driver calls `kernel.inspect(expr)`, then `handler.on_inspect_result(ctx, result)`. Handler can inspect multiple times before returning fix.
   - `skip`: mark cell skipped in log, continue.
   - `abort`: stop execution, return partial results.
4. If `attempt >= max_attempts`: driver calls one final time. Handler should return abort or skip.

**Patch-forward semantics:** Fixes operate on current kernel state. The model can't rewrite history — it can emit correction cells that fix state going forward. `target_index` is an escape hatch for the harness to request replay from an earlier point if the plugin determines the error is rooted upstream (requires kernel restart + replay).

**@scratch for inspection:** During recovery, the model can return a @scratch cell to mean "evaluate this and tell me what you get." The handler parses this as an `inspect` action. The driver runs `kernel.inspect()` and feeds the result back. This lets the model debug before attempting a fix.

### 4. Execution Plugin API

Hook protocol for coaching/tracking systems. Kibitzer subscribes to these hooks; the core system doesn't depend on Kibitzer.

```python
@dataclass
class PluginAdvice:
    hints: list[str]            # coaching strings for recovery prompt
    doc_context: list[str]      # relevant documentation sections
    suggestion: str | None      # advisory (not prescriptive)

class ExecutionPlugin(Protocol):
    def on_cell_start(self, cell: Cell, index: int) -> None: ...
    def on_cell_success(self, cell: Cell, result: CellResult) -> None: ...
    def on_cell_error(self, cell: Cell, error: str, scope: dict) -> PluginAdvice: ...
    def on_recovery_result(self, cell: Cell, success: bool, attempt: int) -> None: ...
```

**Design constraints:**
- Plugins are optional — system works with an empty plugin list
- Plugins observe and advise; they don't control execution flow
- Multiple plugins can be registered; advice is merged (hints concatenated)
- Plugin methods must not raise — errors are logged and ignored
- The full plugin API (lifecycle, registration, discovery) is a separate spec; this defines only the cell execution hooks needed for this feature

**Kibitzer integration path:** Kibitzer implements `ExecutionPlugin` using its existing `classify_failure()`, `get_correction_hints()`, and `report_generation()` machinery. It manages its own session. No KibitzerSession is passed into the execution path.

### 5. Streaming Driver

The orchestrator connecting parser → kernel → recovery → plugins.

```python
@dataclass
class CellExecutionEvent:
    cell: Cell
    cell_index: int
    result: CellResult | None
    status: str                 # "executed" | "recovered" | "skipped" | "pending"
    recovery_attempts: int
    generation: int             # which model stream this came from (for multi-continue)

class StreamingDriver:
    def __init__(
        self,
        kernel: KernelInterface,
        recovery: RecoveryHandler,
        plugins: list[ExecutionPlugin] = [],
    ): ...

    async def feed(self, chunk: str) -> list[CellExecutionEvent]: ...
    async def flush(self) -> list[CellExecutionEvent]: ...
    def interrupt(self) -> None: ...

    @property
    def execution_log(self) -> list[CellExecutionEvent]: ...
    @property
    def rendered_output(self) -> str: ...
    @property
    def generation(self) -> int: ...
```

**Per-cell flow:**
1. Streaming parser yields a Cell
2. Notify plugins: `on_cell_start(cell, index)`
3. Call `kernel.execute_cell(cell)`
4. Success: notify plugins `on_cell_success()`, append to log, accumulate output
5. Failure:
   - Notify plugins `on_cell_error()` → collect PluginAdvice
   - Build RecoveryContext with plugin advice
   - Call `recovery.on_cell_error(ctx)`
   - Handle action (fix → re-execute, inspect → kernel.inspect → loop, skip/abort)
   - On fix success: notify plugins `on_recovery_result(success=True)`
   - On exhaust/abort: call `interrupt()`, return partial results

**Interrupt semantics:** Signals the model stream to stop generating. Called when recovery is exhausted or handler returns abort. Remaining unparsed content is discarded; execution log marks unexecuted cells as "pending."

**@continue interaction:**
1. Kernel executes continue sentinel → driver detects it
2. Driver pauses (stops pulling from parser)
3. Returns event with `status="continue_requested"` + scope snapshot
4. Harness constructs continuation prompt, starts new model stream
5. Harness calls `driver.feed()` with new stream — same kernel state, incremented `generation`

**Multiple continues:** Each continue starts a new generation. The execution log tracks which generation produced each cell, preserving the full multi-turn history in the artifact.

### 6. Format Layer

Converters between the three representations. Cell is the shared interface.

```python
# Markdown → Cells
def parse_document(markdown: str) -> tuple[Frontmatter, list[Cell]]  # batch
# StreamingCellParser for incremental

# Cells → Markdown
def render_markdown(log: list[CellExecutionEvent], frontmatter: Frontmatter) -> str

# Cells + Results → Notebook (.ipynb)
def to_notebook(log: list[CellExecutionEvent], frontmatter: Frontmatter) -> dict

# Notebook → Cells
def from_notebook(nb: dict) -> tuple[Frontmatter, list[Cell]]
```

**Notebook artifact contents:**

Each CellExecutionEvent becomes a notebook cell:
- `cell_type`: mapped to notebook cell type (code or markdown)
- `source`: cell content (or recovery-fixed version)
- `outputs`: captured stdout for that cell
- `metadata.lackpy`: cell_type, annotation_args, status, recovery_attempts, generation

**Re-execution:** Load .ipynb → `from_notebook()` → feed cells to fresh kernel with `NoRecoveryHandler`. This is "Restart and Run All" — deterministic, no model in the loop.

**Markdown round-trip:** `render_markdown()` produces a clean literate document from the (possibly recovered) execution log. This is what the model *should* have written — useful for showing corrected output, feeding back to model, or storing as canonical source.

## Backward Compatibility

The existing `LiterateInterpreter.execute()` becomes a thin wrapper:
1. Parse document (batch)
2. Create StreamingDriver with LightweightKernel + NoRecoveryHandler
3. Feed all cells at once
4. Return InterpreterExecutionResult from execution log

This preserves the current API for callers that don't need streaming or recovery. The current test suite continues to pass unchanged.

## Scope and Dependencies

**In scope:**
- KernelInterface protocol + LightweightKernel implementation
- StreamingCellParser
- StreamingDriver
- RecoveryHandler protocol + NoRecoveryHandler + InferenceRecoveryHandler
- ExecutionPlugin hook protocol (just the cell-level hooks)
- Format converters (markdown ↔ Cell ↔ .ipynb)
- Backward-compatible LiterateInterpreter wrapper

**Out of scope (separate specs):**
- Full plugin API (lifecycle, registration, discovery)
- Jupyter kernel backend
- Migration of other interpreters to KernelInterface
- Kibitzer plugin implementation (uses existing Kibitzer APIs, separate integration work)
- nsjail interaction with the kernel (handled by sandbox layer)

## Testing Strategy

- **Kernel tests:** Unit tests for LightweightKernel — execute, inspect, scope, restart, static analysis catches
- **Parser tests:** StreamingCellParser with chunked input, edge cases (unclosed fences, non-lackpy fences, frontmatter)
- **Driver tests:** Integration tests with mock kernel + mock handler — verify flow, recovery loops, continue semantics, interrupt
- **Recovery tests:** Handler protocol tests with canned responses — fix, inspect, skip, abort paths
- **Format tests:** Round-trip tests — markdown → cells → notebook → cells → markdown
- **Backward compat tests:** Existing test_interpreter.py suite passes unchanged against the wrapper
- **Live model tests:** Optional (require Ollama) — end-to-end streaming with actual model recovery
