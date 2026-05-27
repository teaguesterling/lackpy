# Interpreter types — grounded design (explore/interpreter-types)

Dug into whether the RFC's interpreter taxonomy (one-shot / incremental / composite ×
language profile) is real or speculative, by mapping it onto lackpy's *existing*
interpreters. Finding: **the taxonomy is descriptive — the abstractions already exist in
the code, unnamed and (for the kernel) trapped inside literate.** Formalizing the base
types is consolidation, not invention.

## Two axes (from RFC 0001 Addendum E)

- **Language profile** (`lackpy-lang`): grammar + validator + grader + spec. Today
  restricted-Python; generalizes to profiles (a micro-DSL is a different language).
- **Execution model** (runtime): *how* source runs. The base interpreter types below.

A concrete interpreter = profile × model.

## The execution models, mapped to what exists today

| model | existing interpreter(s) | shape in the code |
|---|---|---|
| **one-shot** | `PythonInterpreter`, `AstSelectInterpreter` | `validate` + async `execute → InterpreterExecutionResult`; stateless. **The current `Interpreter` protocol models exactly this.** |
| **composite — delegating** | `PluckerInterpreter`, `PssInterpreter` | hold **one** sub-interpreter (`self._python = PythonInterpreter()`), transform the context (`_with_plucker_kit` → new kit), **delegate** validate/execute, then annotate the result. A decorator over another interpreter. |
| **incremental / kernel** | `literate/kernel/` (`KernelInterface`) | `execute_cell(cell, idx) → CellResult` + `get_namespace()`; `CellResult.namespace_delta`; a **persistent namespace dict**; `StreamingDriver`, `static_analysis` (names vs the live namespace), `recovery` (patch-forward). A full incremental interpreter — but a *protocol scoped inside literate*. |
| **composite — decomposing** | `literate` (the interpreter) | parse a document into cells → drive them through the kernel (incremental) → assemble. **Composite *over* incremental.** |

Two observations that sharpen the RFC:

1. **Composite has two sub-shapes**, both already present:
   - **delegating** (plucker/pss): wrap **one** sub-interpreter + a context transform (a decorator).
   - **decomposing** (literate): split into **many** units and drive them (usually through an incremental kernel) + assemble.
2. **The incremental abstraction already exists** as literate's `KernelInterface` — it's
   general (execute a unit, persist namespace, return a delta) and is only literate-scoped
   by accident of history.

## Proposed protocols (consolidate, don't invent)

All under the runtime, atop `lackpy-lang`. Sketches:

```python
# (exists) one-shot — today's protocol
class Interpreter(Protocol):
    name: str; description: str
    def validate(self, program, ctx) -> InterpreterValidationResult: ...
    async def execute(self, program, ctx) -> InterpreterExecutionResult: ...

# LIFT literate's KernelInterface to a runtime base
class IncrementalInterpreter(Interpreter, Protocol):
    def open_session(self, ctx) -> "Session": ...           # persistent namespace
    def execute_cell(self, session, cell, index) -> CellResult: ...   # returns namespace_delta
    def get_namespace(self, session) -> dict: ...
    def close(self, session) -> None: ...
    # one-shot execute() = open → execute_cell(whole program) → close

# FORMALIZE the plucker/literate composite patterns
class CompositeInterpreter(Interpreter, Protocol):
    sub_interpreters: Sequence[Interpreter]
    def decompose(self, program, ctx) -> Sequence[Block]: ...   # 1 block (delegating) | N (decomposing)
    # delegating: decompose→[whole], route to one sub with a transformed ctx, annotate
    # decomposing: decompose→[cells], route each (often to an IncrementalInterpreter), assemble
```

- `PluckerInterpreter`/`PssInterpreter` become `CompositeInterpreter`s with one sub + a
  context transform — deleting the hand-rolled delegation boilerplate.
- literate's kernel becomes *the* reference `IncrementalInterpreter`; literate becomes a
  `CompositeInterpreter` (decomposing) that drives it.
- A registry advertising each interpreter's `(language, model, accepts→emits)` is the
  MCP-schema/OPTIONS surface (RFC Addendum C).

## Open questions

- **Protocol shape:** sub-protocols (above) vs capability flags on one `Interpreter`
  (`stateful: bool`, `composite: bool`). Sub-protocols read cleaner; flags are lighter.
- **validate for incremental:** literate already does per-cell `static_analysis` against the
  live namespace — so incremental `validate` is naturally *per-cell*, not whole-program.
- **kit / policy / sandbox across a session:** state persists across cells; does policy
  re-resolve per cell (umwelt context could carry cell index/event), and does one nsjail
  span the session or one per cell?
- **Where the base types live:** runtime package, atop `lackpy-lang` (confirms Addendum D —
  this whole axis is runtime; lang stays pure language).

## Next (not done here — this branch is the design dig)

A prototype: lift `KernelInterface` → `IncrementalInterpreter`, add `CompositeInterpreter`,
re-express `PluckerInterpreter` on it (smallest real consolidation), behind the unchanged
`Interpreter` protocol so nothing else moves. Run the interpreter test suite.
