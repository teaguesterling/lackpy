# Kibitzer Integration Design

*How lackpy uses Kibitzer as both mode enforcer and inferencer coach.*

## Two Integration Points

### 1. Lackpy as a tool call (outer view)

When an outer agent calls `lackpy.delegate`, Kibitzer sees it as a single tool call. This is handled by the outer agent's hooks — not by lackpy itself. Lackpy doesn't need to do anything here.

### 2. Kibitzer as lackpy's harness (inner view)

This is the interesting one. Lackpy uses Kibitzer's Python API to:

- **Validate** the generated program's tool calls against mode policy before execution
- **Track** each tool call during execution for pattern detection
- **Coach** the error correction chain with suggestions from pattern analysis
- **Report** generation outcomes for trust scoring by agent-riggs

## Integration Architecture

```
delegate(intent, kit, ...)
  |
  v
KibitzerSession.load()
  |
  v
Generate program
  |
  v
Validate AST (lackpy's own validator)
  |
  v
validate_calls(planned_calls)  <-- Kibitzer checks mode policy
  |
  +-- Violations? --> Reject or adjust program
  |
  v
Execute program
  |
  For each tool call:
  |   after_call(tool, input, success)  <-- Kibitzer tracks
  |
  v
get_suggestions()  <-- Kibitzer coaches
  |
  +-- Suggestions feed into error correction chain
  |
  v
KibitzerSession.save()
```

## Service Layer Integration

```python
# In LackpyService.__init__
try:
    from kibitzer import KibitzerSession
    self._kibitzer = KibitzerSession(project_dir=self._workspace)
except ImportError:
    self._kibitzer = None

# In delegate()
async def delegate(self, intent, kit, params, ...):
    session = self._kibitzer

    if session:
        session.load()

    # ... generate program ...

    # Validate planned calls against Kibitzer mode policy
    if session and exec_result is None:
        planned = [
            {"tool": call, "input": {}}
            for call in validation.calls
        ]
        violations = session.validate_calls(planned)
        if violations:
            return {
                "success": False,
                "error": "Mode policy violation",
                "violations": [v.reason for v in violations],
            }

    # ... execute program ...
    # (runner reports each call via after_call)

    # Get coaching suggestions
    if session:
        suggestions = session.get_suggestions()
        # Available for error correction chain or return to caller
        session.save()
```

## Runner Integration

The runner needs to report each tool call to Kibitzer as it happens:

```python
def make_traced(name, fn, trace, kibitzer_session=None):
    def wrapper(*args, **kwargs):
        # ... existing tracing ...
        try:
            result = fn(*args, **kwargs)
            if kibitzer_session:
                kibitzer_session.after_call(name, call_args, success=True)
            return result
        except Exception as e:
            if kibitzer_session:
                kibitzer_session.after_call(name, call_args, success=False)
            raise
    return wrapper
```

## Error Correction Integration

Kibitzer's suggestions feed into the error correction chain. After a failed validation or execution, the correction chain can ask Kibitzer for coaching:

```python
# In the correction chain, after a failed attempt:
if session:
    suggestions = session.get_suggestions(mark_given=False)
    # Don't consume dedup budget — we may retry multiple times
    if suggestions:
        # Feed suggestions as additional hints to the fixer prompt
        error_hints.extend(suggestions)
```

This is where Kibitzer becomes the coach for the 1.5B model. Kibitzer sees patterns the model can't see about itself:

- "You've called read 5 times without using the result"
- "Too many consecutive failures — try a simpler approach"
- "This path is read-only in the current mode"

## Seeding Kibitzer

Lackpy needs to tell Kibitzer about its tools so Kibitzer can make informed policy decisions. The current `validate_calls` API checks tool names against mode policy, but Kibitzer needs to know:

- What tools exist (names, descriptions)
- What grade each tool has (read vs write vs exec)
- What paths each tool accesses (for path guard)

### Proposed: `session.register_tools()`

```python
# Seed Kibitzer with the resolved kit's tools
session.register_tools([
    {
        "name": "read",
        "grade": {"w": 1, "d": 1},
        "description": "Read file contents",
        "effects": "read",      # read | write | exec | none
    },
    {
        "name": "edit",
        "grade": {"w": 3, "d": 3},
        "description": "Replace text in a file",
        "effects": "write",
    },
])
```

This lets Kibitzer's path guard and mode controller make grade-aware decisions without hardcoding tool names.

### Proposed: `session.register_context()`

```python
# Seed Kibitzer with generation context
session.register_context({
    "source": "lackpy",
    "intent": intent,
    "kit": kit_names,
    "generation_tier": gen_result.provider_name,
    "retry_count": correction_attempts,
})
```

This gives Kibitzer's coach enough context to generate relevant suggestions. Without it, the coach only sees individual tool calls — it doesn't know this is a lackpy-generated program being retried.

### Proposed: `session.validate_program()`

Higher-level than `validate_calls` — validates an entire program's planned behavior:

```python
result = session.validate_program({
    "calls": [{"tool": "read", "input": {"path": "src/main.py"}}, ...],
    "grade": {"w": 1, "d": 1},
    "estimated_calls": 15,      # from AST analysis
    "has_loops": True,
    "modifies_files": False,
})

if result.denied:
    print(result.reason)
    # "Program makes 15 read calls in a loop — consider batching"
    # "Grade w=1 but program attempts edit — grade violation"
```

This is program-level policy, not call-level. Kibitzer can enforce:
- Grade ceiling (the program's kit grade must not exceed the mode's allowed grade)
- Call budget (too many calls suggests a poorly generated program)
- Resource patterns (loops over file reads, excessive globbing)

### Proposed: `session.report_generation()`

After a complete delegate cycle, report the generation outcome:

```python
session.report_generation({
    "intent": intent,
    "program": gen_result.program,
    "provider": gen_result.provider_name,
    "correction_attempts": attempts,
    "correction_strategies": ["deterministic_cleanup", "few_shot_correction"],
    "success": exec_result.success,
    "output": exec_result.output,
    "trace_summary": {
        "calls": len(trace.entries),
        "files_read": trace.files_read,
        "files_modified": trace.files_modified,
    },
})
```

This feeds agent-riggs (via the event log) and lets Kibitzer's coach learn about generation patterns:
- Which providers succeed most often
- Which intents need retries
- Which correction strategies work
- Which programs should be promoted to templates

## API Suggestions for Kibitzer

Summary of proposed additions to the Kibitzer Python API, in priority order:

### 1. `register_tools(tools: list[dict])` — HIGH

Let the caller describe its tool set. This is essential for grade-aware mode enforcement. Without it, Kibitzer can only check tool names against a hardcoded list.

### 2. `validate_program(program_info: dict) -> CallResult` — MEDIUM

Program-level validation beyond individual call checks. Enables grade ceiling enforcement and resource budgeting.

### 3. `register_context(context: dict)` — MEDIUM

Give the coach information about the current task context. Enables more relevant suggestions (e.g., "this is a retry" vs "this is a first attempt").

### 4. `report_generation(report: dict)` — LOW (for now)

Feed generation outcomes into the event log. Primarily for agent-riggs trust scoring and template promotion. Can be deferred until agent-riggs integration.

### 5. Grade mapping — DISCUSSION

How do lackpy grades (w=0-3, d=0-3) map to Kibitzer modes? Options:

- **A)** Kibitzer modes declare a max grade: `implement.max_grade_w = 3`, `review.max_grade_w = 1`
- **B)** Lackpy maps its own grades to Kibitzer's writable/readonly path concepts
- **C)** Both systems keep their own grade semantics; the mapping is in lackpy's integration code

I lean **A** — it's cleanest if Kibitzer understands grades natively. The grade is a more general concept than writable paths.

## Graceful Degradation

Kibitzer is an optional dependency. The integration is wrapped in try/except:

```python
try:
    from kibitzer import KibitzerSession
    _HAS_KIBITZER = True
except ImportError:
    _HAS_KIBITZER = False
```

When Kibitzer is not available:
- `validate_calls` is skipped (lackpy's AST validator is still the primary safety layer)
- `after_call` is skipped (tracing still works via lackpy's own Trace)
- `get_suggestions` returns nothing (error correction chain uses its own hints)
- No mode enforcement beyond what the caller specifies via sandbox/kit

Lackpy's AST validator is always the primary security boundary. Kibitzer adds policy enforcement (modes, paths, budgets) on top.

## Configuration

Add `kibitzer` to optional dependencies in `pyproject.toml`:

```toml
[project.optional-dependencies]
kibitzer = ["kibitzer"]
full = ["lackpy[ollama,sandbox,anthropic,mcp,fledgling,blq,kibitzer]"]
```

No config needed in `.lackpy/config.toml` — Kibitzer reads its own `.kibitzer/config.toml`. The integration is automatic when both packages are installed in the same project.
