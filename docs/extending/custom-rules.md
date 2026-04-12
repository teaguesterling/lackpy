# Custom Validation Rules

Custom rules let you add application-specific constraints on top of lackpy's built-in AST checks. They are applied at the end of the validation pipeline, after all core checks pass.

---

## Interface

A rule is any callable with this signature:

```python
(tree: ast.Module) -> list[str]
```

Return an empty list to signal that the rule passes. Return one or more error strings to signal failure. The error strings are included in `ValidationResult.errors` unchanged.

Rules can only **tighten** the core checks — they cannot allow constructs that the core validator forbids.

---

## Built-in rules

lackpy ships four ready-to-use rules in `lackpy.lang.rules`:

### `no_loops`

Rejects any `for` loop. Use this when you want programs to be purely functional — every result must come from a single expression or a sequence of assignments.

```python
from lackpy.lang.rules import no_loops

result = svc.validate('for f in find_files("*.py"): print(f)', kit=["find_files"], rules=[no_loops])
# ValidationResult(valid=False, errors=["For-loop forbidden (line 1)"])
```

### `max_depth(n)`

Limits the nesting depth of control structures (`if`, `for`, `with`). Depth 0 means the module level; depth 1 means one level of nesting:

```python
from lackpy.lang.rules import max_depth

rule = max_depth(1)

# Valid — one level deep
program = """
if condition:
    result = read_file(path)
"""
result = svc.validate(program, kit=["read_file"], rules=[rule])

# Invalid — two levels deep
program2 = """
if condition:
    if other:
        result = read_file(path)
"""
result2 = svc.validate(program2, kit=["read_file"], rules=[rule])
# errors: ["Nesting depth 2 exceeds limit 1 (line 3)"]
```

### `max_calls(n)`

Limits the total number of function calls in the program (kit tools + builtins):

```python
from lackpy.lang.rules import max_calls

result = svc.validate(
    'a = read_file("a")\nb = read_file("b")\nc = read_file("c")',
    kit=["read_file"],
    rules=[max_calls(2)],
)
# errors: ["Too many calls: 3 exceeds limit 2"]
```

### `no_nested_calls`

Forbids passing a call result directly as an argument to another call. Encourages explicit intermediate variables, which makes traces easier to read:

```python
from lackpy.lang.rules import no_nested_calls

# Invalid
result = svc.validate(
    'lines = read_file(find_path("config"))',
    kit=["read_file", "find_path"],
    rules=[no_nested_calls],
)
# errors: ["Nested call at line 1: assign inner call to a variable first"]

# Valid equivalent
result2 = svc.validate(
    'path = find_path("config")\nlines = read_file(path)',
    kit=["read_file", "find_path"],
    rules=[no_nested_calls],
)
```

---

## Writing your own rule

### Simple rule

A rule that forbids the `print` builtin in generated programs:

```python
import ast

def no_print(tree: ast.Module) -> list[str]:
    errors = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "print":
                errors.append(f"print() is not allowed (line {node.lineno})")
    return errors
```

Use it:

```python
result = svc.validate('print("hello")', kit=[], rules=[no_print])
# ValidationResult(valid=False, errors=["print() is not allowed (line 1)"])
```

### Rule that checks variable names

A rule that enforces a naming convention — all assigned variables must be `snake_case`:

```python
import ast
import re

_SNAKE_CASE = re.compile(r'^[a-z][a-z0-9_]*$')

def snake_case_variables(tree: ast.Module) -> list[str]:
    errors = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if not _SNAKE_CASE.match(target.id):
                        errors.append(
                            f"Variable '{target.id}' at line {node.lineno} "
                            f"must be snake_case"
                        )
    return errors
```

---

## Parameterized rules (factory pattern)

When a rule needs configuration, wrap it in a factory function:

```python
import ast
from typing import Callable

Rule = Callable[[ast.Module], list[str]]

def only_tools(allowed: set[str]) -> Rule:
    """Restrict the program to a specific set of tool names."""
    def _check(tree: ast.Module) -> list[str]:
        errors = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                name = node.func.id
                if name not in allowed and name not in {"len", "str", "print"}:
                    errors.append(
                        f"Call to '{name}' at line {node.lineno} "
                        f"is not in the allowed set: {sorted(allowed)}"
                    )
        return errors
    return _check
```

Use it:

```python
result = await svc.delegate(
    "find all Python files and count them",
    kit=["find_files"],
    rules=[only_tools({"find_files"})],
)
```

This is the same pattern used by `max_depth`, `max_calls`, and the built-in rules.

---

## Applying rules

Rules can be passed anywhere `extra_rules` is accepted:

```python
# validate
svc.validate(program, kit="filesystem", rules=[no_loops, max_calls(10)])

# generate (enforced during dispatch — invalid generations are retried)
await svc.generate(intent, kit="filesystem", rules=[no_nested_calls])

# run_program (validated before execution)
await svc.run_program(program, kit="filesystem", rules=[max_depth(2)])

# delegate (enforced during dispatch and before execution)
await svc.delegate(intent, kit="filesystem", rules=[no_loops, no_nested_calls])
```

Rules passed to `delegate` are forwarded to both `InferenceDispatcher.generate` (so the LLM is prompted with valid constraints and retried on rule failures) and to the pre-execution `validate` call.
