# Language Spec

## Design philosophy

lackpy programs are Jupyter notebook cells, not scripts. They:

- Use pre-loaded names from the kit (tools) and builtins — there is no import system.
- Assign tool results to variables and compose them using expressions.
- Return a value by placing an expression on the last line.
- Are validated entirely at the AST level before any code runs.

The guiding principle is: **make the dangerous things impossible, not just forbidden**. A string `"import os"` is forbidden by the AST check — not by a blacklist scan of the source text.

---

## Allowed constructs

### Structural

| Construct | AST nodes |
|-----------|-----------|
| Assignments | `Assign`, `AugAssign` |
| `if` / `elif` / `else` | `If` |
| `for` loops (restricted) | `For` |
| `with` blocks | `With` |
| Expression statements | `Expr` |

### Expressions

| Construct | AST nodes |
|-----------|-----------|
| Function calls | `Call` |
| Name lookups | `Name` |
| Attribute access | `Attribute` |
| Subscript / slice | `Subscript`, `Slice` |
| Literals | `Constant` |
| Lists, dicts, tuples, sets | `List`, `Dict`, `Tuple`, `Set` |
| List / dict / set comprehensions | `ListComp`, `DictComp`, `SetComp` |
| Conditional expression | `IfExp` |
| f-strings | `JoinedStr`, `FormattedValue` |
| Binary ops | `BinOp` with `Add`, `Sub`, `Mult`, `Div`, `Mod`, `FloorDiv` |
| Comparisons | `Compare` with `Eq`, `NotEq`, `Lt`, `LtE`, `Gt`, `GtE`, `Is`, `IsNot`, `In`, `NotIn` |
| Boolean ops | `BoolOp` with `And`, `Or` |
| Unary ops | `UnaryOp` with `Not`, `USub`, `UAdd` |
| Starred args | `Starred` |

### Allowed builtins

```python
len, sorted, reversed, enumerate, zip, range,
min, max, sum, any, all, abs, round,
str, int, float, bool, list, dict, set, tuple,
isinstance, print
```

These are the only names available from the standard `builtins` module at runtime.

---

## Forbidden constructs

| Construct | AST node | Why |
|-----------|----------|-----|
| `import` | `Import` | Arbitrary module access |
| `from ... import` | `ImportFrom` | Arbitrary module access |
| `def` | `FunctionDef` | Code definition escapes namespace control |
| `async def` | `AsyncFunctionDef` | Same as `def` |
| `class` | `ClassDef` | Object system bypasses namespace |
| `lambda` | `Lambda` | Anonymous function, same issue as `def` |
| `while` | `While` | Unbounded loops |
| `try` / `except` | `Try`, `ExceptHandler` | Error suppression can mask violations |
| `raise` | `Raise` | Exception injection |
| `global` | `Global` | Scope escape |
| `nonlocal` | `Nonlocal` | Scope escape |
| `yield` / `yield from` | `Yield`, `YieldFrom` | Generator creation |
| `await` | `Await` | Async context bypass |
| `async for` | `AsyncFor` | Async context bypass |
| `async with` | `AsyncWith` | Async context bypass |
| `assert` | `Assert` | Inconsistent in optimized mode |
| `del` | `Delete` | Namespace mutation |
| `match` | `Match` | Not in allowed set (Python 3.10+) |

---

## Forbidden names

The following names are rejected even if they appear as a `Name` node (i.e. as a plain variable reference, not just a call):

```python
__import__, open,
globals, locals, vars, dir,
getattr, setattr, delattr, hasattr,
__builtins__, __build_class__,
breakpoint, exit, quit,
type, super, classmethod, staticmethod, property,
memoryview, bytearray, bytes,
map, filter, reduce,
input
```

`open` is forbidden because file I/O should go through kit tools (which are traced and namespaced). `getattr` / `setattr` and friends are forbidden because they provide a reflection escape hatch. `map`, `filter`, and `reduce` encourage a functional style that is harder to trace; use list comprehensions instead.

---

## Additional checks

### Namespace check

Every function call of the form `name(...)` (where `name` is an `ast.Name` node) must be either in the kit's tool names or in `ALLOWED_BUILTINS`. Method calls (`obj.method(...)`) are not namespace-checked — they are allowed if the receiver is a valid expression.

### For-loop iterator

```python
# Valid
for item in read_lines(path):
    ...

for item in items:
    ...

# Invalid — literal list as iterator
for item in [1, 2, 3]:
    ...
```

The iterator expression in a `for` loop must be either a `Call` or a `Name`. This prevents programs from embedding arbitrary data as loop iterators. If you need to iterate over a literal, assign it first:

```python
items = [1, 2, 3]
for item in items:
    ...
```

### Dunder strings

Any string constant containing `__` is rejected:

```python
# Invalid — contains dunder
name = "__class__"
```

This prevents `__` strings from being used as arguments to reflection functions, even if those functions were somehow available.

---

## Machine-readable spec

The current grammar is also available as JSON:

```bash
lackpy spec
```

```json
{
  "allowed_nodes": ["Module", "Expr", "Assign", "..."],
  "forbidden_nodes": ["Import", "FunctionDef", "..."],
  "forbidden_names": ["__import__", "open", "..."],
  "allowed_builtins": ["len", "sorted", "..."]
}
```

---

## Custom rules

The built-in checks can be supplemented with custom rules. A rule is a callable `(ast.Module) -> list[str]` — return an empty list to pass, or one or more error strings to fail.

See [Extending: Custom Rules](../extending/custom-rules.md) for details.
