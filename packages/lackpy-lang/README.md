# lackpy-lang

The **lackpy language** as a standalone, dependency-free distribution: the grammar
(allowed/forbidden AST nodes, builtins, names), the validator, the grader, and the
language spec. This is the pure-language leaf of [lackpy](https://github.com/teague/lackpy) —
it has **no runtime dependencies** and pulls in none of lackpy's interpreters, policy
layer, service, or MCP surface.

## Why a separate package?

So consumers that only need to *reason about* lackpy programs — validate them, grade
them, or describe the language — can do so without installing lackpy's full runtime.
For example, a coaching/policy tool can classify a generated program against the
canonical grammar by importing `lackpy.lang` directly, instead of mirroring the
definitions by hand.

```python
from lackpy.lang import validate, compute_grade, get_spec

result = validate("x = sorted(items)\nprint(x[0])")
assert result.valid
```

## Namespace

`lackpy-lang` and `lackpy` share the `lackpy` namespace via
[PEP 420](https://peps.python.org/pep-0420/) (implicit namespace packages): neither
ships a top-level `lackpy/__init__.py`. Installing `lackpy` pulls in `lackpy-lang`
automatically; installing `lackpy-lang` alone gives you exactly `lackpy.lang`.

## Development (monorepo)

```bash
# editable install of both, from the lackpy repo root:
pip install -e ./packages/lackpy-lang -e .
```
