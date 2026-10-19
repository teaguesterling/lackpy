# Language Spec

A lackpy program is a plain Python 3 program. It runs unchanged under CPython
with only the standard library, from the workspace directory. It gives the same
output there as inside lackpy. Inside lackpy it can only do less: a refusal is
an error, never a different result.

This page is a summary. Two generated sources hold the full rules, so this page
does not copy them:

- The [style guide](../reference/style-guide.md) lists every rule, with its
  tier, the reason for it, and an approved and a not-approved example.
  `lackpy.lang.spec.format_style_guide()` generates it from the rule registry.
- The pre-prompt is the page that teaches a model to write lackpy.
  `lackpy.lang.spec.format_preprompt()` generates it. The
  [tutorial](../tutorial.md) runs its three worked examples.

---

## The two tiers

Each rule has one of two tiers:

- A **safety** rule protects the sandbox. Nothing waives it.
- A **style** rule keeps programs short and uniform. A waiver can lift it.

A waiver is a comment, `# ste: <rule-id>, <why>`, on one statement. The why is
required. A waiver that names a safety rule is an error.

Without the waiver, this program breaks `style.call-nesting` (a call nests at
most 3 deep). With it, the program is valid:

```python
words = ["a", "b", "a"]
print(len(sorted(set(words))))  # ste: style.call-nesting, a short count
```

---

## What a program can use

- **Imports:** `pathlib` (`Path`), `json`, `re`, `collections`, `math`,
  `statistics`, `hashlib`, `datetime`. Only some members of each module are
  approved. The style guide has the list.
- **Files:** `pathlib.Path` only, under the [path policy](paths.md). `open()`
  and `os` are not approved.
- **Control flow:** `if`, `for`, `break`, `continue`, `assert`, top-level
  `def`, and `try` with a named error.
- **Builtins:** an approved set, such as `len`, `sorted`, `print`, `range`,
  `enumerate`, `zip`, `min`, `max`, `sum`. The style guide has the list.

## What a program cannot use

| Not in the language | Use instead |
| --- | --- |
| `while`, `class`, `global`, `nonlocal` | a `for` loop, a dict, a returned value |
| `except Exception:`, a bare `except:` | name the error: `except KeyError:` |
| `finally:`, `else:` on a `try` | put cleanup code after the `try` block |
| `raise`, `exit()` | `assert cond, "why it must hold"` |
| `open(p)`, `json.load(f)` | `Path(p).read_text()`, `json.loads(text)` |
| `map`, `filter`, `format` | a comprehension, an f-string |
| `getattr`, `eval`, `exec`, `__import__` | nothing: reflection is not in the language |

This program shows a handler that the validator refuses:

```python title="Refused: except Exception"
import json

try:
    data = json.loads("x")
except Exception:
    data = None
print(data)
```

```text
Forbidden except of 'Exception' at line 4 (catch a subclass of Exception that is not a limit signal, for example KeyError)
```

---

## Machine-readable spec

`lackpyctl spec` prints the spec as JSON. It holds the allowed node types, the
allowed builtins and the rules:

```bash
lackpyctl spec
```

The MCP server gives the same data as the resource `lackpy://language-spec`.

---

## Custom rules

A custom rule makes the validator stricter. A rule is a callable that takes an
`ast.Module` and returns a list of error strings. See
[Extending: Custom Rules](../extending/custom-rules.md).
