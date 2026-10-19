# lackpy Style Guide

A lackpy program is a plain Python 3 program. It runs unchanged under CPython with only the standard library, from the workspace directory. It gives the same output there as inside lackpy. Inside lackpy it can only do less. A sandbox refusal is an error, never a different result.

Every rule has one of two tiers. A *safety* rule protects the sandbox, and nothing waives it. A *style* rule keeps programs short and uniform. A waiver can lift a style rule.

A waiver is a comment of the form `# ste: <rule-id>, <why>`. The why is required. A waiver covers one statement. Put it at the end of the statement's first line, or on the line directly above. For a compound statement, a waiver covers the header only, not the body.

A waiver names style rules only. A waiver that names a safety rule gives `meta.waiver-safety`. A misplaced or malformed waiver gives `meta.waiver-unknown`.

## Part 1. Program rules

### safety.parse

The program must parse and compile as Python.

**Tier:** safety

**Why:** The other rules read the syntax tree. A program that does not parse has no tree. A program that does not compile cannot run.

**Not approved:**

```python
x = (
```

**Approved:**

```python
x = 1
```

### safety.node

Use only the syntax that the language allows.

**Tier:** safety

**Why:** Each allowed node type is a reviewed part of the sandbox. An unknown node type can open a path out of it.

**Not approved:**

```python
class Point:
    pass
```

**Approved:**

```python
x = [1, 2, 3]
```

### safety.forbidden-name

Do not use a forbidden name, such as open or getattr.

**Tier:** safety

**Why:** These names reach the file system, the interpreter, or object internals outside the sandbox.

**Not approved:**

```python
f = open('notes.txt')
```

**Approved:**

```python
n = len([1, 2])
```

### safety.dunder-name

Do not use a name that starts with two underscores. You can read `__name__` and `__file__`.

**Tier:** safety

**Why:** Dunder names lead to the builtins, the import machinery and the module namespace. `__name__` and `__file__` are only strings.

**Not approved:**

```python
x = __builtins__
```

**Approved:**

```python
if __name__ == '__main__':
    print(1)
```

### safety.type-form

Use type only as `type(<expr>).__name__`.

**Tier:** safety

**Why:** type returns a class. A class leads to object and to every subclass. The name of the class is a string, and it is safe.

**Not approved:**

```python
t = type(3)
```

**Approved:**

```python
n = type(3).__name__
```

### safety.attr-private

Do not use an attribute whose name starts with an underscore. You can read `module.__version__` and `type(<expr>).__name__`.

**Tier:** safety

**Why:** Underscore attributes lead from a value to its type, its module and its globals. That chain is the main way out of a Python sandbox.

**Not approved:**

```python
name = 'a'.__class__
```

**Approved:**

```python
name = 'a'.upper()
```

### safety.attr-store

Do not assign to or delete an attribute.

**Tier:** safety

**Why:** An attribute store on a class or object that the host passes in changes the host for every later program. A name is local to the program.

**Not approved:**

```python
x = []
x.y = 1
```

**Approved:**

```python
x = {}
x['y'] = 1
```

### safety.attr-denied

Do not use a denied attribute, such as format or f_globals.

**Tier:** safety

**Why:** These attributes have no underscore, but they expose frames and globals, or they turn a string into an attribute lookup.

**Not approved:**

```python
text = '{}'.format(1)
```

**Approved:**

```python
text = f'{1}'
```

### safety.import

Import only a dictionary module, at the top level, with import m or from m import x.

**Tier:** safety

**Why:** An import gives the program a module. Each dictionary module and member is reviewed. Other modules reach the file system, processes or the interpreter.

**Not approved:**

```python
import os
names = os.listdir('.')
```

**Approved:**

```python
import json
data = json.loads('[1]')
```

### safety.module-attr

Use an imported module only as module.member, with an approved member. Do not assign to it.

**Tier:** safety

**Why:** A module holds more than its approved members, for example the modules that it imports. A module used as a value hides which member the program reads. An assignment to a module member changes the module for the host.

**Not approved:**

```python
import json
data = json.load('[1]')
```

**Approved:**

```python
import json
data = json.loads('[1]')
```

### safety.unknown-call

Call only an allowed builtin, a name that the caller allows, a program def or an imported member.

**Tier:** safety

**Why:** The validator must know each function that a program calls. An unknown name can be a missing tool or a guess.

**Not approved:**

```python
x = frobnicate(1)
```

**Approved:**

```python
from collections import Counter
c = Counter([1, 1])
n = len(c)
```

### safety.call-target

Call a function by its name, or call a method on a value.

**Tier:** safety

**Why:** A call of a subscript or of a call result hides which function runs. The validator cannot check it.

**Not approved:**

```python
funcs = [len]
n = funcs[0]([1])
```

**Approved:**

```python
n = len([1])
```

### safety.except-base

Catch only a named error class, such as KeyError or json.JSONDecodeError, or a tuple of them. Do not catch Exception, BaseException, RecursionError, RuntimeError or MemoryError.

**Tier:** safety

**Why:** A bare except, Exception and BaseException also catch the signals that stop the program, such as a time limit or a recursion limit. A handler of RecursionError can retry without end. A name that the program binds, or a call, can hide any class.

**Not approved:**

```python
d = {}
try:
    x = d['k']
except Exception:
    x = 0
```

**Approved:**

```python
d = {}
try:
    x = d['k']
except KeyError:
    x = 0
```

### safety.try-finally

Do not use finally on a try. Use a with statement for cleanup.

**Tier:** safety

**Why:** A return, continue or break in finally discards the exception in flight, such as a recursion limit or a sandbox refusal.

**Not approved:**

```python
d = {}
try:
    x = d['k']
except KeyError:
    x = 0
finally:
    print('done')
```

**Approved:**

```python
d = {}
try:
    x = d['k']
except KeyError:
    x = 0
print('done')
```

### style.assert-message

Give every assert a message that is a string or an f-string.

**Tier:** style

**Why:** The message says what the assert protects. A name or a call hides it.

**Not approved:**

```python
x = 1
assert x
```

**Approved:**

```python
x = 1
assert x == 1, "x is one"
```

### style.for-iter

A for loop iterates over a call, a variable, or an expression with no call.

**Tier:** style

**Why:** A call or a name shows what the loop reads. A call buried in a larger expression hides it.

**Not approved:**

```python
x = [3, 1]
for k in sorted(x)[:3]:
    print(k)
```

**Approved:**

```python
x = [3, 1]
for k in x[:3]:
    print(k)
```

### style.lambda-position

Use lambda only as a key= value, a defaultdict factory, or the repl of re.sub.

**Tier:** style

**Why:** A lambda in a key= argument is short and has one job. A lambda in other places hides logic.

**Not approved:**

```python
xs = [1]
m = sorted(xs, lambda r: r)
```

**Approved:**

```python
words = sorted(['b', 'a'], key=lambda w: w)
```

### style.lambda-assign

Do not assign a lambda to a name.

**Tier:** style

**Why:** A named lambda is a function in disguise. Use a comprehension or a key= argument.

**Not approved:**

```python
f = lambda x: x
```

**Approved:**

```python
words = sorted(['b', 'a'], key=lambda w: w)
```

### style.def-top-level

Write def only at the top level of the program.

**Tier:** style

**Why:** A def inside a block or another def hides its scope.

**Not approved:**

```python
def outer():
    def inner():
        return 1
    return inner()
```

**Approved:**

```python
def inner():
    return 1
def outer():
    return inner()
```

### style.def-before-use

Call a program function only after its def ends.

**Tier:** style

**Why:** Order shows the dependencies. It also removes recursion through direct calls, because a function is not complete inside its own body.

**Not approved:**

```python
def f(n):
    return f(n)
```

**Approved:**

```python
def f(n):
    return n
x = f(1)
```

### style.def-signature

Take plain parameters. Use no `*args` or `**kwargs`, and only constant defaults.

**Tier:** style

**Why:** A default is evaluated once, when the def runs. A constant default has no hidden state. `*args` and `**kwargs` hide the parameters.

**Not approved:**

```python
def f(a=[]):
    return a
```

**Approved:**

```python
def f(a=None):
    return a
```

### style.def-return-mode

Return a value on every path, or never return a value.

**Tier:** style

**Why:** A function that sometimes returns a value and sometimes None makes the caller check both. A path that ends without a return gives None.

**Not approved:**

```python
def f(x):
    if x:
        return 1
    return
```

**Approved:**

```python
def f(x):
    if x:
        return 1
    return 0
```

### style.def-returns-function

Do not return a function.

**Tier:** style

**Why:** A returned function carries hidden state and is called later, where the validator cannot follow it.

**Not approved:**

```python
def f():
    return lambda: 1
```

**Approved:**

```python
def f():
    return 1
```

### style.def-length

Keep a function to 12 statements, nested ones included.

**Tier:** style

**Why:** A short function fits in one read. A long one needs a split into named steps.

**Not approved:**

```python
def f():
    a0 = 0
    a1 = 1
    a2 = 2
    a3 = 3
    a4 = 4
    a5 = 5
    a6 = 6
    a7 = 7
    a8 = 8
    a9 = 9
    a10 = 10
    a11 = 11
    a12 = 12
```

**Approved:**

```python
def f():
    a0 = 0
    a1 = 1
    a2 = 2
    a3 = 3
    a4 = 4
    a5 = 5
    a6 = 6
    a7 = 7
    a8 = 8
    a9 = 9
    a10 = 10
    a11 = 11
```

### style.def-shadow

Do not name a def after a builtin or an import binding. Define a name once.

**Tier:** style

**Why:** A def that reuses a builtin name changes what that name means for the rest of the program. A second def of one name changes what earlier calls reach.

**Not approved:**

```python
def sorted(x):
    return x
```

**Approved:**

```python
def sort_rows(x):
    return x
```

### style.try-body

Put at most 3 statements in a try body.

**Tier:** style

**Why:** A small body ties the handler to the statement that can fail.

**Not approved:**

```python
d = {}
try:
    a = d['a']
    b = d['b']
    c = d['c']
    e = d['e']
except KeyError:
    a = b = c = e = 0
```

**Approved:**

```python
d = {}
try:
    a = d['a']
except KeyError:
    a = 0
```

### style.try-else

Do not use else on a try. Put the statements after the try.

**Tier:** style

**Why:** else adds a path that the reader must follow.

**Not approved:**

```python
d = {}
try:
    x = d['k']
except KeyError:
    x = 0
else:
    print(x)
```

**Approved:**

```python
d = {}
try:
    x = d['k']
except KeyError:
    x = 0
print(x)
```

### style.try-handler

An except may hold only print calls and assignments. It may end with continue (in a for body), `return <value>` (in a def) or pass. An except that does not continue or return must assign every name that the try body assigns.

**Tier:** style

**Why:** After a try, every name that the body assigns must have a value, or the handler must leave the flow. Each shape gives one sentence in the procedure, for example 'If d[k] raises KeyError, use 0 for x.'

**Not approved:**

```python
d = {}
try:
    x = d['k']
except KeyError:
    print('missing')
print(x)
```

**Approved:**

```python
d = {}
try:
    x = d['k']
except KeyError:
    print('missing')
    x = 0
print(x)
```

### style.genexp-consumed

Pass a generator expression straight to a call. Do not store it.

**Tier:** style

**Why:** A stored generator runs later, once, and hides its cost. A call that consumes it makes the order of work clear.

**Not approved:**

```python
xs = [1, 2]
g = (x for x in xs)
print(g)
```

**Approved:**

```python
xs = [1, 2]
n = sum(x for x in xs)
print(n)
```

### style.attr-chain

Use at most 3 names in a dotted chain (a.b.c).

**Tier:** style

**Why:** A long chain reaches through objects the reader must hold in mind. A named step says what each part is.

**Not approved:**

```python
x = {}
y = x.a.b.c
```

**Approved:**

```python
x = {}
b = x.a.b
y = b.c
```

### style.method-chain

Chain at most 3 method calls.

**Tier:** style

**Why:** Each call in a chain changes the value. A long chain hides the type at each step.

**Not approved:**

```python
s = ' a b '
y = s.strip().lower().split().pop()
```

**Approved:**

```python
s = ' a b '
words = s.strip().lower().split()
y = words.pop()
```

### style.call-nesting

Nest calls at most 3 deep in arguments.

**Tier:** style

**Why:** print(len(sorted(x))) still reads at once. A fourth level makes the reader unwrap from the inside.

**Not approved:**

```python
x = [3, 1]
print(len(sorted(list(x))))
```

**Approved:**

```python
x = [3, 1]
n = len(sorted(list(x)))
print(n)
```

### style.io-position

In a chain, make the I/O call the first method call.

**Tier:** style

**Why:** I/O is the step that can fail. Put it at the start of a chain, so the reader finds it. An I/O call may be an argument.

**Not approved:**

```python
from pathlib import Path
p = 'a'
t = Path(p).read_text().read_text()
```

**Approved:**

```python
from pathlib import Path
p = 'a'
t = Path(p).read_text().strip()
```

### style.comprehension-nested

Do not put a comprehension inside a comprehension.

**Tier:** style

**Why:** Two loops in one expression make the reader track two scopes. Name the inner result.

**Not approved:**

```python
rows = [[1], [2]]
y = [[a for a in r] for r in rows]
```

**Approved:**

```python
rows = [[1], [2]]
y = [list(r) for r in rows]
```

### style.comprehension-multi-for

Use one for clause in a comprehension.

**Tier:** style

**Why:** Two for clauses read in an order that surprises people. Use a for loop.

**Not approved:**

```python
rows = [[1], [2]]
y = [a for r in rows for a in r]
```

**Approved:**

```python
rows = [[1], [2]]
y = []
for r in rows:
    y.extend(r)
```

### style.elif-chain

Do not chain 3 or more if branches. Use a dict lookup.

**Tier:** style

**Why:** A long if/elif chain hides that every branch picks one value. A dict states the mapping.

**Not approved:**

```python
x = 1
if x == 1:
    y = 'a'
elif x == 2:
    y = 'b'
else:
    y = 'c'
```

**Approved:**

```python
x = 1
y = {1: 'a', 2: 'b'}.get(x, 'c')
```

## Part 2. Dictionary

### Approved builtins

| Kind | Names |
| --- | --- |
| Functions and types | `abs`, `all`, `any`, `bool`, `chr`, `dict`, `enumerate`, `float`, `frozenset`, `hex`, `int`, `isinstance`, `iter`, `len`, `list`, `max`, `min`, `next`, `ord`, `print`, `range`, `repr`, `reversed`, `round`, `set`, `sorted`, `str`, `sum`, `tuple`, `zip` |
| Exception classes | Every built-in subclass of `Exception`. |

`safety.except-base` forbids catching `Exception`, `MemoryError`, `RecursionError`, `RuntimeError`.

### Approved modules and members

| Module | Members |
| --- | --- |
| `json` | `JSONDecodeError`, `dumps`, `loads` |
| `re` | `DOTALL`, `I`, `IGNORECASE`, `M`, `MULTILINE`, `S`, `VERBOSE`, `X`, `compile`, `error`, `escape`, `findall`, `finditer`, `fullmatch`, `match`, `search`, `split`, `sub`, `subn` |
| `collections` | `Counter`, `OrderedDict`, `defaultdict`, `deque` |
| `math` | `comb`, `exp`, `sqrt` |
| `statistics` | `mean`, `median`, `pstdev`, `quantiles`, `stdev` |
| `hashlib` | `sha1`, `sha256` |
| `datetime` | `UTC`, `date`, `datetime`, `timedelta`, `timezone` |
| `pathlib` | `Path` |

### Path members

`exists`, `glob`, `is_dir`, `is_file`, `iterdir`, `joinpath`, `mkdir`, `name`, `parent`, `parts`, `read_bytes`, `read_text`, `relative_to`, `rglob`, `stem`, `suffix`, `with_suffix`, `write_bytes`, `write_text`

The `/` operator joins paths, for example `Path(d) / name`.

### Unapproved words

| Word | Approved alternative |
| --- | --- |
| `open` | use Path(p).read_text() or Path(p).write_text(s) |
| `json.load` | use json.loads(Path(p).read_text()) |
| `json.dump` | use Path(p).write_text(json.dumps(obj)) |
| `glob` | use Path(d).glob(pattern) |
| `os` | use pathlib.Path |
| `os.path` | use pathlib.Path |
| `map` | use a comprehension |
| `filter` | use a comprehension with if |
| `format` | use an f-string |

Part 3 (intent rules) arrives with Plan 3.
