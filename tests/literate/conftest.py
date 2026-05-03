"""Shared fixtures for literate interpreter tests."""

import pytest


@pytest.fixture
def simple_doc():
    return """\
---
echo: true
output: auto
---
# Hello

```lackpy @hidden
name = "World"
```

Greeting: {name}

```lackpy
result = 2 + 2
result
```

The answer is {result}."""


@pytest.fixture
def gather_doc():
    return """\
```lackpy @gather
info1 = "gathered first"
```

```lackpy @gather
info2 = "gathered second"
```

```lackpy @continue
```

Results: {info1} and {info2}"""
