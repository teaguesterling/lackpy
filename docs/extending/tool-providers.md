# Writing Tool Providers

A **tool provider** resolves `ToolSpec` records into Python callables. By writing a provider you can back lackpy tools with any data source — REST APIs, databases, language servers, or anything else — without modifying lackpy itself.

---

## Protocol

A provider is any object with these three attributes:

```python
class MyProvider:
    @property
    def name(self) -> str:
        """Unique provider identifier. Matches ToolSpec.provider."""
        ...

    def available(self) -> bool:
        """Return True if the provider can currently serve requests."""
        ...

    def resolve(self, tool_spec: ToolSpec) -> Callable[..., Any]:
        """Return a callable for the given ToolSpec."""
        ...
```

There is no abstract base class — Python duck typing is used. `available()` is not called by the current toolbox implementation, but it is good practice to implement it in case a future version adds lazy provider activation.

---

## Complete example — REST API provider

This provider wraps a hypothetical REST API. Each tool is configured with a URL and HTTP method in `provider_config`.

```python
# my_project/providers/rest_provider.py
from __future__ import annotations

from typing import Any, Callable
import urllib.request
import json

from lackpy.kit.toolbox import ToolSpec


class RestProvider:
    @property
    def name(self) -> str:
        return "rest"

    def available(self) -> bool:
        return True  # or check connectivity

    def resolve(self, tool_spec: ToolSpec) -> Callable[..., Any]:
        url = tool_spec.provider_config.get("url")
        method = tool_spec.provider_config.get("method", "GET").upper()
        if not url:
            raise ValueError(
                f"RestProvider requires 'url' in provider_config for tool '{tool_spec.name}'"
            )

        def _call(**kwargs: Any) -> Any:
            if method == "GET":
                # Append kwargs as query parameters
                params = "&".join(f"{k}={v}" for k, v in kwargs.items())
                full_url = f"{url}?{params}" if params else url
                with urllib.request.urlopen(full_url) as resp:  # noqa: S310
                    return json.loads(resp.read())
            elif method == "POST":
                data = json.dumps(kwargs).encode()
                req = urllib.request.Request(url, data=data, method="POST")
                req.add_header("Content-Type", "application/json")
                with urllib.request.urlopen(req) as resp:  # noqa: S310
                    return json.loads(resp.read())
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

        # Rename to match the tool for cleaner tracebacks
        _call.__name__ = tool_spec.name
        return _call
```

### Registration

```python
import asyncio
from lackpy import LackpyService
from lackpy.kit.toolbox import ToolSpec, ArgSpec
from my_project.providers.rest_provider import RestProvider

async def main():
    svc = LackpyService()

    # Register the provider
    svc.toolbox.register_provider(RestProvider())

    # Register tools that use it
    svc.toolbox.register_tool(ToolSpec(
        name="search_issues",
        provider="rest",
        provider_config={
            "url": "https://api.example.com/issues",
            "method": "GET",
        },
        description="Search issues by query string",
        args=[ArgSpec(name="query", type="str", description="Search query")],
        returns="list[dict]",
        grade_w=1,
        effects_ceiling=0,
    ))

    result = await svc.delegate(
        "search for open bugs",
        kit=["search_issues"],
    )
    print(result["output"])

asyncio.run(main())
```

---

## Built-in providers reference

### `BuiltinProvider` (name: `"builtin"`)

Implements four filesystem primitives. The tool name must match exactly:

| Tool name | Signature | Description |
|-----------|-----------|-------------|
| `read` | `read(path: str) -> str` | Read file contents |
| `glob` | `glob(pattern: str) -> list[str]` | Glob from current directory |
| `write` | `write(path: str, content: str) -> bool` | Write file (creates if missing) |
| `edit` | `edit(path: str, old_str: str, new_str: str) -> bool` | Replace first occurrence |

### `PythonProvider` (name: `"python"`)

Wraps any importable Python function. Requires `module` and `function` in `provider_config`:

```python
ToolSpec(
    name="word_count",
    provider="python",
    provider_config={
        "module": "my_tools.text",
        "function": "word_count",
    },
    ...
)
```

The module is imported lazily when `resolve()` is first called.
