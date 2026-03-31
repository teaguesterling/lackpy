# Writing Inference Providers

An **inference provider** generates lackpy program text from a natural language intent and a namespace description. By implementing the provider protocol you can plug in any LLM, retrieval system, or deterministic generator.

---

## Protocol

```python
class MyInferenceProvider:
    @property
    def name(self) -> str:
        """Unique provider name, reported in GenerationResult.provider_name."""
        ...

    def available(self) -> bool:
        """Return True if the provider can currently generate programs.
        The dispatcher skips unavailable providers silently.
        """
        ...

    async def generate(
        self,
        intent: str,
        namespace_desc: str,
        config: dict | None = None,
        error_feedback: list[str] | None = None,
    ) -> str | None:
        """Generate a program string.

        Args:
            intent: The natural language request.
            namespace_desc: Formatted tool descriptions (from Toolbox.format_description).
            config: Reserved for future per-call configuration.
            error_feedback: Validation errors from the previous attempt, if this is a
                            retry call. The provider should incorporate these into its
                            generation strategy.

        Returns:
            A program string, or None to signal that this provider cannot handle
            the request (the dispatcher moves to the next provider).
        """
        ...
```

`generate` is `async` — providers that call synchronous backends should use `asyncio.to_thread` or `loop.run_in_executor` for blocking I/O.

---

## Complete example — cache provider

This provider caches successful generations keyed on `(intent, namespace_desc)`. It acts as an in-memory tier 0 that persists across multiple calls within a session.

```python
# my_project/providers/cache_provider.py
from __future__ import annotations

import hashlib


class CacheProvider:
    def __init__(self) -> None:
        self._cache: dict[str, str] = {}

    @property
    def name(self) -> str:
        return "cache"

    def available(self) -> bool:
        return bool(self._cache)  # skip if empty

    async def generate(
        self,
        intent: str,
        namespace_desc: str,
        config: dict | None = None,
        error_feedback: list[str] | None = None,
    ) -> str | None:
        # Don't return cached program on a retry — the previous result failed
        if error_feedback:
            return None
        key = _cache_key(intent, namespace_desc)
        return self._cache.get(key)

    def store(self, intent: str, namespace_desc: str, program: str) -> None:
        key = _cache_key(intent, namespace_desc)
        self._cache[key] = program


def _cache_key(intent: str, namespace_desc: str) -> str:
    payload = f"{intent.strip().lower()}|{namespace_desc}"
    return hashlib.sha256(payload.encode()).hexdigest()
```

### Integration

Insert the cache provider before the LLM providers. After a successful `delegate`, store the result:

```python
import asyncio
from lackpy import LackpyService
from my_project.providers.cache_provider import CacheProvider

async def main():
    svc = LackpyService()
    cache = CacheProvider()

    # Insert before LLM providers (index 2, after templates and rules)
    svc._inference_providers.insert(2, cache)

    intent = "find all Python files"
    kit = ["glob"]

    result = await svc.delegate(intent, kit=kit)
    if result["success"]:
        # Warm the cache for next time
        resolved = svc._resolve_kit(kit)
        cache.store(intent, resolved.description, result["program"])

    print(result["output"])

asyncio.run(main())
```

---

## Integration notes

- **Provider order** matters. Providers are tried in the order they appear in `svc._inference_providers`. Insert high-confidence providers earlier.
- **`available()` is called on every dispatch.** Keep it cheap — avoid network calls.
- **`None` vs raising.** Return `None` to pass to the next provider. Only raise for unrecoverable errors that should abort the entire dispatch.
- **Retry semantics.** The dispatcher calls `generate` twice per provider: once without `error_feedback`, and once with feedback if the first result failed validation. Return `None` on the retry if your provider cannot incorporate feedback.
- **Thread safety.** The dispatcher is `async` but Python's GIL means most providers are safe. Use locks if you maintain mutable shared state.

---

## Prompt construction helper

For LLM-backed providers, lackpy provides `build_system_prompt()` which constructs the exact system prompt used by the built-in Ollama and Anthropic providers:

```python
from lackpy.infer.prompt import build_system_prompt, format_params_description

system = build_system_prompt(
    namespace_desc=namespace_desc,
    params_desc=None,  # or format_params_description(params)
)
```

The prompt instructs the model to:

- Output only the cell contents (no markdown fences, no explanation)
- Use only names from the kernel namespace
- Assign tool results to variables
- Not use `import`, `def`, `class`, `while`, `try`, or `lambda`

Using `build_system_prompt` ensures your provider generates programs consistent with the validator's expectations.

---

## Output sanitization

Model output often includes markdown code fences and preamble sentences. Use `sanitize_output` to strip these before passing to the validator:

```python
from lackpy.infer.sanitize import sanitize_output

raw = await my_llm_call(prompt)
program = sanitize_output(raw)
```

`sanitize_output` strips:

1. Leading lines matching preamble phrases (`"here's"`, `"here is"`, `"the following"`, `"the solution"`)
2. Opening and closing ```` ``` ```` fences (with optional language identifier)
