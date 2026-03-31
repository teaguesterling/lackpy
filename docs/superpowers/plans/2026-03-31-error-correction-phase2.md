# Error Correction Chain — Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-retry logic in the inference dispatcher with a multi-strategy error correction chain: deterministic cleanup, few-shot correction, fresh fixer prompt, then provider fallthrough.

**Architecture:** New `infer/correction.py` module with a `CorrectionChain` class and individual strategy functions. The chain is called by the dispatcher after initial generation fails validation. Each strategy transforms or regenerates the program and re-validates. The chain also captures attempt history for creation_log provenance.

**Tech Stack:** Python stdlib (ast module for AST transforms). No new dependencies.

---

## File Structure

```
src/lackpy/infer/
|-- correction.py    # CorrectionChain + strategy implementations
|-- cleanup.py       # Deterministic AST-level cleanup (strip imports, rewrite open)
|-- fixer.py         # Fresh fixer prompt construction
|-- dispatch.py      # Modified to use CorrectionChain (existing file)
|-- sanitize.py      # Extended with import stripping (existing file)
|-- hints.py         # Unchanged (existing file)

tests/infer/
|-- test_correction.py
|-- test_cleanup.py
|-- test_fixer.py
```

---

### Task 1: Deterministic Cleanup

**Files:**
- Create: `src/lackpy/infer/cleanup.py`
- Create: `tests/infer/test_cleanup.py`

- [ ] **Step 1: Write failing tests**

`tests/infer/test_cleanup.py`:
```python
"""Tests for deterministic AST-level cleanup."""

from lackpy.infer.cleanup import deterministic_cleanup


class TestStripImports:
    def test_strips_import(self):
        code = "import os\nfiles = glob('*.py')\nfiles"
        result = deterministic_cleanup(code)
        assert "import" not in result
        assert "glob(" in result

    def test_strips_from_import(self):
        code = "from os import path\nfiles = glob('*.py')"
        result = deterministic_cleanup(code)
        assert "from os" not in result
        assert "glob(" in result

    def test_strips_multiple_imports(self):
        code = "import os\nimport sys\nfrom pathlib import Path\nx = read('f')"
        result = deterministic_cleanup(code)
        assert "import" not in result
        assert "read(" in result

    def test_preserves_non_import_lines(self):
        code = "x = read('test.py')\nlen(x)"
        result = deterministic_cleanup(code)
        assert result == code


class TestRewriteOpen:
    def test_rewrites_open_read(self):
        code = "content = open('test.py').read()\nlen(content)"
        result = deterministic_cleanup(code)
        assert "open(" not in result
        assert "read(" in result

    def test_rewrites_open_in_loop(self):
        code = "for f in files:\n    content = open(f).read()\n    print(content)"
        result = deterministic_cleanup(code)
        assert "open(" not in result
        assert "read(f)" in result

    def test_leaves_read_alone(self):
        code = "content = read('test.py')\ncontent"
        result = deterministic_cleanup(code)
        assert result == code


class TestRewritePathCalls:
    def test_rewrites_os_path_basename(self):
        code = "name = os.path.basename(filepath)"
        result = deterministic_cleanup(code)
        assert "os.path" not in result
        assert "split" in result or "rsplit" in result

    def test_rewrites_os_path_join(self):
        code = "full = os.path.join(directory, filename)"
        result = deterministic_cleanup(code)
        assert "os.path" not in result


class TestCombined:
    def test_full_cleanup(self):
        code = "import os\n\nfiles = glob('src/*.py')\nfor f in files:\n    content = open(f).read()\n    name = os.path.basename(f)\n    print(f'{name}: {len(content)}')"
        result = deterministic_cleanup(code)
        assert "import" not in result
        assert "open(" not in result
        assert "os.path" not in result
        assert "glob(" in result
        assert "read(" in result

    def test_empty_after_cleanup(self):
        code = "import os\nimport sys"
        result = deterministic_cleanup(code)
        assert result.strip() == ""

    def test_already_clean(self):
        code = "files = glob('*.py')\nfor f in files:\n    content = read(f)\n    print(f)"
        result = deterministic_cleanup(code)
        assert result == code
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/infer/test_cleanup.py -v`
Expected: FAIL (import error)

- [ ] **Step 3: Implement deterministic cleanup**

`src/lackpy/infer/cleanup.py`:
```python
"""Deterministic AST-level cleanup for common model mistakes.

These transforms are safe because:
- Import lines have no effect in the sandbox (no modules available)
- open(path).read() is semantically equivalent to read(path)
- os.path.basename(x) is equivalent to x.rsplit('/', 1)[-1]
- os.path.join(a, b) is equivalent to f"{a}/{b}"
"""

from __future__ import annotations

import ast


def deterministic_cleanup(program: str) -> str:
    """Apply safe, deterministic fixes to common model mistakes.

    Performs text-level import stripping, then AST-level rewrites
    for open() and os.path calls.

    Args:
        program: Raw program source (already fence-stripped by sanitizer).

    Returns:
        Cleaned program source. May be empty if the program was all imports.
    """
    # Text-level: strip import lines
    lines = program.split("\n")
    lines = [l for l in lines if not l.strip().startswith(("import ", "from "))]
    program = "\n".join(lines).strip()

    if not program:
        return ""

    # AST-level rewrites
    try:
        tree = ast.parse(program)
    except SyntaxError:
        return program  # can't parse — return text-cleaned version

    tree = _OpenRewriter().visit(tree)
    tree = _OsPathRewriter().visit(tree)
    ast.fix_missing_locations(tree)

    try:
        return ast.unparse(tree)
    except Exception:
        return program  # unparse failed — return text-cleaned version


class _OpenRewriter(ast.NodeTransformer):
    """Rewrite open(path).read() -> read(path)."""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)

        # Match: open(path).read()
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "read"
            and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Name)
            and node.func.value.func.id == "open"
            and not node.args  # .read() has no args
        ):
            # Replace with read(path)
            open_args = node.func.value.args
            return ast.Call(
                func=ast.Name(id="read", ctx=ast.Load()),
                args=open_args,
                keywords=[],
            )

        # Match: open(path, 'r').read()
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "read"
            and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Name)
            and node.func.value.func.id == "open"
            and len(node.func.value.args) >= 1
        ):
            path_arg = node.func.value.args[0]
            return ast.Call(
                func=ast.Name(id="read", ctx=ast.Load()),
                args=[path_arg],
                keywords=[],
            )

        # Match: open(path).readlines()
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "readlines"
            and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Name)
            and node.func.value.func.id == "open"
        ):
            path_arg = node.func.value.args[0]
            # Replace with read(path).splitlines()
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Call(
                        func=ast.Name(id="read", ctx=ast.Load()),
                        args=[path_arg],
                        keywords=[],
                    ),
                    attr="splitlines",
                    ctx=ast.Load(),
                ),
                args=[],
                keywords=[],
            )

        return node


class _OsPathRewriter(ast.NodeTransformer):
    """Rewrite os.path.X() calls to string operations."""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)

        # Match os.path.basename(x) -> x.rsplit('/', 1)[-1]
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "basename"
            and _is_os_path(node.func.value)
            and len(node.args) == 1
        ):
            arg = node.args[0]
            return ast.Subscript(
                value=ast.Call(
                    func=ast.Attribute(
                        value=arg, attr="rsplit", ctx=ast.Load(),
                    ),
                    args=[ast.Constant(value="/"), ast.Constant(value=1)],
                    keywords=[],
                ),
                slice=ast.Constant(value=-1),
                ctx=ast.Load(),
            )

        # Match os.path.join(a, b) -> f"{a}/{b}"
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "join"
            and _is_os_path(node.func.value)
            and len(node.args) >= 2
        ):
            # Build f-string: f"{a}/{b}" or f"{a}/{b}/{c}"
            parts: list[ast.expr] = []
            for i, arg in enumerate(node.args):
                if i > 0:
                    parts.append(ast.Constant(value="/"))
                parts.append(ast.FormattedValue(
                    value=arg, conversion=-1, format_spec=None,
                ))
            return ast.JoinedStr(values=parts)

        return node


def _is_os_path(node: ast.expr) -> bool:
    """Check if node is os.path."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "path"
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/infer/test_cleanup.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/lackpy/infer/cleanup.py tests/infer/test_cleanup.py
git commit -m "feat(infer): add deterministic cleanup — strip imports, rewrite open/os.path"
```

---

### Task 2: Fresh Fixer Prompt

**Files:**
- Create: `src/lackpy/infer/fixer.py`
- Create: `tests/infer/test_fixer.py`

- [ ] **Step 1: Write failing tests**

`tests/infer/test_fixer.py`:
```python
"""Tests for the fresh fixer prompt."""

from lackpy.infer.fixer import build_fixer_prompt, build_fixer_messages


class TestBuildFixerPrompt:
    def test_contains_namespace(self):
        prompt = build_fixer_prompt("  read(path) -> str: Read file")
        assert "read(path)" in prompt

    def test_contains_fixer_framing(self):
        prompt = build_fixer_prompt("")
        assert "fix" in prompt.lower() or "rewrite" in prompt.lower()

    def test_no_generation_framing(self):
        prompt = build_fixer_prompt("")
        assert "Jupyter" not in prompt
        assert "notebook" not in prompt


class TestBuildFixerMessages:
    def test_includes_broken_code(self):
        messages = build_fixer_messages(
            intent="read file test.py",
            broken_program="import os\nopen('test.py').read()",
            errors=["Forbidden AST node: Import", "Forbidden name: 'open'"],
            namespace_desc="  read(path) -> str: Read file",
        )
        assert any("import os" in m["content"] for m in messages)

    def test_includes_errors(self):
        messages = build_fixer_messages(
            intent="read file test.py",
            broken_program="import os",
            errors=["Forbidden AST node: Import"],
            namespace_desc="",
        )
        assert any("Import" in m["content"] for m in messages)

    def test_includes_intent(self):
        messages = build_fixer_messages(
            intent="count lines in files",
            broken_program="x = 1",
            errors=["err"],
            namespace_desc="",
        )
        assert any("count lines" in m["content"] for m in messages)

    def test_message_structure(self):
        messages = build_fixer_messages(
            intent="test", broken_program="x = 1",
            errors=["err"], namespace_desc="",
        )
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/infer/test_fixer.py -v`
Expected: FAIL (import error)

- [ ] **Step 3: Implement fixer prompt**

`src/lackpy/infer/fixer.py`:
```python
"""Fresh fixer prompt for code correction.

A separate system prompt optimized for correcting broken programs,
not generating new ones. Uses a fresh conversation context to avoid
anchoring the model to its previous mistakes.
"""

from __future__ import annotations

from ..lang.grammar import ALLOWED_BUILTINS

_FIXER_TEMPLATE = """\
You are fixing code for a restricted Python environment. \
The code below was generated but contains errors.

Rewrite the code using ONLY these functions:
{namespace_desc}

Builtins: {builtins_list}

Output ONLY the fixed code — no explanation, no markdown fences."""


def build_fixer_prompt(namespace_desc: str) -> str:
    """Build the system prompt for the fixer conversation.

    This prompt is deliberately different from the generation prompt —
    it's shorter, focused on correction, and doesn't mention Jupyter
    or notebooks. A fresh framing helps the model break out of patterns.

    Args:
        namespace_desc: Formatted tool namespace string.

    Returns:
        System prompt for the fixer.
    """
    builtins_list = ", ".join(sorted(ALLOWED_BUILTINS))
    return _FIXER_TEMPLATE.format(
        namespace_desc=namespace_desc,
        builtins_list=builtins_list,
    )


def build_fixer_messages(
    intent: str,
    broken_program: str,
    errors: list[str],
    namespace_desc: str,
) -> list[dict[str, str]]:
    """Build the full message list for a fixer conversation.

    Args:
        intent: The original natural language intent.
        broken_program: The program that failed validation.
        errors: Validation error strings.
        namespace_desc: Formatted tool namespace string.

    Returns:
        List of message dicts: [system, user].
    """
    system = build_fixer_prompt(namespace_desc)

    error_text = "\n".join(f"- {e}" for e in errors if e != "--- Suggestions ---")

    user_content = (
        f"Intent: {intent}\n\n"
        f"Broken code:\n{broken_program}\n\n"
        f"Errors:\n{error_text}\n\n"
        f"Rewrite using only the available functions."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/infer/test_fixer.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/lackpy/infer/fixer.py tests/infer/test_fixer.py
git commit -m "feat(infer): add fresh fixer prompt for code correction strategy"
```

---

### Task 3: Correction Chain

**Files:**
- Create: `src/lackpy/infer/correction.py`
- Create: `tests/infer/test_correction.py`

- [ ] **Step 1: Write failing tests**

`tests/infer/test_correction.py`:
```python
"""Tests for the correction chain."""

import pytest

from lackpy.infer.correction import CorrectionChain, CorrectionAttempt


class FakeProvider:
    def __init__(self, name: str, responses: list[str | None]):
        self._name = name
        self._responses = list(responses)

    @property
    def name(self) -> str:
        return self._name

    def available(self) -> bool:
        return True

    async def generate(self, intent, namespace_desc, config=None, error_feedback=None):
        if self._responses:
            return self._responses.pop(0)
        return None


class TestDeterministicCleanupStrategy:
    @pytest.mark.asyncio
    async def test_strips_imports(self):
        chain = CorrectionChain()
        result = await chain.correct(
            program="import os\nfiles = glob('*.py')\nfiles",
            errors=["Forbidden AST node: Import"],
            namespace_desc="  glob(pattern) -> list: Find files",
            intent="find python files",
            allowed_names={"glob"},
            provider=None,
        )
        assert result is not None
        assert result.program is not None
        assert "import" not in result.program
        assert result.strategy == "deterministic_cleanup"


class TestFewShotStrategy:
    @pytest.mark.asyncio
    async def test_uses_provider_retry(self):
        provider = FakeProvider("ollama", ["x = 1"])
        chain = CorrectionChain()
        result = await chain.correct(
            program="import os\ndef f(): pass",  # two errors, cleanup won't fully fix
            errors=["Forbidden AST node: Import", "Forbidden AST node: FunctionDef"],
            namespace_desc="",
            intent="do something",
            allowed_names=set(),
            provider=provider,
        )
        assert result is not None
        assert result.program == "x = 1"
        assert result.strategy in ("few_shot_correction", "deterministic_cleanup")


class TestFreshFixerStrategy:
    @pytest.mark.asyncio
    async def test_fresh_fixer_produces_valid(self):
        # Provider returns None for few-shot, then valid code for fixer
        provider = FakeProvider("ollama", [None, "y = 2"])
        chain = CorrectionChain()
        result = await chain.correct(
            program="def f(): pass",
            errors=["Forbidden AST node: FunctionDef"],
            namespace_desc="",
            intent="do something",
            allowed_names=set(),
            provider=provider,
        )
        assert result is not None
        assert result.program == "y = 2"
        assert result.strategy == "fresh_fixer"


class TestChainExhaustion:
    @pytest.mark.asyncio
    async def test_returns_none_when_all_fail(self):
        provider = FakeProvider("ollama", [None, None])
        chain = CorrectionChain()
        result = await chain.correct(
            program="def f(): pass\nclass C: pass",
            errors=["Forbidden AST node: FunctionDef", "Forbidden AST node: ClassDef"],
            namespace_desc="",
            intent="do something",
            allowed_names=set(),
            provider=provider,
        )
        assert result is None


class TestAttemptHistory:
    @pytest.mark.asyncio
    async def test_records_attempts(self):
        chain = CorrectionChain()
        result = await chain.correct(
            program="import os\nx = 1",
            errors=["Forbidden AST node: Import"],
            namespace_desc="",
            intent="test",
            allowed_names=set(),
            provider=None,
        )
        assert result is not None
        assert len(chain.attempts) >= 1
        assert all(isinstance(a, CorrectionAttempt) for a in chain.attempts)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/infer/test_correction.py -v`
Expected: FAIL (import error)

- [ ] **Step 3: Implement the correction chain**

`src/lackpy/infer/correction.py`:
```python
"""Multi-strategy error correction chain for generated programs.

Strategies are tried in order:
1. Deterministic cleanup (strip imports, rewrite open/os.path)
2. Few-shot correction (same provider, error feedback, higher temp)
3. Fresh fixer prompt (same provider, new conversation)

Each strategy transforms or regenerates the program and re-validates.
The chain records all attempts for creation_log provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..lang.validator import validate
from .cleanup import deterministic_cleanup
from .fixer import build_fixer_messages
from .hints import enrich_errors
from .sanitize import sanitize_output


@dataclass
class CorrectionAttempt:
    """Record of a single correction attempt.

    Attributes:
        strategy: Name of the strategy that produced this attempt.
        program: The program text after this attempt.
        errors: Validation errors (empty if valid).
        accepted: Whether this attempt passed validation.
    """
    strategy: str
    program: str
    errors: list[str]
    accepted: bool


@dataclass
class CorrectionResult:
    """Result of a successful correction.

    Attributes:
        program: The corrected, valid program.
        strategy: Which strategy succeeded.
        attempts: Total number of attempts across all strategies.
    """
    program: str
    strategy: str
    attempts: int


class CorrectionChain:
    """Multi-strategy correction chain for invalid generated programs.

    Call correct() with a broken program. The chain tries deterministic
    cleanup, few-shot correction, and fresh fixer prompt in order.
    Returns a CorrectionResult if any strategy produces a valid program,
    or None if all strategies fail.

    After calling correct(), the attempts list contains all tried
    corrections (for creation_log provenance).
    """

    def __init__(self) -> None:
        self.attempts: list[CorrectionAttempt] = []

    async def correct(
        self,
        program: str,
        errors: list[str],
        namespace_desc: str,
        intent: str,
        allowed_names: set[str],
        provider: Any | None = None,
        extra_rules: list | None = None,
    ) -> CorrectionResult | None:
        """Try all correction strategies in order.

        Args:
            program: The program that failed validation.
            errors: Validation errors from the failed program.
            namespace_desc: Formatted tool namespace for prompts.
            intent: The original natural language intent.
            allowed_names: Allowed callable names for re-validation.
            provider: The inference provider for model-based strategies.
            extra_rules: Additional validation rules.

        Returns:
            CorrectionResult if a strategy produces a valid program, else None.
        """
        self.attempts = []

        # Strategy 1: Deterministic cleanup
        cleaned = deterministic_cleanup(program)
        if cleaned and cleaned != program:
            validation = validate(cleaned, allowed_names=allowed_names, extra_rules=extra_rules)
            self.attempts.append(CorrectionAttempt(
                strategy="deterministic_cleanup",
                program=cleaned,
                errors=validation.errors,
                accepted=validation.valid,
            ))
            if validation.valid:
                return CorrectionResult(
                    program=cleaned,
                    strategy="deterministic_cleanup",
                    attempts=len(self.attempts),
                )
            # Update for subsequent strategies
            program = cleaned
            errors = validation.errors

        # Strategy 2: Few-shot correction (needs a provider)
        if provider is not None:
            enriched = enrich_errors(errors, namespace_desc)
            raw = await provider.generate(
                intent, namespace_desc, error_feedback=enriched,
            )
            if raw is not None:
                candidate = sanitize_output(raw)
                candidate = deterministic_cleanup(candidate)
                if candidate:
                    validation = validate(candidate, allowed_names=allowed_names, extra_rules=extra_rules)
                    self.attempts.append(CorrectionAttempt(
                        strategy="few_shot_correction",
                        program=candidate,
                        errors=validation.errors,
                        accepted=validation.valid,
                    ))
                    if validation.valid:
                        return CorrectionResult(
                            program=candidate,
                            strategy="few_shot_correction",
                            attempts=len(self.attempts),
                        )

        # Strategy 3: Fresh fixer prompt (needs a provider)
        if provider is not None:
            messages = build_fixer_messages(
                intent=intent,
                broken_program=program,
                errors=errors,
                namespace_desc=namespace_desc,
            )
            # Use the provider's _chat if available (Ollama), otherwise generate()
            raw = await _call_fixer(provider, messages)
            if raw is not None:
                candidate = sanitize_output(raw)
                candidate = deterministic_cleanup(candidate)
                if candidate:
                    validation = validate(candidate, allowed_names=allowed_names, extra_rules=extra_rules)
                    self.attempts.append(CorrectionAttempt(
                        strategy="fresh_fixer",
                        program=candidate,
                        errors=validation.errors,
                        accepted=validation.valid,
                    ))
                    if validation.valid:
                        return CorrectionResult(
                            program=candidate,
                            strategy="fresh_fixer",
                            attempts=len(self.attempts),
                        )

        return None


async def _call_fixer(provider: Any, messages: list[dict]) -> str | None:
    """Call a provider with pre-built fixer messages.

    Tries _chat() first (Ollama/Anthropic have it). Falls back to
    using the user message content as a generate() call.
    """
    try:
        if hasattr(provider, "_chat"):
            response = await provider._chat(messages, temperature=0.4)
            content = response.get("message", {}).get("content", "")
            return content.strip() if content else None
        elif hasattr(provider, "_create_message"):
            system = messages[0]["content"]
            user_messages = messages[1:]
            response = await provider._create_message(system, user_messages)
            content = response.content[0].text
            return content.strip() if content else None
        else:
            return None
    except Exception:
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/infer/test_correction.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/lackpy/infer/correction.py tests/infer/test_correction.py
git commit -m "feat(infer): add CorrectionChain — deterministic cleanup, few-shot, fresh fixer"
```

---

### Task 4: Integrate Correction Chain into Dispatcher

**Files:**
- Modify: `src/lackpy/infer/dispatch.py`
- Modify: `tests/infer/test_dispatch.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/infer/test_dispatch.py`:

```python
class TestCorrectionChain:
    @pytest.mark.asyncio
    async def test_deterministic_cleanup_fixes_imports(self):
        """Provider returns code with imports — cleanup should strip them."""
        d = InferenceDispatcher(providers=[
            FakeProvider("ollama", "import os\nx = 1"),
        ])
        result = await d.generate("test", namespace_desc="", allowed_names=set())
        assert result.program == "x = 1"

    @pytest.mark.asyncio
    async def test_correction_chain_recorded_in_result(self):
        """Result should include correction metadata when chain was used."""
        d = InferenceDispatcher(providers=[
            FakeProvider("ollama", "import os\nx = 1"),
        ])
        result = await d.generate("test", namespace_desc="", allowed_names=set())
        assert result.program == "x = 1"
        assert result.correction_strategy is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/infer/test_dispatch.py -v`
Expected: new tests FAIL

- [ ] **Step 3: Modify dispatcher to use CorrectionChain**

Update `src/lackpy/infer/dispatch.py`:

Add `correction_strategy: str | None = None` and `correction_attempts: int = 0` fields to `GenerationResult`.

Replace the inline retry logic with a CorrectionChain call. The new flow for each provider:

1. Generate
2. Sanitize + deterministic cleanup
3. Validate
4. If valid → return
5. If invalid → run correction chain
6. If chain succeeds → return
7. If chain fails → next provider

```python
from .correction import CorrectionChain

@dataclass
class GenerationResult:
    program: str
    provider_name: str
    generation_time_ms: float
    correction_strategy: str | None = None
    correction_attempts: int = 0
    attempts_log: list | None = None  # for creation_log


class InferenceDispatcher:
    def __init__(self, providers: list[Any]) -> None:
        self._providers = providers

    async def generate(self, intent, namespace_desc, allowed_names,
                       params_desc=None, extra_rules=None) -> GenerationResult:
        start = time.perf_counter()
        errors_by_provider: dict[str, list[str]] = {}

        for provider in self._providers:
            if not provider.available():
                continue

            raw = await provider.generate(intent, namespace_desc)
            if raw is None:
                continue

            program = sanitize_output(raw)
            # Apply deterministic cleanup immediately
            from .cleanup import deterministic_cleanup
            program = deterministic_cleanup(program)

            validation = validate(program, allowed_names=allowed_names, extra_rules=extra_rules)
            if validation.valid:
                elapsed = (time.perf_counter() - start) * 1000
                return GenerationResult(
                    program=program, provider_name=provider.name,
                    generation_time_ms=elapsed,
                )

            # Run correction chain
            chain = CorrectionChain()
            correction = await chain.correct(
                program=program, errors=validation.errors,
                namespace_desc=namespace_desc, intent=intent,
                allowed_names=allowed_names, provider=provider,
                extra_rules=extra_rules,
            )

            if correction is not None:
                elapsed = (time.perf_counter() - start) * 1000
                return GenerationResult(
                    program=correction.program,
                    provider_name=provider.name,
                    generation_time_ms=elapsed,
                    correction_strategy=correction.strategy,
                    correction_attempts=correction.attempts,
                    attempts_log=[
                        {"strategy": a.strategy, "program": a.program,
                         "errors": a.errors, "accepted": a.accepted}
                        for a in chain.attempts
                    ],
                )

            errors_by_provider[provider.name] = validation.errors

        provider_names = [p.name for p in self._providers if p.available()]
        raise RuntimeError(
            f"All {len(provider_names)} providers failed to produce a valid program. "
            f"Tried: {', '.join(provider_names)}. Last errors: {errors_by_provider}"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/infer/test_dispatch.py -v`
Expected: all PASS

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/lackpy/infer/dispatch.py tests/infer/test_dispatch.py
git commit -m "feat(infer): integrate CorrectionChain into dispatcher — replaces inline retry"
```

---

### Task 5: Wire Correction Attempts into Creation Log

**Files:**
- Modify: `src/lackpy/service.py`
- Create: `tests/test_correction_integration.py`

- [ ] **Step 1: Write failing test**

`tests/test_correction_integration.py`:
```python
"""Integration test: correction chain feeds into creation log."""

import pytest
from pathlib import Path

from lackpy.service import LackpyService
from lackpy.kit.toolbox import ToolSpec, ArgSpec


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "test.txt").write_text("hello")
    config_dir = tmp_path / ".lackpy"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        '[inference]\norder = ["templates", "rules"]\n\n[kit]\ndefault = "debug"\n'
    )
    return tmp_path


@pytest.fixture
def service(workspace):
    return LackpyService(workspace=workspace)


class TestCorrectionInDelegate:
    @pytest.mark.asyncio
    async def test_delegate_returns_correction_info(self, service):
        result = await service.delegate("read file test.txt", kit=["read"])
        assert result["success"]
        # correction_strategy may be None (rules provider doesn't need correction)
        assert "correction_strategy" in result

    @pytest.mark.asyncio
    async def test_create_lackey_with_correction_log(self, service, workspace):
        gen = await service.generate("read file test.txt", kit=["read"])

        # Build a creation log that includes correction attempts
        creation_log = [
            {"role": "user", "content": "read file test.txt"},
        ]
        if gen.attempts_log:
            for attempt in gen.attempts_log:
                creation_log.append({
                    "role": "assistant",
                    "content": attempt["program"],
                    "accepted": attempt["accepted"],
                    "errors": attempt.get("errors"),
                    "strategy": attempt.get("strategy"),
                })
        creation_log.append({
            "role": "assistant",
            "content": gen.program,
            "accepted": True,
        })

        path = await service.create_lackey(
            program=gen.program, name="ReadTest", tools=["read"],
            creation_log=creation_log, output_dir=workspace,
        )
        assert path.exists()
        content = path.read_text()
        assert "creation_log" in content
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_correction_integration.py -v`
Expected: FAIL (missing correction_strategy key in delegate result)

- [ ] **Step 3: Add correction metadata to delegate result**

In `src/lackpy/service.py`, in the `delegate` method, after getting `gen_result`, add correction metadata to the return dict:

Add these keys to the return dict:
```python
            "correction_strategy": gen_result.correction_strategy,
            "correction_attempts": gen_result.correction_attempts,
```

This requires that `GenerationResult` has these fields — which Task 4 adds.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_correction_integration.py -v`
Expected: all PASS

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/lackpy/service.py tests/test_correction_integration.py
git commit -m "feat: wire correction metadata into delegate results and creation log"
```
