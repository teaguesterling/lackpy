# Distillation Callback Implementation Plan

> **Status: IMPLEMENTED** — 2026-04-20. See outcome notes below.

**Goal:** Let Kibitzer post-process its own query results through a callback that lackpy provides, producing concise correction hints sized for small models — instead of dumping raw pattern data into the prompt.

---

## Outcome

The final implementation diverged from the original plan. Instead of a `DistillCallback` protocol with `DistillContext`/`DistillResult` dataclasses, kibitzer v0.4.0 shipped a **doc context pipeline** with a different (simpler) architecture:

- **Kibitzer side** (`kibitzer.docs`): `DocSection`, `DocResult`, `DocRefinement` (select/present callbacks), `register_docs()`, `get_doc_context()`. Kibitzer retrieves doc sections and refines them through consumer-provided callbacks.
- **Lackpy side**: Three integration points instead of one callback:
  1. `service._register_kibitzer_docs()` — registers tool doc refs + a `DocRefinement` at init
  2. `infer/distill.py` — `build_doc_refinement()` with a `_select_sections` callback that picks failure-mode-relevant doc sections
  3. `infer/correction.py` — consumes `doc_context` from `get_correction_hints(tool=)` and folds sections into error enrichment

Key difference from the plan: the circular dependency was avoided through `register_docs()` + `DocRefinement` callbacks rather than a `DistillCallback` protocol. Kibitzer owns retrieval and calls the select callback; lackpy provides the filter logic. No protocol needed.

---

## Original Plan

> The plan below was the pre-implementation design. It is preserved for context but does not reflect what shipped.

**Architecture (as planned):** Kibitzer defines a `DistillCallback` protocol (a callable signature). Lackpy implements it at session init, injecting its own doc system and failure mode knowledge. When Kibitzer needs to produce a correction hint (via `get_prompt_hints` or `get_correction_hints`), it passes raw context through the callback, which reads relevant tool docs on demand and returns a short, targeted string. This breaks the circular dependency: Kibitzer never imports lackpy; it just calls the function it was given.

**Tech Stack:** Python 3.12, dataclasses, typing.Protocol, pytest

---

## File Structure

### Kibitzer side (kibitzer package)

| File | Responsibility |
|------|----------------|
| `src/kibitzer/distill.py` (create) | `DistillCallback` protocol, `DistillContext` dataclass, `DistillResult` dataclass |
| `src/kibitzer/session.py` (modify) | Accept callback at init, invoke it in `get_prompt_hints()` and `get_correction_hints()` |
| `tests/test_distill.py` (create) | Unit tests for protocol, context construction, callback invocation |

### Lackpy side (lackpy package)

| File | Responsibility |
|------|----------------|
| `src/lackpy/infer/distill.py` (create) | `build_distill_callback()` factory that returns a `DistillCallback`-compatible callable; rule-based distillation logic; doc resolution |
| `src/lackpy/service.py` (modify) | Wire callback into KibitzerSession at init time |
| `src/lackpy/infer/hints.py` (modify) | Use distilled hints instead of raw enrichment when available |
| `tests/test_distill.py` (create) | Unit tests for the distillation callback |
| `tests/test_distill_integration.py` (create) | Integration test: Kibitzer + lackpy callback end-to-end |

---

## Task 1: Define the DistillCallback protocol in Kibitzer

**Files:**
- Create: `src/kibitzer/distill.py`
- Test: `tests/test_distill.py`

This task is done entirely in the **kibitzer** repo (`/mnt/aux-data/teague/Projects/kibitzer`).

- [ ] **Step 1: Write the failing test for DistillContext and DistillResult**

```python
# tests/test_distill.py
from kibitzer.distill import DistillContext, DistillResult


class TestDistillContext:
    def test_required_fields(self):
        ctx = DistillContext(
            failure_mode="stdlib_leak",
            errors=["Forbidden name: 'open'"],
            program="content = open('f.txt').read()",
            intent="read the file",
            tool_docs={"read_file": "docs/tools/read_file.md"},
            kit_docs=[],
        )
        assert ctx.failure_mode == "stdlib_leak"
        assert ctx.errors == ["Forbidden name: 'open'"]
        assert ctx.tool_docs == {"read_file": "docs/tools/read_file.md"}

    def test_optional_fields_default(self):
        ctx = DistillContext(
            failure_mode="stdlib_leak",
            errors=["Forbidden name: 'open'"],
            program="content = open('f.txt').read()",
            intent="read the file",
            tool_docs={},
            kit_docs=[],
        )
        assert ctx.model is None
        assert ctx.history_count is None
        assert ctx.attempt == 1


class TestDistillResult:
    def test_has_hint_and_doc_used(self):
        result = DistillResult(
            hint="Use read_file(path) instead of open().",
            docs_used=["docs/tools/read_file.md"],
        )
        assert result.hint == "Use read_file(path) instead of open()."
        assert result.docs_used == ["docs/tools/read_file.md"]

    def test_empty_result(self):
        result = DistillResult(hint="", docs_used=[])
        assert result.hint == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/aux-data/teague/Projects/kibitzer && python -m pytest tests/test_distill.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'kibitzer.distill'`

- [ ] **Step 3: Implement DistillContext, DistillResult, and DistillCallback**

```python
# src/kibitzer/distill.py
"""Distillation callback protocol for post-processing Kibitzer results.

Kibitzer defines the protocol; consumers (e.g. lackpy) implement it.
This avoids circular imports — Kibitzer never imports the consumer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass
class DistillContext:
    """Raw context Kibitzer passes to the distillation callback.

    Attributes:
        failure_mode: Classified failure mode (from failure_modes taxonomy),
            or None if not yet classified.
        errors: Validation or execution error strings.
        program: The failed program source.
        intent: Original user intent.
        tool_docs: Mapping of tool name to relative doc path for tools
            in the current kit. The callback can resolve these on demand.
        kit_docs: Kit-level doc paths.
        model: Model name that produced the failure, if known.
        history_count: How many times this model has hit this failure mode,
            if known from pattern tracking.
        attempt: Which correction attempt this is (1-indexed).
        extra: Arbitrary provider-specific data.
    """

    failure_mode: str | None
    errors: list[str]
    program: str
    intent: str
    tool_docs: dict[str, str]
    kit_docs: list[str]
    model: str | None = None
    history_count: int | None = None
    attempt: int = 1
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DistillResult:
    """Distilled output: a concise hint and which docs were consulted.

    Attributes:
        hint: A short correction hint (1-3 sentences) sized for small models.
            Empty string means no actionable distillation was possible.
        docs_used: List of doc paths that were read during distillation.
            Useful for observability and caching.
    """

    hint: str
    docs_used: list[str] = field(default_factory=list)


class DistillCallback(Protocol):
    """Protocol for distillation callbacks.

    Implementors receive raw Kibitzer context and return a concise,
    targeted correction hint. The callback may read doc files, query
    tool metadata, or apply rule-based logic — Kibitzer doesn't care
    how the distillation happens, only that it gets a short result.
    """

    def __call__(self, ctx: DistillContext) -> DistillResult: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/aux-data/teague/Projects/kibitzer && python -m pytest tests/test_distill.py -v`
Expected: PASS (all 4 tests)

- [ ] **Step 5: Commit**

```bash
cd /mnt/aux-data/teague/Projects/kibitzer
git add src/kibitzer/distill.py tests/test_distill.py
git commit -m "feat: DistillCallback protocol for post-processing query results"
```

---

## Task 2: Wire the callback into KibitzerSession

**Files:**
- Modify: `src/kibitzer/session.py`
- Test: `tests/test_distill.py` (extend)

This task is done entirely in the **kibitzer** repo.

- [ ] **Step 1: Write the failing test for callback registration and invocation**

```python
# tests/test_distill.py (append to existing file)
from kibitzer.session import KibitzerSession
from kibitzer.distill import DistillContext, DistillResult


class TestSessionDistillCallback:
    def test_register_distill_callback(self, tmp_path):
        session = KibitzerSession(project_dir=tmp_path)
        session.load()

        results = []

        def my_callback(ctx: DistillContext) -> DistillResult:
            results.append(ctx)
            return DistillResult(hint="Use read_file(path).", docs_used=[])

        session.register_distill_callback(my_callback)
        assert session._distill_callback is my_callback

    def test_distill_called_in_get_correction_hints(self, tmp_path):
        session = KibitzerSession(project_dir=tmp_path)
        session.load()

        captured = []

        def my_callback(ctx: DistillContext) -> DistillResult:
            captured.append(ctx)
            return DistillResult(
                hint="Use read_file(path) instead of open().",
                docs_used=["docs/tools/read_file.md"],
            )

        session.register_distill_callback(my_callback)

        result = session.get_correction_hints(
            failure_mode="stdlib_leak",
            model="qwen2.5-coder:1.5b",
            attempt=1,
        )
        assert "distilled" in result
        assert result["distilled"]["hint"] == "Use read_file(path) instead of open()."
        assert len(captured) == 1
        assert captured[0].failure_mode == "stdlib_leak"

    def test_no_callback_means_no_distilled_field(self, tmp_path):
        session = KibitzerSession(project_dir=tmp_path)
        session.load()
        result = session.get_correction_hints(
            failure_mode="stdlib_leak",
            model="qwen2.5-coder:1.5b",
            attempt=1,
        )
        assert "distilled" not in result

    def test_callback_error_is_swallowed(self, tmp_path):
        session = KibitzerSession(project_dir=tmp_path)
        session.load()

        def broken_callback(ctx: DistillContext) -> DistillResult:
            raise RuntimeError("boom")

        session.register_distill_callback(broken_callback)
        result = session.get_correction_hints(
            failure_mode="stdlib_leak",
            model=None,
            attempt=1,
        )
        assert "distilled" not in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/aux-data/teague/Projects/kibitzer && python -m pytest tests/test_distill.py::TestSessionDistillCallback -v`
Expected: FAIL with `AttributeError: 'KibitzerSession' object has no attribute 'register_distill_callback'`

- [ ] **Step 3: Add `register_distill_callback` and invoke it in `get_correction_hints`**

In `src/kibitzer/session.py`, add to `__init__`:

```python
self._distill_callback: DistillCallback | None = None
```

Add the registration method (after `register_context`):

```python
def register_distill_callback(self, callback: Any) -> None:
    """Register a callback for post-processing correction hints.

    The callback receives a DistillContext with raw failure data and
    returns a DistillResult with a concise hint. Kibitzer never imports
    the callback's implementation — the caller injects it.
    """
    self._distill_callback = callback
```

In `get_correction_hints`, after building the `result` dict (before the `return result` at line 551), add:

```python
if self._distill_callback is not None:
    try:
        from kibitzer.distill import DistillContext
        distill_ctx = DistillContext(
            failure_mode=failure_mode,
            errors=[],
            program="",
            intent="",
            tool_docs={},
            kit_docs=[],
            model=model,
            history_count=result["history"]["count"] if result["history"] else None,
            attempt=attempt,
        )
        distill_result = self._distill_callback(distill_ctx)
        result["distilled"] = {
            "hint": distill_result.hint,
            "docs_used": distill_result.docs_used,
        }
    except Exception:
        pass
```

Note: The `errors`, `program`, `intent`, `tool_docs`, and `kit_docs` fields are empty here because `get_correction_hints` doesn't have that context. The caller (lackpy's correction chain) will populate them when it constructs the `DistillContext` directly. This minimal invocation is a fallback for consumers that only use `get_correction_hints` without the richer context.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/aux-data/teague/Projects/kibitzer && python -m pytest tests/test_distill.py -v`
Expected: PASS (all 8 tests)

- [ ] **Step 5: Commit**

```bash
cd /mnt/aux-data/teague/Projects/kibitzer
git add src/kibitzer/session.py tests/test_distill.py
git commit -m "feat: register_distill_callback on KibitzerSession"
```

---

## Task 3: Wire the callback with richer context in `get_prompt_hints`

**Files:**
- Modify: `src/kibitzer/session.py`
- Test: `tests/test_distill.py` (extend)

This task is done entirely in the **kibitzer** repo. `get_prompt_hints` is the other consumer — it generates prompt-level hints from failure patterns. With the callback, it can distill each pattern into a targeted hint instead of using the static `_FAILURE_HINT_MAP`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_distill.py (append)

class TestPromptHintsDistillation:
    def test_distill_callback_enriches_prompt_hints(self, tmp_path):
        session = KibitzerSession(project_dir=tmp_path)
        session.load()

        def my_callback(ctx: DistillContext) -> DistillResult:
            return DistillResult(
                hint=f"Distilled: avoid {ctx.failure_mode}",
                docs_used=[],
            )

        session.register_distill_callback(my_callback)

        # Manually inject a failure pattern so get_prompt_hints has data
        session.report_generation({
            "intent": "read file",
            "success": False,
            "failure_mode": "stdlib_leak",
            "model": "test-model",
        })

        hints = session.get_prompt_hints(model="test-model", window=10, min_confidence=0.0)
        distilled = [h for h in hints if h.get("distilled")]
        assert len(distilled) >= 1
        assert "Distilled: avoid stdlib_leak" in distilled[0]["content"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/aux-data/teague/Projects/kibitzer && python -m pytest tests/test_distill.py::TestPromptHintsDistillation -v`
Expected: FAIL — no `distilled` key in hints

- [ ] **Step 3: Add callback invocation in `get_prompt_hints`**

In `src/kibitzer/session.py`, inside the `get_prompt_hints` method, after building each hint dict (around line 470-483), add a branch that tries the callback first:

```python
# Inside the for-loop over patterns, replace the hint construction:
if self._distill_callback is not None:
    try:
        from kibitzer.distill import DistillContext
        distill_ctx = DistillContext(
            failure_mode=pattern["pattern"],
            errors=[],
            program="",
            intent=pattern.get("sample_intent", ""),
            tool_docs={},
            kit_docs=[],
            model=pattern.get("model"),
            history_count=pattern["count"],
        )
        distill_result = self._distill_callback(distill_ctx)
        if distill_result.hint:
            hints.append({
                "type": "negative_constraint",
                "content": distill_result.hint,
                "confidence": round(confidence, 2),
                "source": f"failure_pattern:{pattern['pattern']}",
                "distilled": True,
                "docs_used": distill_result.docs_used,
            })
            continue
    except Exception:
        pass

# Fall through to existing static hint map logic
known = _FAILURE_HINT_MAP.get(pattern["pattern"])
# ... existing code ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/aux-data/teague/Projects/kibitzer && python -m pytest tests/test_distill.py -v`
Expected: PASS (all 9 tests)

- [ ] **Step 5: Commit**

```bash
cd /mnt/aux-data/teague/Projects/kibitzer
git add src/kibitzer/session.py tests/test_distill.py
git commit -m "feat: distill callback in get_prompt_hints"
```

---

## Task 4: Implement the distillation callback in lackpy

**Files:**
- Create: `src/lackpy/infer/distill.py`
- Test: `tests/test_distill.py`

This is the core logic — a rule-based distiller that reads tool docs on demand and produces concise hints.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_distill.py
import pytest
from unittest.mock import patch
from lackpy.infer.distill import build_distill_callback


@pytest.fixture
def doc_reader():
    """Simulate doc resolution: returns canned content for known paths."""
    docs = {
        "docs/tools/read_file.md": (
            "# read_file\n\n"
            "## Signature\n\n"
            "```python\n"
            "read_file(path: str) -> str\n"
            "```\n\n"
            "## Notes\n\n"
            "- Raises FileNotFoundError if the path does not exist.\n"
        ),
        "docs/tools/edit_file.md": (
            "# edit_file\n\n"
            "## Signature\n\n"
            "```python\n"
            "edit_file(path: str, old_str: str, new_str: str) -> bool\n"
            "```\n"
        ),
    }
    return lambda path: docs.get(path)


class TestDistillCallback:
    def test_stdlib_leak_suggests_read_file(self, doc_reader):
        from kibitzer.distill import DistillContext
        callback = build_distill_callback(doc_reader)
        ctx = DistillContext(
            failure_mode="stdlib_leak",
            errors=["Forbidden name: 'open'"],
            program="content = open('f.txt').read()",
            intent="read the file",
            tool_docs={"read_file": "docs/tools/read_file.md"},
            kit_docs=[],
        )
        result = callback(ctx)
        assert "read_file" in result.hint
        assert "docs/tools/read_file.md" in result.docs_used

    def test_unknown_failure_mode_returns_generic(self, doc_reader):
        from kibitzer.distill import DistillContext
        callback = build_distill_callback(doc_reader)
        ctx = DistillContext(
            failure_mode="some_unknown_mode",
            errors=["Something went wrong"],
            program="x = 1",
            intent="do something",
            tool_docs={},
            kit_docs=[],
        )
        result = callback(ctx)
        assert isinstance(result.hint, str)

    def test_implement_not_orchestrate(self, doc_reader):
        from kibitzer.distill import DistillContext
        callback = build_distill_callback(doc_reader)
        ctx = DistillContext(
            failure_mode="implement_not_orchestrate",
            errors=["Forbidden node: FunctionDef"],
            program="def read(path):\n    return open(path).read()",
            intent="read the config file",
            tool_docs={"read_file": "docs/tools/read_file.md"},
            kit_docs=[],
        )
        result = callback(ctx)
        assert "read_file" in result.hint.lower() or "call" in result.hint.lower()

    def test_missing_doc_still_produces_hint(self, doc_reader):
        from kibitzer.distill import DistillContext
        callback = build_distill_callback(doc_reader)
        ctx = DistillContext(
            failure_mode="stdlib_leak",
            errors=["Forbidden name: 'open'"],
            program="content = open('f.txt').read()",
            intent="read the file",
            tool_docs={"read_file": "docs/tools/nonexistent.md"},
            kit_docs=[],
        )
        result = callback(ctx)
        assert isinstance(result.hint, str)
        assert len(result.hint) > 0

    def test_hint_is_concise(self, doc_reader):
        from kibitzer.distill import DistillContext
        callback = build_distill_callback(doc_reader)
        ctx = DistillContext(
            failure_mode="stdlib_leak",
            errors=["Forbidden name: 'open'"],
            program="content = open('f.txt').read()",
            intent="read the file",
            tool_docs={"read_file": "docs/tools/read_file.md"},
            kit_docs=[],
        )
        result = callback(ctx)
        assert len(result.hint) < 300
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_distill.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lackpy.infer.distill'`

- [ ] **Step 3: Implement the distillation callback**

```python
# src/lackpy/infer/distill.py
"""Distillation callback: post-process Kibitzer results into concise hints.

The callback is injected into KibitzerSession at init time, breaking
the circular dependency. Kibitzer calls it with raw context; this module
reads relevant tool docs on demand and produces short, targeted hints
sized for small models.
"""

from __future__ import annotations

import re
from typing import Any, Callable


def build_distill_callback(
    doc_reader: Callable[[str], str | None],
) -> Callable:
    """Build a distillation callback that resolves docs via doc_reader.

    Args:
        doc_reader: A callable that takes a relative doc path and returns
            the file content as a string, or None if not found. Typically
            ``service.resolve_doc``.

    Returns:
        A callable matching the ``DistillCallback`` protocol.
    """
    from kibitzer.distill import DistillContext, DistillResult

    def _distill(ctx: DistillContext) -> DistillResult:
        docs_used: list[str] = []

        # Try to extract the signature block from the most relevant tool doc
        signature = _find_relevant_signature(ctx, doc_reader, docs_used)

        hint = _build_hint(ctx, signature)
        return DistillResult(hint=hint, docs_used=docs_used)

    return _distill


def _find_relevant_signature(
    ctx: Any,
    doc_reader: Callable[[str], str | None],
    docs_used: list[str],
) -> str | None:
    """Extract the Signature section from the most relevant tool doc."""
    if not ctx.tool_docs:
        return None

    target_tool = _pick_relevant_tool(ctx)
    if target_tool is None:
        return None

    doc_path = ctx.tool_docs.get(target_tool)
    if doc_path is None:
        return None

    content = doc_reader(doc_path)
    if content is None:
        return None

    docs_used.append(doc_path)
    return _extract_signature(content)


def _pick_relevant_tool(ctx: Any) -> str | None:
    """Pick the tool most relevant to the failure from the kit."""
    error_text = " ".join(ctx.errors).lower()

    if ctx.failure_mode in ("stdlib_leak", "implement_not_orchestrate"):
        if "open" in error_text and "read_file" in ctx.tool_docs:
            return "read_file"
        if "glob" in error_text and "find_files" in ctx.tool_docs:
            return "find_files"

    if ctx.failure_mode == "key_hallucination":
        for tool_name in ctx.tool_docs:
            if tool_name in error_text:
                return tool_name

    if ctx.tool_docs:
        return next(iter(ctx.tool_docs))

    return None


def _extract_signature(doc_content: str) -> str | None:
    """Pull the ```python ... ``` block from a ## Signature section."""
    match = re.search(
        r"## Signature\s*\n\s*```python\s*\n(.+?)```",
        doc_content,
        re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return None


_FAILURE_TEMPLATES: dict[str, str] = {
    "stdlib_leak": (
        "Use {tool}. All file operations must go through kit tools, "
        "not Python builtins like open()."
    ),
    "implement_not_orchestrate": (
        "Call {tool} directly — do not define functions or classes. "
        "The tools are already implemented; just call them."
    ),
    "path_prefix": (
        "Paths are relative to the workspace root. "
        "Do not add directory prefixes."
    ),
    "syntax_artifact": (
        "Output ONLY valid Python code. No annotations, arrows, or prose."
    ),
    "jupyter_confusion": (
        "Write a complete Python program using the available tools. "
        "Do not output bare filenames or cell markers."
    ),
    "key_hallucination": (
        "{tool} returns: {signature}. Check the return type and "
        "access the correct fields."
    ),
    "wrong_output": (
        "The code ran but produced the wrong result. "
        "Re-read the intent and check the logic."
    ),
}


def _build_hint(ctx: Any, signature: str | None) -> str:
    """Compose the distilled hint from failure mode + doc signature."""
    tool_name = _pick_relevant_tool(ctx) or "the available tools"

    template = _FAILURE_TEMPLATES.get(ctx.failure_mode or "")

    if template:
        hint = template.format(
            tool=tool_name,
            signature=signature or "(see tool description)",
        )
    else:
        hint = f"Error: {'; '.join(ctx.errors[:2])}"

    if signature and ctx.failure_mode in (
        "stdlib_leak", "implement_not_orchestrate", "key_hallucination"
    ):
        hint += f" Signature: {signature}"

    return hint
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_distill.py -v`
Expected: PASS (all 5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/lackpy/infer/distill.py tests/test_distill.py
git commit -m "feat: rule-based distillation callback for Kibitzer"
```

---

## Task 5: Wire the callback into LackpyService

**Files:**
- Modify: `src/lackpy/service.py:143-161` (`_init_kibitzer`)
- Test: `tests/test_distill_integration.py` (create)

- [ ] **Step 1: Write the failing integration test**

```python
# tests/test_distill_integration.py
"""Integration test: Kibitzer distillation callback wired through service."""

import pytest

try:
    from kibitzer import KibitzerSession
    from kibitzer.distill import DistillContext, DistillResult
    HAS_KIBITZER = True
except ImportError:
    HAS_KIBITZER = False

from lackpy.service import LackpyService


@pytest.mark.skipif(not HAS_KIBITZER, reason="kibitzer not installed")
class TestDistillIntegration:
    def test_callback_registered_at_init(self, tmp_path):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        config_dir = workspace / ".lackpy"
        config_dir.mkdir()
        (config_dir / "kits").mkdir()
        (config_dir / "config.toml").write_text(
            '[inference]\norder = ["templates", "rules"]\n[kit]\ndefault = "none"\n'
        )
        svc = LackpyService(workspace=workspace)
        assert svc._kibitzer is not None
        assert svc._kibitzer._distill_callback is not None

    def test_callback_resolves_docs(self, tmp_path):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        config_dir = workspace / ".lackpy"
        config_dir.mkdir()
        (config_dir / "kits").mkdir()
        (config_dir / "config.toml").write_text(
            '[inference]\norder = ["templates", "rules"]\n[kit]\ndefault = "none"\n'
        )
        # Create a doc file in the workspace
        docs_dir = workspace / "docs" / "tools"
        docs_dir.mkdir(parents=True)
        (docs_dir / "read_file.md").write_text(
            "# read_file\n\n## Signature\n\n```python\n"
            "read_file(path: str) -> str\n```\n"
        )

        svc = LackpyService(workspace=workspace)
        callback = svc._kibitzer._distill_callback

        ctx = DistillContext(
            failure_mode="stdlib_leak",
            errors=["Forbidden name: 'open'"],
            program="content = open('f.txt').read()",
            intent="read the file",
            tool_docs={"read_file": "docs/tools/read_file.md"},
            kit_docs=[],
        )
        result = callback(ctx)
        assert "read_file" in result.hint
        assert "docs/tools/read_file.md" in result.docs_used
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_distill_integration.py -v`
Expected: FAIL with `AttributeError: '_distill_callback'`

- [ ] **Step 3: Wire the callback in `_init_kibitzer`**

In `src/lackpy/service.py`, modify `_init_kibitzer` (lines 143-161). After the `register_tools` call and before the except block, add:

```python
# Register distillation callback so Kibitzer can post-process
# its own query results through our doc system
try:
    from .infer.distill import build_distill_callback
    callback = build_distill_callback(self.resolve_doc)
    self._kibitzer.register_distill_callback(callback)
except Exception:
    pass  # older kibitzer without distill support
```

The full `_init_kibitzer` becomes:

```python
def _init_kibitzer(self) -> None:
    """Initialize Kibitzer session if available."""
    if not _HAS_KIBITZER:
        return
    try:
        self._kibitzer = KibitzerSession(project_dir=self._workspace)
        self._kibitzer.load()
        self._kibitzer.register_tools([
            {
                "name": spec.name,
                "grade": {"w": spec.grade_w, "d": spec.effects_ceiling},
                "description": spec.description,
                "effects": "write" if spec.grade_w >= 3 else "read" if spec.grade_w >= 1 else "none",
            }
            for spec in self.toolbox.list_tools()
        ])
        try:
            from .infer.distill import build_distill_callback
            callback = build_distill_callback(self.resolve_doc)
            self._kibitzer.register_distill_callback(callback)
        except Exception:
            pass
    except Exception:
        self._kibitzer = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_distill_integration.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `python -m pytest tests/ --ignore=tests/eval --ignore=tests/interpreters/test_ast_select.py -q`
Expected: All existing tests still pass

- [ ] **Step 6: Commit**

```bash
git add src/lackpy/service.py tests/test_distill_integration.py
git commit -m "feat: wire distillation callback into KibitzerSession at init"
```

---

## Task 6: Use distilled hints in the correction chain

**Files:**
- Modify: `src/lackpy/infer/correction.py:144-176`
- Modify: `src/lackpy/infer/steps/few_shot.py:25-60`
- Test: `tests/test_distill.py` (extend)

The correction chain currently calls `get_correction_hints` and manually interprets the result. With the distillation callback wired in, `get_correction_hints` now includes a `distilled` key. Use it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_distill.py (append)

class TestCorrectionChainDistillation:
    def test_distilled_hint_used_in_enrichment(self, doc_reader):
        """Verify that when get_correction_hints returns a distilled hint,
        it gets included in the error enrichment for few-shot retry."""
        from lackpy.infer.hints import enrich_errors_with_distillation

        enriched = enrich_errors_with_distillation(
            errors=["Forbidden name: 'open'"],
            namespace_desc="  read_file(path) -> str: Read file contents",
            distilled_hint="Use read_file(path) instead of open(). Signature: read_file(path: str) -> str",
        )
        assert any("read_file(path)" in e for e in enriched)

    def test_no_distilled_hint_falls_back(self, doc_reader):
        from lackpy.infer.hints import enrich_errors_with_distillation

        enriched = enrich_errors_with_distillation(
            errors=["Forbidden name: 'open'"],
            namespace_desc="  read_file(path) -> str: Read file contents",
            distilled_hint=None,
        )
        # Should still have the basic enrichment from enrich_errors
        assert any("read_file" in e for e in enriched)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_distill.py::TestCorrectionChainDistillation -v`
Expected: FAIL with `ImportError: cannot import name 'enrich_errors_with_distillation'`

- [ ] **Step 3: Add `enrich_errors_with_distillation` to hints.py**

In `src/lackpy/infer/hints.py`, add after the existing `enrich_errors` function:

```python
def enrich_errors_with_distillation(
    errors: list[str],
    namespace_desc: str,
    distilled_hint: str | None,
) -> list[str]:
    """Augment errors with a distilled hint if available, else fall back to pattern matching."""
    if distilled_hint:
        enriched = list(errors)
        enriched.append("--- Correction (from observed patterns) ---")
        enriched.append(distilled_hint)
        return enriched
    return enrich_errors(errors, namespace_desc)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_distill.py::TestCorrectionChainDistillation -v`
Expected: PASS (both tests)

- [ ] **Step 5: Update the correction chain to use distilled hints**

In `src/lackpy/infer/correction.py`, replace lines 142-176 (the kibitzer hint enrichment block inside Strategy 2) with:

```python
# Strategy 2: Few-shot correction via provider.generate() with error feedback
if provider is not None:
    from .hints import enrich_errors_with_distillation

    distilled_hint = None

    if kibitzer_session is not None:
        get_hints = getattr(kibitzer_session, "get_correction_hints", None)
        if get_hints is not None:
            try:
                from .failure_modes import classify_failure
                failure_mode = classify_failure(
                    gate_passed=False,
                    gate_errors=validation.errors,
                    exec_error=None,
                    sanitized_program=cleaned,
                )
                model_name = getattr(provider, "_model", None)
                signal = get_hints(
                    failure_mode=failure_mode,
                    model=model_name,
                    attempt=len(self.attempts),
                )
                if isinstance(signal, dict):
                    # Prefer distilled hint from callback
                    distilled = signal.get("distilled")
                    if distilled:
                        distilled_hint = distilled.get("hint")

                    # Escalation for repeated failures
                    history = signal.get("history")
                    if isinstance(history, dict) and history.get("count", 0) >= 3:
                        suffix = (
                            f" IMPORTANT: This model has failed this way "
                            f"{history['count']} times before."
                        )
                        if distilled_hint:
                            distilled_hint += suffix
                        else:
                            distilled_hint = suffix

                    # Known fix from Kibitzer (legacy path)
                    if not distilled_hint:
                        known_fix = signal.get("fix")
                        if known_fix:
                            distilled_hint = known_fix
            except Exception:
                pass

    enriched = enrich_errors_with_distillation(
        validation.errors, namespace_desc, distilled_hint,
    )
    raw = await provider.generate(
        intent, namespace_desc, error_feedback=enriched
    )
```

The rest of the method (lines 180-228) stays the same.

- [ ] **Step 6: Run full test suite**

Run: `python -m pytest tests/ --ignore=tests/eval --ignore=tests/interpreters/test_ast_select.py -q`
Expected: All tests pass

- [ ] **Step 7: Commit**

```bash
git add src/lackpy/infer/hints.py src/lackpy/infer/correction.py
git commit -m "feat: use distilled hints in correction chain"
```

---

## Task 7: Pass doc references through the correction chain context

**Files:**
- Modify: `src/lackpy/infer/correction.py:95-105` (signature)
- Modify: `src/lackpy/infer/dispatch.py` (caller)
- Modify: `src/lackpy/service.py` (delegate, where correction is invoked)
- Test: `tests/test_distill.py` (extend)

Currently the correction chain doesn't know which docs are available for the kit. We need to pass the `ResolvedKit.docs` and tool doc refs so the distillation callback has them when invoked from `get_correction_hints`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_distill.py (append)

class TestCorrectionChainDocContext:
    def test_correction_chain_receives_tool_docs(self):
        """The correction chain should have access to tool doc references."""
        from lackpy.infer.correction import CorrectionChain
        import inspect
        sig = inspect.signature(CorrectionChain.correct)
        assert "tool_docs" in sig.parameters
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_distill.py::TestCorrectionChainDocContext -v`
Expected: FAIL with `AssertionError`

- [ ] **Step 3: Add `tool_docs` parameter to CorrectionChain.correct**

In `src/lackpy/infer/correction.py`, update the `correct` method signature (line 95-105):

```python
async def correct(
    self,
    program: str,
    errors: list[str],
    namespace_desc: str,
    intent: str,
    allowed_names: set[str],
    provider=None,
    extra_rules: list | None = None,
    kibitzer_session=None,
    tool_docs: dict[str, str] | None = None,
) -> CorrectionResult | None:
```

Then in the kibitzer block where we build the `DistillContext` via `get_correction_hints`, we can now populate the `tool_docs` field. Update the `get_hints` call site to also pass tool context by updating the `DistillContext` construction in the Kibitzer callback. Since the callback is already registered at the session level, we enrich the `get_correction_hints` result with doc context here in the correction chain:

After `signal = get_hints(...)`, before checking `distilled`, add:

```python
# Enrich the distilled context with doc refs if callback
# didn't have them at session level
if isinstance(signal, dict) and not signal.get("distilled"):
    # Try direct distillation with full context
    distill_cb = getattr(kibitzer_session, "_distill_callback", None)
    if distill_cb and tool_docs:
        try:
            from kibitzer.distill import DistillContext as DC
            rich_ctx = DC(
                failure_mode=failure_mode,
                errors=validation.errors,
                program=cleaned,
                intent=intent,
                tool_docs=tool_docs or {},
                kit_docs=[],
                model=model_name,
                history_count=(
                    signal.get("history", {}).get("count")
                    if isinstance(signal.get("history"), dict) else None
                ),
                attempt=len(self.attempts),
            )
            rich_result = distill_cb(rich_ctx)
            if rich_result.hint:
                signal["distilled"] = {
                    "hint": rich_result.hint,
                    "docs_used": rich_result.docs_used,
                }
        except Exception:
            pass
```

- [ ] **Step 4: Update callers to pass tool_docs**

In `src/lackpy/infer/dispatch.py`, find where `CorrectionChain.correct()` is called and add `tool_docs`:

Search for `chain.correct(` and add `tool_docs={name: spec.docs for name, spec in resolved.tools.items() if spec.docs}` using the resolved kit available in scope. (The exact line depends on how the dispatcher calls the chain — check the actual call site.)

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ --ignore=tests/eval --ignore=tests/interpreters/test_ast_select.py -q`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add src/lackpy/infer/correction.py src/lackpy/infer/dispatch.py
git commit -m "feat: pass tool doc refs through correction chain for rich distillation"
```

---

## Task 8: Update documentation

**Files:**
- Modify: `docs/concepts/kits.md`

- [ ] **Step 1: Add a section on distillation to the Kibitzer integration docs**

In `docs/concepts/kits.md`, append to the "Kibitzer integration" subsection at the end of the "Tool documentation" section:

```markdown
### Distillation callback

When Kibitzer needs to produce correction hints (during `get_prompt_hints` or `get_correction_hints`), it passes raw failure context through a distillation callback that lackpy provides at session init. The callback:

1. Receives a `DistillContext` with the failure mode, errors, program, intent, and tool/kit doc references
2. Reads only the relevant tool doc (e.g., the signature block for the tool that was misused)
3. Returns a `DistillResult` with a concise hint (1-3 sentences) and a list of doc paths that were consulted

This keeps Kibitzer's output sized for small models — instead of dumping all failure patterns into the prompt, only the targeted correction reaches the model.

The callback is injected via `KibitzerSession.register_distill_callback()`, which accepts any callable matching the `DistillCallback` protocol. Kibitzer never imports lackpy; it just calls the function it was given.

```python
# Automatic — happens in LackpyService._init_kibitzer():
from lackpy.infer.distill import build_distill_callback
callback = build_distill_callback(service.resolve_doc)
kibitzer_session.register_distill_callback(callback)
```
```

- [ ] **Step 2: Commit**

```bash
git add docs/concepts/kits.md
git commit -m "docs: distillation callback architecture"
```

---

## Summary

| Task | Package | What it does |
|------|---------|-------------|
| 1 | kibitzer | Define `DistillCallback` protocol, `DistillContext`, `DistillResult` |
| 2 | kibitzer | Wire callback into `KibitzerSession.get_correction_hints` |
| 3 | kibitzer | Wire callback into `KibitzerSession.get_prompt_hints` |
| 4 | lackpy | Implement rule-based distillation callback with doc resolution |
| 5 | lackpy | Wire callback into `LackpyService._init_kibitzer` |
| 6 | lackpy | Use distilled hints in correction chain + FewShotCorrectStep |
| 7 | lackpy | Pass doc references through correction chain for rich context |
| 8 | lackpy | Update documentation |
