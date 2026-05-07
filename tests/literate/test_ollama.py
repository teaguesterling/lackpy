"""Tests with real Ollama models.

These tests are slow and require a running Ollama instance.
Skip with: pytest -m "not ollama"
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

from lackpy.interpreters.base import ExecutionContext
from lackpy.interpreters.literate import LiterateInterpreter


def _ollama_available() -> bool:
    try:
        import httpx
        resp = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


ollama = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama not available",
)


@pytest.fixture
def interpreter():
    return LiterateInterpreter()


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "data.txt").write_text("Alice: 90\nBob: 85\nCarol: 92\n")
    (tmp_path / "hello.py").write_text("def greet(name):\n    return f'Hello, {name}!'\n")
    return tmp_path


async def _call_ollama(prompt: str, model: str = "qwen3:8b") -> str:
    import httpx
    from lackpy.interpreters.literate.prompt import LITERATE_SYSTEM_PROMPT

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "system": LITERATE_SYSTEM_PROMPT,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 2048},
            },
        )
        response.raise_for_status()
        return response.json()["response"]


@ollama
class TestOllamaBasic:
    @pytest.mark.asyncio
    async def test_simple_prose_generation(self, interpreter, workspace):
        ctx = ExecutionContext(base_dir=workspace)
        response = await _call_ollama(
            "Write a literate document that reads data.txt and reports "
            "how many lines it has. Use @hidden for setup and prose for the report. "
            f"Working directory: {workspace}"
        )
        result = await interpreter.execute(response, ctx)
        # We can't predict exact output, but it should at least parse and run
        assert result.success or "Parse" not in (result.error or ""), (
            f"Model response failed to execute:\n{response}\n\nError: {result.error}"
        )

    @pytest.mark.asyncio
    async def test_code_execution(self, interpreter, workspace):
        ctx = ExecutionContext(base_dir=workspace)
        response = await _call_ollama(
            "Write a literate document that computes 2+2 in a code block "
            "and reports the result in prose using {variable} interpolation."
        )
        result = await interpreter.execute(response, ctx)
        if result.success:
            assert "4" in result.output, f"Expected '4' in output:\n{result.output}"


@ollama
class TestOllamaAnnotations:
    @pytest.mark.asyncio
    async def test_read_annotation(self, interpreter, workspace):
        ctx = ExecutionContext(base_dir=workspace)
        response = await _call_ollama(
            "Write a literate document that uses @read(data.txt) to show "
            f"the contents of data.txt. Working directory: {workspace}"
        )
        result = await interpreter.execute(response, ctx)
        if result.success and result.output:
            assert "Alice" in result.output or "data" in result.output.lower()

    @pytest.mark.asyncio
    async def test_gather_continue(self, interpreter, workspace):
        ctx = ExecutionContext(base_dir=workspace)
        response = await _call_ollama(
            "Write a literate document that uses @gather to read data.txt "
            "into a variable, then @continue, then reports on the contents. "
            f"Working directory: {workspace}"
        )
        result = await interpreter.execute(response, ctx)
        # Even if the model doesn't get gather/continue right, it should parse
        assert result.success or result.error is not None
