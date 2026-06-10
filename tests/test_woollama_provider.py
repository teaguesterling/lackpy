"""WoollamaProvider — lackpy program generation with the model call delegated to
woollama.core. Mocks woollama.core.complete so the provider's prompt/few-shot
logic and routing are exercised without a live model."""
from __future__ import annotations

import pytest

from lackpy.infer.providers.woollama import WoollamaProvider


@pytest.fixture
def calls(monkeypatch):
    import woollama.core as wc

    recorded: list = []

    async def fake_complete(model, messages, **kw):
        recorded.append({"model": model, "messages": messages, "kw": kw})
        return "  kernel.select('x')  "

    monkeypatch.setattr(wc, "complete", fake_complete)
    return recorded


def test_available():
    assert WoollamaProvider().name == "woollama"
    assert WoollamaProvider().available() is True   # woollama.core is importable


async def test_generate_routes_model_and_params(calls):
    p = WoollamaProvider(model="ollama/qwen2.5-coder:1.5b", temperature=0.2)
    out = await p.generate("count rows", namespace_desc="kernel.select(x)")
    assert out == "kernel.select('x')"              # stripped
    call = calls[-1]
    assert call["model"] == "ollama/qwen2.5-coder:1.5b"
    assert call["kw"]["params"] == {"temperature": 0.2}
    assert [m["role"] for m in call["messages"]] == ["system", "user"]
    assert "kernel.select(x)" in call["messages"][0]["content"]
    assert call["messages"][1]["content"] == "count rows"


async def test_retry_is_few_shot_at_higher_temperature(calls):
    p = WoollamaProvider(temperature=0.2, retry_temperature=0.6)
    await p.generate("count rows", "ns")            # seed _last_output
    await p.generate("count rows", "ns", error_feedback=["use the kernel namespace"])
    call = calls[-1]
    assert call["kw"]["params"] == {"temperature": 0.6}
    assert [m["role"] for m in call["messages"]] == ["system", "user", "assistant", "user"]


async def test_per_call_key_and_base_url_forwarded(calls):
    p = WoollamaProvider(model="openai/gpt-x", api_key="sk-x", base_url="http://proxy/v1")
    await p.generate("x", "ns")
    call = calls[-1]
    assert call["kw"]["api_key"] == "sk-x" and call["kw"]["base_url"] == "http://proxy/v1"
