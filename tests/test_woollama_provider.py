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


def test_woollama_plugin_registered_from_config(tmp_path):
    """The `woollama` plugin wires a WoollamaProvider into the service's inference
    tiers (config-driven, alongside the deterministic templates/rules tiers)."""
    from lackpy.config import LackpyConfig
    from lackpy.service import LackpyService

    cfg = LackpyConfig(
        inference_order=["woollama"],
        inference_providers={"woollama": {
            "plugin": "woollama", "model": "ollama/qwen2.5-coder:1.5b",
            "temperature": 0.1}},
    )
    svc = LackpyService(workspace=tmp_path, config=cfg)
    woollama_tiers = [p for p in svc._inference_providers
                      if type(p).__name__ == "WoollamaProvider"]
    assert len(woollama_tiers) == 1
    assert woollama_tiers[0]._model == "ollama/qwen2.5-coder:1.5b"
    assert woollama_tiers[0]._temperature == 0.1


@pytest.mark.parametrize("retired", ["ollama", "anthropic"])
def test_retired_plugins_register_no_provider(tmp_path, caplog, retired):
    """The per-vendor `ollama`/`anthropic` plugins were retired in the woollama
    consolidation: they register no inference tier and warn with the migration."""
    import logging

    from lackpy.config import LackpyConfig
    from lackpy.service import LackpyService

    cfg = LackpyConfig(
        inference_order=[retired],
        inference_providers={retired: {"plugin": retired, "model": "m"}},
    )
    with caplog.at_level(logging.WARNING):
        svc = LackpyService(workspace=tmp_path, config=cfg)

    # Only the deterministic tiers remain — no model provider was registered.
    names = {p.name for p in svc._inference_providers}
    assert names == {"templates", "rules"}
    assert "woollama" in caplog.text and retired in caplog.text
