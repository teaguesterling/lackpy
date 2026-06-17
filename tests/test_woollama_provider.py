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


async def test_options_and_params_forwarded(calls):
    # options (ollama-native, e.g. num_ctx) + params (OpenAI top-level) flow through;
    # the provider's temperature is merged into params and wins.
    p = WoollamaProvider(model="ollama/m", temperature=0.2,
                         options={"num_ctx": 8192}, params={"max_tokens": 256})
    await p.generate("x", "ns")
    call = calls[-1]
    assert call["kw"]["options"] == {"num_ctx": 8192}
    assert call["kw"]["params"] == {"max_tokens": 256, "temperature": 0.2}


async def test_complete_accepts_lackpy_kwargs():
    """Guard the real woollama.core.complete contract lackpy depends on.

    woollama-core is pre-1.0 and fast-moving, AND it's a compiled (Rust/pyo3)
    extension — so `inspect`-based signature checks don't work and the mocked tests
    above (which use a `**kw`-swallowing fake) would miss a breaking change. Instead
    we CALL the real function exactly as the provider does, against an unreachable
    endpoint: it must return an awaitable and fail on TRANSPORT, not reject a kwarg
    with TypeError. Runs in CI (woollama-core is a core dependency); fails loudly on
    drift in the (model, messages, options, params, api_key, base_url) contract.
    """
    import inspect

    core = pytest.importorskip("woollama.core")
    coro = core.complete(
        "ollama/contract-probe", [{"role": "user", "content": "x"}],
        options={"num_ctx": 8}, params={"temperature": 0},
        api_key=None, base_url="http://127.0.0.1:9/v1",   # unreachable → transport error
    )
    assert inspect.isawaitable(coro), "complete(...) must return an awaitable (lackpy awaits it)"
    with pytest.raises(Exception) as excinfo:
        await coro
    assert not isinstance(excinfo.value, TypeError), (
        f"woollama.core.complete rejected a kwarg lackpy passes: {excinfo.value}"
    )


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
