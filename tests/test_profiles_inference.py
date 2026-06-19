"""Phase 2 — per-profile inference wiring (RFC 0002 incr. 5).

The mechanism that lets a profile select a per-task model/temperature: ``generate``/
``delegate`` take ``model``/``temperature`` overrides, the woollama LLM tier is cloned
per call when one is set (omitted → the global list, unchanged — invariant 8), and the
active model is recorded explicitly (no ``getattr(provider, "_model")`` reflection).
"""
from __future__ import annotations

import pytest

from lackpy.config import LackpyConfig
from lackpy.infer.providers.woollama import WoollamaProvider
from lackpy.profiles import Profile, resolve_profile
from lackpy.service import LackpyService


def test_with_overrides_clones_model_temp_preserves_rest():
    p = WoollamaProvider(model="ollama/a", temperature=0.2, retry_temperature=0.7,
                         api_key="k", base_url="u", options={"num_ctx": 8192},
                         params={"max_tokens": 64})
    q = p.with_overrides(model="ollama/b")
    assert q._model == "ollama/b"
    assert q._temperature == 0.2 and q._retry_temperature == 0.7   # preserved
    assert q._api_key == "k" and q._base_url == "u"
    assert q._options == {"num_ctx": 8192} and q._params == {"max_tokens": 64}
    # temperature-only override keeps the model
    r = p.with_overrides(temperature=0.9)
    assert r._model == "ollama/a" and r._temperature == 0.9


@pytest.fixture
def svc(tmp_path):
    cfg = LackpyConfig(
        inference_order=["llm"],
        inference_providers={"llm": {"plugin": "woollama", "model": "ollama/test-model"}},
        config_dir=tmp_path / ".lackpy",
    )
    return LackpyService(workspace=tmp_path, config=cfg)


def test_llm_model_recorded_at_init_no_reflection(svc):
    # The configured model is captured explicitly — what PolicyContext now reads.
    assert svc._llm_model == "ollama/test-model"


def test_providers_for_identity_when_no_override(svc):
    # Invariant 8: no override → the exact same global list, zero allocation.
    assert svc._providers_for(None, None) is svc._inference_providers


def test_providers_for_overrides_only_the_llm_tier(svc):
    out = svc._providers_for("ollama/other", 0.5)
    assert out is not svc._inference_providers
    woollama = [p for p in out if p.name == "woollama"]
    assert woollama and woollama[0]._model == "ollama/other" and woollama[0]._temperature == 0.5
    # deterministic tiers still present and untouched
    assert {"templates", "rules"} <= {p.name for p in out}


def test_profile_model_feeds_the_override(svc, tmp_path):
    # End-to-end of the Phase 1 → Phase 2 seam: a profile's resolved model is exactly
    # what generate()/delegate() would pass as the per-call override.
    rp = resolve_profile(Profile(tools="none", model="ollama/from-profile", temperature=0.4), svc.toolbox)
    assert rp.model == "ollama/from-profile" and rp.temperature == 0.4
    overridden = svc._providers_for(rp.model, rp.temperature)
    llm = [p for p in overridden if p.name == "woollama"][0]
    assert llm._model == "ollama/from-profile" and llm._temperature == 0.4
