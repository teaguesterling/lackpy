"""Phase 3 — profiles wired into the service (RFC 0002 incr. 5).

`profile=` replaces `kit=` across the service; a named `[profiles.<name>]` table resolves
its tools *and* inference through the service, and the kit_* methods become profile_*.
"""
from __future__ import annotations

import pytest

from lackpy.config import LackpyConfig
from lackpy.service import LackpyService


@pytest.fixture
def svc(tmp_path):
    cfg = LackpyConfig(
        inference_order=["llm"],
        inference_providers={"llm": {"plugin": "woollama", "model": "ollama/base-model"}},
        profiles={
            "reader": {"tools": ["read_file", "find_files"]},
            "careful": {"tools": ["read_file"], "model": "ollama/big-model", "mode": "spm"},
        },
        config_dir=tmp_path / ".lackpy",
    )
    return LackpyService(workspace=tmp_path, config=cfg)


def test_named_profile_resolves_its_tools(svc):
    rp = svc._resolve_profile("reader")
    assert set(rp.tools.tools) == {"read_file", "find_files"}
    assert rp.model is None  # 'reader' omits inference → falls back downstream


def test_named_profile_carries_inference_into_the_override(svc):
    rp = svc._resolve_profile("careful")
    assert rp.model == "ollama/big-model" and rp.mode == "spm"
    # the model feeds the per-call provider override (Phase 2 seam)
    llm = [p for p in svc._providers_for(rp.model, rp.temperature) if p.name == "woollama"][0]
    assert llm._model == "ollama/big-model"


def test_bare_tool_list_is_the_degenerate_profile(svc):
    # 'a kit is a profile': a bare list resolves identically to a named tools-only profile.
    assert set(svc._resolve_profile(["read_file"]).tools.tools) == {"read_file"}


def test_profile_info_renamed_and_works(svc):
    info = svc.profile_info("reader")
    assert set(info["tools"]) == {"read_file", "find_files"}
    assert "grade" in info and "description" in info


def test_profile_create_list_roundtrip(svc):
    svc.profile_create("custom", ["read_file"], description="just read")
    names = {p["name"] for p in svc.profile_list()}
    assert "custom" in names
    assert set(svc.profile_info("custom").get("tools", {})) == {"read_file"}


def test_get_config_uses_profile_default(svc):
    cfg = svc.get_config()
    assert "profile_default" in cfg and "kit_default" not in cfg


def test_old_kit_kwarg_is_gone(svc):
    # Hard removal, no alias: kit= is no longer accepted.
    with pytest.raises(TypeError):
        svc.validate("x = 1", kit=["read_file"])
