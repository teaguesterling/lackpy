"""Tests for prompt construction."""

from lackpy.infer.prompt import build_system_prompt


def test_includes_namespace():
    prompt = build_system_prompt(namespace_desc="  read(path) -> str: Read file contents")
    assert "read(path)" in prompt
    assert "Kernel namespace:" in prompt


def test_includes_builtins():
    prompt = build_system_prompt(namespace_desc="")
    assert "len" in prompt
    assert "print" in prompt


def test_includes_params_when_provided():
    prompt = build_system_prompt(namespace_desc="", params_desc="  schema: str — the config schema")
    assert "schema: str" in prompt
    assert "Pre-set variables" in prompt


def test_excludes_params_section_when_empty():
    prompt = build_system_prompt(namespace_desc="")
    assert "Pre-set variables" not in prompt


def test_includes_restrictions():
    prompt = build_system_prompt(namespace_desc="")
    assert "Not available" in prompt
