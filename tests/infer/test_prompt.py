"""Tests for prompt construction."""

from lackpy.infer.prompt import build_system_prompt


def test_includes_namespace():
    prompt = build_system_prompt(namespace_desc="  read_file(path) -> str: Read file contents")
    assert "read_file(path)" in prompt
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


def test_includes_orchestration_guidance():
    prompt = build_system_prompt(namespace_desc="")
    assert "orchestrate tools" in prompt

def test_includes_read_guidance():
    prompt = build_system_prompt(namespace_desc="")
    assert "read_file(path)" in prompt

def test_no_forbidden_list():
    prompt = build_system_prompt(namespace_desc="")
    assert "Not available" not in prompt
