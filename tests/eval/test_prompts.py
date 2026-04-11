"""Tests for prompt variants."""

from scripts.prompt_eval.prompts import (
    PROMPT_VARIANTS,
    get_prompt,
    list_variant_ids,
)


def test_every_interpreter_has_four_variants():
    for interp in ("python", "ast-select", "pss", "plucker"):
        assert interp in PROMPT_VARIANTS
        assert len(PROMPT_VARIANTS[interp]) == 4


def test_variant_ids_are_consistent():
    expected = {"baseline", "specialized", "specialized_fewshot", "specialized_fewshot_constraints"}
    assert set(list_variant_ids()) == expected
    for interp in ("python", "ast-select", "pss", "plucker"):
        assert set(PROMPT_VARIANTS[interp].keys()) == expected


def test_get_prompt_returns_string():
    for interp in ("python", "ast-select", "pss", "plucker"):
        for variant_id in list_variant_ids():
            s = get_prompt(interp, variant_id, namespace_desc="tools: foo")
            assert isinstance(s, str)
            assert len(s) > 50


def test_specialized_prompts_mention_interpreter_language():
    assert "selector" in get_prompt("ast-select", "specialized", "desc").lower()
    pss_prompt = get_prompt("pss", "specialized", "desc").lower()
    assert "sheet" in pss_prompt or "show:" in pss_prompt or "rule" in pss_prompt
    plucker_prompt = get_prompt("plucker", "specialized", "desc").lower()
    assert "source" in plucker_prompt or "chain" in plucker_prompt


def test_constraints_variant_mentions_negatives():
    for interp in ("python", "ast-select", "pss", "plucker"):
        p = get_prompt(interp, "specialized_fewshot_constraints", "desc").lower()
        assert "do not" in p or "never" in p or "no " in p


def test_fewshot_variant_contains_examples():
    """specialized_fewshot should actually have example content beyond what specialized has."""
    for interp in ("python", "ast-select", "pss", "plucker"):
        base_len = len(get_prompt(interp, "specialized", "desc"))
        fewshot_len = len(get_prompt(interp, "specialized_fewshot", "desc"))
        assert fewshot_len > base_len, f"{interp} fewshot variant is not longer than specialized"


def test_get_prompt_raises_on_unknown():
    import pytest
    with pytest.raises(KeyError):
        get_prompt("unknown-interpreter", "baseline", "desc")
    with pytest.raises(KeyError):
        get_prompt("python", "unknown-variant", "desc")
