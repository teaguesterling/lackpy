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


# ── Retrieval-augmented prompting ──

from lackpy.infer.retrieval import Example
from lackpy.infer.prompt import collect_example_pool
from lackpy.kit.toolbox import ToolSpec


def test_no_examples_without_pool():
    prompt = build_system_prompt(namespace_desc="", intent="find async functions")
    assert "Relevant examples" not in prompt


def test_no_examples_without_intent():
    pool = [Example("async functions", ".func:async", {"async"})]
    prompt = build_system_prompt(namespace_desc="", example_pool=pool)
    assert "Relevant examples" not in prompt


def test_includes_retrieved_examples():
    pool = [
        Example("async functions", ".func:async", {"async"}),
        Example("nested functions", ".func .func", {"nested"}),
        Example("constructors", ".class .func#__init__", {"constructor"}),
    ]
    prompt = build_system_prompt(
        namespace_desc="",
        intent="find async functions",
        example_pool=pool,
    )
    assert "Relevant examples" in prompt
    assert ".func:async" in prompt


def test_retrieval_filters_by_relevance():
    pool = [
        Example("async functions", ".func:async", {"async"}),
        Example("sql strings", ".str[peek*=SELECT]", {"sql", "select"}),
    ]
    prompt = build_system_prompt(
        namespace_desc="",
        intent="find async functions",
        example_pool=pool,
    )
    assert ".func:async" in prompt
    # Unrelated example should not appear when min_score=1 filters it out
    assert ".str[peek*=SELECT]" not in prompt


def test_collect_example_pool_from_tool_specs():
    specs = [
        ToolSpec(
            name="tool_a",
            provider="builtin",
            examples=[
                {"intent": "do A", "code": "a()", "tags": ["a"]},
                {"intent": "do A twice", "code": "a(); a()", "tags": ["a"]},
            ],
        ),
        ToolSpec(
            name="tool_b",
            provider="builtin",
            examples=[
                {"intent": "do B", "code": "b()", "tags": ["b"]},
            ],
        ),
        ToolSpec(name="tool_c", provider="builtin"),  # no examples
    ]
    pool = collect_example_pool(specs)
    assert len(pool) == 3
    assert any(ex.code == "a()" for ex in pool)
    assert any(ex.code == "b()" for ex in pool)


def test_collect_skips_invalid_examples():
    specs = [
        ToolSpec(
            name="tool_a",
            provider="builtin",
            examples=[
                {"intent": "good", "code": "a()", "tags": []},
                {"intent": "", "code": "b()"},  # empty intent, skipped
                {"code": "c()"},  # missing intent, skipped
            ],
        ),
    ]
    pool = collect_example_pool(specs)
    assert len(pool) == 1
    assert pool[0].code == "a()"
