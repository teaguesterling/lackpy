"""Tests for prompt construction.

stepy100 Task 8: the default prompt teaches the new language. The kit-era
tests pinned "Kernel namespace:", "orchestrate tools" and "read_file(path)";
those are retired with the kit (Plan 2, Task 7).
"""

from lackpy.infer.prompt import SKELETON, build_system_prompt, format_params_description
from lackpy.lang.spec import format_preprompt, format_style_guide

# Names of the retired kit and phrases of the kit-era framing.
KIT_WORDS = ["read_file", "write_file", "edit_file", "find_files", "find_definitions",
             "find_callers", "Kernel namespace", "kernel namespace", "orchestrate",
             "Jupyter", "pre-loaded", "Pre-set variables"]


def test_default_prompt_follows_the_measured_eval_structure():
    """Lesson, then guide, then skeleton, then the task framing: the order of the
    eval prompt that measured Qwen3-30B at 27/30 valid
    (scripts/eval/ste_usability/prompts.py, PROMPT_VARIANT=pre)."""
    prompt = build_system_prompt(namespace_desc="")
    lesson = prompt.index(format_preprompt())
    guide = prompt.index(format_style_guide())
    skeleton = prompt.index(SKELETON)
    framing = prompt.index("```python fenced block")
    assert lesson < guide < skeleton < framing


def test_no_kit_names_in_default_prompt():
    from lackpy.interpreters.python import PythonInterpreter
    for prompt in (build_system_prompt(namespace_desc=""),
                   build_system_prompt(namespace_desc="", interpreter=PythonInterpreter())):
        for word in KIT_WORDS:
            assert word not in prompt, word


def test_namespace_desc_is_not_taught():
    # No run binds a tool, so a namespace description must not reach the model.
    prompt = build_system_prompt(namespace_desc="  read_file(path) -> str: Read file contents")
    assert "read_file" not in prompt


def test_python_interpreter_gets_the_lackpy_prompt():
    from lackpy.interpreters.python import PythonInterpreter
    interp = PythonInterpreter()
    prompt = build_system_prompt(namespace_desc="", interpreter=interp)
    assert prompt.startswith(interp.system_prompt_hint())
    assert format_preprompt() in prompt
    assert format_style_guide() in prompt
    assert SKELETON in prompt


def test_dsl_interpreter_keeps_its_own_framing():
    class _Dsl:
        name = "ast-select"

        def system_prompt_hint(self):
            return "Write one CSS selector."

    prompt = build_system_prompt(namespace_desc="", interpreter=_Dsl())
    assert prompt.startswith("Write one CSS selector.")
    assert format_preprompt() not in prompt


def test_includes_builtins():
    prompt = build_system_prompt(namespace_desc="")
    assert "len" in prompt
    assert "print" in prompt


def test_includes_params_literal_when_provided():
    desc = format_params_description({"prefix": "hello", "limit": 3})
    prompt = build_system_prompt(namespace_desc="", params_desc=desc)
    assert "params = {'prefix': 'hello', 'limit': 3}" in prompt
    assert 'params["prefix"]' in prompt
    # The literal comes after the skeleton, before the framing.
    assert prompt.index(SKELETON) < prompt.index("params = {") < prompt.index("```python fenced block")


def test_excludes_params_section_when_empty():
    prompt = build_system_prompt(namespace_desc="")
    assert "params = {" not in prompt


def test_no_forbidden_list():
    prompt = build_system_prompt(namespace_desc="")
    assert "Not available" not in prompt


class TestFormatParamsDescription:
    def test_describes_the_literal(self):
        desc = format_params_description({"prefix": "hello", "limit": 3})
        assert "params = {'prefix': 'hello', 'limit': 3}" in desc
        assert "prefix: str" in desc
        assert "limit: int" in desc

    def test_the_literal_line_parses_to_the_values(self):
        import ast
        values = {"root": "src", "names": ["a", "b"], "on": True}
        line = next(ln for ln in format_params_description(values).splitlines()
                    if ln.startswith("params = "))
        assert ast.literal_eval(line.removeprefix("params = ")) == values

    def test_value_wrapper_is_unwrapped(self):
        # The single-key {"value": x} wrapper is the one the service accepts (review M7).
        assert "params = {'n': 5}" in format_params_description({"n": {"value": 5}})

    def test_a_dict_value_is_a_dict_param(self):
        # The kit-era {"value", "type", "description"} metadata shape is retired.
        desc = format_params_description({"cfg": {"value": 1, "type": "int"}})
        assert "params = {'cfg': {'value': 1, 'type': 'int'}}" in desc
        assert "cfg: dict" in desc


# ── Retrieval-augmented prompting ──

from lackpy.infer.retrieval import Example
from lackpy.infer.prompt import collect_example_pool
from lackpy.tools.toolbox import ToolSpec


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
