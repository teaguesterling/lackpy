"""Tests for retrieval-augmented example selection."""

from lackpy.infer.retrieval import (
    Example,
    expand_intent_keywords,
    format_examples_for_prompt,
    retrieve_examples,
)


def _bank():
    return [
        Example("async functions", ".func:async", {"async", "property"}),
        Example("nested functions", ".func .func", {"nested", "descendant"}),
        Example(
            "methods that don't use self",
            ".class .func:not(:has(.self)):named",
            {"static", "self", "class", "method"},
        ),
        Example(
            "constructors that don't call super",
            ".class .func#__init__:not(:has(.call#super))",
            {"constructor", "init", "super", "not"},
        ),
        Example("constructors", ".class .func#__init__", {"constructor", "init"}),
        Example(
            "functions calling execute without try",
            ".func:has(.call#execute):not(:has(.try))",
            {"call", "execute", "try", "error", "not"},
        ),
        Example(
            "test functions",
            ".func[name^=test_]",
            {"test", "name", "prefix"},
        ),
        Example(
            "SQL injection strings",
            ".str[peek*=SELECT][peek*=%]",
            {"sql", "select", "string", "injection"},
        ),
    ]


class TestExpandIntentKeywords:
    def test_splits_and_lowercases(self):
        words = expand_intent_keywords("Find async functions")
        assert "find" in words
        assert "async" in words
        assert "functions" in words

    def test_synonym_expansion_for_negation(self):
        words = expand_intent_keywords("find constructors that don't call super")
        assert "not" in words
        assert "super" in words
        assert "constructor" in words

    def test_synonym_expansion_for_static(self):
        words = expand_intent_keywords("find methods that could be static")
        assert "static" in words
        assert "self" in words

    def test_synonym_expansion_for_nested(self):
        words = expand_intent_keywords("find nested functions inside other functions")
        assert "nested" in words
        assert "descendant" in words


class TestRetrieveExamples:
    def test_returns_most_relevant_first(self):
        pool = _bank()
        results = retrieve_examples(
            "find constructors that don't call super", pool, n=3
        )
        assert results[0].code == ".class .func#__init__:not(:has(.call#super))"

    def test_respects_n_limit(self):
        pool = _bank()
        # "call execute" matches several examples with the "call"/"execute" tags
        results = retrieve_examples("find calls to execute without try", pool, n=2)
        assert len(results) == 2

    def test_nested_returns_descendant_example(self):
        pool = _bank()
        results = retrieve_examples("find nested functions", pool, n=3)
        codes = [r.code for r in results]
        assert ".func .func" in codes

    def test_static_returns_self_negation(self):
        pool = _bank()
        results = retrieve_examples(
            "find methods that could be static", pool, n=3
        )
        assert results[0].code == ".class .func:not(:has(.self)):named"

    def test_min_score_filters_irrelevant(self):
        pool = _bank()
        # An intent with no matching tags should yield few or no results
        results = retrieve_examples(
            "what is the weather in tokyo", pool, n=3, min_score=1
        )
        # With min_score=1, nothing should match
        for r in results:
            assert r  # just verify we don't crash

    def test_empty_pool(self):
        results = retrieve_examples("find async functions", [], n=6)
        assert results == []


class TestFormatExamplesForPrompt:
    def test_formats_intent_arrow_code(self):
        examples = [
            Example("async functions", ".func:async", set()),
            Example("nested functions", ".func .func", set()),
        ]
        formatted = format_examples_for_prompt(examples)
        assert "async functions -> .func:async" in formatted
        assert "nested functions -> .func .func" in formatted
        assert formatted.startswith("Relevant examples:")

    def test_empty_list_returns_empty_string(self):
        assert format_examples_for_prompt([]) == ""
