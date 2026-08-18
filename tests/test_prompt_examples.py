"""collect_example_pool: examples must be reachable by retrieval, not just stored."""

from lackpy.infer.prompt import collect_example_pool, build_system_prompt


class _Spec:
    def __init__(self, examples):
        self.examples = examples


def test_untagged_examples_are_reachable():
    """Untagged examples default to their own intent words as tags.

    ``retrieve_examples`` scores by tag overlap and drops anything scoring zero,
    so an untagged example was silently unreachable however relevant it was --
    accepted, stored on the spec, and never selected. That is invisible from the
    caller's side, which makes it a trap for config-declared examples in
    particular.
    """
    spec = _Spec([{"intent": "how many errors did the run produce",
                   "code": "len(events(severity='error')['events'])"}])
    pool = collect_example_pool([spec])
    assert pool and pool[0].tags, "an untagged example must still carry tags"

    prompt = build_system_prompt(
        "  events(severity) -> dict: ...", None,
        intent="return the number of error events from the run",
        example_pool=pool)
    assert "events(severity='error')" in prompt, "example did not reach the prompt"


def test_explicit_tags_win():
    spec = _Spec([{"intent": "unrelated wording", "code": "x", "tags": ["sql"]}])
    assert collect_example_pool([spec])[0].tags == {"sql"}


def test_malformed_examples_dropped():
    spec = _Spec([{"intent": "no code"}, {"code": "no intent"}, {}])
    assert collect_example_pool([spec]) == []
