"""Retrieval-augmented example selection for inference prompts.

Small language models (1.5B-7B) lose accuracy when prompts contain too many
examples — attention dilutes and the model drops parts of the target pattern.
This module selects the most relevant examples for a given intent, keeping
the prompt focused.

The retrieval is keyword-based with synonym expansion. It doesn't require
embeddings or an external index — just tags attached to examples at
registration time.

See: docs/superpowers/notes/retrieval-beats-stuffing.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


# Synonym groups — expansion maps user intent words to the tags used
# on examples. Keep short; these are hints, not a full ontology.
_SYNONYMS: dict[str, set[str]] = {
    "without": {"not", "without", "missing"},
    "don't": {"not", "without"},
    "no": {"not", "without"},
    "missing": {"not", "missing", "without"},
    "static": {"static", "self"},
    "nested": {"nested", "descendant", "inside"},
    "inside": {"nested", "descendant", "inside"},
    "inner": {"nested", "inside"},
    "constructor": {"constructor", "init"},
    "__init__": {"constructor", "init"},
    "super": {"super", "constructor"},
    "async": {"async"},
    "call": {"call", "containment"},
    "calls": {"call", "containment"},
    "calling": {"call", "containment"},
    "contain": {"containment"},
    "contains": {"containment"},
    "test": {"test"},
    "assert": {"test", "assert"},
    "sql": {"sql", "select"},
    "select": {"sql", "select"},
    "fetch": {"call", "fetch"},
    "try": {"try", "error"},
    "except": {"try", "error"},
    "error": {"try", "error"},
    "exception": {"try", "error"},
    "class": {"class"},
    "method": {"method", "class"},
    "function": {"function"},
    "loop": {"loop"},
    "return": {"return"},
    "import": {"import"},
    "variable": {"variable"},
    "var": {"variable"},
    "string": {"string", "str"},
    "hardcoded": {"hardcoded", "string"},
    "url": {"url", "http"},
    "http": {"url", "http"},
    "password": {"password", "secret"},
    "secret": {"password", "secret"},
    "name": {"name"},
    "prefix": {"prefix", "name"},
    "starts": {"prefix", "name"},
    "property": {"property"},
    "attribute": {"attribute"},
}


@dataclass
class Example:
    """A tagged example for retrieval-augmented prompting.

    Attributes:
        intent: Natural language description of what the example demonstrates.
        code: The code this example produces (typically a lackpy program or tool call).
        tags: Keywords used to match this example against user queries.
    """

    intent: str
    code: str
    tags: set[str] = field(default_factory=set)


def expand_intent_keywords(intent: str) -> set[str]:
    """Tokenize an intent and expand tokens using the synonym map.

    Lowercases and splits on whitespace, then for each word adds any
    synonyms from the expansion map. Substring matches also apply so
    ``don't`` inside a larger string still contributes its synonyms.
    """
    words: set[str] = set()
    lower = intent.lower()
    tokens = lower.replace("(", " ").replace(")", " ").replace(",", " ").split()

    for tok in tokens:
        # Strip common punctuation
        clean = tok.strip(".,;:'\"!?")
        words.add(clean)

    # Substring-based expansion — catches "don't" even if stuck to punctuation
    for trigger, synonyms in _SYNONYMS.items():
        if trigger in lower:
            words |= synonyms

    return words


def retrieve_examples(
    intent: str,
    pool: Iterable[Example],
    n: int = 6,
    min_score: int = 1,
) -> list[Example]:
    """Select the top-N examples most relevant to an intent.

    Scoring: number of tags on each example that overlap with the
    intent's expanded keyword set. Ties preserve original pool order.
    Only examples with ``score >= min_score`` are returned.

    Args:
        intent: Natural language query.
        pool: Candidate examples (typically drawn from tool specs or a
            workspace-level bank).
        n: Maximum number of examples to return.
        min_score: Minimum overlap score to include. Default 1 means
            examples with zero keyword overlap are excluded entirely.
            Set to 0 to include unrelated examples as fallbacks when
            nothing matches.

    Returns:
        Up to n examples with score >= min_score, ranked by descending
        overlap score.
    """
    intent_words = expand_intent_keywords(intent)
    scored: list[tuple[int, int, Example]] = []
    for order, ex in enumerate(pool):
        score = len(ex.tags & intent_words)
        if score >= min_score:
            scored.append((score, order, ex))

    # Sort by -score, then by original order (stable for ties)
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [ex for _score, _order, ex in scored[:n]]


def format_examples_for_prompt(examples: list[Example]) -> str:
    """Render selected examples as a prompt section.

    Format: one line per example, ``intent -> code``. This layout tested
    well with qwen2.5-coder models and keeps the prompt compact.
    """
    if not examples:
        return ""
    lines = ["Relevant examples:"]
    for ex in examples:
        lines.append(f"  {ex.intent} -> {ex.code}")
    return "\n".join(lines)
