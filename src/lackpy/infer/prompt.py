"""System prompt construction for inference providers.

The default prompt teaches the lackpy language (stepy100): the pre-prompt, the
style guide, the skeleton, then the task framing. That is the structure of the
eval prompt that measured Qwen3-30B-A3B-Instruct at 27/30 valid
(scripts/eval/ste_usability/prompts.py with PROMPT_VARIANT=pre; FINDINGS.md).
"""

from __future__ import annotations

from typing import Any

from ..lang.grammar import ALLOWED_BUILTINS
from ..lang.spec import format_preprompt, format_style_guide
from .retrieval import (Example, expand_intent_keywords,
                        format_examples_for_prompt, retrieve_examples)

# The skeleton every generated program starts from. Same text as the measured
# eval's SKELETON (scripts/eval/ste_usability/prompts.py); keep the two equal.
SKELETON = '''# /// intent
# goal   = "<the task, one sentence>"
# reads  = []
# writes = []
# output = "none"          # none | printed | value
# ///

# imports. Approved: pathlib (Path), json, re, collections, math,
#          statistics, hashlib, datetime

# implementation
'''

INTRO = """\
You write lackpy programs. A lackpy program is plain Python 3 that follows the
lackpy style guide below. The validator rejects any program that breaks a rule."""

_PROMPT = """\
{intro}

=== LESSON ===
{lesson}
=== END LESSON ===

=== STYLE GUIDE ===
{guide}
=== END STYLE GUIDE ===

Start from this skeleton and fill in the intent block:

{skeleton}
{params_section}{examples_section}\
The program runs with the workspace directory as its current directory.
The next message is the task.

Reply with the complete program in one ```python fenced block and nothing else.
"""

_PARAMS_SECTION = """\
The program takes parameters. Put this literal at the top level, after the
imports and before any use, exactly as shown. Read a value as params["{first}"].
Do not assign to params again.

{params_desc}

"""

# Interpreters for other languages (CSS selectors, SQL) keep their own framing.
_SPECIALIZED_TEMPLATE = """\
{interpreter_hint}

{namespace_desc}

Builtins: {builtins_list}
{params_section}{examples_section}"""


def build_system_prompt(
    namespace_desc: str,
    params_desc: str | None = None,
    intent: str | None = None,
    example_pool: list[Example] | None = None,
    n_examples: int = 6,
    interpreter: Any = None,
) -> str:
    """Build the system prompt for inference providers.

    For a lackpy program (no interpreter, or the ``python`` interpreter) the
    prompt is: the intro (the interpreter's ``system_prompt_hint()`` when it
    has one), ``format_preprompt()``, ``format_style_guide()``, the skeleton,
    the params literal when params are given, retrieved examples, and the
    task framing. ``namespace_desc`` is not taught: no run binds a tool since
    the kit was removed (stepy100 Plan 2, Task 7).

    An interpreter of another language (its ``name`` is not ``"python"``) with
    a ``system_prompt_hint()`` keeps the specialized framing: its hint, the
    namespace description and the builtins.

    Args:
        namespace_desc: Tool namespace description; used only by the
            specialized (non-Python) framing.
        params_desc: Output of :func:`format_params_description`, or None.
        intent: Natural language intent. Required for example retrieval.
        example_pool: Candidate examples to retrieve from. If None or empty,
            no examples section is added.
        n_examples: Maximum number of examples to include.
        interpreter: An interpreter instance, or None.

    Returns:
        The complete system prompt string ready to send to an inference provider.
    """
    examples_section = ""
    if intent and example_pool:
        selected = retrieve_examples(intent, example_pool, n=n_examples)
        if selected:
            examples_section = format_examples_for_prompt(selected) + "\n\n"

    hint = None
    if interpreter is not None:
        hint_fn = getattr(interpreter, "system_prompt_hint", None)
        if hint_fn is not None:
            hint = hint_fn()

    if hint and getattr(interpreter, "name", "python") != "python":
        params_section = ""
        if params_desc:
            params_section = f"\nParameters:\n{params_desc}\n\n"
        return _SPECIALIZED_TEMPLATE.format(
            interpreter_hint=hint,
            namespace_desc=namespace_desc,
            builtins_list=", ".join(sorted(ALLOWED_BUILTINS)),
            params_section=params_section,
            examples_section="\n" + examples_section if examples_section else "",
        )

    params_section = ""
    if params_desc:
        # The second line is "# <first key>: <type>" (format_params_description).
        lines = params_desc.splitlines()
        key = lines[1].removeprefix("# ").rsplit(": ", 1)[0] if len(lines) > 1 else "name"
        params_section = _PARAMS_SECTION.format(first=key, params_desc=params_desc)
    return _PROMPT.format(
        intro=hint or INTRO,
        lesson=format_preprompt().rstrip(),
        guide=format_style_guide().rstrip(),
        skeleton=SKELETON,
        params_section=params_section,
        examples_section=examples_section,
    )


def collect_example_pool(tool_specs: list) -> list[Example]:
    """Gather all examples from a list of ToolSpec objects into a flat pool.

    Each tool may contribute zero or more examples. Returns an empty list if
    none are defined. Example dicts are converted to Example objects; entries
    missing required fields are skipped.

    **Tags default to the example's own intent words.** ``retrieve_examples``
    scores by tag overlap and drops anything scoring zero, so an untagged example
    is silently unreachable no matter how relevant it is — it is accepted, stored
    on the spec, and never selected. Hand-tagging is reasonable for examples
    registered in Python; it is a trap for examples declared in config, where the
    omission is invisible. Deriving tags from the intent makes the common case
    work and an explicit ``tags`` list still wins.
    """
    pool: list[Example] = []
    for spec in tool_specs:
        for ex in getattr(spec, "examples", None) or []:
            intent = ex.get("intent", "")
            code = ex.get("code", "")
            if not intent or not code:
                continue
            tags = set(ex.get("tags") or expand_intent_keywords(intent))
            pool.append(Example(intent=intent, code=code, tags=tags))
    return pool


def format_params_description(params: dict) -> str:
    """Describe the program's ``params`` literal for the prompt.

    The first line is the literal itself, ``params = {...}``, built from the
    passed values with ``repr``. The runner replaces the values of that literal
    with the passed ones (stepy100 Task 5), so the program reads each one as
    ``params["name"]``. One ``# name: type`` comment line per key follows,
    so a model that copies the block still writes valid Python.

    A value of the single-key form ``{"value": x}`` is unwrapped, as the
    service does (review M7); any other dict is a dict parameter.

    Args:
        params: Mapping of parameter name to value. Each value must have a
            literal form (the runner's rule).

    Returns:
        A newline-joined description, starting with the literal line.
    """
    values = {name: (v["value"] if isinstance(v, dict) and v.keys() == {"value"} else v)
              for name, v in params.items()}
    lines = [f"params = {values!r}"]
    lines += [f"# {name}: {type(v).__name__}" for name, v in values.items()]
    return "\n".join(lines)
