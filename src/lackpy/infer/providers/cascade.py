"""Cascade inference — compound model with fast-path + fallback.

Tries models in speed order using raw completion mode (not chat template).
First model to produce a valid program wins. Validation is mechanical
(AST parse + sandbox check), not model-based.

Raw completion mode uses the ollama `/api/generate` endpoint with
`raw: true`, bypassing the model's chat template. The prompt is
formatted as pattern-completion examples:

    find all functions -> find_names('src/**/*.py', '.fn')
    find all classes -> find_names('src/**/*.py', '.class')
    {user_intent} ->

This produces 5-10x better results than chat-template prompting for
small code models (tested across qwen2.5-coder 0.5b-7b, granite,
codegemma). Chat-template models explain; completion-mode models
generate.

Configuration::

    [inference.providers.cascade]
    plugin = "cascade"
    host = "http://localhost:11435"
    tiers = [
        {model = "qwen2.5-coder:3b", max_tokens = 80, timeout = 15},
        {model = "qwen2.5-coder:7b", max_tokens = 120, timeout = 60},
    ]
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any

from ...lang.validator import validate
from ..cleanup import deterministic_cleanup
from ..sanitize import sanitize_output


@dataclass
class Tier:
    """One model in the cascade."""
    model: str
    max_tokens: int = 80
    timeout: int = 15


# Default few-shot examples — pattern-completion format.
# These are the core patterns that teach the model the tool-calling shape.
# Additional examples come from ToolSpec.examples via the namespace_desc
# parameter, but these bootstraps ensure the model always sees the format.
_DEFAULT_EXAMPLES = """\
find all function names -> find_names('src/**/*.py', '.fn')
find all classes -> find_names('src/**/*.py', '.class')
show me the main function -> view('src/**/*.py', '.fn#main')
find methods of the Auth class -> find_names('src/auth.py', '.class#Auth .fn')
count functions in cli.py -> n = len(find_names('squackit/cli.py', '.fn'))
find the most complex functions -> complexity('src/**/*.py', '.fn')
read lines 1-20 of server.py -> read_source('squackit/server.py', '1-20')
get an overview of the codebase -> explore()
tell me about validate_token -> investigate('validate_token')
review changes since main -> review('main', 'HEAD')
search for cache across the codebase -> search('cache')
show docs about authentication -> doc_outline('docs/**/*.md', search='authentication')
show recent commits -> recent_changes(10)
what files changed since main -> file_changes('main', 'HEAD')
find functions starting with test_ -> source = "tests/**/*.py"
selector = ".fn[name^='test_']"
result = find_names(source, selector)
result"""


def _build_completion_prompt(intent: str, examples: str | None = None,
                             context: str | None = None) -> str:
    """Build a completion-style prompt the model just finishes.

    If *context* is provided (typically namespace_desc from the kit),
    it is prepended as reference material the model can consult.
    This is the integration point for tool documentation (e.g. selector
    syntax, parameter descriptions) and Kibitzer hints.
    """
    parts = []
    if context:
        parts.append(context.strip())
        parts.append("")
    parts.append(examples or _DEFAULT_EXAMPLES)
    parts.append(f"{intent} ->")
    return "\n".join(parts)


def _extract_code(raw: str) -> str:
    """Extract code from a model's raw completion output.

    The model should produce code immediately after `->`. But it may
    also emit markdown fences, explanation prose, or continuation
    examples. We take everything up to the first stop signal.
    """
    lines = []
    for line in raw.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            break
        if "->" in stripped and lines:
            break
        if stripped.startswith(("To ", "Here", "This ", "Note", "You can", "# ")):
            break
        if stripped:
            lines.append(stripped)
        elif lines:
            break
    return "\n".join(lines)


def _query_raw(host: str, model: str, prompt: str,
               max_tokens: int, timeout: int,
               temperature: float = 0.2) -> tuple[str, int]:
    """Query ollama in raw completion mode. Returns (text, token_count)."""
    data = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "raw": True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": 2048,
            "stop": ["\n\n"],
        },
    }).encode()
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    return body.get("response", ""), body.get("eval_count", 0)


class CascadeProvider:
    """Compound inference provider that tries models in speed order.

    Each tier is a (model, max_tokens, timeout) tuple. The cascade
    generates with each tier's model in raw completion mode, validates
    the output mechanically, and returns the first valid result.

    If no tier produces valid output, returns None and lets the
    dispatcher try the next provider (or correction chain).
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        tiers: list[dict] | None = None,
        examples: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._host = host
        self._tiers = [
            Tier(**t) if isinstance(t, dict) else t
            for t in (tiers or [
                {"model": "qwen2.5-coder:3b", "max_tokens": 80, "timeout": 15},
                {"model": "qwen2.5-coder:7b", "max_tokens": 120, "timeout": 60},
            ])
        ]
        self._examples = examples
        self._last_attempts: list[dict] = []

    @property
    def name(self) -> str:
        return "cascade"

    def available(self) -> bool:
        """Check if at least one tier's model is reachable."""
        try:
            with urllib.request.urlopen(
                f"{self._host}/api/tags", timeout=5
            ) as resp:
                body = json.loads(resp.read())
            available_models = {m["name"] for m in body.get("models", [])}
            return any(t.model in available_models for t in self._tiers)
        except Exception:
            return False

    async def generate(
        self,
        intent: str,
        namespace_desc: str,
        config: dict | None = None,
        error_feedback: list[str] | None = None,
        system_prompt_override: str | None = None,
        interpreter: object | None = None,
    ) -> str | None:
        """Try each tier in order. First valid program wins.

        Uses raw completion mode — bypasses chat template entirely.
        The namespace_desc is currently unused (the completion prompt
        has its own examples). Future: extract tool names from
        namespace_desc to build dynamic examples.

        The error_feedback parameter is also unused — correction is
        handled by trying the next tier, not by retrying the same
        model with feedback.
        """
        # Build the allowed names set from namespace_desc for validation.
        # Parse tool signatures like "  find_names(source: str, ...) -> ..."
        import re
        allowed = set()
        for line in namespace_desc.split("\n"):
            m = re.match(r"\s*(\w+)\(", line)
            if m:
                allowed.add(m.group(1))

        prompt = _build_completion_prompt(intent, self._examples, context=namespace_desc)
        self._last_attempts = []

        for tier in self._tiers:
            try:
                raw, tokens = _query_raw(
                    self._host, tier.model, prompt,
                    tier.max_tokens, tier.timeout,
                )
            except Exception as e:
                self._last_attempts.append({
                    "model": tier.model, "error": str(e),
                    "code": "", "valid": False,
                })
                continue

            # Extract and clean
            code = _extract_code(raw)
            code = sanitize_output(code)
            code = deterministic_cleanup(code)

            # Validate
            validation = validate(code, allowed_names=allowed)
            self._last_attempts.append({
                "model": tier.model, "code": code,
                "valid": validation.valid,
                "errors": validation.errors if not validation.valid else [],
                "tokens": tokens,
            })

            if validation.valid:
                return code

        # All tiers failed
        return None

    @property
    def last_attempts(self) -> list[dict]:
        """Log of cascade attempts from the last generate() call.

        Useful for debugging which tier resolved (or why all failed).
        """
        return list(self._last_attempts)
