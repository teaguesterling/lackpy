"""Priority-ordered inference dispatch across provider plugins."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..lang.validator import validate
from .hints import enrich_errors
from .sanitize import sanitize_output


@dataclass
class GenerationResult:
    """Result of generating a lackpy program from an intent.

    Attributes:
        program: The generated and validated lackpy program source.
        provider_name: Name of the provider that produced the program.
        generation_time_ms: Total wall-clock time for generation in milliseconds.
    """

    program: str
    provider_name: str
    generation_time_ms: float


class InferenceDispatcher:
    """Priority-ordered dispatcher that tries inference providers in sequence.

    Iterates through providers in order, attempts generation with each available
    provider, and returns the first result that passes validation. If the first
    attempt from a provider fails validation, one retry with enriched error
    feedback is attempted before moving on.

    Args:
        providers: Ordered list of provider plugin instances. Each must have
            ``name``, ``available()``, and ``generate()`` attributes.
    """

    def __init__(self, providers: list[Any]) -> None:
        self._providers = providers

    async def generate(self, intent: str, namespace_desc: str, allowed_names: set[str],
                       params_desc: str | None = None, extra_rules: list | None = None) -> GenerationResult:
        """Generate a valid lackpy program from a natural language intent.

        Tries each available provider in priority order, validating the output
        after each attempt. On validation failure, one retry with enriched error
        feedback is issued before moving to the next provider.

        Args:
            intent: Natural language description of the desired program.
            namespace_desc: Formatted tool namespace string for the prompt.
            allowed_names: Set of allowed callable names for validation.
            params_desc: Optional description of pre-set parameter variables.
            extra_rules: Additional validation rules beyond the core checks.

        Returns:
            A GenerationResult from the first provider that produces valid output.

        Raises:
            RuntimeError: If all available providers fail to produce a valid program.
        """
        start = time.perf_counter()
        errors_by_provider: dict[str, list[str]] = {}

        for provider in self._providers:
            if not provider.available():
                continue

            raw = await provider.generate(intent, namespace_desc)
            if raw is None:
                continue

            program = sanitize_output(raw)
            validation = validate(program, allowed_names=allowed_names, extra_rules=extra_rules)
            if validation.valid:
                elapsed = (time.perf_counter() - start) * 1000
                return GenerationResult(program=program, provider_name=provider.name, generation_time_ms=elapsed)

            errors_by_provider[provider.name] = validation.errors
            enriched = enrich_errors(validation.errors, namespace_desc)
            raw = await provider.generate(intent, namespace_desc, error_feedback=enriched)
            if raw is None:
                continue

            program = sanitize_output(raw)
            validation = validate(program, allowed_names=allowed_names, extra_rules=extra_rules)
            if validation.valid:
                elapsed = (time.perf_counter() - start) * 1000
                return GenerationResult(program=program, provider_name=provider.name, generation_time_ms=elapsed)

            errors_by_provider[provider.name] = validation.errors

        provider_names = [p.name for p in self._providers if p.available()]
        raise RuntimeError(
            f"All {len(provider_names)} providers failed to produce a valid program. "
            f"Tried: {', '.join(provider_names)}. Last errors: {errors_by_provider}"
        )
