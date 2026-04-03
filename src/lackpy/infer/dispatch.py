"""Priority-ordered inference dispatch across provider plugins."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..lang.validator import validate
from .cleanup import deterministic_cleanup
from .correction import CorrectionChain
from .sanitize import sanitize_output


@dataclass
class GenerationResult:
    """Result of generating a lackpy program from an intent.

    Attributes:
        program: The generated and validated lackpy program source.
        provider_name: Name of the provider that produced the program.
        generation_time_ms: Total wall-clock time for generation in milliseconds.
        correction_strategy: The correction strategy used, or None if no correction needed.
        correction_attempts: Number of correction attempts made (0 if none needed).
        attempts_log: Full log of correction attempts, or None if no correction was run.
    """

    program: str
    provider_name: str
    generation_time_ms: float
    correction_strategy: str | None = None
    correction_attempts: int = 0
    attempts_log: list | None = None


class InferenceDispatcher:
    """Priority-ordered dispatcher that tries inference providers in sequence.

    Iterates through providers in order, attempts generation with each available
    provider, and returns the first result that passes validation. If the first
    attempt from a provider fails validation, the CorrectionChain is invoked
    (deterministic cleanup, few-shot retry, fresh fixer) before moving on.

    Args:
        providers: Ordered list of provider plugin instances. Each must have
            ``name``, ``available()``, and ``generate()`` attributes.
    """

    def __init__(self, providers: list[Any]) -> None:
        self._providers = providers

    def get_provider(self) -> Any:
        """Return the first available provider."""
        for provider in self._providers:
            if provider.available():
                return provider
        raise RuntimeError("No inference providers available")

    def get_providers(self) -> list:
        """Return all available providers in priority order."""
        return [p for p in self._providers if p.available()]

    async def generate(self, intent: str, namespace_desc: str, allowed_names: set[str],
                       params_desc: str | None = None, extra_rules: list | None = None) -> GenerationResult:
        """Generate a valid lackpy program from a natural language intent.

        Tries each available provider in priority order, validating the output
        after each attempt. On validation failure, the CorrectionChain is invoked
        (deterministic cleanup → few-shot retry → fresh fixer) before moving to
        the next provider.

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
            program = deterministic_cleanup(program)
            validation = validate(program, allowed_names=allowed_names, extra_rules=extra_rules)
            if validation.valid:
                elapsed = (time.perf_counter() - start) * 1000
                return GenerationResult(program=program, provider_name=provider.name, generation_time_ms=elapsed)

            errors_by_provider[provider.name] = validation.errors

            chain = CorrectionChain()
            correction = await chain.correct(
                program=program,
                errors=validation.errors,
                namespace_desc=namespace_desc,
                intent=intent,
                allowed_names=allowed_names,
                provider=provider,
                extra_rules=extra_rules,
            )
            if correction is not None:
                elapsed = (time.perf_counter() - start) * 1000
                return GenerationResult(
                    program=correction.program,
                    provider_name=provider.name,
                    generation_time_ms=elapsed,
                    correction_strategy=correction.strategy,
                    correction_attempts=correction.attempts,
                    attempts_log=chain.attempts,
                )

            errors_by_provider[provider.name] = (
                chain.attempts[-1].errors if chain.attempts else validation.errors
            )

        provider_names = [p.name for p in self._providers if p.available()]
        raise RuntimeError(
            f"All {len(provider_names)} providers failed to produce a valid program. "
            f"Tried: {', '.join(provider_names)}. Last errors: {errors_by_provider}"
        )
