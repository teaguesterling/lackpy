"""Priority-ordered inference dispatch across provider plugins."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..lang.validator import validate
from .sanitize import sanitize_output


@dataclass
class GenerationResult:
    program: str
    provider_name: str
    generation_time_ms: float


class InferenceDispatcher:
    def __init__(self, providers: list[Any]) -> None:
        self._providers = providers

    async def generate(self, intent: str, namespace_desc: str, allowed_names: set[str],
                       params_desc: str | None = None, extra_rules: list | None = None) -> GenerationResult:
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
            raw = await provider.generate(intent, namespace_desc, error_feedback=validation.errors)
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
