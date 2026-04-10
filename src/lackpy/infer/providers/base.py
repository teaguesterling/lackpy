"""Base protocol for inference providers."""

from __future__ import annotations

from typing import Protocol


class InferenceProvider(Protocol):
    @property
    def name(self) -> str: ...
    def available(self) -> bool: ...
    async def generate(
        self, intent: str, namespace_desc: str,
        config: dict | None = None, error_feedback: list[str] | None = None,
        system_prompt_override: str | None = None,
    ) -> str | None:
        """Generate a program from intent.

        Args:
            intent: Natural language description of what to generate.
            namespace_desc: Formatted tool namespace string.
            config: Provider-specific configuration.
            error_feedback: Optional error feedback for retry attempts.
            system_prompt_override: If provided, use this exact string as
                the system prompt instead of building one from namespace_desc.
                Used by steps that need retrieval-augmented or otherwise
                pre-built prompts.
        """
        ...
