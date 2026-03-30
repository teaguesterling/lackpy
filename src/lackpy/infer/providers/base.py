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
    ) -> str | None: ...
