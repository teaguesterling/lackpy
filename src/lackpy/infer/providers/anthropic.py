"""Tier 3: Anthropic API fallback inference."""

from __future__ import annotations

from ..prompt import build_system_prompt


class AnthropicProvider:
    def __init__(self, model: str = "claude-haiku-4-5-20251001", max_tokens: int = 1024) -> None:
        self._model = model
        self._max_tokens = max_tokens

    @property
    def name(self) -> str:
        return "anthropic"

    def available(self) -> bool:
        try:
            import anthropic  # noqa: F401
            return True
        except ImportError:
            return False

    async def _create_message(self, system: str, messages: list[dict]) -> object:
        import anthropic
        client = anthropic.AsyncAnthropic()
        return await client.messages.create(
            model=self._model, max_tokens=self._max_tokens,
            system=system, messages=messages,
        )

    async def generate(self, intent: str, namespace_desc: str,
                       config: dict | None = None, error_feedback: list[str] | None = None,
                       system_prompt_override: str | None = None,
                       interpreter: object | None = None) -> str | None:
        if not self.available():
            return None
        system = system_prompt_override or build_system_prompt(namespace_desc, interpreter=interpreter)
        user_msg = intent
        if error_feedback:
            user_msg += "\n\nPrevious attempt had these errors, please fix:\n" + "\n".join(f"- {e}" for e in error_feedback)
        messages = [{"role": "user", "content": user_msg}]
        try:
            response = await self._create_message(system, messages)
            content = response.content[0].text
            return content.strip() if content else None
        except Exception:
            return None
