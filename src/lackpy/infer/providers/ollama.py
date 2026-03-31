"""Tier 2: Ollama local model inference."""

from __future__ import annotations

from ..prompt import build_system_prompt


class OllamaProvider:
    def __init__(self, host: str = "http://localhost:11434", model: str = "qwen2.5-coder:1.5b",
                 temperature: float = 0.2, keep_alive: str = "30m") -> None:
        self._host = host
        self._model = model
        self._temperature = temperature
        self._keep_alive = keep_alive

    @property
    def name(self) -> str:
        return "ollama"

    def available(self) -> bool:
        try:
            import ollama  # noqa: F401
            return True
        except ImportError:
            return False

    async def _chat(self, messages: list[dict], **kwargs) -> dict:
        import ollama
        client = ollama.AsyncClient(host=self._host)
        return await client.chat(
            model=self._model, messages=messages,
            options={"temperature": self._temperature},
            keep_alive=self._keep_alive, **kwargs,
        )

    async def generate(self, intent: str, namespace_desc: str,
                       config: dict | None = None, error_feedback: list[str] | None = None) -> str | None:
        if not self.available():
            return None
        system = build_system_prompt(namespace_desc)
        user_msg = intent
        if error_feedback:
            user_msg += "\n\nPrevious attempt had these errors, please fix:\n" + "\n".join(f"- {e}" for e in error_feedback)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user_msg}]
        try:
            response = await self._chat(messages)
            content = response["message"]["content"]
            return content.strip() if content else None
        except Exception:
            return None
