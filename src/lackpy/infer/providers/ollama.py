"""Tier 2: Ollama local model inference."""

from __future__ import annotations

from ..prompt import build_system_prompt


class OllamaProvider:
    def __init__(self, host: str = "http://localhost:11434", model: str = "qwen2.5-coder:1.5b",
                 temperature: float = 0.2, retry_temperature: float = 0.6,
                 keep_alive: str = "30m") -> None:
        self._host = host
        self._model = model
        self._temperature = temperature
        self._retry_temperature = retry_temperature
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

    async def _chat(self, messages: list[dict], temperature: float | None = None, **kwargs) -> dict:
        import ollama
        client = ollama.AsyncClient(host=self._host)
        return await client.chat(
            model=self._model, messages=messages,
            options={"temperature": temperature or self._temperature},
            keep_alive=self._keep_alive, **kwargs,
        )

    async def generate(self, intent: str, namespace_desc: str,
                       config: dict | None = None, error_feedback: list[str] | None = None,
                       system_prompt_override: str | None = None,
                       interpreter: object | None = None) -> str | None:
        if not self.available():
            return None
        system = system_prompt_override or build_system_prompt(namespace_desc, interpreter=interpreter)

        is_retry = error_feedback and self._last_output
        if is_retry:
            # Few-shot error correction: show the model its bad output
            # and the correction as a conversation, not appended text.
            # Higher temperature helps break out of the same pattern.
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": intent},
                {"role": "assistant", "content": self._last_output},
                {"role": "user", "content": (
                    "That code won't work in this environment. "
                    + " ".join(
                        h for h in error_feedback
                        if h != "--- Suggestions ---"
                    )
                    + " Rewrite using only the kernel namespace."
                )},
            ]
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": intent},
            ]

        try:
            response = await self._chat(
                messages,
                temperature=self._retry_temperature if is_retry else None,
            )
            content = response["message"]["content"]
            self._last_output = content.strip() if content else None
            return self._last_output
        except Exception:
            self._last_output = None
            return None

    _last_output: str | None = None
