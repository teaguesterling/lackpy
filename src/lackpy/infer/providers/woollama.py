"""woollama-backed inference: delegate the model call to ``woollama.core``.

lackpy keeps everything that makes it lackpy — prompt construction, the few-shot
error-correction conversation, and the dispatcher / validation / CorrectionChain
around generation. It only stops doing its OWN per-provider model HTTP: the raw
"send these messages, get text" call goes to :func:`woollama.core.complete`,
which owns provider/model routing, config, transport, ollama-native ``num_ctx``,
and per-call key/base-url overrides.

One ``model`` string of the form ``"<provider>/<model>"`` (e.g.
``"ollama/qwen2.5-coder:1.5b"`` or ``"anthropic/claude-haiku-4-5"``) reaches every
woollama-known backend — so lackpy gets multi-provider model management without
maintaining a provider per vendor. Implements the same ``InferenceProvider``
protocol (``name`` / ``available`` / ``generate``) as the other tiers.

Sampling and context knobs flow through from config: ``options`` carries
ollama-native fields (notably ``num_ctx``, which routes the turn to the native
``/api/chat`` endpoint that honors it), and ``params`` carries OpenAI top-level
fields (e.g. ``max_tokens``, ``top_p``). ``temperature`` is managed here (a higher
``retry_temperature`` is used on a correction pass).
"""
from __future__ import annotations

import logging

from ..prompt import build_system_prompt

logger = logging.getLogger(__name__)


class WoollamaProvider:
    def __init__(self, model: str = "ollama/qwen2.5-coder:1.5b",
                 temperature: float = 0.2, retry_temperature: float = 0.6,
                 api_key: str | None = None, base_url: str | None = None,
                 options: dict | None = None, params: dict | None = None) -> None:
        self._model = model
        self._temperature = temperature
        self._retry_temperature = retry_temperature
        self._api_key = api_key
        self._base_url = base_url
        # ollama-native knobs (e.g. num_ctx — setting it routes ollama/* to the
        # native /api/chat path); and extra OpenAI top-level fields (e.g. max_tokens).
        self._options = options
        self._params = params

    @property
    def name(self) -> str:
        return "woollama"

    def available(self) -> bool:
        try:
            import woollama.core  # noqa: F401
            return True
        except ImportError:
            return False

    async def generate(self, intent: str, namespace_desc: str,
                       config: dict | None = None, error_feedback: list[str] | None = None,
                       system_prompt_override: str | None = None,
                       interpreter: object | None = None) -> str | None:
        if not self.available():
            return None
        from woollama.core import complete

        system = system_prompt_override or build_system_prompt(
            namespace_desc, interpreter=interpreter)

        is_retry = error_feedback and self._last_output
        if is_retry:
            # Few-shot error correction: show the model its bad output and the
            # correction as a conversation (mirrors the OllamaProvider tier).
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": intent},
                {"role": "assistant", "content": self._last_output},
                {"role": "user", "content": (
                    "That code won't work in this environment. "
                    + " ".join(h for h in error_feedback if h != "--- Suggestions ---")
                    + " Rewrite using only the kernel namespace.")},
            ]
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": intent},
            ]

        temperature = self._retry_temperature if is_retry else self._temperature
        # temperature (with retry override) wins over any temperature in config params.
        call_params = {**(self._params or {}), "temperature": temperature}
        try:
            content = await complete(
                self._model, messages,
                options=self._options, params=call_params,
                api_key=self._api_key, base_url=self._base_url)
            self._last_output = content.strip() if content else None
            return self._last_output
        except Exception as e:
            # Returning None hands control to the next tier, but the *reason* must
            # not vanish: a misconfigured backend (bad model string, missing key,
            # unreachable host) otherwise surfaces only as the dispatcher's generic
            # "all providers failed". woollama raises InferenceError with a useful
            # kind/status/message; log it so the failure is diagnosable.
            logger.warning("woollama.core.complete failed (model=%r): %s: %s",
                           self._model, type(e).__name__, e)
            self._last_output = None
            return None

    _last_output: str | None = None
