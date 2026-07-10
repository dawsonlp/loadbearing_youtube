"""Anthropic provider (optional). Requires ``pip install anthropic`` and ANTHROPIC_API_KEY."""

from __future__ import annotations

import os

from .base import LLMProvider, LLMResponse, ProviderError

# Sensible fallbacks used only when live discovery is unavailable.
_KNOWN_MODELS = [
    "claude-3-5-haiku-latest",
    "claude-3-5-sonnet-latest",
    "claude-3-7-sonnet-latest",
]


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    default_model = "claude-3-5-haiku-latest"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        super().__init__(model or os.getenv("ANTHROPIC_MODEL"))
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not set.")
        try:
            import anthropic
        except ImportError as exc:
            raise ProviderError("Install the Anthropic extra: pip install 'loadbearing[anthropic]'") from exc
        self._client = anthropic.Anthropic(api_key=self.api_key)

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 2000,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> LLMResponse:
        # Anthropic has no dedicated JSON mode; nudge via the prompt instead.
        user = prompt
        if json_mode:
            user = prompt + "\n\nRespond with a single valid JSON object and nothing else."

        kwargs = dict(
            model=self.model,
            system=system or "",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": user}],
        )

        def _run(with_temperature: bool) -> str:
            call = dict(kwargs)
            if with_temperature:
                call["temperature"] = temperature
            # Stream and accumulate: with a large max_tokens the SDK refuses a
            # non-streaming call (it may exceed the 10-minute limit), so we
            # always stream and reassemble the final message.
            with self._client.messages.stream(**call) as stream:
                message = stream.get_final_message()
            return "".join(b.text for b in message.content if b.type == "text").strip()

        try:
            text = _run(with_temperature=True)
        except Exception as exc:
            # Newer models (e.g. claude-sonnet-5) deprecate `temperature` and
            # reject the request. Retry once without it rather than failing.
            if "temperature" in str(exc).lower():
                text = _run(with_temperature=False)
            else:
                raise
        return LLMResponse(text=text, provider=self.name, model=self.model)

    @classmethod
    def is_available(cls) -> bool:
        if not os.getenv("ANTHROPIC_API_KEY"):
            return False
        try:
            import anthropic  # noqa: F401
            return True
        except ImportError:
            return False

    @classmethod
    def list_models(cls) -> list[str]:
        if not os.getenv("ANTHROPIC_API_KEY"):
            return []
        try:
            import anthropic

            client = anthropic.Anthropic()
            return sorted(m.id for m in client.models.list())
        except Exception:
            return _KNOWN_MODELS
