"""OpenAI provider (optional). Requires ``pip install openai`` and OPENAI_API_KEY."""

from __future__ import annotations

import os

from .base import LLMProvider, LLMResponse, ProviderError


class OpenAIProvider(LLMProvider):
    name = "openai"
    default_model = "gpt-4o-mini"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        super().__init__(model or os.getenv("OPENAI_MODEL"))
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ProviderError("OPENAI_API_KEY is not set.")
        try:
            import openai
        except ImportError as exc:
            raise ProviderError("Install the OpenAI extra: pip install 'loadbearing[openai]'") from exc
        self._client = openai.OpenAI(api_key=self.api_key)

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 2000,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(**kwargs)
        return LLMResponse(
            text=(resp.choices[0].message.content or "").strip(),
            provider=self.name,
            model=self.model,
        )

    @classmethod
    def is_available(cls) -> bool:
        if not os.getenv("OPENAI_API_KEY"):
            return False
        try:
            import openai  # noqa: F401
            return True
        except ImportError:
            return False

    @classmethod
    def list_models(cls) -> list[str]:
        if not os.getenv("OPENAI_API_KEY"):
            return []
        try:
            import openai

            client = openai.OpenAI()
            return sorted(m.id for m in client.models.list() if "gpt" in m.id)
        except Exception:
            return ["gpt-4o", "gpt-4o-mini"]
