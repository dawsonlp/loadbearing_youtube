"""Ollama provider — zero extra dependencies, talks to the local daemon over
its HTTP API. This is the default provider and works fully offline."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .base import LLMProvider, LLMResponse, ProviderError


def _host() -> str:
    return os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def _post(path: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        _host() + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def _get(path: str, timeout: float) -> dict:
    req = urllib.request.Request(_host() + path)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


class OllamaProvider(LLMProvider):
    name = "ollama"
    default_model = "llama3.2"

    def __init__(self, model: str | None = None):
        super().__init__(model or os.getenv("OLLAMA_MODEL"))

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 2000,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> LLMResponse:
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"
        try:
            data = _post("/api/generate", payload, timeout=600)
        except urllib.error.URLError as exc:
            raise ProviderError(
                f"Could not reach Ollama at {_host()}: {exc}. Is `ollama serve` running?"
            ) from exc
        return LLMResponse(text=data.get("response", "").strip(), provider=self.name, model=self.model)

    @classmethod
    def is_available(cls) -> bool:
        try:
            _get("/api/tags", timeout=1.0)
            return True
        except Exception:
            return False

    @classmethod
    def list_models(cls) -> list[str]:
        try:
            data = _get("/api/tags", timeout=2.0)
            return sorted(m["name"] for m in data.get("models", []))
        except Exception:
            return []
