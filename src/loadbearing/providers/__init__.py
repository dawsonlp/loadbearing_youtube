"""Pluggable, discoverable LLM provider registry.

Register a new backend by subclassing :class:`LLMProvider` and adding a
``@register`` decorator (or calling :func:`register`). Everything else — CLI
listing, discovery, selection — picks it up automatically.
"""

from __future__ import annotations

import os

from .base import LLMProvider, LLMResponse, ProviderError

_REGISTRY: dict[str, type[LLMProvider]] = {}


def register(cls: type[LLMProvider]) -> type[LLMProvider]:
    if not getattr(cls, "name", ""):
        raise ValueError(f"{cls!r} must define a non-empty `name`")
    _REGISTRY[cls.name] = cls
    return cls


def registered_providers() -> dict[str, type[LLMProvider]]:
    return dict(_REGISTRY)


def default_provider_name() -> str:
    return os.getenv("LOADBEARING_PROVIDER", "ollama").lower()


def get_provider(name: str | None = None, model: str | None = None, **kwargs) -> LLMProvider:
    """Instantiate a provider by name (defaults to env/``ollama``)."""
    key = (name or default_provider_name()).lower()
    if key not in _REGISTRY:
        raise ProviderError(
            f"Unknown provider {key!r}. Registered: {', '.join(sorted(_REGISTRY)) or '(none)'}"
        )
    return _REGISTRY[key](model=model, **kwargs)


def discover() -> list[dict]:
    """Report each registered provider's availability and models (for the
    ``providers`` command). Never raises."""
    out = []
    for name, cls in sorted(_REGISTRY.items()):
        available = False
        try:
            available = cls.is_available()
        except Exception:
            available = False
        out.append(
            {
                "name": name,
                "available": available,
                "default_model": cls.default_model,
                "models": cls.list_models() if available else [],
            }
        )
    return out


# Register the built-in providers. Import order defines listing order only.
from .ollama_provider import OllamaProvider  # noqa: E402
from .openai_provider import OpenAIProvider  # noqa: E402
from .anthropic_provider import AnthropicProvider  # noqa: E402

register(OllamaProvider)
register(OpenAIProvider)
register(AnthropicProvider)

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "ProviderError",
    "register",
    "registered_providers",
    "get_provider",
    "discover",
    "default_provider_name",
]
