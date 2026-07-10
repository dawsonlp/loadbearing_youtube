"""LLM provider abstraction.

A provider is a thin, uniform wrapper over one backend's text-completion call.
Providers are pluggable (register with ``@register``), configurable (model and
credentials via args or env), and discoverable (``is_available`` /
``list_models``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str


class ProviderError(RuntimeError):
    """Raised when a provider cannot be constructed or a call fails."""


class LLMProvider(ABC):
    #: registry key, e.g. "ollama"
    name: str = ""
    #: model used when neither arg nor env supplies one
    default_model: str = ""

    def __init__(self, model: str | None = None):
        self.model = model or self.default_model

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 2000,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Return a single completion for ``prompt``."""

    # --- discovery -------------------------------------------------------
    @classmethod
    def is_available(cls) -> bool:
        """Whether this provider is usable right now (creds present, daemon
        reachable, …). Cheap and must never raise."""
        return True

    @classmethod
    def list_models(cls) -> list[str]:
        """Best-effort list of usable model names. Never raises."""
        return []
