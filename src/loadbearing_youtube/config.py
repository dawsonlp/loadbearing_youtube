"""Runtime configuration, resolved from environment (and an optional .env)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _load_dotenv() -> None:
    """Load a .env if python-dotenv is installed; silently skip otherwise."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass


@dataclass
class Settings:
    provider: str = "ollama"
    model: str | None = None
    languages: list[str] = field(default_factory=lambda: ["en"])
    max_chars: int = 12000
    max_tokens: int = 8000

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        langs = os.getenv("LOADBEARING_LANGUAGES", "en")
        return cls(
            provider=os.getenv("LOADBEARING_PROVIDER", "ollama").lower(),
            model=os.getenv("LOADBEARING_MODEL") or None,
            languages=[x.strip() for x in langs.split(",") if x.strip()],
            max_chars=int(os.getenv("LOADBEARING_MAX_CHARS", "12000")),
            max_tokens=int(os.getenv("LOADBEARING_MAX_TOKENS", "8000")),
        )
