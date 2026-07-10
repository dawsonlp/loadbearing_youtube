import pytest

from loadbearing_youtube.providers import (
    discover,
    get_provider,
    registered_providers,
)
from loadbearing_youtube.providers.base import ProviderError


def test_builtin_providers_registered():
    names = set(registered_providers())
    assert {"ollama", "openai", "anthropic"} <= names


def test_unknown_provider_raises():
    with pytest.raises(ProviderError):
        get_provider("does-not-exist")


def test_discover_shape():
    report = discover()
    names = {p["name"] for p in report}
    assert {"ollama", "openai", "anthropic"} <= names
    for p in report:
        assert set(p) == {"name", "available", "default_model", "models"}
        assert isinstance(p["available"], bool)


def test_cloud_providers_unconfigured_without_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from loadbearing_youtube.providers.openai_provider import OpenAIProvider
    from loadbearing_youtube.providers.anthropic_provider import AnthropicProvider

    assert OpenAIProvider.is_available() is False
    assert AnthropicProvider.is_available() is False
