"""Pluggable transcript sources.

A source knows how to turn a URL into a :class:`~loadbearing.models.Transcript`.
YouTube is the only one implemented today, but the registry keeps the door open
for podcasts, Vimeo, local files, etc.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..models import Transcript


class TranscriptError(RuntimeError):
    """Raised when a transcript cannot be retrieved."""


@runtime_checkable
class TranscriptSource(Protocol):
    name: str

    def matches(self, url: str) -> bool:
        """Return True if this source can handle ``url``."""

    def fetch(self, url: str, languages: list[str]) -> Transcript:
        """Fetch a transcript for ``url``."""


_SOURCES: list[TranscriptSource] = []


def register_source(source: TranscriptSource) -> TranscriptSource:
    _SOURCES.append(source)
    return source


def resolve_source(url: str) -> TranscriptSource:
    for source in _SOURCES:
        if source.matches(url):
            return source
    raise TranscriptError(f"No transcript source knows how to handle: {url}")


def fetch_transcript(url: str, languages: list[str] | None = None) -> Transcript:
    """Front door: pick the right source and fetch."""
    return resolve_source(url).fetch(url, languages or ["en"])
