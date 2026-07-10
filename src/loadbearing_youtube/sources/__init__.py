"""Transcript sources."""

from .base import (
    TranscriptError,
    TranscriptSource,
    fetch_transcript,
    register_source,
    resolve_source,
)
from . import youtube  # noqa: F401  (registers YouTubeSource on import)

__all__ = [
    "TranscriptError",
    "TranscriptSource",
    "fetch_transcript",
    "register_source",
    "resolve_source",
]
